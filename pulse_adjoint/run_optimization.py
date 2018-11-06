#!/usr/bin/env python
# c) 2001-2017 Simula Research Laboratory ALL RIGHTS RESERVED
# Authors: Henrik Finsberg
# END-USER LICENSE AGREEMENT
# PLEASE READ THIS DOCUMENT CAREFULLY. By installing or using this
# software you agree with the terms and conditions of this license
# agreement. If you do not accept the terms of this license agreement
# you may not install or use this software.

# Permission to use, copy, modify and distribute any part of this
# software for non-profit educational and research purposes, without
# fee, and without a written agreement is hereby granted, provided
# that the above copyright notice, and this license agreement in its
# entirety appear in all copies. Those desiring to use this software
# for commercial purposes should contact Simula Research Laboratory AS:
# post@simula.no
#
# IN NO EVENT SHALL SIMULA RESEARCH LABORATORY BE LIABLE TO ANY PARTY
# FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,
# INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# "PULSE-ADJOINT" EVEN IF SIMULA RESEARCH LABORATORY HAS BEEN ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE. THE SOFTWARE PROVIDED HEREIN IS
# ON AN "AS IS" BASIS, AND SIMULA RESEARCH LABORATORY HAS NO OBLIGATION
# TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
# SIMULA RESEARCH LABORATORY MAKES NO REPRESENTATIONS AND EXTENDS NO
# WARRANTIES OF ANY KIND, EITHER IMPLIED OR EXPRESSED, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS
import numpy as np
import dolfin
import dolfin_adjoint
from .setup_optimization import (make_mechanics_problem,
                                 get_measurements)

# from pulse.dolfin_utils import MyReducedFunctional, get_constant, read_hdf5

# from .utils import (Text, print_line,
#                     print_head, logger,
#                     get_lv_marker,
#                     UnableToChangePressureExeption)

from .io_utils import (get_simulated_pressure,
                       check_group_exists,
                       contract_point_exists)

from .forward_runner import (ActiveForwardRunner,
                             PassiveForwardRunner)
from .optimization_targets import (RegionalStrainTarget,
                                   Regularization, VolumeTarget)
from pulse import numpy_mpi
from . import config
from pulse import kinematics

from .store_results import write_opt_results_to_h5
from .optimal_control import OptimalControl
from pulse.mechanicsproblem import SolverDidNotConverge


def run_unloaded_optimization(params, patient):

    # Run an inital optimization as we are used to
    params["unload"] = False
    params["h5group"] = "initial_passive"

    unload_params = params["Unloading_parameters"].to_dict()
    estimate_initial_guess = unload_params.pop('estimate_initial_guess', True)

    if params["Patient_parameters"]["geometry_index"] == "-1" \
       and estimate_initial_guess:
        msg = ("You cannot estimate the initial guess when using "
               "end-diastolic geometry as reference.")
        logger.warning(msg)
        estimate_initial_guess = False

    # Interpolation of displacement does not work with dolfin-adjoint
    # and can thus only be used when non-gradient based optimization
    # is used. Interpolation is the only resonable thins to do. Everything else
    # will produce a different volume after invoking ALE.move.
    # Hence, we force interpolation if scalar material parameters
    vol_approx = params["volume_approx"]
    if params["matparams_space"] == "R_0":
        params["volume_approx"] = "interpolate"

    measurements = get_measurements(params, patient)
    problem, active_control, passive_control \
        = make_mechanics_problem(params, patient.geometry)

    if check_group_exists(params["sim_file"], params["h5group"]):

        # Load the initial guess
        h5group = "/".join([params["h5group"],
                            config.PASSIVE_INFLATION_GROUP,
                            "/optimal_control"])
        read_hdf5(h5name=params["sim_file"],
                  func=passive_control, h5group=h5group)
        logger.info(Text.green("Fetch initial guess for material paramters"))

    else:

        if estimate_initial_guess:
            logger.info(Text.blue("\nRun Passive Optimization"))
            rd, passive_control \
                = run_passive_optimization_step(params,
                                                problem,
                                                measurements,
                                                passive_control)
            
            logger.info("\nSolve optimization problem.......")
            params, rd, opt_result = solve_oc_problem(params, rd,
                                                      passive_control,
                                                      return_solution=True,
                                                      store_solution=True)
    params["unload"] = True
    params["h5group"] = ""
    pfd = patient.data.passive_duration
    geo_idx = int(params["Patient_parameters"]["geometry_index"])
    geo_idx = geo_idx if geo_idx >= 0 else pfd-1
    
    if patient.geometry.is_biv:

        pressures = zip(patient.data.pressure[:pfd], patient.data.RVP[:pfd])
        volumes = zip(patient.data.volume[:pfd], patient.data.RVV[:pfd])

    else:
        pressures = patient.data.pressure[:pfd]
        volumes = patient.data.volume[:pfd]

    from unloading import UnloadedMaterial
    estimator \
        = UnloadedMaterial(geo_idx, pressures, volumes,
                           params, passive_control,
                           optimize_matparams=params["optimize_matparams"],
                           **unload_params)

    estimator.unload_material()
    params["volume_approx"] = vol_approx
    params["h5group"] = ""


def run_passive_optimization(params, patient):
    """
    Main function for the passive phase

    :param dict params: adjoin_contraction_parameters
    :param patient: A patient instance
    :type patient: :py:class`patient_data.Patient`

    **Example of usage**::

      # Setup compiler parameters
      setup_general_parameters()
      params = setup_adjoint_contraction_parameter()
      params['phase'] = 'passive_inflation'
      patient = initialize_patient_data(param['Patient_parameters'])
      run_passive_optimization(params, patient)


    """
    logger.info(Text.blue("\nRun Passive Optimization"))

    measurements = get_measurements(params, patient)
    problem, active_control, passive_control \
        = make_mechanics_problem(params, patient.geometry)

    rd, passive_control = run_passive_optimization_step(params,
                                                        problem,
                                                        measurements,
                                                        passive_control)

    logger.info("\nSolve optimization problem.......")
    return solve_oc_problem(params, rd, passive_control)


def run_passive_optimization_step(params,
                                  problem,
                                  measurements,
                                  passive_control):
    """FIXME! briefly describe function
    """
    optimization_targets, bcs \
        = load_targets(params, problem,
                       measurements)

    # Initialize the solver for the Forward problem
    for_run = PassiveForwardRunner(problem,
                                   bcs,
                                   optimization_targets,
                                   params,
                                   passive_control)

    # Update the weights for the functional
    if params["adaptive_weights"]:
        # Solve the forward problem with guess results (just for printing)
        logger.info(Text.blue("\nForward solution at guess parameters"))
        forward_result, _ = for_run(passive_control, False)

        weights = {}
        for k, v in for_run.opt_weights.iteritems():
            weights[k] = v/(10*forward_result["func_value"])
        for_run.opt_weights.update(**weights)
        logger.info("\nUpdate weights for functional")
        logger.info(for_run._print_functional())

    # Stop recording
    logger.debug(Text.yellow("Stop annotating"))
    dolfin.parameters["adjoint"]["stop_annotating"] = True

    # Initialize MyReducedFuctional
    rd = MyReducedFunctional(for_run, passive_control,
                             relax=params["passive_relax"],
                             verbose=params["verbose"])

    return rd, passive_control


def run_active_optimization(params, patient):
    """FIXME! briefly describe function
    """

    logger.info(Text.blue("\nRun Active Optimization"))

    measurements = get_measurements(params, patient)
    problem, active_control, passive_control \
        = make_mechanics_problem(params, patient.geometry)

    # Loop over contract points
    i = 0
    logger.info(("Number of contract points: "
                 "{}").format(patient.data.num_contract_points))

    # We need to use a while loop here because we might fail to converge at
    # some time point. In that case we will interpolate between the previous
    # succeded time point and the failed time point in hope to reach
    # convergence. As as result, the number of contract points may
    # change during the simulation
    while i < patient.data.num_contract_points:
        params["active_contraction_iteration_number"] = i

        if not contract_point_exists(params):

            # Number of times we have interpolated in order
            # to be able to change the pressure
            attempts = 0
            pressure_change = False

            while (not pressure_change and attempts < 8):

                try:
                    rd, gamma = run_active_optimization_step(params,
                                                             problem,
                                                             measurements,
                                                             active_control)
                except UnableToChangePressureExeption:
                    logger.info("Unable to change pressure. Exception caught")
                    measurements = interpolate_data(params, patient, i)
                    attempts += 1

                else:
                    pressure_change = True

                    # If you want to apply a different initial guess than
                    # the pevious value, assign this now and evaluate.

                    active_initial_guess(params, gamma, patient)

                    logger.info("\nSolve optimization problem.......")
                    solve_oc_problem(params, rd, gamma)
                    dolfin_adjoint.adj_reset()

            if not pressure_change:
                raise RuntimeError("Unable to increasure")

        else:

            # Make sure to do interpolation if that was done earlier
            plv = get_simulated_pressure(params)
            if not plv == measurements["pressure"][i+1]:
                measurements = interpolate_data(params, patient, i)
                i -= 1
        i += 1


def interpolate_data(params, patient, i):

    logger.info("Interpolate.")
    patient.data.interpolate_data(i+patient.passive_duration-1)

    # Update the measurements
    measurements = get_measurements(params, patient)
    return measurements


def active_initial_guess(params, gamma, patient=None):

    if params["initial_guess"] == "previous":
        return

    elif params["initial_guess"] == "zero":
        zero = get_constant(gamma.value_size(),
                            gamma.value_rank(), 0.0)

        # Do we need dolfin-adjoint here?
        g = dolfin_adjoint.Function(gamma.function_space(),
                                    name="gamma_initial_guess")
        g.assign(zero)

    elif params["initial_guess"] == "smooth":

        # We find a constant that represents the previous state

        if params["gamma_space"] == "regional":

            # Sum all regional values with weights given by the size
            # of the regions
            meshvols = patient.geometry.meshvols
            meshvol = patient.meshvol
            g_arr = numpy_mpi.gather_broadcast(gamma.vector().array())
            val = sum(np.multiply(g_arr, meshvols))/float(meshvol)
            c = get_constant(gamma.value_size(), gamma.value_rank(), val)

        else:

            # Project the activation parameter onto the real line
            g_proj = dolfin_adjoint.project(gamma,
                                            dolfin.FunctionSpace(patient.
                                                                 geometry.mesh,
                                                                 "R", 0))
            val = numpy_mpi.gather_broadcast(g_proj.vector().array())[0]
            c = get_constant(gamma.value_size(), gamma.value_rank(), val)

        # Do we need dolfin-adjoint here?
        g = dolfin_adjoint.Function(gamma.function_space(), name="gamma2")
        g.assign(c)

    rd(g)


def run_active_optimization_step(params, problem,
                                 measurements, active_control):
    """FIXME! briefly describe function

    """
    acin = params["active_contraction_iteration_number"]
    # Get initial guess for gamma
    if acin == 0:

        zero = get_constant(active_control.value_size(),
                            active_control.value_rank(), 0.0)
        active_control.assign(zero)

    else:

        # Use gamma from the previous point as initial guess
        # Load gamma from previous point
        g_temp = dolfin.Function(active_control.function_space())
        with dolfin.HDF5File(dolfin.mpi_comm_world(),
                             params["sim_file"], "r") as h5file:
            h5file.read(g_temp, ("active_contraction/contract_point_{}"
                                 "/optimal_control")
                        .format(acin-1))

        active_control.assign(g_temp)

    # Load targets
    optimization_targets, bcs \
        = load_targets(params, problem, measurements)

    for_run = ActiveForwardRunner(problem,
                                  bcs,
                                  optimization_targets,
                                  params,
                                  active_control)

    # Update weights so that the initial value of the
    # functional is 0.1
    if params["adaptive_weights"]:
        # Solve the forward problem with guess results (just for printing)
        logger.info(Text.blue("\nForward solution at guess parameters"))
        forward_result, _ = for_run(active_control, False)

        weights = {}
        for k, v in for_run.opt_weights.iteritems():
            weights[k] = v/(10*forward_result["func_value"])
        for_run.opt_weights.update(**weights)
        logger.info("Update weights for functional")
        logger.info(for_run._print_functional())

    # Stop recording
    logger.debug(Text.yellow("Stop annotating"))
    dolfin.parameters["adjoint"]["stop_annotating"] = True

    rd = MyReducedFunctional(for_run, active_control,
                             relax=params["active_relax"],
                             verbose=params["verbose"])

    return rd, active_control


def store(params, rd, opt_result):

    solver = rd.for_run.cphm.problem

    if params["phase"] == config.PHASES[0]:

        h5group = "/".join([params["h5group"],
                            config.PASSIVE_INFLATION_GROUP])
    else:

        h5group = "/".join([params["h5group"],
                            config.ACTIVE_CONTRACTION_GROUP
                            .format(params["active_contraction_iteration_number"])])

    write_opt_results_to_h5(h5group,
                            params,
                            rd.for_res,
                            solver,
                            opt_result)


def solve_oc_problem(params, rd, paramvec,
                     return_solution=False, store_solution=True):
    """Solve the optimal control problem

    :param params: Application parameters
    :param rd: The reduced functional
    :param paramvec: The control parameter(s)

    """

    # Create optimal control problem
    oc_problem = OptimalControl()
    oc_problem.build_problem(params, rd, paramvec)

    x = oc_problem.get_initial_guess()
    nvar = len(x)

    if params["phase"] == config.PHASES[0] and \
       not params["optimize_matparams"]:

        rd(x)
        rd.for_res["initial_control"] = rd.initial_paramvec,
        rd.for_res["optimal_control"] = rd.paramvec

        if store_solution:
            store(params, rd, {})

        if return_solution:
            return params, rd, {}

    else:

        logger.info("\n"+"".center(72, "-"))
        logger.info("Solve optimal contol problem".center(72, "-"))
        logger.info("".center(72, "-"))

        # Some flags
        done = False
        paramvec_start = paramvec.copy(deepcopy=True)
        niter = 0

        par_max = np.max(numpy_mpi.
                         gather_broadcast(paramvec_start.vector().array()))
        par_min = np.min(numpy_mpi.
                         gather_broadcast(paramvec_start.vector().array()))
        gamma_max = float(params["Optimization_parameters"]["gamma_max"])
        mat_min = float(params["Optimization_parameters"]["matparams_min"])

        while not done and niter < 10:
            # Evaluate the reduced functional in case the solver
            # chrashes at the first point.
            # If this is not done, and the solver crashes in the first point
            # then Dolfin adjoit has no recording and will raise an exception.

            # If this fails, there is no hope.
            try:

                rd(paramvec)
            except SolverDidNotConverge:

                if len(rd.controls_lst) > 0:
                    numpy_mpi.assign_to_vector(paramvec.vector(),
                                               rd.controls_lst[-1].array())
                else:
                    msg = ("Unable to converge. "
                           "Choose a different initial guess")
                    logger.error(msg)
                try:
                    rd(paramvec)
                except Exception as ex:
                    logger.error(ex)
                    msg = ("Unable to converge. "
                           "Try changing the scales and restart")
                    logger.error(msg)

            # Create optimal control problem
            oc_problem = OptimalControl()
            oc_problem.build_problem(params, rd, paramvec)

            try:
                # Try to solve the problem
                rd, opt_result = oc_problem.solve()

            except SolverDidNotConverge:

                logger.warning(Text.red("Solver failed - reduce step size"))
                # If the solver did not converge assign the state from
                # previous iteration and reduce the step size and try again
                rd.reset()
                rd.derivative_scale /= 2.0

                # There might be many reasons for why the sovler is not converging, 
                # but most likely it happens because the optimization algorithms try to
                # evaluate the function in a point in the parameter space, which is close
                # to the boundary. One thing we can do is to reduce the mangnitude of the
                # gradient (but keeping the direction) so that the step size reduces.
                # Another thing we can do is to actually change the bounds so that
                # the algorithm do not go into the nasty parts of the parameters space.
                # Usually the main problem is that the optimziation tries an activation that
                # is too strong (high gamma max) in the active phase, or at material parameter
                # set that is too soft (low material parameters) in the passive phase
                params["Optimization_parameters"]["gamma_max"] \
                    = np.max([par_max, 0.9*params["Optimization_parameters"]["gamma_max"]])
                params["Optimization_parameters"]["matparams_min"] \
                    = np.min([par_min, 2*params["Optimization_parameters"]["matparams_min"]])

            else:
                params["Optimization_parameters"]["gamma_max"] = gamma_max
                params["Optimization_parameters"]["matparams_min"] = mat_min
                rd.derivative_scale = 1.0
                done = True

            niter += 1
                    

        if not done:
            opt_result = {}
            control_idx = np.argmin(rd.func_values_lst)
            x = numpy_mpi.gather_broadcast(rd.controls_lst[control_idx].array())
            msg = "Unable to solve problem. Choose the best value"
            logger.warning(msg)
        else:
            x = np.array([opt_result.pop("x")]) if nvar == 1 \
                else numpy_mpi.gather_broadcast(opt_result.pop("x"))

        optimum = dolfin_adjoint.Function(paramvec.function_space(), name="optimum")
        numpy_mpi.assign_to_vector(optimum.vector(),
                                   numpy_mpi.gather_broadcast(x))

        logger.info(Text.blue("\nForward solution at optimal parameters"))
        numpy_mpi.assign_to_vector(paramvec.vector(),
                                   numpy_mpi.gather_broadcast(x))

        rd.for_res["initial_control"] = rd.initial_paramvec,
        rd.for_res["optimal_control"] = rd.paramvec

        print_optimization_report(params, rd.paramvec, rd.initial_paramvec,
                                  rd.ini_for_res, rd.for_res, opt_result)

        if store_solution:
            store(params, rd, opt_result)

        if return_solution:
            return params, rd, opt_result


def print_optimization_report(params, opt_controls, init_controls,
                              ini_for_res, opt_for_res, opt_result=None):

    from numpy_mpi import gather_broadcast

    if opt_result:
        logger.info("\nOptimization terminated...")
        logger.info("\tFunction Evaluations: {}".format(opt_result["nfev"]))
        logger.info("\tGradient Evaluations: {}".format(opt_result["njev"]))
        logger.info("\tNumber of iterations: {}".format(opt_result["nit"]))
        logger.info("\tNumber of crashes: {}".format(opt_result["ncrash"]))
        logger.info("\tRun time: {:.2f} seconds".format(opt_result["run_time"]))

    logger.info("\nFunctional Values")
    logger.info(" "*7+"\t"+print_head(ini_for_res, False))

    if "grad_norm" not in opt_result \
       or len(opt_result["grad_norm"]) == 0:
        grad_norm_ini = 0.0
        grad_norm_opt = 0.0
    else:
        grad_norm_ini = opt_result["grad_norm"][0]
        grad_norm_opt = opt_result["grad_norm"][-1]

    logger.info("{:7}\t{}".format("Initial",
                                  print_line(ini_for_res,
                                             grad_norm=grad_norm_ini)))
    logger.info("{:7}\t{}".format("Optimal",
                                  print_line(opt_for_res,
                                             grad_norm=grad_norm_opt)))
    
    if params["phase"] == config.PHASES[0]:
        logger.info("\nMaterial Parameters")
        logger.info("Initial {}".format(init_controls))
        logger.info("Optimal {}".format(numpy_mpi.gather_broadcast(opt_controls.vector().array())))
    else:
        logger.info("\nContraction Parameter")
        logger.info("\tMin\tMean\tMax")
        logger.info("Initial\t{:.5f}\t{:.5f}\t{:.5f}".format(init_controls.min(),
                                                             init_controls.mean(),
                                                             init_controls.max()))
        opt_controls_arr \
            = numpy_mpi.gather_broadcast(opt_controls.vector().array())
        logger.info("Optimal\t{:.5f}\t{:.5f}\t{:.5f}".format(opt_controls_arr.min(), 
                                                             opt_controls_arr.mean(), 
                                                             opt_controls_arr.max()))


def load_target_data(measurements, params, optimization_targets):
    """Load the target data into dolfin functions.
    The target data will be loaded into the optiization_targets

    :param measurements: The target measurements
    :param params: Application parameters
    :param optimization_targer: A dictionary with the targets
    :returns: object with target data
    :rtype: object
    """

    logger.debug(Text.blue("Loading Target Data"))

    # The point in the acitve phase (0 if passive)
    acin = params["active_contraction_iteration_number"]
    # Load boundary conditions
    bcs = {}

    for p in ['pressure', 'rv_pressure']:

        if p not in measurements:
            continue

        if params["phase"] == config.PHASES[1]:
            bcs[p] = measurements[p][acin:2+acin]
        else:
            bcs[p] = measurements[p]

    # Load the target data into dofin functions
    for key, val in params["Optimization_weigths"].items():

        # There is no data for regularization
        if key == "regularization":
            continue

        # If target is included in the optimization
        if val > 0.0:
            # Load the target data
            for it, p in enumerate(bcs['pressure']):

                optimization_targets[key].\
                    load_target_data(measurements[key], it+acin)

    return optimization_targets, bcs


def get_optimization_targets(params, problem):
    """FIXME! briefly describe function
    """
    logger.debug("Get optimization targets")

    p = params["Optimization_weigths"]

    # p = params["Optimization_targets"]
    # mesh = solver_parameters["mesh"]
    if params["phase"] == config.PHASES[0]:
        spacestr = params["matparams_space"]
    else:
        spacestr = params["gamma_space"]

    reg_par = p["regularization"]

    targets = {"regularization": Regularization(problem.geometry.mesh,
                                                spacestr,
                                                reg_par,
                                                mshfun=problem.geometry.cfun)}

    if p["volume"] > 0.0:
        logger.debug("Load volume target")
        marker = get_lv_marker(problem.geometry)
        logger.debug(("Make surface meausure for LV endo "
                      "with marker {}").format(marker))

        ds = problem.geometry.ds(marker)

        logger.debug("Load VolumeTarget")
        targets["volume"] = VolumeTarget(mesh=problem.geometry.mesh,
                                         dmu=ds, chamber="LV",
                                         approx=params["volume_approx"])

    if p["rv_volume"] > 0.0:
        logger.debug("Load RV volume target")
        marker = problem.geometry.markers["ENDO_RV"][0]
        logger.debug(("Make surface meausure for RV endo "
                      "with marker {}").format(marker))

        ds = problem.geometry.ds(marker)

        logger.debug("Load VolumeTarget")
        targets["rv_volume"] = VolumeTarget(mesh=problem.geometry.mesh,
                                            dmu=ds, chamber="RV",
                                            approx=params["volume_approx"])

    if p["regional_strain"] > 0.0:

        logger.debug("Load regional strain target")
        dx = problem.geometry.dx

        load_displacemet = (params["unload"] and
                            not params["strain_reference"] == "unloaded") or \
                           (not params["unload"]
                            and params["strain_reference"] == "ED")

        if load_displacemet and params["phase"] == config.PHASES[1]:
            # We need to recompute strains wrt reference as diastasis

            logger.debug(("Load displacment for recomputing strain with "
                          "respect to different reference"))
            
            if params["strain_reference"] == "0":
                group = "1"
            else:
                pass
                # strain reference = "ED"

                #FIXME!! passive filling duration
                # if params["unload"]:
                #     group = str(solver_parameters["passive_filling_duration"])
                # else:
                #     group = str(solver_parameters["passive_filling_duration"]-1)

            u = dolfin_adjoint.Function(problem.state_space.sub(0).collapse())

            logger.debug("Load displacement from state number {}.".format(group))
            with dolfin.HDF5File(dolfin.mpi_comm_world(),
                                 params["sim_file"], 'r') as h5file:

                # Get previous state
                group = "/".join([params["h5group"],
                                  config.PASSIVE_INFLATION_GROUP,
                                  "displacement", group])
                h5file.read(u, group)

            if params["strain_approx"] in ["project", "interpolate"]:

                V = dolfin.VectorFunctionSpace(problem.geometry.mesh,
                                               "CG", 1)
    
                if params["strain_approx"] == "project":
                    logger.debug("Project displacement")
                    u = dolfin_adjoint.project(u, V, name="u_project")
                    logger.debug("Interpolate displacement")
                    u = dolfin_adjoint.interpolate(u, V, name="u_interpolate")

            F_ref = kinematics.DeformationGradient(u)

        else:
            logger.debug("Do not recompute strains with respect than difference reference")
            F_ref = dolfin.Identity(3)

        logger.debug("Get RegionalStrainTarget")
        targets["regional_strain"] = \
            RegionalStrainTarget(mesh=problem.geometry.mesh,
                                 crl_basis=problem.geometry.crl_basis,
                                 dmu=dx,
                                 weights=None,
                                 tensor=params["strain_tensor"],
                                 F_ref=F_ref,
                                 approx=params["strain_approx"],
                                 map_strain=params["map_strain"])

    return targets


def load_targets(params, problem, measurements):
    """FIXME! briefly describe function

    """
    logger.debug(Text.blue("Load optimization targets"))
    # Solve calls are not registred by libajoint
    logger.debug(Text.yellow("Stop annotating"))
    dolfin.parameters["adjoint"]["stop_annotating"] = True

    # Load optimization targets
    optimization_targets \
        = get_optimization_targets(params, problem)
    
    # Load target data
    optimization_targets, bcs = \
        load_target_data(measurements, params, optimization_targets)

    # Start recording for dolfin adjoint
    logger.debug(Text.yellow("Start annotating"))
    dolfin.parameters["adjoint"]["stop_annotating"] = False

    return optimization_targets, bcs
