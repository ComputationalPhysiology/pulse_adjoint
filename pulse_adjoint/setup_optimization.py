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
# for commercial purposes should contact Simula Research Laboratory AS: post@simula.no
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
import logging
import pulse
from pulse import numpy_mpi
from pulse.dolfin_utils import (
    RegionalParameter,
    MixedParameter,
    BaseExpression,
    VertexDomain,
)

from .dolfinimport import *
from .utils import Object, Text, print_line, print_head
from .adjoint_contraction_args import *
from .setup_parameters import *


def merge_control(patient, control_str):

    sfun = dolfin.MeshFunction("size_t", patient.mesh, patient.sfun.dim())
    arr = patient.sfun.array()
    sfun.set_values(arr)
    if control_str != "":
        for v in control_str.split(":"):
            vals = sorted(np.array(v.split(","), dtype=arr.dtype))
            min_val = vals[0]
            for vi in vals[1:]:
                sfun.array()[sfun.array() == vi] = min_val

    return sfun


def get_material_model(material_model):

    assert material_model in pulse.material_model_names
    idx = pulse.material_model_names.index(material_model)
    return pulse.material_models[idx]


def update_unloaded_patient(params, patient):

    # Make sure to load the new referece geometry
    from mesh_generation import load_geometry_from_h5

    h5group = "/".join([_f for _f in [params["h5group"], "unloaded"] if _f])
    geo = load_geometry_from_h5(
        params["sim_file"], h5group, comm=patient.mesh.mpi_comm()
    )
    setattr(patient, "original_geometry", getattr(patient, "mesh"))

    for k, v in geo.__dict__.items():
        if hasattr(patient, k):
            delattr(patient, k)

        setattr(patient, k, v)

    return patient


def initialize_patient_data(patient_parameters):
    """
    Make an instance of patient from :py:module`patient_data`
    baed on th given parameters

    *Parameters*

    patient_parameters: dict
        the parameters 
    
    *Returns*

    patient: :py:class`patient_data.Patient`
        A patient instance

    **Example of usage**::
    
      params = setup_patient_parameters()
      patient = initialize_patient_data(params)

    """

    logger.info("Initialize patient data")
    from .patient_data import Patient

    patient = Patient(**patient_parameters)

    return patient


def check_patient_attributes(patient):
    """
    Check that the object contains the minimum 
    required attributes. 
    """

    msg = "Patient is missing attribute {}"

    # Mesh
    if not hasattr(patient, "mesh"):
        raise AttributeError(msg.format("mesh"))
    else:
        dim = patient.mesh.topology().dim()

    ## Microstructure

    # Fibers
    if not hasattr(patient, "fiber"):

        no_fiber = True
        if hasattr(patient, "e_f"):
            rename_attribute(patient, "e_f", "fiber")
            no_fiber = False

        if no_fiber:

            idx_arr = np.where([item.startswith("fiber") for item in dir(patient)])[0]
            if len(idx_arr) == 0:
                raise AttributeError(msg.format("fiber"))
            else:
                att = dir(patient)[idx_arr[0]]
                rename_attribute(patient, att, "fiber")

    # Sheets
    if not hasattr(patient, "sheet"):
        if hasattr(patient, "e_s"):
            rename_attribute(patient, "e_s", "sheet")
        else:
            setattr(patient, "sheet", None)

    # Cross-sheet
    if not hasattr(patient, "sheet_normal"):
        if hasattr(patient, "e_sn"):
            rename_attribute(patient, "e_sn", "sheet_normal")
        else:
            setattr(patient, "sheet_normal", None)

    ## Local basis

    # Circumferential
    if not hasattr(patient, "circumferential") and hasattr(patient, "e_circ"):
        rename_attribute(patient, "e_circ", "circumferential")

    # Radial
    if not hasattr(patient, "radial") and hasattr(patient, "e_rad"):
        rename_attribute(patient, "e_rad", "radial")

    # Longitudinal
    if not hasattr(patient, "longitudinal") and hasattr(patient, "e_long"):
        rename_attribute(patient, "e_long", "longitudinal")

    ## Markings

    # Markers
    if not hasattr(patient, "markers"):
        raise AttributeError(msg.format("markers"))

    # Facet fuction
    if not hasattr(patient, "ffun"):

        no_ffun = True
        if hasattr(patient, "facets_markers"):
            rename_attribute(patient, "facets_markers", "ffun")
            no_ffun = False

        if no_ffun:
            setattr(
                patient,
                "ffun",
                dolfin.MeshFunction("size_t", patient.mesh, 2, patient.mesh.domains()),
            )

    # Cell markers
    if dim == 3 and not hasattr(patient, "sfun"):

        no_sfun = True
        if no_sfun and hasattr(patient, "strain_markers"):
            rename_attribute(patient, "strain_markers", "sfun")
            no_sfun = False

        if no_sfun:
            setattr(
                patient,
                "sfun",
                dolfin.MeshFunction("size_t", patient.mesh, 3, patient.mesh.domains()),
            )

    ## Other

    # Weigts on strain semgements
    if not hasattr(patient, "strain_weights"):
        setattr(patient, "strain_weights", None)

    # Mesh type
    if not hasattr(patient, "mesh_type"):
        # If markers are according to fiberrules,
        # rv should be marked with 20
        if 20 in set(patient.ffun.array()):
            setattr(patient, "mesh_type", lambda: "biv")
        else:
            setattr(patient, "mesh_type", lambda: "lv")

    if not hasattr(patient, "passive_filling_duration"):
        setattr(patient, "passive_filling_duration", 1)


def save_patient_data_to_simfile(patient, sim_file):

    from pulse.geometry_utils import save_geometry_to_h5

    fields = []
    for att in ["fiber", "sheet", "sheet_normal"]:
        if hasattr(patient, att):
            fields.append(getattr(patient, att))

    local_basis = []
    for att in ["circumferential", "radial", "longitudinal"]:
        if hasattr(patient, att):
            local_basis.append(getattr(patient, att))

    meshfunctions = {}
    for dim, name in enumerate(['vfun', 'efun', 'ffun', 'cfun']):
        meshfunctions[dim] = getattr(patient, name)

    save_geometry_to_h5(
        patient.mesh, sim_file, "", patient.markers, fields, local_basis
    )


def get_simulated_strain_traces(phm):
    simulated_strains = {
        strain: np.zeros(17) for strain in list(STRAIN_NUM_TO_KEY.values())
    }
    strains = phm.strains
    for direction in range(3):
        for region in range(17):
            simulated_strains[STRAIN_NUM_TO_KEY[direction]][
                region
            ] = numpy_mpi.gather_broadcast(strains[region].vector().get_local())[
                direction
            ]
    return simulated_strains


def make_solver_params(params, patient, measurements=None):

    paramvec, gamma, matparams = make_control(params, patient)
    return make_solver_parameters(
        params, patient, matparams, gamma, paramvec, measurements
    )


def make_solver_parameters(
    params,
    patient,
    matparams,
    gamma=dolfin_adjoint.Constant(0.0),
    paramvec=None,
    measurements=None,
):

    ##  Material
    Material = get_material_model(params["material_model"])

    f0 = getattr(patient, "fiber", getattr(patient, "f0", None))
    s0 = getattr(patient, "sheet", getattr(patient, "s0", None))
    n0 = getattr(patient, "sheet_normal", getattr(patient, "n0", None))

    material = Material(
        f0=f0, activation=gamma, parameters=matparams, s0=s0, n0=n0, **params
    )

    if measurements is None:
        p_lv_ = 0.0
        p_rv_ = 0.0
    else:
        p_lv_ = measurements["pressure"][0]
        if "rv_pressure" in measurements:
            p_rv_ = measurements["rv_pressure"][0]

    # Neumann BC
    neuman_bc = []

    p_lv = dolfin_adjoint.Constant(p_lv_, name="LV_endo_pressure")

    if "ENDO_LV" in patient.markers:

        p_rv = dolfin_adjoint.Constant(p_rv_, name="RV_endo_pressure")

        neumann_bc = [
            [p_lv, patient.markers["ENDO_LV"][0]],
            [p_rv, patient.markers["ENDO_RV"][0]],
        ]

        pressure = {"p_lv": p_lv, "p_rv": p_rv}
    else:
        neumann_bc = [[p_lv, patient.markers["ENDO"][0]]]
        pressure = {"p_lv": p_lv}

    pericard = dolfin_adjoint.Constant(params["pericardium_spring"])
    robin_bc = [[pericard, patient.markers["EPI"][0]]]

    if params["base_bc"] == "from_seg_base":

        # Direchlet BC at the Base
        try:
            mesh_verts = patient.mesh_verts
            seg_verts = measurements.seg_verts
        except:
            raise ValueError(
                (
                    "No mesh vertices found. Fix base "
                    + "is the only applicable Direchlet BC"
                )
            )

        endoring = VertexDomain(mesh_verts)
        base_it = dolfin.Expression("t", t=0.0, name="base_iterator")

        # Expression for defining the boundary conditions
        base_bc_y = BaseExpression(
            mesh_verts, seg_verts, "y", base_it, name="base_expr_y"
        )
        base_bc_z = BaseExpression(
            mesh_verts, seg_verts, "z", base_it, name="base_expr_z"
        )

        def base_bc(W):
            """
            Fix base in the x = 0 plane, and fix the vertices at 
            the endoring at the base according to the segmeted surfaces. 
            """
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)

            bc = [
                dolfin.DirichletBC(
                    V.sub(0),
                    dolfin_adjoint.Constant(0.0),
                    patient.ffun,
                    patient.markers["BASE"][0],
                ),
                DirichletBC(V.sub(1), base_bc_y, endoring, "pointwise"),
                DirichletBC(V.sub(2), base_bc_z, endoring, "pointwise"),
            ]
            return bc

    elif params["base_bc"] == "fixed":

        base_bc_y = None
        base_bc_z = None
        base_it = None

        def base_bc(W):
            """Fix the basal plane.
            """
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            bc = [
                dolfin.DirichletBC(
                    V,
                    dolfin_adjoint.Constant((0, 0, 0)),
                    patient.ffun,
                    patient.markers["BASE"][0],
                )
            ]
            return bc

    else:

        if not (params["base_bc"] == "fix_x"):
            logger.warning("Unknown Base BC {}".format(params["base_bc"]))
            logger.warning("Fix base in x direction")

        def base_bc(W):
            """Make Dirichlet boundary conditions where the base is allowed to slide
            in the x = 0 plane.
            """
            V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
            bc = [
                dolfin.DirichletBC(
                    V.sub(0), 0, patient.ffun, patient.markers["BASE"][0]
                )
            ]
            return bc

        # Apply a linear sprint robin type BC to limit motion
        # base_spring = Function(V_real, name = "base_spring")
        base_spring = dolfin_adjoint.Constant(params["base_spring"])
        robin_bc += [[base_spring, patient.markers["BASE"][0]]]

    # Circumferential, Radial and Longitudinal basis vector
    crl_basis = {}
    for att, att1 in [
        ("circumferential", "c0"),
        ("radial", "r0"),
        ("longitudinal", "l0"),
    ]:
        # if hasattr(patient, att):
        crl_basis[att] = getattr(patient, att, getattr(patient, att1, None))

    solver_parameters = {
        "mesh": patient.mesh,
        "facet_function": patient.ffun,
        "facet_normal": dolfin.FacetNormal(patient.mesh),
        "crl_basis": crl_basis,
        "mesh_function": getattr(patient, "sfun", getattr(patient, "cfun", None)),
        "markers": patient.markers,
        "passive_filling_duration": getattr(patient, "passive_filling_duration", 1),
        "strain_weights": getattr(patient, "strain_weights", None),
        "state_space": "P_2:P_1",
        "compressibility": {
            "type": params["compressibility"],
            "lambda": params["incompressibility_penalty"],
        },
        "material": material,
        "bc": {"dirichlet": base_bc, "neumann": neumann_bc, "robin": robin_bc},
        "solve": setup_solver_parameters(),
    }

    if params["phase"] in [PHASES[0], PHASES[2]]:
        return solver_parameters, pressure, paramvec
    elif params["phase"] == PHASES[1]:
        return solver_parameters, pressure, gamma
    else:
        return solver_parameters, pressure


def make_control(params, patient):

    ##  Contraction parameter
    if params["gamma_space"] == "regional":
        sfun = merge_control(patient, params["merge_active_control"])
        gamma = RegionalParameter(sfun)
    else:
        gamma_family, gamma_degree = params["gamma_space"].split("_")
        gamma_space = dolfin.FunctionSpace(
            patient.mesh, gamma_family, int(gamma_degree)
        )

        gamma = dolfin_adjoint.Function(gamma_space, name="activation parameter")

    ##  Material parameters

    # Create an object for each single material parameter
    if params["matparams_space"] == "regional":
        sfun = merge_control(patient, params["merge_passive_control"])
        paramvec_ = RegionalParameter(sfun)

    else:

        family, degree = params["matparams_space"].split("_")
        matparams_space = dolfin.FunctionSpace(patient.mesh, family, int(degree))
        paramvec_ = dolfin_adjoint.Function(matparams_space, name="matparam vector")

    # If we want to estimate more than one parameter

    # Number of passive parameters to optimize
    npassive = sum([not v for v in list(params["Fixed_parameters"].values())])

    if npassive <= 1:
        # If there is only one parameter, just pick the same object
        paramvec = paramvec_

        # If there is none then
        if npassive == 0:
            logger.debug("All material paramters are fixed")
            params["optimize_matparams"] = False

    else:

        # Otherwise, we make a mixed parameter
        paramvec = MixedParameter(paramvec_, npassive)
        # Make an iterator for the function assigment
        nopts_par = 0

    if params["phase"] in [PHASES[1]]:
        # Load the parameters from the result file

        # Open simulation file
        with dolfin.HDF5File(
            dolfin.mpi_comm_world(), params["sim_file"], "r"
        ) as h5file:

            # Get material parameter from passive phase file
            h5file.read(paramvec, PASSIVE_INFLATION_GROUP + "/optimal_control")

    matparams = params["Material_parameters"].to_dict()

    for par, val in matparams.items():

        # Check if material parameter should be fixed
        if not params["Fixed_parameters"][par]:
            # If not, then we need to put the parameter into some dolfin function

            # Use the materal parameters from the parameters as initial guess
            if params["phase"] in [PHASES[0], PHASES[2]]:

                # val_const = dolfin_adjoint.Function(paramvec.function_space())
                # numpy_mpi.assign_to_vector(val_const.vector(),
                #                            val * np.ones(len(val_const)))
                val_const = (
                    dolfin_adjoint.Constant(val)
                    if paramvec_.value_size() == 1
                    else dolfin_adjoint.Constant([val] * paramvec_.value_size())
                )

                if npassive <= 1:
                    paramvec.assign(val_const)

                else:
                    paramvec.assign_sub(val_const, nopts_par)

            if npassive <= 1:
                matparams[par] = paramvec

            else:
                matparams[par] = split(paramvec)[nopts_par]
                nopts_par += 1

    # Print the material parameter to stdout
    logger.info("\nMaterial Parameters")
    nopts_par = 0

    for par, v in matparams.items():
        if isinstance(v, (float, int)):
            logger.info("\t{}\t= {:.3f}".format(par, v))
        else:

            if npassive <= 1:
                v_ = numpy_mpi.gather_broadcast(v.vector().get_local())

            else:
                v_ = numpy_mpi.gather_broadcast(
                    paramvec.split(deepcopy=True)[nopts_par].vector().get_local()
                )
                nopts_par += 1

            sp_str = "(mean), spatially resolved" if len(v_) > 1 else ""
            logger.info("\t{}\t= {:.3f} {}".format(par, v_.mean(), sp_str))

    return paramvec, gamma, matparams


def get_measurements(params, patient):
    """Get the measurement to be used as BC 
    or targets in the optimization

    :param params: Application parameter
    :param patient: class with the patient data
    :returns: The target data
    :rtype: dict

    """

    # Parameters for the targets
    p = params["Optimization_targets"]
    measurements = {}

    # Find the start and end of the measurements
    if params["phase"] == PHASES[0]:  # Passive inflation
        # We need just the points from the passive phase
        start = 0
        end = patient.passive_filling_duration

        pvals = params["Passive_optimization_weigths"]

    elif params["phase"] == PHASES[1]:
        # We need just the points from the active phase
        start = patient.passive_filling_duration - 1
        end = patient.num_points

        pvals = params["Active_optimization_weigths"]

        if params["unload"]:
            start += 1

    else:
        # We need all the points
        start = 0
        end = patient.num_points

        # pvals = params["Passive_optimization_weigths"]
        pvals = params["Active_optimization_weigths"]

    if params["unload"]:
        end += 1

    p["volume"] = pvals["volume"] > 0 or params["phase"] == "all"
    p["rv_volume"] = hasattr(patient, "RVV") and (
        pvals["rv_volume"] > 0 or params["phase"] == "all"
    )

    p["regional_strain"] = hasattr(patient, "strain") and (
        pvals["regional_strain"] > 0 or params["phase"] == "all"
    )

    ## Pressure

    # We need the pressure as a BC
    pressure = np.array(patient.pressure)

    # Compute offsets
    # Choose the pressure at the beginning as reference pressure
    if params["unload"]:
        reference_pressure = 0.0
        pressure = np.append(0.0, pressure)
    else:
        reference_pressure = pressure[0]
    logger.info("LV Pressure offset = {} kPa".format(reference_pressure))

    # Here the issue is that we do not have a stress free reference mesh.
    # The reference mesh we use is already loaded with a certain
    # amount of pressure, which we remove.
    pressure = np.subtract(pressure, reference_pressure)
    measurements["pressure"] = pressure[start:end]

    if hasattr(patient, "RVP"):
        rv_pressure = np.array(patient.RVP)
        if params["unload"]:
            reference_pressure = 0.0
            rv_pressure = np.append(0.0, rv_pressure)
        else:
            reference_pressure = rv_pressure[0]
        logger.info("RV Pressure offset = {} kPa".format(reference_pressure))

        rv_pressure = np.subtract(rv_pressure, reference_pressure)
        measurements["rv_pressure"] = rv_pressure[start:end]

    ## Volume
    if p["volume"]:
        # Calculate difference bwtween calculated volume, and volume given from echo
        volume_offset = get_volume_offset(patient, params)
        logger.info("LV Volume offset = {} cm3".format(volume_offset))
        logger.info("Measured LV volume = {}".format(patient.volume[0]))

        # Subtract this offset from the volume data
        volume = np.subtract(patient.volume, volume_offset)
        logger.info("Computed LV volume = {}".format(volume[0]))
        if params["unload"]:
            volume = np.append(-1, volume)

        measurements["volume"] = volume[start:end]

    if p["rv_volume"]:
        # Calculate difference bwtween calculated volume, and volume given from echo
        volume_offset = get_volume_offset(patient, params, "rv")
        logger.info("RV Volume offset = {} cm3".format(volume_offset))
        logger.info("Measured RV volume = {}".format(patient.RVV[0]))

        # Subtract this offset from the volume data
        volume = np.subtract(patient.RVV, volume_offset)
        logger.info("Computed RV volume = {}".format(volume[0]))
        if params["unload"]:
            volume = np.append(-1, volume)

        measurements["rv_volume"] = volume[start:end]

    if p["regional_strain"]:

        strain = {}
        if hasattr(patient, "strain"):
            for region in list(patient.strain.keys()):

                s = patient.strain[region]
                if params["unload"]:
                    s = [(0.0, 0.0, 0.0)] + s

                strain[region] = s[start:end]

        else:
            msg = (
                "\nPatient do not have strain as attribute."
                + "\nStrain will not be used"
            )
            p["regional_strain"] = False
            logger.warning(msg)

        measurements["regional_strain"] = strain

    return measurements


def get_volume(patient, unload=False, chamber="lv", u=None):

    if unload:
        mesh = patient.original_geometry
        ffun = dolfin.MeshFunction("size_t", mesh, 2, mesh.domains())
    else:
        mesh = patient.mesh
        ffun = patient.ffun

    if chamber == "lv":
        if "ENDO_LV" in patient.markers:
            endo_marker = patient.markers["ENDO_LV"][0]
        else:
            endo_marker = patient.markers["ENDO"][0]

    else:
        endo_marker = patient.markers["ENDO_RV"][0]

    dS = dolfin.Measure("exterior_facet", subdomain_data=ffun, domain=mesh)(endo_marker)

    X = dolfin.SpatialCoordinate(mesh)
    N = dolfin.FacetNormal(mesh)
    if u is None:
        vol_form = (-1.0 / 3.0) * dolfin.dot(X, N)
    else:
        F = dolfin.grad(u) + dolfin.Identity(3)
        J = dolfin.det(F)
        vol_form = (-1.0 / 3.0) * dolfin.dot(X + u, J * inv(F).T * N)

    vol = dolfin.assemble(vol_form * dS)
    return vol


def get_volume_offset(patient, params, chamber="lv"):

    if params["Patient_parameters"]["geometry_index"] == "-1":
        idx = patient.passive_filling_duration - 1
    else:
        idx = int(params["Patient_parameters"]["geometry_index"])

    if chamber == "lv":
        volume = patient.volume[idx]
    else:
        volume = patient.RVV[idx]

    logger.info("Measured = {}".format(volume))
    vol = get_volume(patient, params["unload"], chamber)
    return volume - vol


def setup_simulation(params, patient):

    # check_patient_attributes(patient)
    # Load measurements
    measurements = get_measurements(params, patient)
    solver_parameters, pressure, controls = make_solver_params(
        params, patient, measurements
    )

    return measurements, solver_parameters, pressure, controls


class MyReducedFunctional(dolfin_adjoint.ReducedFunctional):
    """
    A modified reduced functional of the `dolfin_adjoint.ReducedFuctionl`

    *Parameters*
    
    for_run: callable
        The forward model, which can be called with the control parameter
        as first argument, and a boolean as second, indicating that annotation is on/off.
    paramvec: :py:class`dolfin_adjoint.function`
        The control parameter
    scale: float
        Scale factor for the functional
    relax: float
        Scale factor for the derivative. Note the total scale factor for the 
        derivative will be scale*relax


    """

    def __init__(self, for_run, paramvec, scale=1.0, relax=1.0, verbose=False):

        self.log_level = logger.level
        self.reset()
        self.for_run = for_run
        self.paramvec = paramvec

        self.initial_paramvec = numpy_mpi.gather_broadcast(
            paramvec.vector().get_local()
        )
        self.scale = scale
        self.derivative_scale = relax
        self.verbose = verbose

    def __call__(self, value, return_fail=False):

        logger.debug("\nEvaluate functional...")
        dolfin_adjoint.adj_reset()
        self.iter += 1

        paramvec_new = dolfin_adjoint.Function(
            self.paramvec.function_space(), name="new control"
        )
        # paramvec_new = RegionalParameter(self.paramvec._meshfunction)

        if isinstance(value, (dolfin.Function, RegionalParameter, MixedParameter)):
            paramvec_new.assign(value)
        elif isinstance(value, float) or isinstance(value, int):
            numpy_mpi.assign_to_vector(paramvec_new.vector(), np.array([value]))
        elif isinstance(value, dolfin_adjoint.enlisting.Enlisted):
            val_delisted = delist(value, self.controls)
            paramvec_new.assign(val_delisted)

        else:
            numpy_mpi.assign_to_vector(
                paramvec_new.vector(), numpy_mpi.gather_broadcast(value)
            )

        logger.debug(Text.yellow("Start annotating"))
        dolfin.parameters["adjoint"]["stop_annotating"] = False

        if self.verbose:
            arr = numpy_mpi.gather_broadcast(paramvec_new.vector().get_local())
            msg = (
                "\nCurrent value of control:"
                + "\n\t{:>8}\t{:>8}\t{:>8}\t{:>8}\t{:>8}".format(
                    "Min", "Mean", "Max", "argmin", "argmax"
                )
                + "\n\t{:>8.2f}\t{:>8.2f}\t{:>8.2f}\t{:>8d}\t{:>8d}".format(
                    np.min(arr),
                    np.mean(arr),
                    np.max(arr),
                    np.argmin(arr),
                    np.argmax(arr),
                )
            )
            logger.info(msg)
        # Change loglevel to avoid to much printing (do not change if in dbug mode)
        change_log_level = (self.log_level == logging.INFO) and not self.verbose

        if change_log_level:
            logger.setLevel(logging.WARNING)

        t = dolfin.Timer("Forward run")
        t.start()

        logger.debug("\nEvaluate forward model")

        self.for_res, crash = self.for_run(paramvec_new, True)

        for_time = t.stop()
        logger.debug(
            (
                "Evaluating forward model done. "
                + "Time to evaluate = {} seconds".format(for_time)
            )
        )
        self.forward_times.append(for_time)

        if change_log_level:
            logger.setLevel(self.log_level)

        if self.first_call:
            # Store initial results
            self.ini_for_res = self.for_res
            self.first_call = False

            # Some printing
            logger.info(print_head(self.for_res))

        control = dolfin_adjoint.Control(self.paramvec)

        dolfin_adjoint.ReducedFunctional.__init__(
            self, dolfin_adjoint.Functional(self.for_res["total_functional"]), control
        )

        if crash:
            # This exection is thrown if the solver uses more than x steps.
            # The solver is stuck, return a large value so it does not get stuck again
            logger.warning(
                Text.red(
                    "Iteration limit exceeded. Return a large value of the functional"
                )
            )
            # Return a big value, and make sure to increment the big value so the
            # the next big value is different from the current one.
            func_value = np.inf
            self.nr_crashes += 1

        else:
            func_value = self.for_res["func_value"]

        grad_norm = (
            None if len(self.grad_norm_scaled) == 0 else self.grad_norm_scaled[-1]
        )

        self.func_values_lst.append(func_value * self.scale)
        self.controls_lst.append(dolfin.Vector(paramvec_new.vector()))

        logger.debug(Text.yellow("Stop annotating"))
        dolfin.parameters["adjoint"]["stop_annotating"] = True

        self.print_line()

        if return_fail:
            return self.scale * func_value, crash

        return self.scale * func_value

    def reset(self):

        logger.setLevel(self.log_level)
        if not hasattr(self, "ini_for_res"):

            self.cache = None
            self.first_call = True
            self.nr_crashes = 0
            self.iter = 0
            self.nr_der_calls = 0
            self.func_values_lst = []
            self.controls_lst = []
            self.forward_times = []
            self.backward_times = []
            self.grad_norm = []
            self.grad_norm_scaled = []
        else:
            if len(self.func_values_lst):
                self.func_values_lst.pop()
            if len(self.controls_lst):
                self.controls_lst.pop()
            if len(self.grad_norm):
                self.grad_norm.pop()
            if len(self.grad_norm_scaled):
                self.grad_norm_scaled.pop()

    def print_line(self):
        grad_norm = (
            None if len(self.grad_norm_scaled) == 0 else self.grad_norm_scaled[-1]
        )

        func_value = self.for_res["func_value"]

        logger.info(print_line(self.for_res, self.iter, grad_norm, func_value))

    def derivative(self, *args, **kwargs):

        logger.debug("\nEvaluate gradient...")
        self.nr_der_calls += 1
        import math

        t = dolfin.Timer("Backward run")
        t.start()

        out = dolfin_adjoint.ReducedFunctional.derivative(self, forget=False)
        back_time = t.stop()
        logger.debug(
            (
                "Evaluating gradient done. "
                + "Time to evaluate = {} seconds".format(back_time)
            )
        )
        self.backward_times.append(back_time)

        for num in out[0].vector().get_local():
            if math.isnan(num):
                raise Exception("NaN in adjoint gradient calculation.")

        # Multiply with some small number to that we take smaller steps
        gathered_out = numpy_mpi.gather_broadcast(out[0].vector().get_local())

        self.grad_norm.append(np.linalg.norm(gathered_out))
        self.grad_norm_scaled.append(
            np.linalg.norm(gathered_out) * self.scale * self.derivative_scale
        )
        logger.debug(
            "|dJ|(actual) = {}\t|dJ|(scaled) = {}".format(
                self.grad_norm[-1], self.grad_norm_scaled[-1]
            )
        )
        return self.scale * gathered_out * self.derivative_scale
