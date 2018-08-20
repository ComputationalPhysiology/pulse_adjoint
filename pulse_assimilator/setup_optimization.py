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
from functools import partial
import numpy as np
import dolfin
import dolfin_adjoint
# import tabulate

from pulse import material as mat
# from .utils import make_logger, get_lv_marker, number_of_passive_controls
from pulse.dolfin_utils import (RegionalParameter, MixedParameter,
                                get_constant)
from pulse.mechanicsproblem import (dirichlet_fix_base,
                               dirichlet_fix_base_directional,
                               MechanicsProblem,
                               BoundaryConditions,
                               NeumannBC, RobinBC)
from pulse import numpy_mpi
from pulse.geometry import HeartGeometry
from pulse import kinematics

from .clinical_data import ClinicalData


from . import setup_parameters
from . import config

# from . import parameters
from . import Patient

# logger = make_logger(__name__, config.log_level)


def initialize_patient_data(params):

    mesh_path = params["mesh_path"]
    data_path = params["data_path"]
    echo_path = params["echo_path"]
    
    geometry = HeartGeometry.from_file(mesh_path)
    data = ClinicalData.from_file(data_path=data_path,
                                  echo_path=echo_path)
    return Patient(geometry=geometry, data=data)

    
def update_unloaded_patient(params, patient):

    # Make sure to load the new referece geometry
    from mesh_generation import load_geometry_from_h5
    h5group = "/".join(filter(None, [params["h5group"], "unloaded"]))
    geo = load_geometry_from_h5(params["sim_file"], h5group,
                                comm=patient.geometry.mesh.mpi_comm())

    patient.geometry.original_geometry = patient.geometry.mesh

    for k, v in geo.__dict__.iteritems():
        if hasattr(patient, k):
            delattr(patient, k)

        setattr(patient, k, v)

    return patient


def save_patient_data_to_simfile(patient, sim_file):

    from mesh_generation.mesh_utils import save_geometry_to_h5

    fields = []
    for att in ['f0', 's0', 'n0']:
        if hasattr(patient.geometry, att):
            fields.append(getattr(patient.geometry, att))

    local_basis = []
    for att in ['c0', 'r0', 'l0']:
        if hasattr(patient.geomtry, att):
            local_basis.append(getattr(patient.geometry, att))

    save_geometry_to_h5(patient.geometry.mesh, sim_file, "",
                        patient.geomtry.markers,
                        fields, local_basis)


def get_simulated_strain_traces(phm):
        simulated_strains = {strain: np.zeros(17)
                             for strain in config.STRAIN_NUM_TO_KEY.values()}

        strains = phm.strains
        for direction in range(3):
            for region in range(17):
                simulated_strains[config.STRAIN_NUM_TO_KEY[direction]][region] \
                    = numpy_mpi.\
                    gather_broadcast(strains[region].
                                     vector().array())[direction]
        return simulated_strains



def find_start_and_end_index(params, data):

    # Find the start and end of the measurements
    if params["phase"] == 'passive_inflation':
        # We need just the points from the passive phase
        start = 0
        end = data.passive_duration

    elif params["phase"] == 'active_contraction':
        # We need just the points from the active phase
        start = data.passive_duration-1
        end = data.num_points

        if params["unload"]:
            start += 1
    else:
        # We need all the points
        start = 0
        end = data.num_points

    if params["unload"]:
        end += 1

    return start, end


# def check_optimization_target_weigths(params, data):

#     pvals = params["Passive_optimization_weigths"] \
#         if params["phase"] == 'passive_inflation' \
#         else params["Active_optimization_weigths"]

#     params["Optimization_targets"]["volume"] \
#         = params["Optimization_targets"]["volume"] and \
#         hasattr(data, "volume") and \
#         pvals["volume"] > 0 or params["phase"] == "all"
#     # FIX this with RV data
#     params["Optimization_targets"]["rv_volume"] \
#         = params["Optimization_targets"]["rv_volume"] and \
#         hasattr(data, "RVV") and \
#         (pvals["rv_volume"] > 0 or params["phase"] == "all")

#     params["Optimization_targets"]["regional_strain"] \
#         = params["Optimization_targets"]["regional_strain"] and \
#         hasattr(data, "strain") and \
#         (pvals["regional_strain"] > 0 or params["phase"] == "all")


def get_measurements(params, patient):
    """Get the measurement to be used as BC
    or targets in the optimization

    :param params: Application parameter
    :param patient: class with the patient data
    :returns: The target data
    :rtype: dict

    """

    # Parameters for the targets
    p = params["Optimization_weigths"]
    measurements = {}

    start, end = find_start_and_end_index(params, patient.data)
    # check_optimization_target_weigths(params, patient.data)

    # Pressure
    # We need the pressure as a BC
    pressure = np.array(patient.data.pressure)

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

    # FIX this
    if getattr(patient.data, "RVP", None) is not None:
        rv_pressure = np.array(patient.data.RVP)
        if params["unload"]:
            reference_pressure = 0.0
            rv_pressure = np.append(0.0, rv_pressure)
        else:
            reference_pressure = rv_pressure[0]
        logger.info("RV Pressure offset = {} kPa".format(reference_pressure))

        rv_pressure = np.subtract(rv_pressure, reference_pressure)
        measurements["rv_pressure"] = rv_pressure[start:end]

    # Volume
    if p["volume"] > 0.0:
        assert hasattr(patient.data, "volume")
        # Calculate difference bwtween calculated volume,
        # and volume given from echo
        volume_offset = get_volume_offset(patient, params)
        logger.info("LV Volume offset = {} cm3".format(volume_offset))
        logger.info("Measured LV volume = {}".format(patient.data.volume[0]))
                    
        # Subtract this offset from the volume data
        volume = np.subtract(patient.data.volume, volume_offset)
        logger.info("Computed LV volume = {}".format(volume[0]))
        if params["unload"]:
            volume = np.append(-1, volume)

        measurements["volume"] = volume[start:end]

    if p["rv_volume"] > 0.0:
        assert hasattr(patient.data, "rv_volume")
        # Calculate difference bwtween calculated volume,
        # and volume given from echo
        volume_offset = get_volume_offset(patient, params, "rv")
        logger.info("RV Volume offset = {} cm3".format(volume_offset))
        logger.info("Measured RV volume = {}".format(patient.data.RVV[0]))

        # Subtract this offset from the volume data
        volume = np.subtract(patient.data.RVV, volume_offset)
        logger.info("Computed RV volume = {}".format(volume[0]))
        if params["unload"]:
            volume = np.append(-1, volume)

        measurements["rv_volume"] = volume[start:end]

    if p["regional_strain"] > 0.0:

        if hasattr(patient.data, "strain"):
            # Fix new strain type!!!!!
            strain = patient.data.strain[start:end]

            if params["unload"]:
                zero_strain = {r: (0.0, 0.0, 0.0) for r
                               in patient.geometry.regions}
                strain = tuple([zero_strain] + [s for s in strain])

        else:
            msg = ("\nPatient do not have strain as attribute."
                   "\nStrain will not be used")
            p["regional_strain"] = False
            logger.warning(msg)

        measurements["regional_strain"] = strain

    return measurements


def get_volume_offset(patient, params, chamber="lv"):

    idx = patient.geometry.geometry_index

    if chamber == "lv":
        volume = patient.data.volume[idx]
    else:
        volume = patient.data.RVV[idx]

    logger.info("Measured = {}".format(volume))
    vol = get_volume(geometry=patient.geometry,
                     unload=params["unload"],
                     chamber=chamber)
    return volume - vol


def make_mechanics_problem(params, geometry, matparams=None):

    # Controls
    active_control = make_active_control(params, geometry)
    passive_control = make_passive_control(params, geometry)

    # Material
    Material = mat.get_material_model(params["material_model"])
    if matparams is None:
        matparams = get_material_parameters(params, passive_control)
    material = Material(activation=active_control,
                        parameters=matparams,
                        f0=geometry.f0,
                        s0=geometry.s0,
                        n0=geometry.n0,
                        **params)

    bcs = get_boundary_conditions(params, geometry)

    problem = MechanicsProblem(geometry, material, bcs)

    return problem, active_control, passive_control


def get_boundary_conditions(params, geometry):

    # Neumann BC
    lv_marker = get_lv_marker(geometry)
    lv_pressure = NeumannBC(traction=dolfin_adjoint.
                            Constant(0.0, name="lv_pressure"),
                            marker=lv_marker, name='lv')
    neumann_bc = [lv_pressure]

    if 'ENDO_RV' in geometry.markers:

        rv_pressure = NeumannBC(traction=dolfin_adjoint.
                                Constant(0.0, name='lv_pressure'),
                                marker=geometry.markers['ENDO_RV'][0],
                                name='rv')

        neumann_bc += [rv_pressure]

    # Robin BC
    if params["pericardium_spring"] > 0.0:

        robin_bc = [RobinBC(value=dolfin.Constant(params["pericardium_spring"]),
                            marker=geometry.markers["EPI"][0])]

    else:
        robin_bc = []

    # Apply a linear sprint robin type BC to limit motion
    if params["base_spring_k"] > 0.0:
        robin_bc += [RobinBC(value=dolfin.Constant(params["base_spring_k"]),
                             marker=geometry.markers["BASE"][0])]

    # Dirichlet BC
    if params["base_bc"] == "fixed":

        dirichlet_bc = [partial(dirichlet_fix_base,
                                ffun=geometry.ffun,
                                marker=geometry.markers["BASE"][0])]

    else:

        if not (params["base_bc"] == "fix_x"):
            logger.warning("Unknown Base BC {}".format(params["base_bc"]))
            logger.warning("Fix base in x direction")

        if params["base_spring_k"] == 0.0:
            logger.warning(('Base is only fixed in one direction '
                            'with no spring term. Problem might be '
                            'underconstrained. Consider to set '
                            'base_spring different from zero'))

        dirichlet_bc = [partial(dirichlet_fix_base_directional,
                                ffun=geometry.ffun,
                                marker=geometry.markers["BASE"][0])]

    boundary_conditions = BoundaryConditions(dirichlet=dirichlet_bc,
                                             neumann=neumann_bc,
                                             robin=robin_bc)

    return boundary_conditions


def make_passive_control(params, geometry):

    # Material parameters

    # Create an object for each single material parameter
    if params["matparams_space"] == "regional":
        cfun = merge_control(geometry, params["merge_passive_control"])
        passive_control_ = RegionalParameter(cfun, name='material control')

    else:

        family, degree = params["matparams_space"].split("_")
        matparams_space = dolfin.FunctionSpace(geometry.mesh,
                                               family, int(degree))
        passive_control_ = dolfin_adjoint.Function(matparams_space,
                                                   name="material control")

    # If we want to estimate more than one parameter
    if number_of_passive_controls(params) <= 1:
        # If there is only one parameter, just pick the same object
        passive_control = passive_control_

        # If there is none then
        if number_of_passive_controls(params) == 0:
            logger.debug("All material paramters are fixed")
            params["optimize_matparams"] = False

    else:

        # Otherwise, we make a mixed parameter
        passive_control = MixedParameter(passive_control,
                                         number_of_passive_controls(params))

    if params["phase"] in [config.PHASES[1]]:
        # Load the material control from the result file

        # Open simulation file
        with dolfin.HDF5File(dolfin.mpi_comm_world(),
                             params["sim_file"], 'r') as h5file:

            # Get material parameter from passive phase file
            h5file.read(passive_control,
                        config.PASSIVE_INFLATION_GROUP + "/optimal_control")

    return passive_control


def get_material_parameters(params, passive_control):
    
    matparams = params["Material_parameters"].to_dict()

    nopts_par = 0
    for par, val in matparams.iteritems():

        # Check if material parameter should be fixed
        if not params["Fixed_parameters"][par]:
            # If not, then we need to put the parameter into
            # some dolfin function. Use the materal parameters from
            # the parameters as initial guess
            if params["phase"] in [config.PHASES[0], config.PHASES[2]]:

                val_const = get_constant(passive_control.value_size(), 0, val)

                if number_of_passive_controls(params) <= 1:
                    passive_control.assign(val_const)

                else:
                    passive_control.assign_sub(val_const, nopts_par)

            if number_of_passive_controls(params) <= 1:
                matparams[par] = passive_control

            else:
                matparams[par] = dolfin.split(passive_control)[nopts_par]
                nopts_par += 1

    print_material_parameters(matparams)
    
    return matparams


def print_material_parameters(matparams):

    material_print = []
    for idx, (par, v) in enumerate(matparams.iteritems()):
        this_par_print = [par]
        if isinstance(v, (float, int)):
            this_par_print.append('{:.3f}'.format(v))

        else:

            if v.function_space().num_sub_spaces() == 0:
                v_ = numpy_mpi.gather_broadcast(v.vector().array())

            else:
                v_ = numpy_mpi.\
                    gather_broadcast(v.split(deepcopy=True)[idx].
                                     vector().array())

            this_par_print.append('{:.3f} (mean)'.format(v_.mean()))

        material_print.append(this_par_print)


    tab = tabulate.tabulate(material_print, headers=['parameter', 'value'])
    logger.info('\n\nMaterial Parameters\n\n{}\n'.format(tab))


def make_active_control(params, geometry):

    # Contraction parameter
    if params["gamma_space"] == "regional":
        sfun = merge_control(geometry, params["merge_active_control"])
        gamma = RegionalParameter(sfun)
    else:
        gamma_family, gamma_degree = params["gamma_space"].split("_")
        gamma_space = dolfin.FunctionSpace(geometry.mesh,
                                           gamma_family, int(gamma_degree))

        gamma = dolfin_adjoint.Function(gamma_space,
                                        name='activation parameter')

    return gamma


def merge_control(geometry, control_str):

    sfun = dolfin.MeshFunction("size_t", geometry.mesh,
                               geometry.cfun.dim())
    sfun.set_values(geometry.cfun.array())
    if control_str != "":
        for v in control_str.split(":"):
            vals = sorted(np.array(v.split(","), dtype=int))
            min_val = vals[0]
            for vi in vals[1:]:
                sfun.array()[sfun.array() == vi] = min_val

    return sfun
