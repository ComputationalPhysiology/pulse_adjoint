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
"""
Unloading will find the reference geometry.
Here we also want to match the volumes or
strains and estimate the material parameteres
based on this
"""

__author__ = "Henrik Finsberg (henriknf@simula.no)"

import numpy as np
import dolfin

from . import unloader
from . import utils
# from unloader import *
# from utils import *



from ..geometry import Geometry
from ..setup_optimization import (make_passive_control,
                                  make_mechanics_problem,
                                  get_measurements,
                                  get_material_parameters)
from ..run_optimization import (run_passive_optimization_step,
                                solve_oc_problem, store)

from ..utils import make_logger, Text
from .. import config
from .. import numpy_mpi
from .. import dolfin_utils
from .. import io_utils
from ..clinical_data import ClinicalData
from .. import Patient

logger = make_logger(__name__, config.log_level)


class UnloadedMaterial(object):
    """
    This class finds the unloaded cofiguration assuming
    that the given geometry is loaded with `p_geo`.
    It iteratively estimate material parameters and unload
    the geometry until the relative difference between the
    current and previous volumes of the referece configuration
    is less than given tolerace

    This method is similar to the one described in [1].

    Parameters
    ----------

    geometry_index : int
        Index from which the given geometry correspond to in the
        given data.
    pressure : list
        List of pressure used for esimating material paramteters, the first
        entry in the list begin the same as the pressure in the geometry
        (`p_geo`). If BiV provide a list of tuples
    volumes : list
        List of volumes used for estimating material parameters, each volume
        corresponding to the volume at the given pressure in the pressure list.
        The offset between the first volume and the volume in the geometry
        will be subtracted.
    params : dict
        Application parameters from pulse_adjoint.setup_parameters.
        setup_adjoint_contraction_parameters.
        Used to setup the solver paramteteres. Note that the path to
        the original mesh should be in this
        dictionary, with key `Patient_parameters/mesh_path`.
        The output file will be saved to `sim_file`
    method : str
        Which method to use for unloading.
        Options = ['fixed_point', 'raghavan', 'hybrid'].
        Default = 'hybrid'. For more info see :func`unloader.py`.
    tol : float
        Relative tolerance for difference in reference volume. Default = 5%.
    maxiter : int
        Maximum number of iterations of unloading/estimate parameters.
    unload_options: dict
        More info see :func`unloader.py`.

    Reference
    ---------
    .. [1] Nikou, Amir, et al. "Effects of using the unloaded configuration
           in predicting the in vivo diastolic properties of the heart.
           " Computer methods in biomechanics
           and biomedical engineering 19.16 (2016): 1714-1720.

    """
    def __init__(self, geometry_index,
                 pressures, volumes,
                 params, paramvec,
                 method='fixed_point',
                 tol=0.05, maxiter=10,
                 continuation=True,
                 unload_options=None,
                 optimize_matparams=True):

        p0 = pressures[0]
        self.it = 0
        self.is_biv = isinstance(p0, tuple) and len(p0) == 2
        self.params = params

        self.continuation = continuation
        self.optimize_matparams = optimize_matparams

        self.geometry_index = geometry_index
        self.calibrate_data(volumes, pressures)

        self._backward_displacement = None

        self.unload_options = UnloadedMaterial.default_unload_options()
        if unload_options is not None:
            self.unload_options.update(**unload_options)

        self._paramvec = paramvec.copy(deepcopy=True)

        # 5% change
        self.tol = tol
        self.maxiter = maxiter

        if method == "hybrid":
            self.MeshUnloader = unloader.Hybrid
        elif method == "fixed_point":
            self.MeshUnloader = unloader.FixedPoint
        elif method == "raghavan":
            self.MeshUnloader = unloader.Raghavan
        else:
            methods = ['fixed_point', 'raghavan', 'hybrid']
            msg = ("Unknown unloading algorithm {}. "
                   "Possible values are {}").format(method, methods)
            raise ValueError(msg)

        msg = "\n\n"+" Start Unloaded Material Estimation  ".center(72, "#")
        msg += ("\n\n\tgeometry_index = {geometry_index}\n"
                "\tpressures = {pressures}\n"
                "\tvolumes = {volumes}\n"
                "\tUnloading algorithm = {method}\n"
                "\ttolerance = {tol}\n"
                "\tmaxiter = {maxiter}\n"
                "\tcontinuation= {continuation}\n\n"
                "".center(72, "#") +
                "\n").format(geometry_index=geometry_index,
                             pressures=self.pressures,
                             volumes=self.volumes,
                             method=method,
                             tol=tol,
                             maxiter=maxiter,
                             continuation=continuation)
        logger.info(msg)

    @staticmethod
    def default_unload_options():
        return dict(maxiter=10,
                    tol=1e-2,
                    regen_fibers=True)

    def calibrate_data(self, volumes, pressures):

        p = self.params["Patient_parameters"]
        geometry = Geometry.from_file(h5name=p["mesh_path"],
                                      h5group=p["mesh_group"])

        if self.is_biv:
            v_lv = dolfin_utils.get_volume(geometry, chamber="lv")
            v_lv_offset = v_lv - np.array(volumes).T[0][self.geometry_index]
            lv_volumes = np.add(np.array(volumes).T[0], v_lv_offset).tolist()
            logger.info("LV volume offset: {} ml".format(v_lv_offset))

            v_rv = dolfin_utils.get_volume(geometry, chamber="rv")
            v_rv_offset = v_rv - np.array(volumes).T[1][self.geometry_index]
            rv_volumes = np.add(np.array(volumes).T[1], v_rv_offset).tolist()
            logger.info("RV volume offset: {} ml".format(v_rv_offset))

            self.volumes = zip(lv_volumes, rv_volumes)

        else:

            v_lv = dolfin_utils.get_volume(geometry, chamber="lv")
            v_lv_offset = v_lv - np.array(volumes).T[0]
            lv_volumes = np.add(np.array(volumes), v_lv_offset).tolist()
            logger.info("LV volume offset: {} ml".format(v_lv_offset))

            self.volumes = lv_volumes

        self.pressures = np.array(pressures).tolist()
        self.p_geo = self.pressures[self.geometry_index]

    def unload(self):

        p = self.params["Patient_parameters"]
        geometry = Geometry.from_file(h5name=p["mesh_path"],
                                      h5group=p["mesh_group"])

        passive_control = make_passive_control(self.params, geometry)
        matparams = get_material_parameters(self.params, passive_control)

        if self.it == 0:
            numpy_mpi.\
                assign_to_vector(passive_control.vector(),
                                 numpy_mpi.
                                 gather_broadcast(self._paramvec.
                                                  vector().array()))

        if self.it > 0:
            logger.info("Load control parmeters")
            utils.load_material_parameter(self.params["sim_file"],
                                          str(self.it-1), passive_control)

        if self.it > 1 and self.continuation:
            utils.continuation_step(self.params, self.it, passive_control)

        logger.info(("Value of control parameters = "
                     "{}".format(numpy_mpi.
                                 gather_broadcast(passive_control.
                                                  vector().array()))))

        unloader = self.MeshUnloader(geometry, self.p_geo,
                                     matparams,
                                     self.params["sim_file"],
                                     options=self.unload_options,
                                     h5group=str(self.it), remove_old=False,
                                     params=self.params,
                                     approx=self.params["volume_approx"],
                                     merge_control=self.params["merge_passive_control"])

        unloader.unload()
        new_geometry = unloader.get_unloaded_geometry()
        backward_displacement = unloader.get_backward_displacement()

        group = "/".join([str(self.it), "unloaded"])
        utils.save_unloaded_geometry(new_geometry,
                                     self.params["sim_file"],
                                     group, backward_displacement)

        group = "unloaded"
        utils.save_unloaded_geometry(new_geometry,
                                     self.params["sim_file"], group)

        return Geometry.from_file(h5name=self.params['sim_file'],
                                  h5group=group)

    def get_backward_displacement(self):

        p = self.params["Patient_parameters"]
        geometry = Geometry.from_file(h5name=p["mesh_path"],
                                      h5group=p["mesh_group"])

        u = dolfin.Function(dolfin.VectorFunctionSpace(geometry.mesh, "CG", 1))

        group = "/".join([str(self.it), "unloaded", "backward_displacement"])

        with dolfin.HDF5File(dolfin.mpi_comm_world(),
                             self.params["sim_file"], "r") as h5file:
            h5file.read(u, group)

        return u

    def get_unloaded_geometry(self):

        group = "/".join([str(self.it), "unloaded"])
        try:
            return Geometry.from_file(h5name=self.params['sim_file'],
                                      h5group=group)
        except IOError as ex:
            logger.warning(ex)
            msg = ("No unloaded geometry found {}:{} \nReturn original "
                   "geometry.").format(self.params["sim_file"],
                                       group)
            logger.warning(msg)

            p = self.params["Patient_parameters"]
            geometry = Geometry.from_file(h5name=p["mesh_path"],
                                          h5group=p["mesh_group"])
            return geometry

    def get_optimal_material_parameter(self):

        paramvec = self._paramvec.copy(deepcopy=True)
        try:
            group = "/".join([str(self.it-1),
                              "passive_inflation", "optimal_control"])
            with dolfin.HDF5File(dolfin.mpi_comm_world(),
                                 self.params["sim_file"], "r") as h5file:
                h5file.read(paramvec, group)
            logger.info(("Load material parameter from {}:{}"
                         "").format(self.params["sim_file"], group))
        except Exception as ex:
            logger.warning(ex)
            logger.info("Could not open and read material parameter")

        return paramvec

    def get_loaded_volume(self, chamber="lv"):

        from pulse_adjoint.setup_optimization import get_volume
        
        geo = self.get_unloaded_geometry()
        V = dolfin.VectorFunctionSpace(geo.mesh, "CG", 2)
        u = dolfin.Function(V)
        try:
            group = "/".join([str(self.it), "passive_inflation",
                              "displacement", "1"])
            with dolfin.HDF5File(dolfin.mpi_comm_world(),
                                 self.params["sim_file"], "r") as h5file:
                h5file.read(u, group)
            logger.info(("Load displacement from {}:{}"
                        "").format(self.params["sim_file"], group))
        except Exception as ex:
            logger.warning(ex)
            logger.info("Could not open and read displacement")

        return get_volume(geo, chamber=chamber, u=u)

    def estimate_material(self):

        p = self.params["Patient_parameters"]
        original_geometry = Geometry.from_file(h5name=p["mesh_path"],
                                               h5group=p["mesh_group"])

        if self.it >= 0:
            group = "/".join([str(self.it), "unloaded"])
            logger.info(("Load geometry from {}:{}"
                        "").format(self.params["sim_file"], group))
            geometry = Geometry.from_file(h5name=self.params['sim_file'],
                                          h5group=group)
        else:
            geometry = original_geometry

        geometry.original_geometry = dolfin.Mesh(original_geometry.mesh)

        start_active = len(self.pressures)
        if self.is_biv:
            pressure = np.array(self.pressures).T[0]
            volume = np.array(self.volumes).T[0]
            RVP = np.array(self.pressures).T[1]
            RVV = np.array(self.volumes).T[1]

        else:
            pressure = self.pressures
            volume = self.volumes
            RVP = None
            RVV = None

        data = ClinicalData(start_active=start_active,
                            pressure=pressure,
                            volume=volume,
                            RVP=RVP,
                            RVV=RVV)
        patient = Patient(geometry=geometry, data=data)

        self.params["h5group"] = str(self.it)

        measurements = get_measurements(self.params, patient)
        problem, active_control, passive_control \
            = make_mechanics_problem(self.params, patient.geometry)

        p_tmp = dolfin.Function(passive_control.function_space())

        if self.it == 0:
            numpy_mpi.\
                assign_to_vector(p_tmp.vector(),
                                 numpy_mpi.
                                 gather_broadcast(self._paramvec.
                                                  vector().array()))
        else:
            # Use the previos value as initial guess
            p_tmp = dolfin.Function(passive_control.function_space())
            utils.load_material_parameter(self.params["sim_file"],
                                          str(self.it-1), p_tmp)

            if self.it > 1 and self.continuation:
                utils.continuation_step(self.params, self.it, p_tmp)

        passive_control.assign(p_tmp)

        logger.info(("Value of control parameters = "
                     + "{}".format(numpy_mpi.
                                   gather_broadcast(passive_control.
                                                    vector().array()))))

        rd, passive_control = run_passive_optimization_step(self.params,
                                                            problem,
                                                            measurements,
                                                            passive_control)

        res = solve_oc_problem(self.params, rd,
                               passive_control, return_solution=True)
        return res

    def exist(self, key="unloaded"):

        group = "/".join([str(self.it), key])

        if io_utils.check_group_exists(self.params["sim_file"], group):
            logger.info(Text.green(("{}, iteration {} - {}"
                                    "").format(key, self.it,
                                               "fetched from database")))
            return True
        else:
            logger.info(Text.blue(("{}, iteration {} - {}"
                                  "").format(key, self.it, "Run")))
            return False

    def copy_passive_inflation(self):

        group = "/".join([str(self.it), "passive_inflation"])
        io_utils.copy_h5group(h5name=self.params["sim_file"],
                              src=group, dst="passive_inflation",
                              comm=dolfin.mpi_comm_world(),
                              overwrite=True)

    def compute_residual(self, it):

        if self.it > 0:

            group1 = "/".join([str(self.it-1), "unloaded"])
            patient1 = Geometry.from_file(h5name=self.params['sim_file'],
                                          h5group=group1)

            group2 = "/".join([str(self.it), "unloaded"])
            patient2 = Geometry.from_file(h5name=self.params['sim_file'],
                                          h5group=group2)

            vol1_lv = dolfin_utils.get_volume(patient1)
            vol2_lv = dolfin_utils.get_volume(patient2)
            lv = abs(vol1_lv-vol2_lv) / vol1_lv

            if self.is_biv:
                vol1_rv = dolfin_utils.get_volume(patient1, chamber="rv")
                vol2_rv = dolfin_utils.get_volume(patient2, chamber="rv")
                rv = (vol1_rv-vol2_rv) / vol2_rv
            else:
                rv = 0.0

            return max(lv, rv)

        else:
            return np.inf

    def update_function_to_new_reference(self, fun, u, mesh=None):
        """
        Assume given function lives on the original
        geometry, and you want to find the function
        on the new referece geometry.
        Since the new referece geometry is topologically
        equal to the old reference, the vector within the
        functions should be identical.

        Note that this is only relevant for functions of
        rank 1, i.e vectors.
        """
        if mesh is None:
            geo = self.get_unloaded_geometry()
            mesh = geo.mesh

        return utils.update_vector_field(fun, mesh, u, str(fun),
                                         normalize=True,
                                         regen_fibers=False)

    def unload_material(self):

        err = np.inf
        res = None

        while self.it < self.maxiter and err > self.tol:

            dolfin.parameters["adjoint"]["stop_annotating"] = True
            if not self.exist("unloaded"):
                self.unload()

            err = self.compute_residual(self.it)
            logger.info("\nCurrent residual:\t{}".format(err))

            dolfin.parameters["adjoint"]["stop_annotating"] = False
            if not self.exist("passive_inflation"):
                res = self.estimate_material()

            self.it += 1

            if not self.optimize_matparams:
                break

        self.it -= 1

        if res is None:
            assert self.it >= 0, \
                "You need to perform at least one iteration with unloading"
            self.copy_passive_inflation()
        else:
            # Store optimization results
            res[0]["h5group"] = ""
            store(*res)

