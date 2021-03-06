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
import os
from pulse.numpy_mpi import *
import pulse

from .utils import logger
from . import load, utils

mesh_types = ["lv", "biv"]
patient_types = ["full", "lv", "biv", "pah", "work", "test"]
curdir = os.path.dirname(os.path.abspath(__file__))


def get_patient_class(patient_type, params):

    msg = "Expected patient_type to be one of the following:" "\n{}, got {}".format(
        patient_types, patient_type
    )
    assert patient_type in patient_types, msg

    assert "mesh_type" in params, "Please provide a mesh_type"
    msg = "Expected mesh_type to be one of the following:" "\n{}, got {}".format(
        mesh_types, params["mesh_type"]
    )
    assert params["mesh_type"] in mesh_types, msg

    if patient_type == "test":

        if params["mesh_type"] == "lv":
            return LVTestPatient(**params)
        else:
            return BiVTestPatient(**params)

    if patient_type in ["pah", "biv"]:
        return BiVPatient(**params)
    if "mesh_type" in params and params["mesh_type"] == "biv":
        return BiVPatient(**params)

    if patient_type in ["full", "work", "lv"]:
        return FullPatient(**params)


def Patient(patient_type, **kwargs):

    return get_patient_class(patient_type, kwargs)


class BasePatient(object):
    def __init__(self, **kwargs):

        self._name = kwargs.pop("patient", "JohnDoe")
        self.h5group = kwargs.pop("mesh_group", None)

        self._set_paths(**kwargs)
        # self._check_paths()

        subsample = kwargs.pop("subsample", False)
        self._set_measurements(subsample)
        self._load_geometry(**kwargs)

        self.regions = np.array(list(set(gather_broadcast(self.sfun.array())))).tolist()

    def get_fiber_angles(self):
        return load.get_fiber_angles(self.paths["mesh_path"], self.h5group)

    def _set_strain_weights(self, **kwargs):

        """Compute weights on the strain regions according to some rule
        
        *Arguments*
          rule (string)
            Either peak value,  drift, combination, equal or custom. If peak value, regions with high strain will 
            have smaller weight. If drift, regions with high drift will have
            higher weight. If combination drift and peak value will both be considered.
            If equal, then all regions will be given the same weight
            If custom then eqaul weigths will be assigned to regions/directions in val
          
          direction (string)
            Either c (circumferential), l (longitidinal), r (radial) or None
            If l, then only the strains in the longitudinal direction will be
            given a weight. All other strains will be given a very small weight

          custom_weights (list of tupples)
            Tu be used if rule = custom.
            Example val = [("l",3), ("c", 5)]. Then region 3 and 5 in direction "l"
            and "c" respectively will be given all the weights
        """
        if hasattr(self, "strain"):

            rule = kwargs.pop("weight_rule", "equal")
            direction = kwargs.pop("weight_direction", "all")
            custom_weights = kwargs.pop("custom_weights", None)

            self.strain_weights = utils.compute_strain_weights(
                self.strain, rule, direction, custom_weights
            )
        else:
            self.strain_weights = None

    def is_biv(self):
        return "ENDO_LV" in self.markers

    def interpolate_data(self, start, n=1):
        """Interpolate data for the pressure, 
        volume and strain between start and start +1,
        and return n new points between to 
        successive ones
        """

        if n == 0:
            return

        # Possible meausurements
        attrs = ["volume", "pressure", "strain", "RVV", "RVP"]

        # The original data at the integers
        xp = list(range(self.num_points))
        # Add x-values for where to interpolate
        x = sorted(xp + np.linspace(start, start + 1, n + 2)[1:-1].tolist())

        for att in attrs:

            if not hasattr(self, att):
                continue

            arr = getattr(self, att)

            if att == "strain":
                for r, s in self.strain.items():
                    f0 = np.interp(x, xp, np.transpose(s)[0]).tolist()
                    f1 = np.interp(x, xp, np.transpose(s)[1]).tolist()
                    f2 = np.interp(x, xp, np.transpose(s)[2]).tolist()
                    self.strain[r] = list(zip(f0, f1, f2))

            else:
                arr_int = np.interp(x, xp, arr).tolist()
                setattr(self, att, arr_int)

        if start < self.passive_filling_duration:
            self.passive_filling_duration += n
            self.num_points += n
        else:
            self.num_contract_points += n
            self.num_points += n

        self.number_of_interpolations += n

    def name(self):
        return self._name

    def mesh_type(self):
        return self._mesh_type

    def _set_paths(self, **kwargs):

        self.paths = {}

        for k, v in kwargs.items():
            if k.endswith("path"):
                self.paths[k] = v

    def _set_measurements(self, subsample):
        """
        Get volume, pressure and strain measurements

        All measuments are shiftent so that they start with
        the reference time given in self.parameters["time"]
        The strains are recomputed with reference taken at
        this time point

        Pressure is measured in kPa and volume is measured in ml 

        """

        d = load.load_measurement(**self.paths)

        for k, v in d.items():
            setattr(self, k, v)

        if self.h5group is None:
            self.h5group = (
                ""
                if not hasattr(self, "passive_filling_begins")
                else str(self.passive_filling_begins)
            )

        self.num_points = len(self.volume)
        self.num_contract_points = self.num_points - self.passive_filling_duration
        self.number_of_interpolations = 0

    def _load_geometry(self, **kwargs):

        if self.paths["mesh_path"] != "":
            geo = load.load_geometry(self.paths["mesh_path"], self.h5group, **kwargs)

            # Mapper for compatability
            mapper = dict(fiber="f0", sheet="s0", sheet_normal="n0", sfun="cfun")

            for k, v in mapper.items():
                if hasattr(geo, v):
                    setattr(self, k, getattr(geo, v))

            for k, v in geo.__dict__.items():
                setattr(self, k, v)

                if k == "markers":
                    for name, (marker, dim) in list(v.items()):
                        if dim in [1, 2]:
                            setattr(self, name, int(marker))

        else:
            logger.warinig("Mesh path does not exist")


class TestPatient(BasePatient):
    def __init__(self, **kwargs):

        self._type = "test"

        # LV pressure
        self.pressure = np.array(list(range(0, 11, 2)))

        # LV volume
        self.volume = np.array([27.387594] * 6)

        # LV regional strain
        z = np.zeros(6)
        self.strain = {i: list(zip(z, z, z)) for i in range(1, 18)}
        self.strain_weights = np.ones((17, 3))

        # Some indices
        self.passive_filling_duration = 3
        self.passive_filling_begins = 0
        self.num_contract_points = 3
        self.num_points = 6
        self.number_of_interpolations = 0

        # Some paths
        self.h5group = ""

        # self._check_paths()
        self._load_geometry(**{"include_sheets": True})

    def _check_paths(self):

        for pname in ["mesh_path"]:
            path = self.paths[pname]
            assert os.path.isfile(path)

    def get_original_echo_surfaces(self):
        raise TypeError("Not working for patient of type {}".format(self._type))


class LVTestPatient(TestPatient):
    def __init__(self, name="ellipsoid", **kwargs):

        assert name in pulse.mesh_paths
        self._name = name
        self._mesh_type = "lv"

        self.paths = {"mesh_path": pulse.mesh_paths[name]}
        TestPatient.__init__(self, **kwargs)


class BiVTestPatient(TestPatient):
    def __init__(self, **kwargs):

        self._name = "biv_test"
        self._mesh_type = "biv"
        # RV pressure
        self.RVP = np.linspace(0, 1, 6)

        # RV volume
        self.RVV = np.array([27.387594] * 6)

        self.paths = {"mesh_path": pulse.mesh_paths["biv_ellipsoid.h5"]}

        TestPatient.__init__(self, **kwargs)
        self.pressure = np.linspace(0, 2, 6)

        self.ENDO = self.ENDO_LV


class FullPatient(BasePatient):
    def __init__(self, **kwargs):
        self._type = "full"
        self._mesh_type = "lv"
        init = kwargs.pop("init", True)
        if init:
            BasePatient.__init__(self, **kwargs)
            self._set_strain_weights(**kwargs)

    def get_fiber_strain(self):

        if not hasattr(self, "_fiber_strain"):
            self._fiber_strain = utils.calculate_fiber_strain(
                self.fiber,
                self.circumferential,
                self.radial,
                self.longitudinal,
                self.sfun,
                self.mesh,
                self.strain,
            )

        return self._fiber_strain

    def get_transformation_matrix(self, time=None):

        msg = "Path to echo file is missing"
        assert "echo_path" in self.paths, msg

        from mesh_generation.mesh_utils import get_round_off_buffer, load_echo_geometry
        from mesh_generation.surface import get_geometric_matrix

        time = self.passive_filling_begins if time is None else time

        round_off = get_round_off_buffer(self.name(), time)
        echo_surfaces = load_echo_geometry(self.paths["echo_path"], time)
        endo_verts_orig = np.ones((echo_surfaces["endo_verts"].shape[0], 4))
        endo_verts_orig[:, :3] = np.copy(echo_surfaces["endo_verts"])

        T = get_geometric_matrix(
            echo_surfaces["strain_mesh"], endo_verts_orig, round_off, second_layer=False
        )
        return T

    def get_original_echo_surfaces(self):

        msg = "Path to echo file is missing"
        assert "echo_path" in self.paths, msg

        return load.get_echo_surfaces(self.paths["echo_path"])

    def _check_paths(self):

        for pname in ["echo_path", "pressure_path", "mesh_path"]:
            path = self.paths[pname]
            assert os.path.isfile(path)


class BiVPatient(BasePatient):
    def __init__(self, **kwargs):
        self._mesh_type = "biv"
        self._type = "biv"

        BasePatient.__init__(self, **kwargs)
        self.ENDO = self.ENDO_LV

        self._set_strain_weights(**kwargs)

    def _check_paths(self):

        for pname in ["pressure_path", "mesh_path"]:
            path = self.paths[pname]
            assert os.path.isfile(path)

    def _set_measurements(self, subsample):
        """
        Get volume, pressure and strain measurements

        All measuments are shiftent so that they start with
        the reference time given in self.parameters["time"]
        The strains are recomputed with reference taken at
        this time point

        Pressure is measured in kPa and volume is measured in ml 

        :param bool subsample: If true, choose every forth point

        .. todo::

           Let the user decide on the subsampling level

        """

        d = load.load_measurement(**self.paths)

        for k, v in d.items():
            if k == "pfd":
                if subsample:
                    self.passive_filling_duration = int(np.ceil(v / 4.0)) + 1
                else:
                    self.passive_filling_duration = v
            else:
                if subsample:
                    setattr(self, k, v[::4])
                else:
                    setattr(self, k, v)

        # Rename LVV and LVP to volume and presure for easier porting
        for a in [("LVV", "volume"), ("LVP", "pressure")]:
            setattr(self, a[1], getattr(self, a[0]))
            delattr(self, a[0])

        if not hasattr(self, "passive_filling_begins"):
            self.passive_filling_begins = 0
            if not hasattr(self, "h5group"):
                self.h5group = ""
        else:
            if not hasattr(self, "h5group"):
                self.h5group = str(self.passive_filling_begins)

        self.num_points = len(self.volume)
        self.num_contract_points = self.num_points - self.passive_filling_duration
        self.number_of_interpolations = 0


if __name__ == "__main__":

    main_dir = "../../impact_stress_article/"

    d = {
        "patient": "Henrik",
        "mesh_type": "lv",
        "patient_type": "full",
        "echo_path": "../h5_data/US_simHenrik.h5",
        "pressure_path": "/".join([main_dir, "pv_data", "Henrik.yml"]),
        "mesh_path": "/".join([main_dir, "meshes", "Henrik_low_res_lv.h5"]),
    }

    patient = Patient(**d)
    from IPython import embed

    embed()
    exit()
    # from IPython import embed; embed()
    # LVTestPatient()
    BiVTestPatient()
