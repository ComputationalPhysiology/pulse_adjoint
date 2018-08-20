import pytest
import logging
import dolfin_adjoint


from pulse_adjoint.example_meshes import mesh_paths
from pulse_adjoint import Patient
# from pulse_adjoint.material import HolzapfelOgden
from pulse_adjoint.geometry import Geometry
from pulse_adjoint.setup_parameters import setup_general_parameters

from test_clinical_data import sample_clinical_data

setup_general_parameters()


def test_adjoint_calculations(geometry, clinical_data):

    patient = Patient(geometry=geometry, data=clinical_data)
    opt_targets = {"volume": 1.0,
                   "regularization": 0.1}

    from utils import setup_params, my_taylor_test
    params = setup_params("passive", "CG_1", "lv",
                          opt_targets, "active_strain")

    # Changing these will make the Taylor test fail
    params["active_relax"] = 1.0
    params["passive_relax"] = 1.0

    params["phase"] = "passive_inflation"

    from pulse_adjoint.setup_optimization import (make_mechanics_problem,
                                                  get_measurements)
    measurements = get_measurements(params, patient)
    problem, active_control, passive_control \
        = make_mechanics_problem(params, patient.geometry)

    from pulse_adjoint.run_optimization import run_passive_optimization_step

    rd, passive_control = run_passive_optimization_step(params,
                                                        problem,
                                                        measurements,
                                                        passive_control)
    passive_control_arr = passive_control.vector().array()
    rd(passive_control_arr)

    dolfin_adjoint.adj_html("passive_forward.html", "forward")
    dolfin_adjoint.adj_html("passive_adjoint.html", "adjoint")
    assert dolfin_adjoint.replay_dolfin(tol=1e-12)

    logging.info("Taylor test")
    my_taylor_test(rd, passive_control)


def test_adjoint_calculations_active(geometry, clinical_data):

    from pulse_adjoint.run_optimization import (run_active_optimization_step,
                                                run_passive_optimization)

    patient = Patient(geometry=geometry, data=clinical_data)
    # opt_targets = ["volume", "regional_strain", "regularization"]
    opt_targets = {"volume": 0.5,
                   "regional_strain": 0.5,
                   "regularization": 0.1}
    from utils import setup_params, my_taylor_test
    params = setup_params("active", "regional", "lv",
                          opt_targets, "active_strain")

    # Changing these will make the Taylor test fail
    params["active_relax"] = 1.0
    params["passive_relax"] = 1.0

    params["phase"] = "passive_inflation"
    params["optimize_matparams"] = False
    run_passive_optimization(params, patient)
    dolfin_adjoint.adj_reset()
    print("AJAJAJAJAAJAJAJAJAJ")
    params = setup_params("active", "regional", "lv",
                          opt_targets, "active_strain")

    # Changing these will make the Taylor test fail
    params["active_relax"] = 1.0
    params["passive_relax"] = 1.0
    params["phase"] = "active_contraction"

    from pulse_adjoint.setup_optimization import (make_mechanics_problem,
                                                   get_measurements)
    measurements = get_measurements(params, patient)
    problem, active_control, passive_control \
        = make_mechanics_problem(params, patient.geometry)

    rd, gamma = run_active_optimization_step(params,
                                             problem,
                                             measurements,
                                             active_control)

    gamma_arr = gamma.vector().array()
    rd(gamma_arr)

    dolfin_adjoint.adj_html("active_forward.html", "forward")
    dolfin_adjoint.adj_html("active_adjoint.html", "adjoint")
    assert dolfin_adjoint.replay_dolfin(tol=1e-12)

    logging.info("Taylor test")
    my_taylor_test(rd, gamma)


if __name__ == "__main__":
    # geo = geometry()
    # data = clinical_data(geo)

    geo = Geometry.from_file(mesh_paths['simple_ellipsoid'])
    data = sample_clinical_data(4, geo.regions)

    # test_adjoint_calculations(geo, data)
    test_adjoint_calculations_active(geo, data)
