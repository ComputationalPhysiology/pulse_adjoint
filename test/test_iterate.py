import pytest
import itertools
from pulse_adjoint.example_meshes import mesh_paths

# from pulse_adjoint.material import HolzapfelOgden
from pulse_adjoint.geometry import Geometry
from pulse_adjoint.setup_parameters2 import setup_general_parameters
from pulse_adjoint.iterate import iterate


import dolfin_adjoint
import dolfin
setup_general_parameters()

cases = itertools.product((False, True), (False, True))


@pytest.fixture
def problem():

    geometry = Geometry.from_file(mesh_paths['simple_ellipsoid'])
    opt_targets = {"volume": 1.0,
                   "regularization": 0.1}

    from utils import setup_params
    params = setup_params("passive", "CG_1", "lv",
                          opt_targets, "active_strain")

    # Changing these will make the Taylor test fail
    params["active_relax"] = 1.0
    params["passive_relax"] = 1.0

    params["phase"] = "passive_inflation"

    from pulse_adjoint.setup_optimization2 import make_mechanics_problem

    problem, active_control, passive_control \
        = make_mechanics_problem(params, geometry)

    return problem


@pytest.mark.parametrize('continuation, annotate', cases)
def test_iterate_pressure(problem, continuation, annotate):

    target_pressure = 1.0
    plv = [p.traction for p in problem.bcs.neumann if p.name == 'lv']
    pressure = {'p_lv': plv[0]}

    dolfin.parameters["adjoint"]["stop_annotating"] = not annotate
    dolfin_adjoint.adj_reset()

    iterate("pressure", problem,
            target_pressure, pressure,
            continuation=continuation)

    if annotate:
        # Check the recording
        assert dolfin_adjoint.replay_dolfin(tol=1e-12)

    # Check that the pressure is correct
    assert float(plv[0]) == target_pressure
    # Check that the state is nonzero
    assert dolfin.norm(problem.state.vector()) > 0


@pytest.mark.parametrize('continuation, annotate', cases)
def test_iterate_gamma(problem, continuation, annotate):

    target_gamma = 0.1
    gamma = problem.material.activation

    dolfin.parameters["adjoint"]["stop_annotating"] = not annotate
    dolfin_adjoint.adj_reset()

    iterate("gamma", problem,
            target_gamma, gamma,
            continuation=continuation)

    assert all(gamma.vector().array() == target_gamma)
    assert dolfin.norm(problem.state.vector()) > 0

    if annotate:
        # dolfin_adjoint.adj_html("active_forward.html", "forward")
        # dolfin_adjoint.adj_html("active_adjoint.html", "adjoint")
        # Check the recording
        assert dolfin_adjoint.replay_dolfin(tol=1e-12)


if __name__ == "__main__":

    for c, a in cases:
        print("Continuation = {}, annotate = {}".format(c, a))
        prob = problem()
        test_iterate_pressure(prob, continuation=c, annotate=a)
        dolfin_adjoint.adj_reset()

        prob = problem()
        test_iterate_gamma(prob, continuation=c, annotate=a)
        dolfin_adjoint.adj_reset()

