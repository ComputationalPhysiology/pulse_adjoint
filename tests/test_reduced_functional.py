import dolfin
import fixtures
import pulse

import pulse_adjoint


def test_create_reduced_functional():

    start_control = 1.0
    target_control = 2.0
    target_pressure = 0.5

    problem, control = fixtures.create_problem("R_0", control_value=start_control)
    problem2, control2 = fixtures.create_problem("R_0", control_value=target_control)

    pressure_data = [target_pressure]

    pulse.iterate.iterate(problem2, problem2.bcs.neumann[0].traction, target_pressure)
    u, p = dolfin.split(problem2.state)

    volume_data = [problem2.geometry.cavity_volume(u=u)]

    # Create volume observersion
    endo = problem.geometry.markers["ENDO"][0]
    volume_model = pulse_adjoint.model_observations.VolumeObservation(
        problem.geometry.mesh, problem.geometry.ds(endo)
    )
    volume_target = pulse_adjoint.OptimizationTarget(volume_data, volume_model)

    pressure_obs = pulse_adjoint.model_observations.BoundaryObservation(
        problem.bcs.neumann[0], pressure_data
    )

    assimilator = pulse_adjoint.Assimilator(
        problem, targets=[volume_target], bcs=pressure_obs, control=control
    )

    rd = assimilator.create_reduced_functional()
    assert abs(rd(target_control)) < 1e-12


if __name__ == "__main__":
    test_create_reduced_functional()
