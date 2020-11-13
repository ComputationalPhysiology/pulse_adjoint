import dolfin
import dolfin_adjoint
import matplotlib.pyplot as plt
import pulse

import pulse_adjoint


def create_problem():

    geometry = pulse.HeartGeometry.from_file(pulse.mesh_paths["simple_ellipsoid"])

    activation_control = dolfin_adjoint.Function(
        dolfin.FunctionSpace(geometry.mesh, "R", 0), name="activation"
    )
    activation_control.assign(dolfin_adjoint.Constant(0.0))
    matparams = pulse.HolzapfelOgden.default_parameters()

    V = dolfin.FunctionSpace(geometry.mesh, "R", 0)
    material_control = dolfin_adjoint.Function(V)
    material_control.assign(dolfin_adjoint.Constant(1.0))
    # control = dolfin_adjoint.Constant(1.0, name="matparam control (a)")
    matparams["a"] = material_control

    f0 = dolfin_adjoint.Function(geometry.f0.function_space())
    f0.assign(geometry.f0)
    s0 = dolfin_adjoint.Function(geometry.s0.function_space())
    s0.assign(geometry.s0)
    n0 = dolfin_adjoint.Function(geometry.n0.function_space())
    n0.assign(geometry.n0)

    material = pulse.HolzapfelOgden(
        activation=activation_control,
        parameters=matparams,
        f0=f0,
        s0=s0,
        n0=n0,
        active_model="active_strain",
    )

    # LV Pressure
    lvp = dolfin_adjoint.Constant(0.0, name="lvp")
    lv_marker = geometry.markers["ENDO"][0]
    lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
    neumann_bc = [lv_pressure]

    # Add spring term at the base with stiffness 1.0 kPa/cm^2
    base_spring = 1.0
    robin_bc = [
        pulse.RobinBC(
            value=dolfin_adjoint.Constant(base_spring, name="base_spring"),
            marker=geometry.markers["BASE"][0],
        )
    ]

    # Fix the basal plane in the longitudinal direction
    # 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
    def fix_basal_plane(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        bc = dolfin_adjoint.DirichletBC(
            V.sub(0),
            dolfin.Constant(0.0, name="fix_base"),
            geometry.ffun,
            geometry.markers["BASE"][0],
        )
        return bc

    dirichlet_bc = [fix_basal_plane]

    # Collect boundary conditions
    bcs = pulse.BoundaryConditions(
        dirichlet=dirichlet_bc, neumann=neumann_bc, robin=robin_bc
    )

    # Create the problem
    problem = pulse.MechanicsProblem(geometry, material, bcs)

    return problem, material_control, activation_control


def main():

    pulse.annotation.annotate = True

    # Set up problem
    problem, material_control, activation_control = create_problem()
    computed_volumes = [problem.geometry.cavity_volume()]

    # Create volume observersion
    endo = problem.geometry.markers["ENDO"][0]
    volume_model = pulse_adjoint.model_observations.VolumeObservation(
        problem.geometry.mesh, problem.geometry.ds(endo)
    )

    passive_volume_data = [2.511304019359619, 2.8]
    active_volume_data = [2.8, 1.5, 1.2, 1.2, 2.8]
    volume_data = passive_volume_data + active_volume_data

    passive_pressure_data = [0, 0.1]
    active_pressure_data = [1.0, 1.2, 1.0, 0.05, 0.1]
    pressure_data = passive_pressure_data + active_pressure_data

    if 0:
        fig, ax = plt.subplots()
        ax.plot(volume_data, pressure_data)
        ax.set_xlabel("Volume [ml]")
        ax.set_ylabel("Pressure [kPa]")
        plt.show()

    # ------------- Passive phase ----------------

    volume_target = pulse_adjoint.OptimizationTarget(
        passive_volume_data[1:], volume_model
    )

    pressure_obs = pulse_adjoint.model_observations.BoundaryObservation(
        problem.bcs.neumann[0],
        passive_pressure_data[1:],
        start_value=passive_pressure_data[0],
    )

    assimilator = pulse_adjoint.Assimilator(
        problem, targets=[volume_target], bcs=pressure_obs, control=material_control
    )

    optimal_control = assimilator.assimilate(min_value=0.1, max_value=10.0, tol=0.01)

    material_control.assign(dolfin.Constant(optimal_control.optimal_control))
    optimzed_material_paramter = optimal_control.optimal_control

    problem.solve()
    u, p = dolfin.split(problem.state)
    computed_volume = problem.geometry.cavity_volume(u=u)
    computed_volumes.append(computed_volume)
    print(f"Target volume: {passive_volume_data[-1]}")
    print(f"Model volume: {computed_volume}")
    print(f"Estimated activation parameters {optimzed_material_paramter}")

    prev_pressure = passive_pressure_data[-1]
    activation_parameters = [0] * len(passive_pressure_data)
    # ------------- Active phase ----------------

    for volume, pressure in zip(active_volume_data, active_pressure_data):
        print(f"Try to fit volume : {volume} with pressure {pressure}")

        volume_target = pulse_adjoint.OptimizationTarget([volume], volume_model)

        pressure_obs = pulse_adjoint.model_observations.BoundaryObservation(
            problem.bcs.neumann[0], [pressure], start_value=prev_pressure
        )

        assimilator = pulse_adjoint.Assimilator(
            problem,
            targets=[volume_target],
            bcs=pressure_obs,
            control=activation_control,
        )

        optimal_control = assimilator.assimilate(min_value=0.0, max_value=0.3, tol=0.01)

        activation_control.assign(dolfin.Constant(optimal_control.optimal_control))
        problem.solve()
        u, p = dolfin.split(problem.state)
        computed_volume = problem.geometry.cavity_volume(u=u)
        computed_volumes.append(computed_volume)
        activation_parameters.append(optimal_control.optimal_control)

        print(f"Target volume: {volume}")
        print(f"Model volume: {computed_volume}")
        print(f"Estimated activation parameters {optimal_control.optimal_control}")

        prev_pressure = pressure

    fig, ax = plt.subplots(2, 1)
    ax[0].plot(volume_data, pressure_data, label="Data")
    ax[0].plot(computed_volumes, pressure_data, label="Data")
    ax[0].set_title(f"Material parameter {optimzed_material_paramter:.2f}")
    ax[1].plot(activation_parameters, marker="o")
    ax[1].set_title("Active strain")
    fig.savefig("results.png")


if __name__ == "__main__":
    main()
