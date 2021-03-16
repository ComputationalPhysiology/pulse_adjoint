# A basic example of how to use pulse_adjoint to assimilate
# clinical volume data into a Bi-ventricular heart model.
# The control parameters are one material parameter and
# and an activation parameter. These parameters are spatially
# resolved on the LV, RV and Septum.
#
#
# This demo shows how to
# * Set up the forward problem (using code from Pulse)
# * Define spatially resolved control parameters on the LV, RV and Septum
# * Define multiple model observations
# * Use the assimilator class in pulse_adjoint to fit the model

# Import the necessary packages
import dolfin
import dolfin_adjoint

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Matplotlib not installed - plotting not possible")
import pulse

import pulse_adjoint


# Define the forward problem
def create_problem():

    # Define the domain
    geometry = pulse.HeartGeometry.from_file("biv_mesh.h5")

    # Define a sptially resolved material parameter
    regional_mat = pulse.RegionalParameter(geometry.cfun)

    # Assign regional values of the material control parameter
    regional_mat.vector()[0] = 0.5  # LVFW
    regional_mat.vector()[1] = 0.5  # SEPT
    regional_mat.vector()[2] = 0.5  # RVFW

    # Define the material control function
    # material_control = dolfin_adjoint.Function(
    #     dolfin.FunctionSpace(geometry.mesh, "CG", 1), name="material"
    # )

    # # Assign regional material parameter to material control function
    # material_control.assign(regional_mat)

    # Define spatially resolved activation control parameter
    regional_activation = pulse.RegionalParameter(geometry.cfun)
    # from IPython import embed

    # embed()
    # exit()
    # Assign regional values of the activation control parameter
    regional_activation.vector()[0] = 0  # LVFW
    regional_activation.vector()[1] = 0  # SEPT
    regional_activation.vector()[2] = 0  # RVFW

    # Define the activation control function
    # activation_control = dolfin_adjoint.Function(
    #     dolfin.FunctionSpace(geometry.mesh, "CG", 1), name="activation"
    # )

    # # Assign regional activation parameter to activation control function
    # activation_control.assign(regional_activation)

    # Define the parameters of the material model
    matparams = pulse.HolzapfelOgden.default_parameters()

    # Update parameter in material model
    matparams["a"] = regional_mat

    # Define the microstructure
    f0 = dolfin_adjoint.Function(geometry.f0.function_space())
    f0.assign(geometry.f0)
    # s0 = dolfin_adjoint.Function(geometry.s0.function_space())
    # s0.assign(geometry.s0)
    # n0 = dolfin_adjoint.Function(geometry.n0.function_space())
    # n0.assign(geometry.n0)

    material = pulse.HolzapfelOgden(
        activation=regional_activation,
        parameters=matparams,
        f0=f0,
        s0=None,
        n0=None,
        active_model="active_strain",
    )

    # LV Pressure
    lvp = dolfin_adjoint.Constant(0.0, name="lvp")
    lv_marker = geometry.markers["ENDO_LV"][0]
    lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")

    # RV Pressure
    rvp = dolfin_adjoint.Constant(0.0, name="rvp")
    rv_marker = geometry.markers["ENDO_RV"][0]
    rv_pressure = pulse.NeumannBC(traction=rvp, marker=rv_marker, name="rv")

    neumann_bc = [lv_pressure, rv_pressure]

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
    problem.solve()
    return problem, regional_mat, regional_activation


# Define the inverse problem
def main():

    pulse.annotation.annotate = True

    # Set up problem
    problem, material_control, activation_control = create_problem()

    # Create a file for checkpointing
    disp_file = dolfin.XDMFFile("displacement.xdmf")
    (U, P) = problem.state.split(deepcopy=True)
    disp_file.write_checkpoint(
        U, "displacement", 0, dolfin.XDMFFile.Encoding.HDF5, False
    )

    # Store the model computed cavity volumes
    computed_volumes_LV = [problem.geometry.cavity_volume(chamber="lv")]
    computed_volumes_RV = [problem.geometry.cavity_volume(chamber="rv")]

    # Create volume observations
    endo_LV = problem.geometry.markers["ENDO_LV"][0]
    endo_RV = problem.geometry.markers["ENDO_RV"][0]

    volume_model_LV = pulse_adjoint.model_observations.VolumeObservation(
        problem.geometry.mesh, problem.geometry.ds(endo_LV)
    )

    volume_model_RV = pulse_adjoint.model_observations.VolumeObservation(
        problem.geometry.mesh, problem.geometry.ds(endo_RV)
    )

    # Define the clinical volume and pressure data
    passive_volume_data_LV = [2.511304019359619, 2.8]
    active_volume_data_LV = [2.8, 1.5, 1.2, 1.2, 2.8]
    volume_data_LV = passive_volume_data_LV + active_volume_data_LV

    passive_volume_data_RV = [2.511304019359619, 2.8]
    active_volume_data_RV = [2.8, 1.5, 1.2, 1.2, 2.8]
    volume_data_RV = passive_volume_data_RV + active_volume_data_RV

    passive_pressure_data_LV = [0, 0.1]
    active_pressure_data_LV = [1.0, 1.2, 1.0, 0.05, 0.1]
    pressure_data_LV = passive_pressure_data_LV + active_pressure_data_LV

    passive_pressure_data_RV = [0, 0.1]
    active_pressure_data_RV = [1.0, 1.2, 1.0, 0.05, 0.1]
    pressure_data_RV = passive_pressure_data_RV + active_pressure_data_RV

    if 0:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.set_title("LV PV Loop")
        ax2.set_title("RV PV Loop")
        ax1.plot(volume_data_LV, pressure_data_LV)
        ax2.plot(volume_data_RV, pressure_data_LV)
        # ax.set_xlabel("Volume [ml]")
        # ax.set_ylabel("Pressure [kPa]")
        plt.show()

    # ------------- Passive phase ----------------

    # Define optimization targets
    volume_target_LV = pulse_adjoint.OptimizationTarget(
        passive_volume_data_LV[1:], volume_model_LV
    )

    volume_target_RV = pulse_adjoint.OptimizationTarget(
        passive_volume_data_RV[1:], volume_model_RV
    )

    # Define regularization term for the passive phase optimization
    material_regularization = pulse_adjoint.Regularization(
        material_control, weight=1e-4, reg_type="L2"
    )

    # Define the boundary data which will drive the model
    pressure_obs_LV = pulse_adjoint.model_observations.BoundaryObservation(
        problem.bcs.neumann[0],
        passive_pressure_data_LV[1:],
        start_value=passive_pressure_data_LV[0],
    )

    pressure_obs_RV = pulse_adjoint.model_observations.BoundaryObservation(
        problem.bcs.neumann[1],
        passive_pressure_data_RV[1:],
        start_value=passive_pressure_data_RV[0],
    )

    # Create an assimilator object collecting the information
    # necessary for the optimization
    assimilator = pulse_adjoint.Assimilator(
        problem,
        targets=[volume_target_LV, volume_target_RV],
        bcs=[pressure_obs_LV, pressure_obs_RV],
        control=material_control,
        regularization=material_regularization,
    )

    # Solve the inverse problem for the passive phase
    optimal_control = assimilator.assimilate(min_value=0.1, max_value=10.0, tol=1e-6)

    # Update the material parameter to the optimized value
    material_control.vector()[:] = optimal_control.optimal_control
    dolfin.File("material.pvd") << material_control
    optimzed_material_paramter = optimal_control.optimal_control

    problem.solve()
    u, p = dolfin.split(problem.state)
    computed_volume_LV = problem.geometry.cavity_volume(u=u, chamber="lv")
    computed_volume_RV = problem.geometry.cavity_volume(u=u, chamber="rv")
    computed_volumes_LV.append(computed_volume_LV)
    computed_volumes_RV.append(computed_volume_RV)
    print(f"Target volume_LV: {passive_volume_data_LV[-1]}")
    print(f"Model volume_LV: {computed_volume_LV}")
    print(f"Target volume_RV: {passive_volume_data_RV[-1]}")
    print(f"Model volume_RV: {computed_volume_RV}")
    print(f"Estimated material parameters {optimzed_material_paramter}")

    (U, P) = problem.state.split(deepcopy=True)
    disp_file.write_checkpoint(
        U, "displacement", 1, dolfin.XDMFFile.Encoding.HDF5, True
    )

    prev_pressure_LV = passive_pressure_data_LV[-1]
    prev_pressure_RV = passive_pressure_data_RV[-1]
    activation_parameters = [0] * len(passive_pressure_data_LV)

    # ------------- Active phase ----------------

    # Define regularization term for the active phase optimization
    gamma_regularization = pulse_adjoint.Regularization(
        activation_control, weight=1e-4, reg_type="L2"
    )

    # Create a file for checkpointing
    gamma_file = dolfin.XDMFFile("activation.xdmf")
    gamma_file.write_checkpoint(
        activation_control, "gamma", 0, dolfin.XDMFFile.Encoding.HDF5, False
    )

    for i, (volume_lv, pressure_lv, volume_rv, pressure_rv) in enumerate(
        zip(
            active_volume_data_LV,
            active_pressure_data_LV,
            active_volume_data_RV,
            active_pressure_data_RV,
        ),
    ):
        print(
            f"Try to fit LV volume : {volume_lv} with LV pressure {pressure_lv}\n"
            f"Try to fit RV volume : {volume_rv} with RV pressure {pressure_rv}"
        )

        volume_target_LV = pulse_adjoint.OptimizationTarget(
            [volume_lv], volume_model_LV
        )
        volume_target_RV = pulse_adjoint.OptimizationTarget(
            [volume_rv], volume_model_RV
        )

        pressure_obs_LV = pulse_adjoint.model_observations.BoundaryObservation(
            problem.bcs.neumann[0], [pressure_lv], start_value=prev_pressure_LV
        )

        pressure_obs_RV = pulse_adjoint.model_observations.BoundaryObservation(
            problem.bcs.neumann[1], [pressure_rv], start_value=prev_pressure_RV
        )

        assimilator = pulse_adjoint.Assimilator(
            problem,
            targets=[volume_target_LV, volume_target_RV],
            bcs=[pressure_obs_LV, prev_pressure_RV],
            control=activation_control,
            regularization=gamma_regularization,
        )

        optimal_control = assimilator.assimilate(min_value=0.0, max_value=0.3, tol=0.01)

        activation_control.vector()[:] = optimal_control.optimal_control
        gamma_file.write_checkpoint(
            activation_control, "gamma", i + 1, dolfin.XDMFFile.Encoding.HDF5, True
        )
        problem.solve()
        u, p = dolfin.split(problem.state)
        computed_volume_LV = problem.geometry.cavity_volume(u=u, chamber="lv")
        computed_volume_RV = problem.geometry.cavity_volume(u=u, chamber="rv")
        computed_volumes_LV.append(computed_volume_LV)
        computed_volumes_RV.append(computed_volume_RV)
        activation_parameters.append(optimal_control.optimal_control)

        print(f"Target volume_LV: {volume_lv}")
        print(f"Model volume_LV: {computed_volume_LV}")
        print(f"Target volume_RV: {volume_rv}")
        print(f"Model volume_RV: {computed_volume_RV}")
        print(f"Estimated activation parameters {optimal_control.optimal_control}")

        prev_pressure_LV = pressure_lv
        prev_pressure_RV = pressure_rv

        (U, P) = problem.state.split(deepcopy=True)
        disp_file.write_checkpoint(
            U, "displacement", i + 2, dolfin.XDMFFile.Encoding.HDF5, True
        )

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("LV PV Loop")
    ax2.set_title("RV PV Loop")
    ax1.plot(volume_data_LV, pressure_data_LV, label="Data")
    ax1.plot(computed_volumes_LV, pressure_data_LV, label="Model")
    ax2.plot(volume_data_RV, pressure_data_RV, label="Data")
    ax2.plot(computed_volumes_RV, pressure_data_RV, label="Model")
    # ax.set_xlabel("Volume")
    # ax.set_ylabel("Pressure")
    ax1.legend()
    ax2.legend()
    fig.savefig("results.png")


if __name__ == "__main__":
    main()
