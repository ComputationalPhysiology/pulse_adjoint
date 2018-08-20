import dolfin
import dolfin_adjoint

import pulse
import pulse_assimilator


def create_problem():

    geometry = pulse.HeartGeometry.from_file(pulse.mesh_paths['simple_ellipsoid'])

    activation = dolfin_adjoint.Function(dolfin.FunctionSpace(geometry.mesh, "R", 0),
                                         name="activation")
    activation.assign(dolfin_adjoint.Constant(0.0))
    matparams = pulse.HolzapfelOgden.default_parameters()

    control = dolfin_adjoint.Constant(1.0, name="matparam control (a)")
    matparams['a'] = control

    f0 = dolfin_adjoint.Function(geometry.f0.function_space())
    f0.assign(geometry.f0)
    s0 = dolfin_adjoint.Function(geometry.s0.function_space())
    s0.assign(geometry.s0)
    n0 = dolfin_adjoint.Function(geometry.n0.function_space())
    n0.assign(geometry.n0)

   
    
    material = pulse.HolzapfelOgden(activation=activation,
                                    parameters=matparams,
                                    f0=f0,
                                    s0=s0,#dolfin_adjoint.Function(geometry.s0),
                                    n0=n0)#dolfin_adjoint.Function(geometry.n0))

    # LV Pressure
    lvp = dolfin_adjoint.Constant(0.0, name="lvp")
    lv_marker = geometry.markers['ENDO'][0]
    lv_pressure = pulse.NeumannBC(traction=lvp,
                                  marker=lv_marker, name='lv')
    neumann_bc = [lv_pressure]

    # Add spring term at the base with stiffness 1.0 kPa/cm^2
    base_spring = 1.0
    robin_bc = [pulse.RobinBC(value=dolfin_adjoint.Constant(base_spring, name="base_spring"),
                              marker=geometry.markers["BASE"][0])]

    # Fix the basal plane in the longitudinal direction
    # 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
    def fix_basal_plane(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        bc = dolfin_adjoint.DirichletBC(V.sub(0),
                                        dolfin.Constant(0.0, name="fix_base"),
                                        geometry.ffun, geometry.markers["BASE"][0])
        return bc

    dirichlet_bc = [fix_basal_plane]

    # Collect boundary conditions
    bcs = pulse.BoundaryConditions(dirichlet=dirichlet_bc,
                                   neumann=neumann_bc,
                                   robin=robin_bc)

    # Create the problem
    problem = pulse.MechanicsProblem(geometry, material, bcs)

    return problem, control


def main():

    pulse.annotate.annotate = True
    # dolfin_adjoint.continue_annotation()
    problem, control = create_problem()

    pressure = [0.1]
    volume = [3.0]

    data = pulse_assimilator.ClinicalData(pressure=pressure,
                                          volume=volume)

    assimilator = pulse_assimilator.Assimilator(problem,
                                                data,
                                                control)

    optimal_control = assimilator.assimilate()

    
    from IPython import embed; embed()
    exit()
    

if __name__ == "__main__":
    main()
