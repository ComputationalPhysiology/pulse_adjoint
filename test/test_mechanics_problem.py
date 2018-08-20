
import dolfin
from pulse_adjoint.setup_parameters import setup_general_parameters
from pulse_adjoint.mechanicsproblem import (MechanicsProblem,
                                            BoundaryConditions, NeumannBC)
from pulse_adjoint.geometry import (Geometry, Marker,
                                    Microstructure, MarkerFunctions)
from pulse_adjoint.dolfin_utils import QuadratureSpace
from pulse_adjoint.models.material import HolzapfelOgden


setup_general_parameters()
mesh = dolfin.UnitCubeMesh(3, 3, 3)


class Free(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > (1.0 - dolfin.DOLFIN_EPS) and on_boundary


class Fixed(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < dolfin.DOLFIN_EPS and on_boundary


ffun = dolfin.MeshFunction("size_t", mesh, 2)
ffun.set_all(0)

fixed = Fixed()
fixed_marker = 1
fixed.mark(ffun, fixed_marker)

free = Free()
free_marker = 2
free.mark(ffun, free_marker)

cfun = dolfin.MeshFunction("size_t", mesh, 3)
cfun.set_all(0)


marker_functions = MarkerFunctions(ffun=ffun, cfun=cfun)

fixed_marker = Marker(name='fixed', value=1, dimension=2)
free_marker = Marker(name='free', value=2, dimension=2)
markers = (fixed_marker, free_marker)

V_f = QuadratureSpace(mesh, 4)

f0 = dolfin.interpolate(
    dolfin.Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
s0 = dolfin.interpolate(
    dolfin.Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
n0 = dolfin.interpolate(
    dolfin.Expression(("0.0", "0.0", "1.0"), degree=1), V_f)


microstructure = Microstructure(f0=f0, s0=s0, n0=n0)

geometry = Geometry(mesh=mesh, markers=markers,
                    marker_functions=marker_functions,
                    microstructure=microstructure)

material_parameters = HolzapfelOgden.default_parameters()

active_model = "active_strain"
compressible_model = "incompressible"

activation = dolfin.Constant(0.0)

material = HolzapfelOgden(active_model=active_model,
                          params=material_parameters,
                          gamma=activation,
                          compressible_model=compressible_model)


def dirichlet_bc(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    return dolfin.DirichletBC(V,
                              dolfin.Constant((0.0, 0.0, 0.0)),
                              fixed)


neumann_bc = NeumannBC(traction=dolfin.Constant(-0.3),
                       marker=free_marker.value)

bcs = BoundaryConditions(dirichlet=(dirichlet_bc,),
                         neumann=(neumann_bc,))

problem = MechanicsProblem(geometry, material, bcs)
problem.solve()

u, p = problem.state.split(deepcopy=True)


# from IPython import embed; embed()
# exit()
