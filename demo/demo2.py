import numpy as np
import dolfin
import dolfin_adjoint

import pulse


# Create mesh
N = 6
mesh = dolfin.UnitCubeMesh(N, N, N)


# Create subdomains
class Free(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > (1.0 - dolfin.DOLFIN_EPS) and on_boundary


class Fixed(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < dolfin.DOLFIN_EPS and on_boundary


# Create a facet fuction in order to mark the subdomains
ffun = dolfin.MeshFunction("size_t", mesh, 2)
ffun.set_all(0)

# Mark the first subdomain with value 1
fixed = Fixed()
fixed_marker = 1
fixed.mark(ffun, fixed_marker)

# Mark the second subdomain with value 2
free = Free()
free_marker = 2
free.mark(ffun, free_marker)

# Create a cell function (but we are not using it)
cfun = dolfin.MeshFunction("size_t", mesh, 3)
cfun.set_all(0)


# Collect the functions containing the markers
marker_functions = pulse.MarkerFunctions(ffun=ffun, cfun=cfun)

# Collect the individual markers
fixed_marker = pulse.Marker(name='fixed', value=1, dimension=2)
free_marker = pulse.Marker(name='free', value=2, dimension=2)
markers = (fixed_marker, free_marker)

# Create mictrotructure
el = dolfin.VectorElement("CG", mesh.ufl_cell(), 1)
V_f = dolfin_adjoint.FunctionSpace(mesh, el)
# V_f = pulse.QuadratureSpace(mesh, 4)


# Fibers
f0 = dolfin_adjoint.interpolate(
    dolfin.Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
# Sheets
s0 = dolfin_adjoint.interpolate(
    dolfin.Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
# Fiber-sheet normal
n0 = dolfin_adjoint.interpolate(
    dolfin.Expression(("0.0", "0.0", "1.0"), degree=1), V_f)

# Collect the mictrotructure
microstructure = pulse.Microstructure(f0=f0, s0=s0, n0=n0)

# Create the geometry
geometry = pulse.Geometry(mesh=mesh, markers=markers,
                          marker_functions=marker_functions,
                          microstructure=microstructure)

# Use the default material parameters
material_parameters = pulse.HolzapfelOgden.default_parameters()

# Select model for active contraction
active_model = "active_strain"
# active_model = "active_stress"

# Set the activation
activation = dolfin_adjoint.Constant(0.1)

control = dolfin_adjoint.Control(activation)
# Create material
material = pulse.HolzapfelOgden(active_model=active_model,
                                params=material_parameters,
                                activation=activation)


# Make Dirichlet boundary conditions
def dirichlet_bc(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    return dolfin_adjoint.DirichletBC(V,
                                      dolfin.Constant((0.0, 0.0, 0.0)),
                                      fixed)


# Make Neumann boundary conditions
neumann_bc = pulse.NeumannBC(traction=dolfin_adjoint.Constant(0.0),
                             marker=free_marker.value)

# Collect Boundary Conditions
bcs = pulse.BoundaryConditions(dirichlet=(dirichlet_bc,),
                               neumann=(neumann_bc,))

# Create problem
problem = pulse.MechanicsProblem(geometry, material, bcs)

# Solve problem
problem.solve()

# Get displacement and hydrostatic pressure
u, p = dolfin.split(problem.state)

F = pulse.DeformationGradient(u)
E = pulse.GreenLagrangeStrain(F)
Es = dolfin.inner(E*s0, s0)

J = dolfin_adjoint.assemble(dolfin.inner(Es, Es) * dolfin.dx)

grad = dolfin_adjoint.compute_gradient(J, control)

grad_val = np.zeros(3)
grad.eval(grad_val, np.zeros(3))
print("Gradient = ", grad_val)
