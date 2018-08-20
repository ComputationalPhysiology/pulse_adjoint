from fenics import *
from dolfin_adjoint import *

parameters["form_compiler"]["quadrature_degree"] = 4
# Create mesh
N = 6
mesh = UnitCubeMesh(N, N, N)


# Create subdomains
class Free(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > (1.0 - DOLFIN_EPS) and on_boundary


class Fixed(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < DOLFIN_EPS and on_boundary


# Create a facet fuction in order to mark the subdomains
ffun = MeshFunction("size_t", mesh, 2)
ffun.set_all(0)

# Mark the first subdomain with value 1
fixed = Fixed()
fixed_marker = 1
fixed.mark(ffun, fixed_marker)

# Mark the second subdomain with value 2
free = Free()
free_marker = 2
free.mark(ffun, free_marker)

# Create mictrotructure
V_f = FunctionSpace(mesh, VectorElement(family="Quadrature",
                                        cell=mesh.ufl_cell(),
                                        degree=4,
                                        quad_scheme="default"))

# Fibers
f0 = interpolate(Expression(("1.0", "0.0", "0.0"), degree=1), V_f)

P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, P2)

u = Function(V, name="state")
v = TestFunction(V)

F = grad(u) + Identity(3)
J = det(F)

f = F*f0

dW = derivative((1.0 * tr(F.T * F) - 3.0)*dx \
                + 1.0  * (inner(f, f) - 1.0) * dx \
                + 10.0*(J * dolfin.ln(J) - J + 1) * dx,
                u, v)

# Make Dirichlet boundary conditions
bc = DirichletBC(V,dolfin.Constant((0.0, 0.0, 0.0)),fixed)

solve(dW == 0, u, bc)
