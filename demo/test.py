from fenics import *
# from fenics_adjoint import *

parameters["form_compiler"]["quadrature_degree"] = 2

n = 30
mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, VectorElement(family="Quadrature",
                                      cell=mesh.ufl_cell(),
                                      degree=2,
                                      quad_scheme="default"))
W = VectorFunctionSpace(mesh, "CG", 2)

u = interpolate(Expression(("sin(2*pi*x[0])", "cos(2*pi*x[1])"), degree=2),  V)
# control = Control(u)

u_next = Function(V)
v = TestFunction(V)

nu = Constant(0.0001)

# nu = Function(V)
# nu.assign(Constant(0.0001))
# nu.vector() = 0.0001

timestep = Constant(0.01)

F = (inner((u_next - u)/timestep, v))*dx

bc = DirichletBC(W, (0.0, 0.0), "on_boundary")

t = 0.0
end = 0.1
while (t <= end):
    solve(F == 0, u_next, bc)
    u.assign(u_next)
    t += float(timestep)

J = assemble(inner(u, u)*dx)
# dJdu, dJdnu = compute_gradient(J, [control, Control(nu)])

