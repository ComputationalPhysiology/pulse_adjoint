import pytest
import dolfin
import numpy as np

from pulse_adjoint.material import HolzapfelOgden
from pulse_adjoint.geometry import (Geometry, Marker, CRLBasis,
                                    Microstructure, MarkerFunctions)
from pulse_adjoint.setup_parameters import setup_general_parameters
from pulse_adjoint.dolfin_utils import QuadratureSpace
from pulse_adjoint.mechanicsproblem import (MechanicsProblem,
                                            BoundaryConditions,
                                            NeumannBC)

from test_clinical_data import sample_clinical_data


# from pulse_adjoint.optimization_targets_new2 import (RegionalStrainTarget,
#                                                     OptimizationTarget,
#                                                     OptTargets)
from pulse_adjoint.assimilator import Assimilator


setup_general_parameters()


class Free(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > (1.0 - dolfin.DOLFIN_EPS) and on_boundary


class Fixed(dolfin.SubDomain):
    def inside(self, x, on_boundary):
        return x[0] < dolfin.DOLFIN_EPS and on_boundary


fixed = Fixed()
fixed_marker = 1

free = Free()
free_marker = 2


def strain_markers_3d(mesh, nregions):
    strain_markers = dolfin.MeshFunction("size_t", mesh, 3)
    strain_markers.set_all(0)
    xs = np.linspace(0, 1, nregions+1)

    region = 0
    for it_x in range(nregions):
        for it_y in range(nregions):
            for it_z in range(nregions):

                region += 1
                domain_str = ""

                domain_str += "x[0] >= {}".format(xs[it_x])
                domain_str += " && x[1] >= {}".format(xs[it_y])
                domain_str += " && x[2] >= {}".format(xs[it_z])
                domain_str += " && x[0] <= {}".format(xs[it_x+1])
                domain_str += " && x[1] <= {}".format(xs[it_y+1])
                domain_str += " && x[2] <= {}".format(xs[it_z+1])

                len_sub = dolfin.CompiledSubDomain(domain_str)
                len_sub.mark(strain_markers, region)

    return strain_markers

@pytest.fixture
def problem():
    N = 2
    mesh = dolfin.UnitCubeMesh(N, N, N)

    V_f = QuadratureSpace(mesh, 4)

    l0 = dolfin.interpolate(
        dolfin.Expression(("1.0", "0.0", "0.0"), degree=1), V_f)
    r0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
    c0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "0.0", "1.0"), degree=1), V_f)

    crl_basis = CRLBasis(l0=l0, r0=r0, c0=c0)

    cfun = strain_markers_3d(mesh, 2)

    ffun = dolfin.MeshFunction("size_t", mesh, 2)
    ffun.set_all(0)
    fixed.mark(ffun, fixed_marker)
    free.mark(ffun, free_marker)

    marker_functions = MarkerFunctions(ffun=ffun, cfun=cfun)
    fixed_marker_ = Marker(name='fixed', value=fixed_marker, dimension=2)
    free_marker_ = Marker(name='free', value=free_marker, dimension=2)

    markers = (fixed_marker_, free_marker_)

    # Fibers
    f0 = dolfin.interpolate(
        dolfin.Expression(("1.0", "0.0", "0.0"), degree=1),  V_f)
    s0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "1.0", "0.0"), degree=1), V_f)
    n0 = dolfin.interpolate(
        dolfin.Expression(("0.0", "0.0", "1.0"), degree=1), V_f)

    microstructure = Microstructure(f0=f0, s0=s0, n0=n0)

    geometry = Geometry(mesh=mesh, markers=markers,
                        marker_functions=marker_functions,
                        microstructure=microstructure,
                        crl_basis=crl_basis)

    material = HolzapfelOgden()

    def dirichlet_bc(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        return dolfin.DirichletBC(V, dolfin.Constant((0.0, 0.0, 0.0)),
                                  fixed)

    neumann_bc = NeumannBC(traction=dolfin.Constant(0.0),
                           marker=free_marker)

    bcs = BoundaryConditions(dirichlet=(dirichlet_bc,),
                             neumann=(neumann_bc,))

    problem = MechanicsProblem(geometry, material, bcs)
    return problem


@pytest.fixture
def clinical_data(problem):
    datasize = 4
    return sample_clinical_data(datasize, problem.geometry.regions)


def test_assimilator(problem, clinical_data):

    # strain_target = RegionalStrainTarget(problem.geometry)
    # targets = OptTargets(regional_strain=strain_target)

    assimilator = Assimilator(problem, clinical_data)
    assimilator.assimilate()

if __name__ == "__main__":

    prob = problem()
    data = clinical_data(prob)
    test_assimilator(prob, data)
    # test_optimization_target(geo)
