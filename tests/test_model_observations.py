import dolfin
import dolfin_adjoint
import numpy as np
import pulse
import pytest

from pulse_adjoint import BoundaryObservation, StrainObservation, VolumeObservation

geo = pulse.HeartGeometry.from_file(pulse.mesh_paths["simple_ellipsoid"])
P2 = dolfin.VectorElement("Lagrange", geo.mesh.ufl_cell(), 2)
P1 = dolfin.FiniteElement("Lagrange", geo.mesh.ufl_cell(), 1)
P2P1 = dolfin.FunctionSpace(geo.mesh, P2 * P1)
V_cg2 = dolfin.VectorFunctionSpace(geo.mesh, "CG", 2)


def norm(v):
    return np.linalg.norm(v)


@pytest.mark.parametrize("approx", ("project", "interpolate", "original"))
def test_volume_CG2(approx):

    u = dolfin_adjoint.Function(V_cg2)
    volume_obs = VolumeObservation(
        mesh=geo.mesh,
        dmu=geo.ds(geo.markers["ENDO"]),
        approx=approx,
        description="Test LV volume",
    )

    v1 = volume_obs()
    assert norm(v1.vector().get_local() - 2.51130402) < 1e-8
    v2 = volume_obs(u)
    assert norm(v2.vector().get_local() - 2.51130402) < 1e-8


@pytest.mark.parametrize("approx", ("project", "interpolate", "original"))
def test_volume_CG1(approx):

    u = dolfin_adjoint.Function(V_cg2)
    volume_obs = VolumeObservation(
        mesh=geo.mesh,
        dmu=geo.ds(geo.markers["ENDO"]),
        approx=approx,
        displacement_space="CG_1",
        description="Test LV volume",
    )

    v1 = volume_obs()
    assert norm(v1.vector().get_local() - 2.51130402) < 1e-8
    v2 = volume_obs(u)
    assert norm(v2.vector().get_local() - 2.51130402) < 1e-8


@pytest.mark.parametrize("approx", ("project", "interpolate", "original"))
def test_volume_P2P1(approx):

    w = dolfin_adjoint.Function(P2P1)
    u, p = dolfin.split(w)
    volume_obs = VolumeObservation(
        mesh=geo.mesh,
        dmu=geo.ds(geo.markers["ENDO"]),
        approx=approx,
        description="Test LV volume",
    )

    v1 = volume_obs()
    assert norm(v1.vector().get_local() - 2.51130402) < 1e-8
    v2 = volume_obs(u)
    assert norm(v2.vector().get_local() - 2.51130402) < 1e-8


def test_strain(isochoric=True):

    mesh = dolfin_adjoint.UnitCubeMesh(3, 3, 3)
    x_dir = dolfin_adjoint.Constant([1.0, 0.0, 0.0])
    y_dir = dolfin_adjoint.Constant([0.0, 1.0, 0.0])
    V = dolfin.VectorFunctionSpace(mesh, "CG", 2)
    # u = dolfin_adjoint.Function(V)

    # Make an incompressible deformation (with J = 1)
    x_strain = -0.1
    y_strain = 1 / (1 + x_strain) - 1
    u = dolfin_adjoint.interpolate(
        dolfin.Expression(
            ("x_strain * x[0]", "y_strain * x[1]", "0.0"),
            x_strain=x_strain,
            y_strain=y_strain,
            degree=2,
        ),
        V,
    )
    x_strain_obs = StrainObservation(
        mesh=mesh, field=x_dir, isochoric=isochoric, strain_tensor="gradu"
    )

    assert abs(x_strain_obs(u).vector().get_local()[0] - x_strain) < 1e-12

    x_strain_obs = StrainObservation(
        mesh=mesh, field=x_dir, isochoric=isochoric, strain_tensor="E"
    )

    assert (
        abs(x_strain_obs(u).vector().get_local()[0] - 0.5 * ((1 + x_strain) ** 2 - 1))
        < 1e-12
    )

    y_strain_obs = StrainObservation(
        mesh=mesh, field=y_dir, isochoric=isochoric, strain_tensor="gradu"
    )

    assert abs(y_strain_obs(u).vector().get_local()[0] - y_strain) < 1e-12

    y_strain_obs = StrainObservation(
        mesh=mesh, field=y_dir, isochoric=isochoric, strain_tensor="E"
    )

    assert (
        abs(y_strain_obs(u).vector().get_local()[0] - 0.5 * ((1 + y_strain) ** 2 - 1))
        < 1e-12
    )


def test_boundary_observation():

    bc = pulse.NeumannBC(
        traction=dolfin_adjoint.Constant(0.0), marker=0, name="test bc"
    )
    pressures = (0.0, 0.1, 0.2)
    bcs = BoundaryObservation(bc=bc, data=pressures)

    for i, b in enumerate(bcs):
        b.assign_bc()
        assert abs(float(b.bc.traction) - pressures[i]) < 1e-12


if __name__ == "__main__":
    test_boundary_observation()
