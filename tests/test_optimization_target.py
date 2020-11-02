import dolfin
import dolfin_adjoint
import numpy as np
import pulse
import pytest

from pulse_adjoint import OptimizationTarget, StrainObservation, VolumeObservation

volumes = [1.0, (1.0, 1.1), np.ones(4), [-1.0, 1.0, 1.1]]
strains = [0.2, (0.03, 0.2), 0.02 * np.ones(4), [-0.02, 0.05, 0.03]]


@pytest.mark.parametrize("volumes", volumes)
def test_volume(volumes):

    geo = pulse.HeartGeometry.from_file(pulse.mesh_paths["simple_ellipsoid"])
    V_cg2 = dolfin.VectorFunctionSpace(geo.mesh, "CG", 2)
    u = dolfin_adjoint.Function(V_cg2)
    volume_obs = VolumeObservation(
        mesh=geo.mesh, dmu=geo.ds(geo.markers["ENDO"]), description="Test LV volume"
    )

    model_volume = volume_obs(u).vector().get_local()[0]
    target = OptimizationTarget(volumes, volume_obs, collect=True)
    if np.isscalar(volumes):
        volumes = (volumes,)

    fs = []
    for t, v in zip(target, volumes):
        fun = dolfin.project(t.assign(u), t._V)
        f = fun.vector().get_local()[0]
        fs.append(f)
        g = (model_volume - v) ** 2
        assert abs(f - g) < 1e-12

    # We collect the data
    assert np.all(
        np.subtract(
            np.squeeze([v.get_local() for v in target.collector["data"]]), volumes
        )
        < 1e-12
    )
    assert np.all(
        np.subtract(
            np.squeeze([v.get_local() for v in target.collector["model"]]), model_volume
        )
        < 1e-12
    )
    assert np.all(np.subtract(target.collector["functional"], fs) < 1e-12)


@pytest.mark.parametrize("strains", strains)
def test_strain(strains):

    mesh = dolfin_adjoint.UnitCubeMesh(3, 3, 3)
    x_dir = dolfin_adjoint.Constant([1.0, 0.0, 0.0])
    V = dolfin.VectorFunctionSpace(mesh, "CG", 2)

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
    strain_obs = StrainObservation(mesh=mesh, field=x_dir)

    model_strain = strain_obs(u).vector().get_local()[0]
    weight = 0.1
    target = OptimizationTarget(strains, strain_obs, weight=weight, collect=False)
    if np.isscalar(strains):
        strains = (strains,)

    for t, s in zip(target, strains):
        fun = dolfin.project(t.assign(u), t._V)
        f = fun.vector().get_local()[0]
        g = (model_strain - s) ** 2
        assert abs(f - weight * g) < 1e-12

    # Nothing is collected
    for k, v in target.collector.items():
        assert len(v) == 0


if __name__ == "__main__":
    test_volume((1.0, 1.1))
