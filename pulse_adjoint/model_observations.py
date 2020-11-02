import dolfin
import dolfin_adjoint
import numpy as np
import pulse
from pulse.dolfin_utils import compute_meshvolume, map_displacement


class BoundaryObservation(object):
    """
    Class of observations that should be enforced as a boundary condition,
    e.g pressure.

    Arguemnts
    ---------
    bc : :py:class:`pulse.NeumannBC`
        The boundary condition
    data : scalar or list
        The data that shoud be enforced on the boudary
    start_value : float
        The starting value
    """

    def __init__(self, bc, data, start_value=0.0):
        self.bc = bc
        if np.isscalar(data):
            self.data = (data,)
        elif isinstance(data, tuple):
            self.data = data
        else:
            self.data = tuple(data)
        self._start_value = start_value
        self.__len__ = len(data)
        self.reset()

    def __repr__(self):
        return ("{self.__class__.__name__}" "({self.bc}, {self.data})").format(
            self=self
        )

    def reset(self):
        self._count = -1
        name = "Start value Boundary {}".format(self.bc)
        self.bc.traction.assign(dolfin_adjoint.Constant(self._start_value, name=name))

    def assign_bc(self):
        data = self.data[self.count]
        name = "Data ({}) Boundary {}".format(self.count, self.bc)
        dolfin_data = dolfin_adjoint.Constant(data, name=name)
        self.bc.traction.assign(dolfin_data)

    @property
    def count(self):
        return max(self._count, 0)

    def __iter__(self):
        return self

    def __next__(self):
        self._count += 1
        if self.count == self.__len__:
            raise StopIteration
        else:
            return self


class ModelObservation(object):
    """
    Base class for model observation.
    """

    def __init__(self, mesh, target_space="R_0", description=""):

        assert isinstance(mesh, dolfin.Mesh)
        self.mesh = mesh

        # The volume of the mesh
        self.meshvol = compute_meshvolume(mesh)

        self.description = description

        # Test and trial functions for the target space
        family, degree = target_space.split("_")
        self._V = dolfin.FunctionSpace(mesh, family, int(degree))
        self._trial = dolfin.TrialFunction(self._V)
        self._test = dolfin.TestFunction(self._V)

    def __repr__(self):
        return "{self.__class__.__name__}({self.description})".format(self=self)


class VolumeObservation(ModelObservation):
    r"""
    Model observation of the endocardial cavity volume.
    Compute the cavity volume using the formula

    .. math::

        V = -\frac{1}{3} \int_{\partial \Omega} \langle (\mathbf{X} + \mathbf{u}),
        J \mathbf{F}^{-T} N \rangle d \mu

    where :math:`\partial \Omega` is the endcardial surface, :math:`\mathbf{u}`
    is the displacement :math:`\mathbf{F} = \mathbf{I} + \nabla_{\mathbf{X}} \mathbf{u}`,
    and :math:`N` is the outward facet normal.

    Arguments
    ---------
    mesh : :py:class:`dolfin.Mesh`
        The mesh.
    dmu : :py:class:`dolfin.Measure`
        A surface measure for the endocardial surface.
    approx : str
        Whether to `project` or `interpolate` the displacement from the
        displacement space to lineart Lagrange elements. Choices:
        ['interpolate', 'project', 'original']
    displacement_space : str
        A string on the form {family}_{degree} for the displacment space.
        Note that is is only relevant if approx is different from original.
        Default: 'CG_2'
    interpolation_space : str
        A string on the form {family}_{degree} used for interpolating the
        displacement before computing the volume.
        Note that is is only relevant if approx is different from original.
        Default: 'CG_1'.
    description : str
        A string that describes this observation. This will be part of the
        `__repr__` method.
    """

    def __init__(
        self,
        mesh,
        dmu,
        approx="project",
        displacement_space="CG_2",
        interpolation_space="CG_1",
        description="",
    ):
        ModelObservation.__init__(
            self, mesh, target_space="R_0", description=description
        )

        approxs = ["project", "interpolate", "original"]
        msg = 'Expected "approx" for be one of {}, got {}'.format(approxs, approx)
        assert approx in approxs, msg
        self._approx = approx

        # These spaces are only used if you want to project
        # or interpolate the displacement before assigning it
        # Space for interpolating the displacement if needed
        family, degree = interpolation_space.split("_")
        self._interpolation_space = dolfin.VectorFunctionSpace(
            mesh, family, int(degree)
        )

        # Displacement space
        family, degree = displacement_space.split("_")
        self._displacement_space = dolfin.VectorFunctionSpace(mesh, family, int(degree))

        self._X = dolfin.SpatialCoordinate(mesh)
        self._N = dolfin.FacetNormal(mesh)
        assert isinstance(dmu, dolfin.Measure)
        self._dmu = dmu

        name = "EndoArea {}".format(self)
        self._endoarea = dolfin_adjoint.Constant(
            dolfin.assemble(dolfin.Constant(1.0) * dmu), name=name
        )

    def __call__(self, u=None):
        """

        Arguments
        ---------
        u : :py:class:`dolfin.Function`
            The displacement
        """

        if u is None:

            volume_form = (-1.0 / 3.0) * dolfin.dot(self._X, self._N)

        else:
            u_int = map_displacement(
                u, self._displacement_space, self._interpolation_space, self._approx
            )

            # Compute volume
            F = pulse.kinematics.DeformationGradient(u_int)
            J = pulse.kinematics.Jacobian(F)
            volume_form = (-1.0 / 3.0) * dolfin.dot(
                self._X + u_int, J * dolfin.inv(F).T * self._N
            )

        volume = dolfin_adjoint.Function(self._V, name="Simulated volume")
        # Make a project for dolfin-adjoint recording
        dolfin_adjoint.solve(
            dolfin.inner(self._trial, self._test) / self._endoarea * self._dmu
            == dolfin.inner(volume_form, self._test) * self._dmu,
            volume,
        )

        return volume


class StrainObservation(ModelObservation):
    r"""
    Model observation of strain in a given region and direction.
    For a given field :math:`mathbf{e}` and strain tensor :math:`\mathbf{A}`
    this computes

    .. math::

        \mathbf{A}_{\mathbf{e}} = \frac{1}{|\Omega|} \int_{\Omega}
        \mathbf{A} \mathbf{e} \cdot \mathbf{e} \mathrm{d} \mu

    where :math:`\mathrm{d} \mu` is a volume measure for the regions
    of interest.

    Arguments
    ---------
    mesh : :py:class:`dolfin.Mesh`
        The mesh.
    field : :py:class:`dolfin.Function`
        The strain field, i.e a vector field (or constant) in the
        direction of the strain
    strain_tensor : str
        Which strain
    dmu : :py:class:`dolfin.Measure`
        A volume measure for the region of interest. If not provided the
        integration will be performed over the entire mesh.
    approx : str
        Whether to `project` or `interpolate` the displacement from the
        displacement space to lineart Lagrange elements. Choices:
        ['interpolate', 'project', 'original']
    F_ref : :py:class:`ulf.Form`
        A reference deformation gradient in case strain should be computed with
        respect to a different reference geometry, i.e that the deformation
        that should be used is :math:`\mathbf{F} \mathbf{F_{\text{ref}}}`.
    isochoric : bool
        If true use the the isochoric part of the deformation graident (i.e
        neglect volumectric strains). Default: True
    displacement_space : str
        A string on the form {family}_{degree} for the displacment space.
        Note that is is only relevant if approx is different from original.
        Default: 'CG_2'
    interpolation_space : str
        A string on the form {family}_{degree} used for interpolating the
        displacement before computing the volume.
        Note that is is only relevant if approx is different from original.
        Default: 'CG_1'.
    description : str
        A string that describes this observation. This will be part of the
        `__repr__` method.

    """

    def __init__(
        self,
        mesh,
        field,
        strain_tensor="E",
        dmu=None,
        approx="original",
        F_ref=None,
        isochoric=True,
        displacement_space="CG_2",
        interpolation_space="CG_1",
        description="",
    ):

        ModelObservation.__init__(
            self, mesh, target_space="R_0", description=description
        )

        # Check that the given field is a vector of the same dim as the mesh
        dim = mesh.geometry().dim()
        # assert isinstance(field, (dolfin.Function, dolfin.Constant)
        assert field.ufl_shape[0] == dim
        approxs = ["project", "interpolate", "original"]
        msg = 'Expected "approx" for be one of {}, got {}'.format(approxs, approx)
        self.field = field

        assert approx in approxs, msg
        self._approx = approx

        self._F_ref = F_ref if F_ref is not None else dolfin.Identity(dim)

        self._isochoric = isochoric

        tensors = ["gradu", "E", "almansi"]
        msg = "Expected strain tensor to be one of {}, got {}".format(
            tensors, strain_tensor
        )
        assert strain_tensor in tensors, msg
        self._tensor = strain_tensor

        if dmu is None:
            dmu = dolfin.dx(domain=mesh)
        assert isinstance(dmu, dolfin.Measure)
        self._dmu = dmu

        self._vol = dolfin_adjoint.Constant(dolfin.assemble(dolfin.Constant(1.0) * dmu))

        # These spaces are only used if you want to project
        # or interpolate the displacement before assigning it
        # Space for interpolating the displacement if needed
        family, degree = interpolation_space.split("_")
        self._interpolation_space = dolfin.VectorFunctionSpace(
            mesh, family, int(degree)
        )

        # Displacement space
        family, degree = displacement_space.split("_")
        self._displacement_space = dolfin.VectorFunctionSpace(mesh, family, int(degree))

    def __call__(self, u=None):

        if u is None:
            u = dolfin.Function(
                self._displacement_space,
                name="Zero displacement from strain observation",
            )

        u_int = map_displacement(
            u, self._displacement_space, self._interpolation_space, self._approx
        )
        # We need to correct for th reference deformation
        F = pulse.kinematics.DeformationGradient(u_int) * dolfin.inv(self._F_ref)

        # Compute the strains
        if self._tensor == "gradu":
            tensor = pulse.kinematics.EngineeringStrain(F, isochoric=self._isochoric)

        elif self._tensor == "E":
            tensor = pulse.kinematics.GreenLagrangeStrain(F, isochoric=self._isochoric)

        elif self._tensor == "almansi":
            tensor = pulse.kinematics.EulerAlmansiStrain(F, isochoric=self._isochoric)

        form = dolfin.inner(tensor * self.field, self.field)
        strain = dolfin_adjoint.Function(self._V, name="Simulated Strain")
        dolfin_adjoint.solve(
            dolfin.inner(self._trial, self._test) / self._vol * self._dmu
            == dolfin.inner(self._test, form) * self._dmu,
            strain,
            solver_parameters={"linear_solver": "gmres"},
        )

        return strain
