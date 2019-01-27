import numpy as np
import dolfin
import dolfin_adjoint
from pulse.dolfin_utils import list_sum, compute_meshvolume
from pulse import numpy_mpi

from .model_observations import ModelObservation
from . import make_logger, annotation


logger = make_logger(__name__, 10)


def L2(f, meshvol=None, dx=None):

    if meshvol is None:
        meshvol = dolfin.Constant(1.0)
    if dx is None:
        dx = dolfin.dx

    return (dolfin.inner(f, f) / meshvol) * dx


def H0(f, meshvol=None, dx=None):

    if meshvol is None:
        meshvol = dolfin.Constant(1.0)
    if dx is None:
        dx = dolfin.dx

    return (dolfin.inner(dolfin.grad(f), dolfin.grad(f)) / meshvol) * dx


def H1(f, meshvol=None, dx=None):
    return L2(f, meshvol, dx) + H0(f, meshvol, dx)


def regional(f, meshvol=None, dx=None):

    if meshvol is None:
        meshvol = dolfin.Constant(1.0)
    if dx is None:
        dx = dolfin.dx

    expr_arr = ["0"] * f.value_size()

    # Sum all the components to find the mean
    expr_arr[0] = "1"
    f_sum = dolfin.dot(
        f, dolfin.Expression(tuple(expr_arr), degree=1)
    )
    expr_arr[0] = "0"

    for i in range(1, f.value_size()):
        expr_arr[i] = "1"
        f_sum += dolfin.dot(
            f, dolfin.Expression(tuple(expr_arr), degree=1)
        )
        expr_arr[i] = "0"

    # Compute the mean
    f_avg = f_sum / f.value_size()

    # Compute the variance
    expr_arr[0] = "1"
    f_reg = (
        dolfin.dot(f, dolfin.Expression(tuple(expr_arr), degree=1))
        - f_avg
    ) ** 2 / f.value_size()

    expr_arr[0] = "0"
    for i in range(1, f.value_size()):
        expr_arr[i] = "1"
        f_reg += (
            dolfin.dot(
                f, dolfin.Expression(tuple(expr_arr), degree=1)
            )
            - f_avg
        ) ** 2 / f.value_size()
        expr_arr[i] = "0"

    # Create a functional term
    return (f_reg / meshvol) * dx


class Regularization(object):
    """
    Class for regularization.
    To get the value of the regularzaion you can use the `form` attribute
    and in the `functional` attribute we also multiply with the given
    weights

    Arguments
    ---------
    f : :py:class:`dolfin.Function`
        The parameter you want to regularize
    mesh : :py:class:`dolfin.Mesh`
        The mesh
    weight : float
        The weight that you should multiply with the form. Default: 1.0.
    reg_type : str
        The type of regularization. Possible choices are ['L2', 'H0', 'H1',
        'regional']. Default: 'L2'
    """

    def __init__(self, f, mesh, weight=1.0, reg_type='L2'):
        self.f = f
        self.weight = dolfin_adjoint.Constant(weight, name='Regularization weight')
        self.reg_type = reg_type

        if mesh is not None:
            self._meshvol = compute_meshvolume(mesh)
            self._dx = dolfin.dx(domain=mesh)
        else:
            self._meshvol = 1.0
            self._dx = dolfin.dx

        self.form = self._form()

    def _form(self):
        if self.reg_type == '':
            return dolfin_adjoint.Constant(0.0)
        elif self.reg_type == 'L2':
            return L2(self.f, self._meshvol, self._dx)
        elif self.reg_type == 'H0':
            return H0(self.f, self._meshvol, self._dx)
        elif self.reg_type == 'H1':
            return H1(self.f, self._meshvol, self._dx)
        elif self.reg_type == 'regional':
            return regional(self.f, self._meshvol, self._dx)
        else:
            raise ValueError('Unknown regularization type {}'.format(self.reg_type))

    @classmethod
    def zero(cls):
        return cls(None, None, weight=0.0, reg_type='')

    @property
    def functional(self):
        return self.weight * self.form
