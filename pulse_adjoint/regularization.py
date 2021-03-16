import dolfin
import dolfin_adjoint

from . import make_logger

logger = make_logger(__name__, 10)


def L2(f):
    return dolfin.inner(f, f)


def H0(f):
    return dolfin.inner(dolfin.grad(f), dolfin.grad(f))


def H1(f, meshvol=None, dx=None):
    return L2(f) + H0(f)


def regional(f):

    expr_arr = ["0"] * f.value_size()

    # Sum all the components to find the mean
    expr_arr[0] = "1"
    f_sum = dolfin.dot(f, dolfin_adjoint.Expression(tuple(expr_arr), degree=1))
    expr_arr[0] = "0"

    for i in range(1, f.value_size()):
        expr_arr[i] = "1"
        f_sum += dolfin.dot(f, dolfin_adjoint.Expression(tuple(expr_arr), degree=1))
        expr_arr[i] = "0"

    # Compute the mean
    f_avg = f_sum / f.value_size()

    # Compute the variance
    expr_arr[0] = "1"
    f_reg = (
        dolfin.dot(f, dolfin_adjoint.Expression(tuple(expr_arr), degree=1)) - f_avg
    ) ** 2 / f.value_size()

    expr_arr[0] = "0"
    for i in range(1, f.value_size()):
        expr_arr[i] = "1"
        f_reg += (
            dolfin.dot(f, dolfin_adjoint.Expression(tuple(expr_arr), degree=1)) - f_avg
        ) ** 2 / f.value_size()
        expr_arr[i] = "0"

    # Create a functional term
    return f_reg


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

    def __init__(self, f, weight=1.0, reg_type="L2"):
        self.f = f
        self.weight = dolfin_adjoint.Constant(weight, name="Regularization weight")
        self.reg_type = reg_type

    @property
    def form(self):
        if self.reg_type == "":
            return dolfin_adjoint.Constant(0.0)
        elif self.reg_type == "L2":
            return L2(self.f)
        elif self.reg_type == "H0":
            return H0(self.f)
        elif self.reg_type == "H1":
            return H1(self.f)
        elif self.reg_type == "regional":
            return regional(self.f)
        else:
            raise ValueError("Unknown regularization type {}".format(self.reg_type))

    @classmethod
    def zero(cls):
        return cls(None, weight=0.0, reg_type="")

    @property
    def functional(self):
        return self.weight * self.form
