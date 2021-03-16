import dolfin_adjoint as da
import pytest

from pulse_adjoint import Regularization


def test_zero_regularization():
    reg = Regularization.zero()
    assert abs(float(reg.functional)) < 1e-12


@pytest.mark.parametrize(
    "value, reg_type, weight, expected_functional_value",
    [
        (1.0, "L2", 1.0, 1.0),
        (1.0, "L2", 0.5, 0.5),
        (1.0, "L2", 0.0, 0.0),
        (1.0, "H1", 1.0, 1.0),
        (1.0, "H1", 0.5, 0.5),
        (1.0, "H1", 0.0, 0.0),
        (1.0, "H0", 1.0, 0.0),
        (1.0, "H0", 0.0, 0.0),
    ],
)
def test_constant_regularization(value, reg_type, weight, expected_functional_value):
    f = da.Constant(value, cell="tetrahedron")
    reg = Regularization(f, weight=weight, reg_type=reg_type)
    assert abs(float(reg.functional) - expected_functional_value) < 1e-12
