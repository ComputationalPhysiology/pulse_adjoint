import dolfin
import dolfin_adjoint
import pulse
import pytest

from pulse_adjoint import Regularization

def test_no_parameter_regularization():
    assert isinstance(Regularization.zero(), Regularization)



