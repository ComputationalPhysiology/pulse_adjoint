from pulse_adjoint import Regularization


def test_zero_regularization():
    reg = Regularization.zero()
    assert abs(float(reg.functional)) < 1e-12
