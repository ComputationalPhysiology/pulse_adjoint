import itertools as it

import numpy as np
import pytest

from pulse_adjoint.optimal_control import OptimalControl

ns = [1, 10]
opt_libs = ["scipy"]  # [, pyOpt, ipopt, moola]


class RD(object):
    def __call__(self, x):
        return np.linalg.norm((x - 0.5)) ** 2

    def derivative(self, x):
        return 2 * (x - 0.5)


@pytest.mark.parametrize("n, opt_lib", it.product(ns, opt_libs))
def test_optimal_control(n, opt_lib):

    tol = 1e-12
    oc_problem = OptimalControl(tol=tol)
    rd = RD()
    x = np.zeros(n)
    oc_problem.build_problem(rd, x)
    res = oc_problem.solve()

    assert np.all(abs(res.initial_control - x) < tol)
    assert np.all(abs(res.optimal_control - 0.5) < tol)


if __name__ == "__main__":
    test_optimal_control(1, "scipy")
    test_optimal_control(10, "scipy")
