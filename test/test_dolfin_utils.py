import unittest
import numpy as np
import dolfin
import dolfin_adjoint

from pulse_adjoint import dolfin_utils


class TestDolfinUtils(unittest.TestCase):

    def test_get_constant(self):

        for value_size in (1, 2):
            for value_rank in (0, 1):

                vals = np.zeros(value_size)
                constant = dolfin_utils.get_constant(value_size,
                                                     value_rank, 1)
                constant.eval(vals, np.zeros(3))
                self.assertTrue(np.all((vals == 1)))

                self.assertIsInstance(constant,
                                      dolfin.Constant)

    def test_reduced_functional(self):
        pass

    def test_base_expression(self):
        pass

    def test_regional_paramters(self):
        pass
    
if __name__ == "__main__":
    unittest.main()
