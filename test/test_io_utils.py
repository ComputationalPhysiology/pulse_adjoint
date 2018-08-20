import unittest
import os
import h5py

from pulse_adjoint import io_utils
from pulse_adjoint import config


class TestPhasesExists(unittest.TestCase):

    def setup_passive_test(self):
        self.params = {"sim_file": "test_passive.h5"}
        with h5py.File(self.params["sim_file"], "w") as f:
            f.create_group(config.PASSIVE_INFLATION_GROUP)

    def cleanup(self):
        os.remove(self.params["sim_file"])

    def test_passive_inflation_exists_false(self):
        self.assertFalse(io_utils.
                         passive_inflation_exists({"sim_file": "dummy.h5"}))

    def test_passive_inflation_exists_true(self):
        self.setup_passive_test()
        self.assertTrue(io_utils.
                        passive_inflation_exists(self.params))
        self.cleanup()

    def setup_active_test(self, include_contract_point=True):
        self.params = {"sim_file": "test_active.h5",
                       "phase": config.PHASES[1],
                       "active_contraction_iteration_number": 0}
        with h5py.File(self.params["sim_file"], "w") as f:

            f.create_group(config.PASSIVE_INFLATION_GROUP)

            f.create_group(config.ACTIVE_CONTRACTION)
            if include_contract_point:
                f[config.ACTIVE_CONTRACTION].\
                    create_group("{}/bcs".format(
                        config.CONTRACTION_POINT.format(0)))

                f[config.ACTIVE_CONTRACTION] \
                    ["{}/bcs/".format(
                        config.CONTRACTION_POINT.format(0))].\
                    create_dataset("pressure", data=[0, 0])

    def test_contract_point_exists_false1(self):
        with self.assertRaises(IOError):
            io_utils.contract_point_exists({"sim_file": "dummy.h5"})

    def test_contract_point_exists_false2(self):
        self.setup_active_test(False)
        self.assertFalse(io_utils.
                         contract_point_exists(self.params))
        self.cleanup()

    def test_contract_point_exists_true(self):
        self.setup_active_test()
        self.assertTrue(io_utils.
                        contract_point_exists(self.params))
        self.cleanup()


if __name__ == '__main__':
    unittest.main()
