import unittest
import os
from typing import List

import torch

from ai2pot.models.mtp.linear_mtp_utils import LinearMtp4Extxyz



TEST_FILES_DIR = os.getenv("AI2POT_PATH")
CHECK_POINT_PATH: str = os.path.join(TEST_FILES_DIR,
                                     "lightning_logs",
                                     "linear_mtp",
                                     "version_34",
                                     "checkpoints",
                                     "epoch=146-step=3675.ckpt")

torch.set_num_threads(16)

class LinearMtp4ExtxyzTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtp4ExtxyzTest (TestSuite) is setting up...")
        self.checkpoint_path: str = CHECK_POINT_PATH
        self.testset_path: str = os.path.join(TEST_FILES_DIR,
                                              "test",
                                              "test_data",
                                              "XYZ",
                                              "11_NEP_potential_PbTe",
                                              "train_m.xyz")
        self.has_virial: bool = False
        self.map_location: str = "cpu"        
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.linear_mtp_extxyz: LinearMtp4Extxyz = LinearMtp4Extxyz(checkpoint_path=self.checkpoint_path,
                                                                              testset_path=self.testset_path,
                                                                              map_location=self.map_location,
                                                                              torch_float_dtype=self.torch_float_dtype)

    def tearDown(self):
        print("LinearMtp4ExtxyzTest (TestSuite) is tearing down...")

    
    def test_plot_ef_diagonal(self):
        self.linear_mtp_extxyz.plot_ef_diagonal(save=True)


    def test_calculate_ef_rmse(self):
        e_rmse, f_rmse = self.linear_mtp_extxyz.calculate_ef_rmse()
        print("RMSE summary:")
        print("\t1. RMSE of energy = {0:.3f} meV".format(e_rmse * 1000))
        print("\t2. RMSE of force = {0:.3f} meV/A".format(f_rmse * 1000))



if __name__ == '__main__':
    unittest.main()
