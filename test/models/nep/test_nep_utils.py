import unittest
import os
from typing import List

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read

from ai2pot.models.nep.nep_utils import (Nep4Extxyz)



TEST_FILES_DIR = os.getenv("AI2POT_PATH")
CHECK_POINT_PATH: str = os.path.join(TEST_FILES_DIR,
                                     "lightning_logs",
                                     "nep",
                                     "version_1",
                                     "checkpoints",
                                     "epoch=199-step=5000.ckpt")
EXTXYZ_PATH: str = os.path.join(TEST_FILES_DIR,
                                "test",
                                "test_data",
                                "XYZ",
                                "11_NEP_potential_PbTe",
                                "train_m.xyz")
EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"

torch.set_num_threads(16)




class Nep4ExtxyzTest(unittest.TestCase):
    def setUp(self):
        print("Nep4ExtxyzTest (TestCase) is setting up...")
        self.checkpoint_path: str = "/data/home/liuhanyu/mycode/AI2Pot/lightning_logs/version_1/checkpoints/epoch=499-step=12500.ckpt"
        self.testset_path: str = EXTXYZ_PATH
        self.map_location: str = "cuda"        
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.nep_extxyz: Nep4Extxyz = Nep4Extxyz(checkpoint_path=self.checkpoint_path,
                                                 testset_path=self.testset_path,
                                                 map_location=self.map_location,
                                                 torch_float_dtype=self.torch_float_dtype)

    def tearDown(self):
        print("Nep4ExtxyzTest (TestCase) is tearing down...")

    
    def test_plot_ef_diagonal(self):
        print("=== q_scaler ===")
        print(1.0 / self.nep_extxyz.nep.q_scaler_tensor)
        print("=== q_scaler ===")
        print(1.0 / self.nep_extxyz.nep.q_scaler_tensor[0].item())
        self.nep_extxyz.plot_ef_diagonal(save=True)


    def test_calculate_ef_rmse(self):
        e_rmse, f_rmse = self.nep_extxyz.calculate_ef_rmse()
        print("RMSE summary:")
        print("\t1. RMSE of energy = {0:.3f} meV".format(e_rmse * 1000))
        print("\t2. RMSE of force = {0:.3f} meV/A".format(f_rmse * 1000))



if __name__ == '__main__':
    unittest.main()
