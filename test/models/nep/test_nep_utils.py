import unittest
import os
from typing import List

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read

from ai2pot.models.nep.nep_utils import (
    NepCalculator,
    Nep4Extxyz)



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
EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"

torch.set_num_threads(16)


"""
class NepCalculatorTest(unittest.TestCase):
    def setUp(self):
        print("NepCalculator (TestSuite) is setting up...")
        self.checkpoint_path: str = "/data/home/liuhanyu/mycode/AI2Pot/lightning_logs/version_4/checkpoints/epoch=199-step=5000.ckpt"
        self.map_location: str = "cpu"
        self.torch_float_dtype: torch._C.dtype = torch.float32
        
        self.nep_calculator: NepCalculator = NepCalculator(checkpoint_path=self.checkpoint_path,
                                                                  map_location=self.map_location,
                                                                  torch_float_dtype=self.torch_float_dtype)
        self.atoms: Atoms = ase_read(filename=EXTXYZ_PATH, index=":")[0]
        self.atoms.calc = self.nep_calculator


    def tearDown(self):
        print("NepCalculator (TestSuite) is tearing down...")


    def test_calculate(self):
        print("Calculator Summary:")
        print("-------------------")
        print("\t1. Energy = ", self.atoms.get_potential_energy())
        print("\t2. forces.shape = ", self.atoms.get_forces().shape)
        print("\t3. descriptors.shape = ",self.nep_calculator.get_property("descriptors", self.atoms).shape)
        #print("\t4. coeffs_gradients.shape = ", self.nep_calculator.get_property("coeffs_gradients", self.atoms).shape)


    def test_predict_atoms_ef(self):
        e, f = self.nep_calculator.predict_ef(atoms=self.atoms)
        e: float = e
        f: np.ndarray = f
        print("\t1. Energy = {0:.3f} eV".format(e))
        print("\t2. Force.shape = ", f.shape)
"""


class Nep4ExtxyzTest(unittest.TestCase):
    def setUp(self):
        print("Nep4ExtxyzTest (TestCase) is setting up...")
        self.checkpoint_path: str = "/data/home/liuhanyu/mycode/AI2Pot/lightning_logs/lightning_logs/version_42/checkpoints/epoch=167-step=25536.ckpt"
        self.testset_path: str = EXTXYZ_PATH
        self.map_location: str = "cuda"        
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.nep_extxyz: Nep4Extxyz = Nep4Extxyz(checkpoint_path=self.checkpoint_path,
                                                 testset_path=self.testset_path,
                                                 map_location=self.map_location,
                                                 torch_float_dtype=self.torch_float_dtype)
        print(self.nep_extxyz.model.coeffs_tensor)

    def tearDown(self):
        print("Nep4ExtxyzTest (TestCase) is tearing down...")

    
    def test_plot_ef_diagonal(self):
        print("=== q_scaler ===")
        print(1.0 / self.nep_extxyz.model.q_scaler_tensor)
        print("=== q_scaler ===")
        self.nep_extxyz.plot_ef_diagonal(save=True)


    def test_calculate_ef_rmse(self):
        e_rmse, f_rmse = self.nep_extxyz.calculate_ef_rmse()
        print("RMSE summary:")
        print("\t1. RMSE of energy = {0:.3f} meV".format(e_rmse * 1000))
        print("\t2. RMSE of force = {0:.3f} meV/A".format(f_rmse * 1000))



if __name__ == '__main__':
    unittest.main()
