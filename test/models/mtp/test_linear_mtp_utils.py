import unittest
import os
from typing import List

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read

from ai2pot.models.mtp.linear_mtp_utils import (LinearMtp4Extxyz,
                                                LinearMtpActiveDR,
                                                LinearMtpCalculator)



TEST_FILES_DIR = os.getenv("AI2POT_PATH")
CHECK_POINT_PATH: str = "/data/home/liuhanyu/mycode/AI2Pot/lightning_logs/lightning_logs/version_59/checkpoints/epoch=199-step=5000.ckpt"
EXTXYZ_PATH: str = os.path.join(TEST_FILES_DIR,
                                "test",
                                "test_data",
                                "XYZ",
                                "11_NEP_potential_PbTe",
                                "train.xyz")
EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
#EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train_802.xyz"
#EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"
#EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/gst/test.xyz"
#EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/gst/train.xyz"
#EXTXYZ_PATH = "/data/home/liuhanyu/ai2pot_paper/2.demo/hea_linear_mtp/train.xyz"

torch.manual_seed(42)
torch.set_num_threads(16)



class LinearMtp4ExtxyzTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtp4ExtxyzTest (TestSuite) is setting up...")
        self.checkpoint_path: str = CHECK_POINT_PATH
        self.testset_path: str = EXTXYZ_PATH
        self.map_location: str = "cuda"
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.linear_mtp_extxyz: LinearMtp4Extxyz = LinearMtp4Extxyz(checkpoint_path=self.checkpoint_path,
                                                                    testset_path=self.testset_path,
                                                                    map_location=self.map_location,
                                                                    torch_float_dtype=self.torch_float_dtype,
                                                                    pbc_xyz=[True, True, True])

    def tearDown(self):
        print("LinearMtp4ExtxyzTest (TestSuite) is tearing down...")


    def est_calculate_parity(self):
        e_dft_array, f_dft_array, e_ml_array, f_ml_array = self.linear_mtp_extxyz.calculate_parity()
        print(e_dft_array.shape)
        print(f_dft_array.shape)
        print(e_ml_array.shape)
        print(f_ml_array.shape)


    def est_calculate_rmse(self):
        e_rmse, f_rmse = self.linear_mtp_extxyz.calculate_rmse()
        print("RMSE summary:")
        print("\t1. RMSE of energy = {0:.3f} meV".format(e_rmse * 1000))
        print("\t2. RMSE of force = {0:.3f} meV/A".format(f_rmse * 1000))



class LinearMtpCalculatorTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtpCalculator (TestSuite) is setting up...")
        self.checkpoint_path: str = CHECK_POINT_PATH
        self.map_location: str = "cuda"
        self.torch_float_dtype: torch._C.dtype = torch.float32
        
        self.linear_mtp_calculator: LinearMtpCalculator = LinearMtpCalculator(checkpoint_path=self.checkpoint_path,
                                                                              map_location=self.map_location,
                                                                              torch_float_dtype=self.torch_float_dtype)
        self.atoms: Atoms = ase_read(filename=EXTXYZ_PATH, index=":")[0]
        self.atoms.calc = self.linear_mtp_calculator


    def tearDown(self):
        print("LinearMtpCalculator (TestSuite) is tearing down...")


    def test_calculate_without_virail(self):
        print("Calculator (without virial) Summary:")
        print("-------------------")
        print("\t1.1. Energy = ", self.atoms.get_potential_energy())
        print("\t1.2. forces.shape = ", self.atoms.get_forces().shape)
        print()

        print("\t2.1. descriptors.shape = ", self.atoms.calc.get_property("descriptors", atoms=self.atoms).shape)


    def test_calculate_with_virial(self):
        print("Calculator (virial) Summary:")
        print("-------------------")
        print("\t1.1. Energy = ", self.atoms.get_potential_energy())
        print("\t1.2. force.shape = ", self.atoms.get_forces().shape)
        print("\t1.3. virial.shape = ", self.atoms.get_stress().shape)

    


"""
class LinearMtpActiveDRTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtpActiveDRTest (TestCase) is setting up...")
        self.checkpoint_path: str = CHECK_POINT_PATH
        self.trainset_path: str = os.path.join(TEST_FILES_DIR,
                                              "test",
                                              "test_data",
                                              "XYZ",
                                              "11_NEP_potential_PbTe",
                                              "train_m.xyz")
        self.map_location: str = "cpu"        
        self.torch_float_dtype: torch._C.dtype = torch.float32
        
        self.linear_mtp_active_dr: LinearMtpActiveDR = LinearMtpActiveDR(checkpoint_path=self.checkpoint_path,
                                                                         extxyz_path=self.trainset_path,
                                                                         map_location=self.map_location,
                                                                         torch_float_dtype=self.torch_float_dtype)

    def tearDown(self):
        print("LinearMtpActiveDRTest (TestCase) is tearing down...")


    def test_get_batch_descriptors(self):
        descriptors_tensor: torch.Tensor = self.linear_mtp_active_dr._get_batch_descriptors()
        print("\t1. descriptors_tensor.shape = ", descriptors_tensor.shape)


    def test_get_pca_2d(self):
        pca_2d = self.linear_mtp_active_dr.get_pca_2d()
        print("\t1. Shape of PCA descriptors = ", pca_2d.shape)
        #self.linear_mtp_active_dr.plot_pca_2d()

    def test_get_pca_2d(self):
        tnse_2d = self.linear_mtp_active_dr.get_tsne_2d()
        print("\t1. Shape of T-SNE descriptors = ", tnse_2d.shape)
        self.linear_mtp_active_dr.plot_tsne_2d()
"""
    


if __name__ == '__main__':
    unittest.main()
