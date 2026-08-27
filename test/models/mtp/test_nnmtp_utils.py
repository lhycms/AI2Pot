import unittest
import os
from typing import List

import numpy as np
import torch
from ase import Atoms
from ase.io import read as ase_read

from ai2pot.models.potential_train import LitNNMtp
from ai2pot.models.mtp.nn_mtp_utils import (NNMtpSerializer,
                                            NNMtp4Extxyz,
                                            NNMtpCalculator)



TEST_FILES_DIR = os.getenv("AI2POT_PATH")
CHECK_POINT_PATH: str = "/data/home/liuhanyu/mycode/AI2Pot/lightning_logs/lightning_logs/version_68/checkpoints/epoch=0-step=165.ckpt"
# "/data/home/liuhanyu/mycode/AI2Pot/lightning_logs/lightning_logs/version_59/checkpoints/epoch=199-step=5000.ckpt"
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
EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/gst/train.xyz"
#EXTXYZ_PATH = "/data/home/liuhanyu/ai2pot_paper/2.demo/hea_linear_mtp/train.xyz"

torch.manual_seed(42)
torch.set_num_threads(16)


class NNMtpSerializerTest(unittest.TestCase):
    def setUp(self):
        print("NNMtpSerializer (TestCase) is setting up...")


    def tearDown(self):
        print("NNMtpSerializer (TestCase) is tearing down...")

    
    def est_serialize(self):
        pt_path: str = "./ai2pot_libtorch.pt"
        NNMtpSerializer.serialize(ckpt_path=CHECK_POINT_PATH,
                                  pt_path=pt_path)
        
        scripted_model = torch.jit.load(pt_path, map_location="cpu")
        scripted_model.eval()

        print(scripted_model._c._method_names())



class NNMtp4ExtxyzTest(unittest.TestCase):
    def setUp(self):
        print("LitNNMtpTest (TestSuite) is setting up...")
        self.checkpoint_path: str = CHECK_POINT_PATH
        self.testset_path: str = EXTXYZ_PATH
        self.map_location: str = "cuda"
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.nnmtp_extxyz: NNMtp4Extxyz = NNMtp4Extxyz(checkpoint_path=self.checkpoint_path,
                                                                    testset_path=self.testset_path,
                                                                    map_location=self.map_location,
                                                                    pbc_xyz=[True, True, True])

    def tearDown(self):
        print("LitNNMtpTest (TestSuite) is tearing down...")


    def est_calculate_parity(self):
        e_dft_array, f_dft_array, e_ml_array, f_ml_array = self.nnmtp_extxyz.calculate_parity()
        print(e_dft_array.shape)
        print(f_dft_array.shape)
        print(e_ml_array.shape)
        print(f_ml_array.shape)


    def est_calculate_rmse(self):
        e_rmse, f_rmse = self.nnmtp_extxyz.calculate_rmse()
        print("RMSE summary:")
        print("\t1. RMSE of energy = {0:.3f} meV".format(e_rmse * 1000))
        print("\t2. RMSE of force = {0:.3f} meV/A".format(f_rmse * 1000))

    
    def est_calculate_descriptors(self):
        descriptors_array, atomic_numbers_array = self.nnmtp_extxyz.calculate_descriptors()
        print("1.1. descriptors_array.shape = ", descriptors_array.shape)
        print("1.2. atomic_numbers_array.shape = ", atomic_numbers_array.shape)


class NNMtpCalculatorTest(unittest.TestCase):
    def setUp(self):
        print("NNMtpCalculator (TestSuite) is setting up...")
        self.checkpoint_path: str = CHECK_POINT_PATH
        self.map_location: str = "cuda"
        self.torch_float_dtype: torch._C.dtype = torch.float32
        
        self.nnmtp_calculator: NNMtpCalculator = NNMtpCalculator(checkpoint_path=self.checkpoint_path,
                                                                              map_location=self.map_location,
                                                                              torch_float_dtype=self.torch_float_dtype)
        self.atoms: Atoms = ase_read(filename=EXTXYZ_PATH, index=":")[2000]
        self.atoms.calc = self.nnmtp_calculator


    def tearDown(self):
        print("NNMtpCalculator (TestSuite) is tearing down...")


    def test_calculate_without_virail(self):
        print("Calculator (without virial) Summary:")
        print("-------------------")
        print("\t1.1. Energy = ", self.atoms.get_potential_energy())
        print("\t1.2. forces.shape = ", self.atoms.get_forces().shape)
        print()

        print("\t2.1. descriptors.shape = ", self.atoms.calc.get_property("descriptors", atoms=self.atoms).shape)


    def est_calculate_with_virial(self):
        print("Calculator (virial) Summary:")
        print("-------------------")
        print("\t1.1. Energy = ", self.atoms.get_potential_energy())
        print("\t1.2. force = \n", self.atoms.get_forces())
        print("\t1.3. virial = \n", self.atoms.get_stress())



if __name__ == '__main__':
    unittest.main()