import os
import time
import unittest
from typing import List

import numpy as np
import torch
from pymatgen.core import Structure
from ai2pot.utils import (MlffInput, MlffToLossInput, MlffToEFLossInput)
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.mtp.linear_mtp_al_utils import LinearMtpMaxvolGenerator


TEST_FILES_DIR = os.getenv("AI2POT_PATH")
CHECK_POINT_PATH: str = "/data/home/liuhanyu/ai2pot_paper_code/5.JCTC_1.2.1/5.1.GST/3.1.1.MTP/bs16/lightning_logs/version_0/checkpoints/last.ckpt"
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


class LinearMtpMaxvolGeneratorTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtpSerializer (TestCase) is setting up...")
        self.linear_mtp_maxvol_generator: LinearMtpMaxvolGenerator = LinearMtpMaxvolGenerator(
            checkpoint_path=CHECK_POINT_PATH,
            candidate_set_path=EXTXYZ_PATH,
            map_location="cuda",
            pbc_xyz=[True, True, True])


    def tearDown(self):
        print("LinearMtpSerializer (TestCase) is tearing down...")
        
    
    def test_generate_candidate_set_matrix(self):
        structure_id_tensor, atom_id_tensor, candidate_matrix_tensor = self.linear_mtp_maxvol_generator._generate_candidate_set_matrix()
        print(structure_id_tensor.shape, atom_id_tensor.shape, candidate_matrix_tensor.shape)


if __name__ == "__main__":
    unittest.main()
    