import os
import time
import unittest
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
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
        
        device_str: str = "cuda"
        self.linear_mtp_maxvol_generator: LinearMtpMaxvolGenerator = LinearMtpMaxvolGenerator(
            checkpoint_path=CHECK_POINT_PATH,
            candidate_set_path=EXTXYZ_PATH,
            map_location=device_str,
            pbc_xyz=[True, True, True])
        
        self.device: torch._C.device = torch.device(device_str)
        
        self.linear_mtp: LinearMtp = self.linear_mtp_maxvol_generator.model
        self.candidate_set_dataloader: DataLoader = self.linear_mtp_maxvol_generator.candidate_set_dataloader


    def tearDown(self):
        print("LinearMtpSerializer (TestCase) is tearing down...")
        
    
    def est_generate_candidate_set_matrix_local(self):
        structure_id_tensor, atom_id_tensor, candidate_matrix_tensor = self.linear_mtp_maxvol_generator._generate_candidate_set_matrix_local()
        print(structure_id_tensor.shape)
        print(atom_id_tensor.shape)
        print(candidate_matrix_tensor.shape)
        
    
    def est_generate_maxvol_matrix_local(self):
        maxvol_structure_id_tensor, maxvol_atom_id_tensor, maxvol_matrix, maxvol_matrix_inv = \
            self.linear_mtp_maxvol_generator.generate_maxvol_matrix_local()
        print(maxvol_structure_id_tensor)
        print(maxvol_atom_id_tensor)
        print(maxvol_matrix)
        print(maxvol_matrix_inv)
    
    
    def est_linear_mtp_predict_gamma_local(self):
        maxvol_structure_id_tensor, maxvol_atom_id_tensor, maxvol_matrix, maxvol_matrix_inv = \
            self.linear_mtp_maxvol_generator.generate_maxvol_matrix_local()

        for batch_idx, batch_data in enumerate(self.candidate_set_dataloader):
            binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                brcs_tensor, btypes_tensor, bnghost_tensor, \
                betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = \
                    [t.to(self.device) for t in batch_data]
            bgamma_strutures_tensor, bgamma_atoms_tensor = self.linear_mtp.predict_gamma_local(
                binum_tensor,
                bilist_tensor,
                bnumneigh_tensor,
                bfirstneigh_tensor,
                brcs_tensor,
                btypes_tensor,
                bnghost_tensor,
                maxvol_matrix_inv)
            
            print(bgamma_strutures_tensor)
            print(bgamma_atoms_tensor.amax())
    
    
    def est_get_gamma_local(self):
        maxvol_structure_id_tensor, maxvol_atom_id_tensor, maxvol_matrix, maxvol_matrix_inv = \
            self.linear_mtp_maxvol_generator.generate_maxvol_matrix_local()
            
        structure_id_tensor, atom_id_tensor, gamma_structures_tensor, gamma_atoms_tensor = self.linear_mtp_maxvol_generator.get_gamma_local(
            maxvol_matrix_inv=maxvol_matrix_inv,
            explore_path=EXTXYZ_PATH)
        print(structure_id_tensor)
        print(atom_id_tensor)
        print(gamma_structures_tensor.shape)
        print(gamma_atoms_tensor)


    def test_generate_candidate_set_matrix_global(self):
        structure_id_tensor, candidate_matrix_tensor = self.linear_mtp_maxvol_generator._generate_candidate_set_matrix_global()
        print(structure_id_tensor.shape)
        print(candidate_matrix_tensor.shape)
        
        


if __name__ == "__main__":
    unittest.main()
    