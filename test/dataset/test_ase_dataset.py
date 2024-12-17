import unittest
import os
from typing import List

import torch
import numpy as np
from torch.utils.data import DataLoader
from ai2pot.data.mlffdataset import ExtxyzDataset


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
CU_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "Cu", "Cu.extxyz")


class AseScDatasetTest(unittest.TestCase):
    def setUp(self):
        print("AseScDatasetTest (TestCase) is setting up...")
        self.rcut: float = 5.0
        self.umax_num_neigh_atoms: int = 100
        self.pbc_xyz: List[bool] = [True, True, True]
        self.sort: bool = False
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.np_float_dtype: np.dtype = np.float32
        self.has_virials: bool = False
    
    def tearDown(self):
        print("AseScDatasetTest (TestCase) is tearding down...")
    
    def test_extxyz(self):
        extxyz_dataset = ExtxyzDataset(filename=CU_EXTXYZ_PATH,
                                       rcut=self.rcut,
                                       umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                       pbc_xyz=self.pbc_xyz,
                                       sort=self.sort,
                                       torch_float_dtype=self.torch_float_dtype,
                                       has_virials=self.has_virials)
        extxyz_dataloader = DataLoader(dataset=extxyz_dataset,
                                       batch_size=32,
                                       shuffle=True)
        for idx, batch in enumerate(extxyz_dataloader):
            print("---- Iteration {0} ----".format(idx))
            print("\tinum.size() = ", batch[0].size())
            print("\tilist.size() = ", batch[1].size())
            print("\tnumneigh.size() = ", batch[2].size())
            print("\tfirstneigh.size() = ", batch[3].size())
            print("\trcs.size() = ", batch[4].size())
            print("\ttypes.size() = ", batch[5].size())
            print("\tnghost.size() = ", batch[6].size())
            print("\tenergies.size() = ", batch[7].size())
            print("\tforces.size() = ", batch[8].size())
            print("---------------------")
    
    
if __name__ == "__main__":
    unittest.main()
    