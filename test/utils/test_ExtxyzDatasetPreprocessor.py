import os
import unittest
from typing import (List, Tuple, Optional)
import torch

import numpy as np
from ai2pot.utils.prepot import ExtxyzDatasetPreprocessor
from ai2pot.data.mlffdataset import ExtxyzDataset


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")

#torch.use_deterministic_algorithms(True)
torch.set_num_threads(16)
torch.manual_seed(2143)


class ExtxyzDatasetPreprocessorTest(unittest.TestCase):
    def setUp(self):
        print("ExtxyzDatasetPreprocessorTest (TestCase) is setting up...")
        
        filename: str = PbTe_EXTXYZ_PATH
        rcut: float = 5.0
        umax_num_neigh_atoms: int = 200
        pbc_xyz: List[bool] = [True, True, True]
        sort: bool = False
        torch_float_dtype: torch._C.dtype = torch.float64
        has_virial: bool = False
        self.extxyz_dataset: ExtxyzDataset = ExtxyzDataset(filename=filename,
                                                      rcut=rcut,
                                                      umax_num_neigh_atoms=umax_num_neigh_atoms,
                                                      pbc_xyz=pbc_xyz,
                                                      sort=sort,
                                                      torch_float_dtype=torch_float_dtype,
                                                      has_virial=has_virial)

    def tearDown(self):
        print("ExtxyzDatasetPreprocessorTest (TestCase) is tearing down...")


    def test_get_dataset_target_statistics(self):
        ei_mean, ei_std, force_std = ExtxyzDatasetPreprocessor.get_dataset_target_statistics(extxyz_dataset=self.extxyz_dataset,
                                                                batch_size=100)
        print("1. ei_mean = ", ei_mean)
        print("2. ei_std = ", ei_std)
        print("3. force_std = ", force_std)



if __name__ == "__main__":
    unittest.main()