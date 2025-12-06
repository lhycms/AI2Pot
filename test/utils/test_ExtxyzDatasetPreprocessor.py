import os
import unittest
from typing import (List, Tuple, Optional)
import torch

import numpy as np
from ai2pot.utils.prepot import ExtxyzDatasetPreprocessor
from ai2pot.data.mlffdataset import ExtxyzDataset
from ai2pot.models.mtp.linear_mtp import LinearMtp


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")
C_EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"

#torch.use_deterministic_algorithms(True)
torch.set_num_threads(16)
torch.manual_seed(21432)


class ExtxyzDatasetPreprocessorTest(unittest.TestCase):
    def setUp(self):
        print("ExtxyzDatasetPreprocessorTest (TestCase) is setting up...")
        
        # 1. Model
        filename: str = PbTe_EXTXYZ_PATH
        self.type_map: List[int] = [82, 52]
        self.chebyshev_size: int = 8
        rmax: float = 6.0
        rmin: float = 0.0
        umax_num_neigh_atoms = 200
        has_virial: bool = False
        self.device: torch._C.device = torch.device("cpu")
        torch_float_dtype: torch._C.dtype = torch.float64
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.linear_mtp: LinearMtp = LinearMtp(mtp_level=16,
                                               type_map=self.type_map,
                                               chebyshev_size=self.chebyshev_size,
                                               rmax=rmax,
                                               rmin=rmin,
                                               umax_num_neigh_atoms=umax_num_neigh_atoms,
                                               q_shifter=None,
                                               q_scaler=None,
                                               fit_virial=has_virial).to(torch_float_dtype)

        # 2. Data
        pbc_xyz: List[bool] = [True, True, True]
        sort: bool = False
        self.extxyz_dataset: ExtxyzDataset = ExtxyzDataset(filename=filename,
                                                      rcut=rmax,
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
        print("1. test_get_dataset_target_statistics:")
        print("\t1.1. ei_mean = ", ei_mean)
        print("\t1.2. ei_std = ", ei_std)
        print("\t1.3. force_std = ", force_std)
    

    def test_get_dataset_q_scaler_maxmin(self):
        descriptor_max_tensor, descriptor_min_tensor = ExtxyzDatasetPreprocessor.get_dataset_descriptors_maxmin(extxyz_dataset=self.extxyz_dataset,
                                                                                                                model=self.linear_mtp,
                                                                                                                batch_size=100)
        
        q_scaler_tensor = ExtxyzDatasetPreprocessor.get_dataset_q_scaler_maxmin(extxyz_dataset=self.extxyz_dataset,
                                                                                model=self.linear_mtp,
                                                                                batch_size=100)
        
        print("2. test_get_dataset_q_scaler_maxmin && test_get_dataset_descriptors_maxmin:")
        print("\t2.1. test_get_dataset_descriptors_maxmin:")
        print(descriptor_max_tensor)
        print(descriptor_min_tensor)
        print("\t2.2. test_get_dataset_q_scaler_maxmin:")
        print(q_scaler_tensor)
        print(descriptor_min_tensor / q_scaler_tensor)
    

    def test_get_dataset_q_scaler_statistics(self):
        q_scaler_tensor = ExtxyzDatasetPreprocessor.get_dataset_q_scaler_statistics(extxyz_dataset=self.extxyz_dataset,
                                                                                    model=self.linear_mtp,
                                                                                    batch_size=100)
        print("4.***")


if __name__ == "__main__":
    unittest.main()
    