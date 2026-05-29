import os
import time
import unittest
from typing import (List)

import numpy as np
import torch

from ai2pot.optimizer.ladar_bfgs import TorchScipyBfgs
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.data.mlffdataset import ExtxyzDataset


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
# os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")

torch.manual_seed(42)
torch.set_num_threads(16)


class TorchScipyBfgsTest(unittest.TestCase):
    def setUp(self):
        print("TorchScipyBfgsTest (TestCase) is setting up...\n")
        ###
        self.type_map: List[int] = ExtxyzDataset.get_type_map(filename=PbTe_EXTXYZ_PATH)
        self.umax_num_neigh_atoms: int = 200
        self.fit_virial: bool = False
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.device: torch._C.device = torch.device("cuda")
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.linear_mtp: LinearMtp = LinearMtp(type_map=self.type_map,
                                               umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                               fit_virial=self.fit_virial,
                                               mtp_level=16,
                                               chebyshev_size=self.chebyshev_size,
                                               rmax=self.rmax,
                                               rmin=self.rmin,
                                               zbl_rmax=0.0,
                                               zbl_rmin=0.0,
                                               zbl_cks_list=None,
                                               zbl_dks_list=None).to(device=self.device).to(self.torch_float_dtype)
        
        self.trainset: ExtxyzDataset = ExtxyzDataset(filename=PbTe_EXTXYZ_PATH,
                                                     rcut=self.rmax,
                                                     umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                     pbc_xyz=[True, True, True],
                                                     sort=False,
                                                     torch_float_dtype=self.torch_float_dtype,
                                                     has_virial=self.fit_virial)

        ###
        self.maxiter: int = 500
        self.torch_scipy_bfgs: TorchScipyBfgs = TorchScipyBfgs(linear_mtp=self.linear_mtp,
                                                               extxyz_dataset=self.trainset,
                                                               e_weight=1.0,
                                                               f_weight=2.0,
                                                               v_weight=0.0,
                                                               fit_virial=False,
                                                               batch_size=25,
                                                               maxiter=self.maxiter,
                                                               gtol=1e-7,
                                                               disp=True,
                                                               use_best_params=True)

    
    def tearDown(self):
        print("TorchScipyBfgsTest (TestCase) is tearing down...\n")
    

    def est_get_x0(self):
        x0: np.ndarray = self.torch_scipy_bfgs._get_x0()
        print("1. x0:")
        print("*" * 80)
        print(x0)

    
    def est_set_x(self):
        x0: np.ndarray = self.torch_scipy_bfgs._get_x0() + 1.0
        self.torch_scipy_bfgs._set_x(x=x0)

    
    def test_loss_and_grad(self):
        x1: np.ndarray = self.torch_scipy_bfgs._get_x0() + 1.0
        total_loss, grad = self.torch_scipy_bfgs._loss_and_grad(x=x1)
        total_loss: float
        grad: np.ndarray
        print(total_loss)
        print(grad)


if __name__ == "__main__":
    unittest.main()
