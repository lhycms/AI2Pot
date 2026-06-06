import unittest
import os
from typing import (List, Tuple, Optional)

import numpy as np
import torch

from ai2pot.optimizer.laddar_bfgs_callback import (
    PrintEFVLossCallback,
    CallbackManager)
from ai2pot.data.mlffdataset import ExtxyzDataset
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_train import LitLinearMtp


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
# os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")

torch.manual_seed(42)
torch.set_num_threads(16)



class PrintEFVLossCallbackTest(unittest.TestCase):
    def setUp(self):
        print("PrintEFVLossCallbackTest (TestCase) is setting up...\n")
        ###
        self.type_map: List[int] = ExtxyzDataset.get_type_map(filename=PbTe_EXTXYZ_PATH)
        self.umax_num_neigh_atoms: int = 200
        self.fit_virial: bool = False
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.device: torch._C.device = torch.device("cuda")
        self.torch_float_dtype: torch._C.dtype = torch.float32

        self.e_weight: float = 2.0
        self.f_weight: float = 3.0
        self.v_weight: float = 0.0

        self.lit_linear_mtp: LitLinearMtp = LitLinearMtp(type_map=self.type_map,
                                                         umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                         fit_virial=self.fit_virial,
                                                         mtp_level=16,
                                                         chebyshev_size=self.chebyshev_size,
                                                         rmax=self.rmax,
                                                         rmin=self.rmin,
                                                         zbl_rmax=0.0,
                                                         zbl_rmin=0.0,
                                                         zbl_cks_list=None,
                                                         zbl_dks_list=None,
                                                         e_wgt_start=self.e_weight,
                                                         e_wgt_end=self.e_weight,
                                                         f_wgt_start=self.f_weight,
                                                         f_wgt_end=self.f_weight,
                                                         v_wgt_start=self.v_weight,
                                                         v_wgt_end=self.v_weight)
        self.trainset: ExtxyzDataset = ExtxyzDataset(filename=PbTe_EXTXYZ_PATH,
                                                     rcut=self.rmax,
                                                     umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                     pbc_xyz=[True, True, True],
                                                     sort=False,
                                                     torch_float_dtype=self.torch_float_dtype,
                                                     has_virial=self.fit_virial)
        
        ###
        self.print_efv_loss_callback: PrintEFVLossCallback = PrintEFVLossCallback()


    def tearDown(self):
        print("PrintEFVLossCallbackTest (TestCase) is tearing down...\n")
    

    def test_on_train_step_end(self):
        self.print_efv_loss_callback.on_train_step_end(lit_linear_mtp=self.lit_linear_mtp,
                                                       trainset=self.trainset)


class CallbackManagerTest(unittest.TestCase):
    def setUp(self):
        print("CallbackManagerTest (TestCase) is setting up...\n")
        ###
        self.type_map: List[int] = ExtxyzDataset.get_type_map(filename=PbTe_EXTXYZ_PATH)
        self.umax_num_neigh_atoms: int = 200
        self.fit_virial: bool = False
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.device: torch._C.device = torch.device("cuda")
        self.torch_float_dtype: torch._C.dtype = torch.float32

        self.e_weight: float = 2.0
        self.f_weight: float = 3.0
        self.v_weight: float = 0.0

        self.lit_linear_mtp: LitLinearMtp = LitLinearMtp(type_map=self.type_map,
                                                         umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                         fit_virial=self.fit_virial,
                                                         mtp_level=16,
                                                         chebyshev_size=self.chebyshev_size,
                                                         rmax=self.rmax,
                                                         rmin=self.rmin,
                                                         zbl_rmax=0.0,
                                                         zbl_rmin=0.0,
                                                         zbl_cks_list=None,
                                                         zbl_dks_list=None,
                                                         e_wgt_start=self.e_weight,
                                                         e_wgt_end=self.e_weight,
                                                         f_wgt_start=self.f_weight,
                                                         f_wgt_end=self.f_weight,
                                                         v_wgt_start=self.v_weight,
                                                         v_wgt_end=self.v_weight)
        self.trainset: ExtxyzDataset = ExtxyzDataset(filename=PbTe_EXTXYZ_PATH,
                                                     rcut=self.rmax,
                                                     umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                     pbc_xyz=[True, True, True],
                                                     sort=False,
                                                     torch_float_dtype=self.torch_float_dtype,
                                                     has_virial=self.fit_virial)
    
        ###
        self.print_efv_loss_callback: PrintEFVLossCallback = PrintEFVLossCallback()
        self.callback_manager: CallbackManager = CallbackManager(callbacks=[self.print_efv_loss_callback,
                                                                            self.print_efv_loss_callback,
                                                                            self.print_efv_loss_callback])
    

    def tearDown(self):
        print("CallbackManagerTest (TestCase) is tearing down...\n")


    def test_on_train_step_end(self):
        self.callback_manager.on_train_step_end(lit_linear_mtp=self.lit_linear_mtp,
                                                trainset=self.trainset)


if __name__ == "__main__":
    unittest.main()
