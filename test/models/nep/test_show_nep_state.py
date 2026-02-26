import unittest
import os
from typing import List

from pymatgen.core import Structure
import numpy as np
import torch
from torch.autograd import gradcheck
from ase import Atoms
from ase.io import read as ase_read

from ai2pot.utils.usepot import MlffToEFLossInput, MlffInput
from ai2pot.models.nep.nep_utils import (
    NepCalculator,
    Nep4Extxyz)
from ai2pot.fromcc import (
    nepToEFLossOp,
    nepToEFOp,
    nepToDescriptorsOp)



class Nep4ExtxyzTest(unittest.TestCase):
    def setUp(self):
        print("Nep4ExtxyzTest (TestCase) is setting up...")
        self.checkpoint_path: str = "/data/home/liuhanyu/mycode/AI2Pot/lightning_logs/lightning_logs/version_9/checkpoints/epoch=199-step=5000.ckpt"
        self.testset_path: str = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
        self.map_location: str = "cuda"
        self.device = torch.device(self.map_location)   
        self.torch_float_dtype: torch._C.dtype = torch.float64
        self.nep_extxyz: Nep4Extxyz = Nep4Extxyz(checkpoint_path=self.checkpoint_path,
                                                 testset_path=self.testset_path,
                                                 map_location=self.map_location,
                                                 torch_float_dtype=self.torch_float_dtype)
        print(self.nep_extxyz.model.coeffs_tensor)
        print(self.nep_extxyz.model.w0_tensor)
        print(self.nep_extxyz.model.b0_tensor)
        print(self.nep_extxyz.model.w1_tensor)
        print(self.nep_extxyz.model.type_bias_tensor)

        self.ntypes: int = 4
        self.type_map_tensor: torch.Tensor = torch.tensor(data=[3, 57, 40, 8], dtype=torch.int32)
        self.structure: Structure = Structure.from_file("/data/home/liuhanyu/mycode/AI2Pot/test/test_data/POSCARs/LiLaZrO3/POSCAR")
        self.mlff_to_loss_input: MlffToEFLossInput = MlffToEFLossInput(type_map=self.type_map_tensor.numpy().tolist(),
                                                                       rcut=self.nep_extxyz.model.rmax_radial,
                                                                       umax_num_neigh_atoms=self.nep_extxyz.model.umax_num_neigh_atoms,
                                                                       dtype=self.torch_float_dtype,
                                                                       device=self.device)
        
        self.mlff_input: MlffInput = MlffInput(type_map=self.type_map_tensor.numpy().tolist(),
                                                rcut=self.nep_extxyz.model.rmax_radial,
                                                umax_num_neigh_atoms=self.nep_extxyz.model.umax_num_neigh_atoms,
                                                dtype=self.torch_float_dtype,
                                                device=self.device)



    def tearDown(self):
        print("Nep4ExtxyzTest (TestCase) is tearing down...")


    def est_nepToEFLoss(self):
        # 1. Parameters
        e_weight: float = 0.0
        f_weight: float = 1e1
        self.nep_extxyz.model.coeffs_tensor.requires_grad_(True)
        self.nep_extxyz.model.w0_tensor.requires_grad_(True)
        self.nep_extxyz.model.b0_tensor.requires_grad_(True)
        self.nep_extxyz.model.w1_tensor.requires_grad_(True)
        self.nep_extxyz.model.type_bias_tensor.requires_grad_(True)
        
        # 2. Run
        input_info: List[torch.Tensor] = self.mlff_to_loss_input.analyse_pymatgen(self.structure,
                                                                             e_weight=e_weight,
                                                                             f_weight=f_weight)

        def nepToEFLossOp1(*args, **kwargs):
            return nepToEFLossOp(*args, **kwargs)[0]
        
        test = gradcheck(func=nepToEFLossOp1,
                         inputs=(e_weight,
                                 f_weight,
                                 input_info[2],
                                 input_info[3],
                                 self.nep_extxyz.model.chebyshev_size,
                                 self.nep_extxyz.model.n_radial_basis,
                                 self.nep_extxyz.model.n_angular_basis,
                                 self.nep_extxyz.model.l_max,
                                 self.nep_extxyz.model.coeffs_tensor,
                                 self.nep_extxyz.model.w0_tensor,
                                 self.nep_extxyz.model.b0_tensor,
                                 self.nep_extxyz.model.w1_tensor,
                                 self.nep_extxyz.model.type_bias_tensor,
                                 input_info[4],
                                 input_info[5],
                                 input_info[6],
                                 input_info[7],
                                 input_info[8],
                                 input_info[9],
                                 self.type_map_tensor,
                                 input_info[10].item(),
                                 self.nep_extxyz.model.rmax_radial,
                                 self.nep_extxyz.model.rmax_angular,
                                 self.nep_extxyz.model.q_scaler_tensor,
                                 self.nep_extxyz.model.zbl_rmax,
                                 self.nep_extxyz.model.zbl_rmin,
                                 self.nep_extxyz.model.zbl_cks_tensor,
                                 self.nep_extxyz.model.zbl_dks_tensor),
                            eps=1e-6,
                            atol=1e-5,
                            rtol=1e-4,
                            nondet_tol=1e-6)
        print("-------------------------------------------------")
        print("* NepToEFLossOp Gradient pass check: ", test)
        print("-------------------------------------------------")


    def test_nepToDescriptors(self):
        input_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        bdescriptors_tensor: torch.Tensor = nepToDescriptorsOp(self.nep_extxyz.model.chebyshev_size,
                                                               self.nep_extxyz.model.n_radial_basis,
                                                               self.nep_extxyz.model.n_angular_basis,
                                                               self.nep_extxyz.model.l_max,
                                                               self.nep_extxyz.model.coeffs_tensor,
                                                               input_info[0],
                                                               input_info[1],
                                                               input_info[2],
                                                               input_info[3],
                                                               input_info[4],
                                                               input_info[5],
                                                               self.type_map_tensor,
                                                               input_info[6].item(),
                                                               self.nep_extxyz.model.rmax_radial,
                                                               self.nep_extxyz.model.rmax_angular)
        print("-------------------------------------------------")
        print("* bdescriptor_tensor: ")
        print("-------------------------------------------------")
        print(bdescriptors_tensor[0][0][0])


if __name__ == '__main__':
    unittest.main()
