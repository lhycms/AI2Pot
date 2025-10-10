import os
import unittest
from typing import List

from pymatgen.core import Structure
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from ai2pot.utils.usepot import MlffToLossInput, MlffInput
from ai2pot.data.mlffdataset import ExtxyzDataset
from ai2pot.fromcc import (
    linearMtpToEFLossOp,
    linearMtpToLossOp,
    linearMtpToEsitesOp,
    mtpParamOp,
    linearMtpToDescriptorsOp
)


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")

#torch.use_deterministic_algorithms(True)
torch.set_num_threads(16)
torch.manual_seed(2143)


class LinearMtpTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtpTest (TestCase) is setting up...\n")
        # 0.
        self.torch_float_dtype: torch._C.dtype = torch.float64
        self.device: torch._C.device = torch.device("cuda")
        
        # 1. 
        self.mtp_level: int = 16
        #self.ntypes: int = 4
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.umax_num_neigh_atoms: int = 200
        self.fit_virial: bool = False
        
        """
        self.ntypes: int = 4
        self.type_map: List[int] = [16, 34, 41, 75]
        self.type_map_tensor = torch.tensor(self.type_map, dtype=torch.int32).to(self.device)
        self.structure: Structure = Structure.from_file(ReNbSSe_POSCAR_PATH)
        """
        self.ntypes: int = 2
        self.type_map: List[int] = [1, 8]
        self.type_map_tensor: torch.Tensor = torch.tensor(data=self.type_map, 
                                                          dtype=torch.int32).to(self.device)
        self.structure: Structure = Structure(lattice=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                                              species=["H", "H", "O"],
                                              coords=[[0, 0, 0],
                                                      [0, 4.0, 0],
                                                      [3.0, 0.0, 0]
                                                      ],
                                              coords_are_cartesian=True)
        print(self.structure)
    
        # 2. ZBL
        self.zbl_rmax: float = 0.0
        self.zbl_rmin: float = 0.0
        self.zbl_cks_tensor: torch.Tensor = torch.zeros(self.ntypes*self.ntypes*4, 
                                                        dtype=self.torch_float_dtype,
                                                        device=self.device)
        self.zbl_dks_tensor: torch.Tensor = torch.zeros(self.ntypes*self.ntypes*4, 
                                                        dtype=self.torch_float_dtype,
                                                        device=self.device)
        for ii in range(self.ntypes):
            for jj in range(self.ntypes):
                idx = ii*self.ntypes + jj
                self.zbl_cks_tensor[idx*4 + 0] = 0.18175
                self.zbl_cks_tensor[idx*4 + 1] = 0.50986
                self.zbl_cks_tensor[idx*4 + 2] = 0.28022
                self.zbl_cks_tensor[idx*4 + 3] = 0.02817
                self.zbl_dks_tensor[idx*4 + 0] = 3.1998
                self.zbl_dks_tensor[idx*4 + 1] = 0.94229
                self.zbl_dks_tensor[idx*4 + 2] = 0.4029
                self.zbl_dks_tensor[idx*4 + 3] = 0.20162
        
        self.mlff_to_loss_input: MlffToLossInput = MlffToLossInput(type_map=self.type_map,
                                                                   rcut=self.rmax,
                                                                   umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                                   dtype=self.torch_float_dtype,
                                                                   device=self.device)
        self.mlff_input: MlffInput = MlffInput(type_map=self.type_map,
                                                rcut=self.rmax,
                                                umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                dtype=self.torch_float_dtype,
                                                device=self.device)
        
        # 2. mtpParamOp
        mtp_param_info = mtpParamOp(self.mtp_level)
        self.alpha_moments_count: int = mtp_param_info[0].item()
        self.alpha_index_basic_tensor: torch.Tensor = mtp_param_info[1].to(self.device)
        self.alpha_index_times_tensor: torch.Tensor = mtp_param_info[2].to(self.device)
        self.alpha_moment_mapping_tensor: torch.Tensor = mtp_param_info[3].to(self.device)
        self.alpha_scalar_moments: int = self.alpha_moment_mapping_tensor.size(0)
        self.nmus: int = mtp_param_info[6].item()
        
        # 3. coeffs, linear coeffs, type bias
        self.coeffs_tensor: torch.Tensor = torch.zeros(self.ntypes*self.ntypes*self.nmus*self.chebyshev_size, 
                                                       dtype=self.torch_float_dtype,
                                                       device=self.device)
        nn.init.normal_(self.coeffs_tensor, mean=0.0, std=0.2)
        self.linear_coeffs_tensor: torch.Tensor = torch.zeros(self.alpha_moment_mapping_tensor.size(0),
                                                              dtype=self.torch_float_dtype,
                                                              device=self.device)
        nn.init.normal_(self.linear_coeffs_tensor, mean=0.0, std=0.1)
        self.type_bias_tensor: torch.Tensor = torch.zeros(self.ntypes,
                                                          dtype=self.torch_float_dtype,
                                                          device=self.device)
        nn.init.normal_(self.type_bias_tensor, mean=0.0, std=0.1)
        
        # q_shifter_tensor, q_scaler_tensor
        self.q_shifter_tensor: torch.Tensor = torch.zeros(self.alpha_scalar_moments,
                                                          dtype=self.torch_float_dtype,
                                                          device=self.device)
        self.q_scaler_tensor: torch.Tensor = torch.randn(self.alpha_scalar_moments,
                                                        dtype=self.torch_float_dtype,
                                                        device=self.device)
        
    
    def tearDown(self):
        print("LinearMtpTest (TestCase) is tearing down...\n")
    
    
    def test_linearMtpToLoss(self):
        # 1. Parameters
        e_weight: float = 1.0
        f_weight: float = 0.1
        v_weight: float = 0.0
        self.coeffs_tensor.requires_grad_(True)
        self.linear_coeffs_tensor.requires_grad_(True)
        self.type_bias_tensor.requires_grad_(True)
        
        # 2. Run
        input_info: List[torch.Tensor] = self.mlff_to_loss_input.analyse_pymatgen(self.structure,
                                                                                  e_weight=e_weight,
                                                                                  f_weight=f_weight,
                                                                                  v_weight=v_weight)

        test = gradcheck(func=linearMtpToEFLossOp,
                         inputs=(e_weight,
                                 f_weight,
                                 input_info[3],
                                 input_info[4],
                                 self.chebyshev_size,
                                 self.coeffs_tensor,
                                 self.linear_coeffs_tensor,
                                 self.type_bias_tensor,
                                 self.alpha_moments_count,
                                 self.alpha_index_basic_tensor,
                                 self.alpha_index_times_tensor,
                                 self.alpha_moment_mapping_tensor,
                                 self.nmus,
                                 input_info[6],
                                 input_info[7],
                                 input_info[8],
                                 input_info[9],
                                 input_info[10],
                                 input_info[11],
                                 self.type_map_tensor,
                                 input_info[12].item(),
                                 self.rmax,
                                 self.rmin,
                                 self.q_shifter_tensor,
                                 self.q_scaler_tensor,
                                 self.zbl_rmax,
                                 self.zbl_rmin,
                                 self.zbl_cks_tensor,
                                 self.zbl_dks_tensor),
                         eps=1e-5,
                         atol=1e-6,
                         rtol=1e-3,
                         nondet_tol=1e-5)
        print("-------------------------------------------------")
        print("* linearMtpToEFLossOp Gradient pass check: ", test)
        print("-------------------------------------------------")


    def est_linearMtpToEsites(self):
        # 1. Parameters
        self.coeffs_tensor.requires_grad_(True)
        self.linear_coeffs_tensor.requires_grad_(True)
        self.type_bias_tensor.requires_grad_(True)

        
        # 2. Run
        input_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)

        test = gradcheck(func=linearMtpToEsitesOp,
                         inputs=(self.chebyshev_size,
                                 self.coeffs_tensor,
                                 self.linear_coeffs_tensor,
                                 self.type_bias_tensor,
                                 self.alpha_moments_count,
                                 self.alpha_index_basic_tensor,
                                 self.alpha_index_times_tensor,
                                 self.alpha_moment_mapping_tensor,
                                 self.nmus,
                                 input_info[0],
                                 input_info[1],
                                 input_info[2],
                                 input_info[3],
                                 input_info[4],
                                 input_info[5],
                                 self.type_map_tensor,
                                 input_info[6].item(),
                                 self.rmax,
                                 self.rmin,
                                 self.q_shifter_tensor,
                                 self.q_scaler_tensor,
                                 self.zbl_rmax,
                                 self.zbl_rmin,
                                 self.zbl_cks_tensor,
                                 self.zbl_dks_tensor),
                         eps=1e-8,
                         atol=1e-6,
                         rtol=1e-3,
                         nondet_tol=1e-2)
        print("-------------------------------------------------")
        print("* linearMtpToEsitesOp Gradient pass check: ", test)
        print("-------------------------------------------------")


    def est_output_descriptors(self):
        input_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        for item in input_info:
            item.to(self.device)

        bdescriptors = linearMtpToDescriptorsOp(
            self.chebyshev_size,
            self.coeffs_tensor,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            input_info[0],
            input_info[1],
            input_info[2],
            input_info[3],
            input_info[4],
            input_info[5],
            self.type_map_tensor,
            input_info[6].item(),
            self.rmax,
            self.rmin)[0]
        print("1. bdescriptor.shape = ", bdescriptors)


if __name__ == "__main__":
    unittest.main()
