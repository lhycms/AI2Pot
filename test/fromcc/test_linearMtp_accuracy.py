import os
import unittest
from typing import List

from pymatgen.core import Structure
import torch
from torch.autograd import gradcheck

from ai2pot.utils.usepot import MlffToLossInput
from ai2pot.fromcc import (
    linearMtpToEFLossOp,
    linearMtpToLossOp,
    mtpParamOp
)


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))


class LinearMtpTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtpTest (TestCase) is setting up...\n")
        # 0.
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.device: torch._C.device = torch.device("cpu")
        
        # 1. 
        self.mtp_level: int = 8
        self.ntypes: int = 4
        self.chebyshev_size: int = 8
        self.rmax: float = 6.0
        self.rmin: float = 0.5
        self.umax_num_neighs: int = 200
        self.fit_virial: bool = False
        
        self.structure: Structure = Structure.from_file(ReNbSSe_POSCAR_PATH)
        
        self.mlff_to_loss_input: MlffToLossInput = MlffToLossInput(rcut=self.rmax,
                                                                   umax_num_neighs=self.umax_num_neighs)
        
        # 2. mtpParamOp
        mtp_param_info = mtpParamOp(self.mtp_level)
        self.alpha_moments_count: int = mtp_param_info[0].item()
        self.alpha_index_basic_tensor: torch.Tensor = mtp_param_info[1]
        self.alpha_index_times_tensor: torch.Tensor = mtp_param_info[2]
        self.alpha_moment_mapping_tensor: torch.Tensor = mtp_param_info[3]
        self.nmus: int = mtp_param_info[6].item()
        
        # 3. DFT reference
        num_atoms: int = len(self.structure.species)
        self.betot_dft_tensor: torch.Tensor = torch.zeros(size=(1,),
                                                     dtype=self.torch_float_dtype,
                                                     device=self.device)
        self.bforce_dft_tensor: torch.Tensor = torch.zeros(size=(1, num_atoms, 3),
                                                      dtype=self.torch_float_dtype,
                                                      device=self.device)
        self.bvirial_dft_tensor: torch.Tensor = torch.zeros(size=(1, 3, 3),
                                                       dtype=self.torch_float_dtype,
                                                       device=self.device)
        
        
        # 4. coeffs, linear coeffs, type bias
        self.coeffs: torch.Tensor = torch.zeros(self.ntypes*self.ntypes*self.nmus*self.chebyshev_size)
        self.coeffs.to(self.torch_float_dtype).to(self.device)
        self.linear_coeffs: torch.Tensor = torch.zeros(self.alpha_moment_mapping_tensor.size(0))
        self.linear_coeffs.to(self.torch_float_dtype).to(self.device)
        self.type_bias: torch.Tensor = torch.zeros(self.ntypes)
        self.type_bias.to(self.torch_float_dtype).to(self.device)        
        
    
    def tearDown(self):
        print("LinearMtpTest (TestCase) is tearing down...\n")
    
    
    def test_linearMtpToLoss(self):
        # 1. Parameters
        e_weight: float = 1.0
        f_weight: float = 0.0
        v_weight: float = 0.0
        self.coeffs.requires_grad_(False)
        self.linear_coeffs.requires_grad_(False)
        self.type_bias.requires_grad_(False)
        
        # 2. Run
        input_info: List[torch.Tensor] = self.mlff_to_loss_input.analyse_pymatgen(self.structure,
                                                                                  e_weight=self.e_weight,
                                                                                  f_weight=self.f_weight,
                                                                                  v_weight=self.v_weight)
        test = gradcheck(func=linearMtpToEFLossOp,
                         inputs=(),
                         eps=1e-8,
                         atol=1e-6)
        print("Gradient pass check: ", test)
    
    
    
if __name__ == "__main__":
    unittest.main()