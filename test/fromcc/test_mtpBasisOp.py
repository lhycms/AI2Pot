import unittest
import os
from typing import List

import torch
from pymatgen.core import Structure
from ai2pot.fromcc import mtpParamOp, mtpBasisOp
from ai2pot.utils.usepot import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")


class MtpBasisOpTest(unittest.TestCase):
    def setUp(self):
        print("MtpBasisOpTest (TestCase) is setting up...")
        
        self.mtp_level: int = 16
        self.ntypes: int = 4
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 2.0
        self.umax_num_neigh: int = 30
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        
        self.mlff_input = MlffInput(rcut=self.rmax,
                                    umax_num_neighs=self.umax_num_neigh,
                                    pbc_xyz=[True, True, True],
                                    sort=False,
                                    dtype=self.dtype,
                                    device=self.device)
        self.structure = Structure.from_file(ReNbSSe_POSCAR)
        
    def tearDown(self):
        print("MtpBasisOpTest (TestCase) is tearing down...")
        
    def test_mtp_param_op(self):
        mtp_param_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        #print(mtp_param_info[0].size())
        #print(mtp_param_info[1].size())
        #print(mtp_param_info[2].size())
        #print(mtp_param_info[3].size())
        #print(mtp_param_info[4].size())
        #print(mtp_param_info[5].size())
        
    def test_der2xyz(self):
        center_idx_modify: int = 4
        neigh_idx_modify: int = 7
        direction_idx_modify: int = 2
        delta: float = 1e-4
        
        mtp_parm_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        nmus: int = mtp_parm_info[5].item()
        coeffs_tensor: torch.Tensor = torch.ones(self.ntypes*self.ntypes*nmus*self.chebyshev_size, dtype=self.dtype)
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        nblist_info[4].requires_grad_(True)
        descriptor = mtpBasisOp(mtp_parm_info[0],
                                mtp_parm_info[1],
                                mtp_parm_info[2],
                                mtp_parm_info[3],
                                mtp_parm_info[4],
                                mtp_parm_info[5],
                                self.ntypes,
                                self.chebyshev_size,
                                coeffs_tensor,
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        result: torch.Tensor = descriptor.sum()
        result.backward()
        
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4][0][center_idx_modify][neigh_idx_modify][direction_idx_modify] += delta
        nblist_info_[4].requires_grad_(True)
        descriptor_ = mtpBasisOp(mtp_parm_info[0],
                                mtp_parm_info[1],
                                mtp_parm_info[2],
                                mtp_parm_info[3],
                                mtp_parm_info[4],
                                mtp_parm_info[5],
                                self.ntypes,
                                self.chebyshev_size,
                                coeffs_tensor,
                                nblist_info_[1],
                                nblist_info_[2],
                                nblist_info_[3],
                                nblist_info_[4],
                                nblist_info_[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        result_: torch.Tensor = descriptor_.sum()
        result_.backward()
        
        print("---------------------------------------------------------------------------------------")
        print("1.1. Sum of descriptors derivative w.r.t. rcs[{0}][{1}][{2}] calculated by custom code:".format(center_idx_modify, neigh_idx_modify, direction_idx_modify))
        print(descriptor.size(), result)
        print(nblist_info[4].grad[0][center_idx_modify][neigh_idx_modify][direction_idx_modify])
        print("1.2. Sum of descriptors derivative w.r.t rcs[{0}][{1}][{2}] calculated by finite difference code:".format(center_idx_modify, neigh_idx_modify, direction_idx_modify))
        print(result_)
        print((result_ - result) / delta)
        print("---------------------------------------------------------------------------------------")


    def test_mtp_basis_der2coeffs(self):
        itype_modify: int = 0
        jtype_modify: int = 1
        mu_modify: int = 0
        xi_modify: int = 1
        delta: float = 1e-4
        mtp_param_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        nmus: int = mtp_param_info[5].item()
        coeff_idx_modify: int = (itype_modify*self.ntypes + jtype_modify)*nmus*self.chebyshev_size + mu_modify*self.chebyshev_size + xi_modify
        
        coeffs_tensor: torch.Tensor = torch.ones(self.ntypes*self.ntypes*nmus*self.chebyshev_size, dtype=self.dtype)
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        coeffs_tensor.requires_grad_(True)
        descriptor = mtpBasisOp(mtp_param_info[0],
                                mtp_param_info[1],
                                mtp_param_info[2],
                                mtp_param_info[3],
                                mtp_param_info[4],
                                mtp_param_info[5],
                                self.ntypes,
                                self.chebyshev_size,
                                coeffs_tensor,
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        result: torch.Tensor = descriptor.sum()
        result.backward()
        
        coeffs_tensor_ = coeffs_tensor.clone().detach()
        coeffs_tensor_[coeff_idx_modify] += delta
        coeffs_tensor_.requires_grad_(True)
        descriptor_ = mtpBasisOp(mtp_param_info[0],
                                mtp_param_info[1],
                                mtp_param_info[2],
                                mtp_param_info[3],
                                mtp_param_info[4],
                                mtp_param_info[5],
                                self.ntypes,
                                self.chebyshev_size,
                                coeffs_tensor_,
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        result_: torch.Tensor = descriptor_.sum()
        result_.backward()
        print("---------------------------------------------------------------------------------------")
        print("2.1. Sum of descriptors detivative w.r.t coeffs[{0}][{1}][{2}][{3}] calculated by custom code:\n".format(itype_modify, jtype_modify, mu_modify, xi_modify))
        print(coeffs_tensor.grad[coeff_idx_modify])
        print("2.2. Sum of descriptors detivative w.r.t coeffs[{0}][{1}][{2}][{3}] calculated by finite difference method:\n".format(itype_modify, jtype_modify, mu_modify, xi_modify))
        print((result_ - result) / delta)
        print("---------------------------------------------------------------------------------------")

    
if __name__ == "__main__":
    unittest.main()
    