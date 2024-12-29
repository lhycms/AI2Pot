import unittest
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from pymatgen.core import Structure
from ai2pot.fromcc import mtpParamOp, mtpBasisOp
from ai2pot.utils.usepot import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")
MoS2_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "MoS2", "POSCAR")

class MtpBasisOpTest(unittest.TestCase):
    def setUp(self):
        print("MtpBasisOpTest (TestCase) is setting up...")
        
        self.mtp_level: int = 16
        self.ntypes: int = 4
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 2.0
        self.umax_num_neigh: int = 30
        self.dtype = torch.float64
        self.device = torch.device("cpu")
        
        self.center_idx_modify: int = 0
        self.neigh_idx_modify: int = 3
        self.descriptor_idx_modify: int = 40
        self.delta: float = 1e-5
                
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
        direction_idx_modify: int = 1
        
        mtp_parm_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        nmus: int = mtp_parm_info[5].item()
        num_descriptor: int = mtp_parm_info[2].size(0)
        coeffs_tensor: torch.Tensor = torch.randn(self.ntypes*self.ntypes*nmus*self.chebyshev_size, dtype=self.dtype) / 10
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
                                nblist_info[0],
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        single_descriptor = descriptor[0, self.center_idx_modify, self.descriptor_idx_modify]
        single_descriptor.backward()
        
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4][0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify] += self.delta
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
                                nblist_info[0],
                                nblist_info_[1],
                                nblist_info_[2],
                                nblist_info_[3],
                                nblist_info_[4],
                                nblist_info_[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        single_descriptor_ = descriptor_[0][self.center_idx_modify][self.descriptor_idx_modify]
        
        
        print("---------------------------------------------------------------------------------------")
        print("1.0. Descriptor[0, {0}, {1}] = {2}".format(self.center_idx_modify, self.descriptor_idx_modify, single_descriptor.item()))
        print("1.1. Descriptor[0, {0}, {1}] derivative w.r.t. rcs[{2}][{3}][{4}] calculated by custom code = {5:.6f}".format(
                self.center_idx_modify,
                self.descriptor_idx_modify,
                self.center_idx_modify, 
                self.neigh_idx_modify, 
                direction_idx_modify,
                nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item()))
        print("1.2. Descriptor[0, {0}, {1}] derivative w.r.t rcs[{2}][{3}][{4}] calculated by finite difference method = {5:.6f}".format(
                self.center_idx_modify,
                self.descriptor_idx_modify,
                self.center_idx_modify, 
                self.neigh_idx_modify, 
                direction_idx_modify,
                ((single_descriptor_ - single_descriptor) / self.delta).item()))
        print("1.3. Difference between custom code and finite difference method = {0:.5f}".format( 
            nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item()
            - ((single_descriptor_ - single_descriptor) / self.delta).item())
        )
        print("---------------------------------------------------------------------------------------")
        

    def test_mtp_basis_der2coeffs(self):
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        itype_modify: int = nblist_info[5][0][self.center_idx_modify].item()
        jtype_modify: int = nblist_info[5][0][self.neigh_idx_modify].item()
        mu_modify: int = 0
        xi_modify: int = 1
        
        mtp_param_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        nmus: int = mtp_param_info[5].item()
        coeff_idx_modify: int = (itype_modify*self.ntypes + jtype_modify)*nmus*self.chebyshev_size + mu_modify*self.chebyshev_size + xi_modify
        
        coeffs_tensor: torch.Tensor = torch.randn(self.ntypes*self.ntypes*nmus*self.chebyshev_size, dtype=self.dtype) / 10
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
                                nblist_info[0],
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        single_descriptor: torch.Tensor = descriptor[0, self.center_idx_modify, self.descriptor_idx_modify]
        single_descriptor.backward()
        
        coeffs_tensor_ = coeffs_tensor.clone().detach()
        coeffs_tensor_[coeff_idx_modify] += self.delta
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
                                nblist_info[0],
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        single_descriptor_: torch.Tensor = descriptor_[0, self.center_idx_modify, self.descriptor_idx_modify]
        single_descriptor_.backward()
        print("---------------------------------------------------------------------------------------")
        print("2.0. Descriptor[0, {0}, {1}] = {2}".format(self.center_idx_modify, self.descriptor_idx_modify, single_descriptor.item()))
        print("2.1. Descriptors[0, {0}, {1}] detivative w.r.t coeffs[{2}][{3}][{4}][{5}] calculated by custom code = {6:.6f}".format(
                self.center_idx_modify,
                self.descriptor_idx_modify,
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                coeffs_tensor.grad[coeff_idx_modify].item()))
        print("2.2. Descriptors[0, {0}, {1}] detivative w.r.t coeffs[{2}][{3}][{4}][{5}] calculated by finite difference method = {6:.6f}".format(
                self.center_idx_modify,
                self.descriptor_idx_modify,
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                ((single_descriptor_ - single_descriptor) / self.delta).item()))
        print("2.3. Difference between custom code and finite difference method = {0:.5f}".format(
            coeffs_tensor.grad[coeff_idx_modify].item() \
            - ((single_descriptor_ - single_descriptor) / self.delta).item()
        ))
        print("---------------------------------------------------------------------------------------")

    
    def test_sum_der2xyz(self):
        direction_idx_modify: int = 1
        
        mtp_parm_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        nmus: int = mtp_parm_info[5].item()
        num_descriptor: int = mtp_parm_info[2].size(0)
        coeffs_tensor: torch.Tensor = torch.randn(self.ntypes*self.ntypes*nmus*self.chebyshev_size, dtype=self.dtype) / 10
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
                                nblist_info[0],
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        sum_descriptor = descriptor.sum()
        sum_descriptor.backward()
        
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4][0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify] += self.delta
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
                                nblist_info[0],
                                nblist_info_[1],
                                nblist_info_[2],
                                nblist_info_[3],
                                nblist_info_[4],
                                nblist_info_[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        sum_descriptor_ = descriptor_.sum()
        
        print("---------------------------------------------------------------------------------------")
        print("3.0. Sum descriptor = {0}".format(sum_descriptor.item()))
        print("3.1. Sum descriptor derivative w.r.t. rcs[{0}][{1}][{2}] calculated by custom code = {3}".format(
                self.center_idx_modify,
                self.neigh_idx_modify,
                direction_idx_modify,
                nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item())
        )
        print("3.2. Sum descriptor derivative w.r.t. rcs[{0}][{1}][{2}] calculated by finite difference method = {3}".format(
                self.center_idx_modify,
                self.neigh_idx_modify,
                direction_idx_modify,
                (sum_descriptor_ - sum_descriptor) / self.delta)
        )
        print("3.3. Difference between custom code and finite difference method = {0:.5f}".format(
            nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item() \
            - (sum_descriptor_ - sum_descriptor) / self.delta)
        )
        print("---------------------------------------------------------------------------------------")
    
    
    def test_sum_der2coeffs(self):
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        itype_modify: int = nblist_info[5][0][self.center_idx_modify].item()
        jtype_modify: int = nblist_info[5][0][self.neigh_idx_modify].item()
        mu_modify: int = 0
        xi_modify: int = 1
        
        mtp_param_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        nmus: int = mtp_param_info[5].item()
        num_descriptor: int = mtp_param_info[2].size(0)
        coeff_idx_modify: int = (itype_modify*self.ntypes + jtype_modify)*nmus*self.chebyshev_size + mu_modify*self.chebyshev_size + xi_modify
        
        linear: nn.Module = nn.Linear(num_descriptor, 1).to(self.dtype)
        
        coeffs_tensor: torch.Tensor = torch.randn(self.ntypes*self.ntypes*nmus*self.chebyshev_size, dtype=self.dtype) / 10
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
                                nblist_info[0],
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        sum_descriptor: torch.Tensor = descriptor.sum()
        print(sum_descriptor.size())
        sum_descriptor.backward()
        
        coeffs_tensor_ = coeffs_tensor.clone().detach()
        coeffs_tensor_[coeff_idx_modify] += self.delta
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
                                nblist_info[0],
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        sum_descriptor_: torch.Tensor = descriptor_.sum()
        sum_descriptor_.backward()
        print("---------------------------------------------------------------------------------------")
        print("4.0. Sum descriptor = {0}".format(sum_descriptor.item()))
        print("4.1. Sum descriptor derivative w.r.t. coeffs[{0}][{1}][{2}][{3}] calculated by custom code = {4}".format(
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                coeffs_tensor.grad[coeff_idx_modify].item())
        )
        print("4.2. Sum descriptor derivative w.r.t. coeffs[{0}][{1}][{2}][{3}] calculated by finite difference method = {4}".format(
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                (sum_descriptor_ - sum_descriptor) / self.delta)
        )
        print("4.3. Difference between custom code and finite difference method = {0:.5f}".format(
            coeffs_tensor.grad[coeff_idx_modify].item() \
            - (sum_descriptor_ - sum_descriptor) / self.delta)
        )
        print("---------------------------------------------------------------------------------------")


    def test_etot_der2rcs(self):
        direction_idx_modify: int = 1
        
        mtp_parm_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        nmus: int = mtp_parm_info[5].item()
        num_descriptor: int = mtp_parm_info[2].size(0)
        
        linear: nn.Module = nn.Linear(num_descriptor, 1).to(self.dtype)
        
        coeffs_tensor: torch.Tensor = torch.randn(self.ntypes*self.ntypes*nmus*self.chebyshev_size, dtype=self.dtype) / 10
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
                                nblist_info[0],
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        etot = linear(descriptor)[0].sum()
        etot.backward()
        
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4][0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify] += self.delta
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
                                nblist_info[0],
                                nblist_info_[1],
                                nblist_info_[2],
                                nblist_info_[3],
                                nblist_info_[4],
                                nblist_info_[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        etot_ = linear(descriptor_)[0].sum()
        
        print("---------------------------------------------------------------------------------------")
        print("5.0. Etot = {0}".format(etot.item()))
        print("5.1. Etot derivative w.r.t. rcs[{0}][{1}][{2}] calculated by custom code = {3}".format(
                self.center_idx_modify,
                self.neigh_idx_modify,
                direction_idx_modify,
                nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item())
        )
        print("5.2. Etot derivative w.r.t. rcs[{0}][{1}][{2}] calculated by finite difference method = {3}".format(
                self.center_idx_modify,
                self.neigh_idx_modify,
                direction_idx_modify,
                (etot_ - etot) / self.delta)
        )
        print("5.3. Difference between custom code and finite difference method = {0:.5f}".format(
            nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item() \
            - (etot_ - etot) / self.delta)
        )
        print("---------------------------------------------------------------------------------------")
    


    def test_etot_der2coeffs(self):
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        itype_modify: int = nblist_info[5][0][self.center_idx_modify].item()
        jtype_modify: int = nblist_info[5][0][self.neigh_idx_modify].item()
        mu_modify: int = 0
        xi_modify: int = 1
        
        mtp_param_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        nmus: int = mtp_param_info[5].item()
        num_descriptor: int = mtp_param_info[2].size(0)
        coeff_idx_modify: int = (itype_modify*self.ntypes + jtype_modify)*nmus*self.chebyshev_size + mu_modify*self.chebyshev_size + xi_modify
        
        linear: nn.Module = nn.Linear(num_descriptor, 1).to(self.dtype)
        
        coeffs_tensor: torch.Tensor = torch.randn(self.ntypes*self.ntypes*nmus*self.chebyshev_size, dtype=self.dtype) / 10
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
                                nblist_info[0],
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        etot: torch.Tensor = linear(descriptor)[0].sum()
        etot.backward()
        
        coeffs_tensor_ = coeffs_tensor.clone().detach()
        coeffs_tensor_[coeff_idx_modify] += self.delta
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
                                nblist_info[0],
                                nblist_info[1],
                                nblist_info[2],
                                nblist_info[3],
                                nblist_info[4],
                                nblist_info[5],
                                self.umax_num_neigh,
                                self.rmax,
                                self.rmin)[0]
        etot_: torch.Tensor = linear(descriptor_)[0].sum()
        etot_.backward()
        print("---------------------------------------------------------------------------------------")
        print("6.0. Etot = {0}".format(etot.item()))
        print("6.1. Etot derivative w.r.t. coeffs[{0}][{1}][{2}][{3}] calculated by custom code = {4}".format(
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                coeffs_tensor.grad[coeff_idx_modify].item())
        )
        print("6.2. Etot derivative w.r.t. coeffs[{0}][{1}][{2}][{3}] calculated by finite difference method = {4}".format(
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                (etot_ - etot) / self.delta)
        )
        print("6.3. Difference between custom code and finite difference method = {0:.5f}".format(
            coeffs_tensor.grad[coeff_idx_modify].item() \
            - (etot_ - etot) / self.delta)
        )
        print("---------------------------------------------------------------------------------------")

    
if __name__ == "__main__":
    unittest.main()
    