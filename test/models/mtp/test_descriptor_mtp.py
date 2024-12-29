import unittest
import os
from typing import List

import torch
import torch.nn as nn
from pymatgen.core import Structure
from ai2pot.models.mtp import DescriptorMtp
from ai2pot.utils.usepot import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")


class MtpDescriptorDerTest(unittest.TestCase):
    def setUp(self):
        print("MtpDescriptorDerTest (TestCase) is setting up...\n")
        
        self.mtp_level: int = 16
        self.ntypes: int = 4
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 2.0
        self.umax_num_neigh: int = 100
        self.dtype = torch.float64
        self.device = torch.device("cpu")
        
        self.center_idx_modify: int = 0
        self.neigh_idx_modify: int = 6
        self.descriptor_idx_modify: int = 40
        self.delta: float = 1e-5
        
        self.mlff_input: MlffInput = MlffInput(rcut=self.rmax,
                                               umax_num_neighs=self.umax_num_neigh,
                                               pbc_xyz=[True, True, True],
                                               sort=False,
                                               dtype=self.dtype,
                                               device=self.device)
        self.structure = Structure.from_file(ReNbSSe_POSCAR)
        self.descriptor_mtp = DescriptorMtp(mtp_level=self.mtp_level,
                                            ntypes=self.ntypes,
                                            chebyshev_size=self.chebyshev_size,
                                            rmax=self.rmax,
                                            rmin=self.rmin,
                                            umax_num_neighs=self.umax_num_neigh).to(self.dtype)
        self.nmus = self.descriptor_mtp.nmus

    
    def tearDown(self):
        print("MtpDescriptorDerTest (TestCase) is tearing down...\n")
    
    
    def test_mtp_basis_der2xyz(self):
        direction_idx_modify: int = 1
        
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        nblist_info[4].requires_grad_(True)
        descriptor = self.descriptor_mtp(*nblist_info[:-1])
        single_descriptor = descriptor[0][self.center_idx_modify][self.descriptor_idx_modify]
        single_descriptor.backward()
        
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4][0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify] += self.delta
        nblist_info_[4].requires_grad_(True)
        descriptor_ = self.descriptor_mtp(*nblist_info_[:-1])
        single_descriptor_ = descriptor_[0][self.center_idx_modify][self.descriptor_idx_modify]
        
        print("---------------------------------------------------------------------------------------")
        print("1.0. Descriptor[0, {0}, {1}] = {2:.5f}".format(self.center_idx_modify, self.descriptor_idx_modify, single_descriptor.item()))
        print("1.1. Descriptor[0, {0}, {1}] derivative w.r.t. rcs[{2}][{3}][{4}] calculated by custom code = {5:.6f}".format(
                self.center_idx_modify,
                self.descriptor_idx_modify,
                self.center_idx_modify, 
                self.neigh_idx_modify, 
                direction_idx_modify,
                nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item())
        )
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
        coeff_idx_modify: int = (itype_modify*self.ntypes + jtype_modify)*self.nmus*self.chebyshev_size + mu_modify*self.chebyshev_size + xi_modify
        
        descriptor = self.descriptor_mtp(*nblist_info[:-1])
        single_descriptor: torch.Tensor = descriptor[0, self.center_idx_modify, self.descriptor_idx_modify]
        single_descriptor.backward()
        
        coeffs_tensor_ = self.descriptor_mtp.coeffs.clone().detach()
        coeffs_tensor_[coeff_idx_modify] += self.delta
        coeffs_tensor_.requires_grad_(True)
        self.descriptor_mtp_ = DescriptorMtp(mtp_level=self.mtp_level,
                                             ntypes=self.ntypes,
                                             chebyshev_size=self.chebyshev_size,
                                             rmax=self.rmax,
                                             rmin=self.rmin,
                                             umax_num_neighs=self.umax_num_neigh).to(self.dtype)
        self.descriptor_mtp_.register_parameter("coeffs", nn.Parameter(coeffs_tensor_, requires_grad=True))
        descriptor_ = self.descriptor_mtp_(*nblist_info[:-1])
        single_descriptor_: torch.Tensor = descriptor_[0, self.center_idx_modify, self.descriptor_idx_modify]
        print("---------------------------------------------------------------------------------------")
        print("2.0. Descriptor[0, {0}, {1}] = {2}".format(self.center_idx_modify, self.descriptor_idx_modify, single_descriptor.item()))
        print("2.1. Descriptors[0, {0}, {1}] detivative w.r.t coeffs[{2}][{3}][{4}][{5}] calculated by custom code = {6:.6f}".format(
                self.center_idx_modify,
                self.descriptor_idx_modify,
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                self.descriptor_mtp.coeffs.grad[coeff_idx_modify].item()))
        print("2.2. Descriptors[0, {0}, {1}] detivative w.r.t coeffs[{2}][{3}][{4}][{5}] calculated by finite difference method = {6:.6f}".format(
                self.center_idx_modify,
                self.descriptor_idx_modify,
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                ((single_descriptor_ - single_descriptor) / self.delta).item()))
        print("2.3. Difference between custom code and finite difference method = {0:.5f}".format(
            self.descriptor_mtp.coeffs.grad[coeff_idx_modify].item() \
            - ((single_descriptor_ - single_descriptor) / self.delta).item())
        )
        print("---------------------------------------------------------------------------------------")
        
    
    def test_sum_der2xyz(self):
        direction_idx_modify: int = 1
        
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        nblist_info[4].requires_grad_(True)
        descriptor = self.descriptor_mtp(*nblist_info[:-1])
        sum_descriptor: torch.Tensor = descriptor.sum()
        sum_descriptor.backward()
        
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4][0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify] += self.delta
        nblist_info_[4].requires_grad_(True)
        descriptor_ = self.descriptor_mtp(*nblist_info_[:-1])
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
        coeff_idx_modify: int = (itype_modify*self.ntypes + jtype_modify)*self.nmus*self.chebyshev_size + mu_modify*self.chebyshev_size + xi_modify
        
        descriptor = self.descriptor_mtp(*nblist_info[:-1])
        sum_descriptor: torch.Tensor = descriptor.sum()
        sum_descriptor.backward()
        
        coeffs_tensor_ = self.descriptor_mtp.coeffs.clone().detach()
        coeffs_tensor_[coeff_idx_modify] += self.delta
        coeffs_tensor_.requires_grad_(True)
        self.descriptor_mtp_ = DescriptorMtp(mtp_level=self.mtp_level,
                                             ntypes=self.ntypes,
                                             chebyshev_size=self.chebyshev_size,
                                             rmax=self.rmax,
                                             rmin=self.rmin,
                                             umax_num_neighs=self.umax_num_neigh).to(self.dtype)
        self.descriptor_mtp_.register_parameter("coeffs", nn.Parameter(coeffs_tensor_, requires_grad=True))
        descriptor_ = self.descriptor_mtp_(*nblist_info[:-1])
        sum_descriptor_: torch.Tensor = descriptor_.sum()
        
        print("---------------------------------------------------------------------------------------")
        print("4.0. Sum descriptor = {0}".format(sum_descriptor.item()))
        print("4.1. Sum descriptor derivative w.r.t. coeffs[{0}][{1}][{2}][{3}] calculated by custom code = {4}".format(
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                self.descriptor_mtp.coeffs.grad[coeff_idx_modify].item())
        )
        print("4.2. Sum descriptor derivative w.r.t. coeffs[{0}][{1}][{2}][{3}] calculated by finite difference method = {4}".format(
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                (sum_descriptor_ - sum_descriptor) / self.delta)
        )
        print("4.3. Difference between custom code and finite difference method = {0:.5f}".format(
            self.descriptor_mtp.coeffs.grad[coeff_idx_modify].item() \
            - ((sum_descriptor_ - sum_descriptor) / self.delta).item())
        )
        print("---------------------------------------------------------------------------------------")
    
    
    def test_etot_der2xyz(self):
        direction_idx_modify: int = 1
        num_descriptor: int = self.descriptor_mtp.num_descriptors
        linear: nn.Module = nn.Sequential(
            nn.Linear(num_descriptor, 80),
            nn.Tanh(),
            nn.Linear(80, 1)
        ).to(self.dtype)
        
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        nblist_info[4].requires_grad_(True)
        descriptor = self.descriptor_mtp(*nblist_info[:-1])
        etot = linear(descriptor)[0].sum()
        etot.backward()
        
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4][0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify] += self.delta
        nblist_info_[4].requires_grad_(True)
        descriptor_ = self.descriptor_mtp(*nblist_info_[:-1])
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
        coeff_idx_modify: int = (itype_modify*self.ntypes + jtype_modify)*self.nmus*self.chebyshev_size + mu_modify*self.chebyshev_size + xi_modify
        
        num_descriptor: int = self.descriptor_mtp.num_descriptors
        linear: nn.Module = nn.Sequential(
            nn.Linear(num_descriptor, 80),
            nn.Tanh(),
            nn.Linear(80, 1)
        ).to(self.dtype)
                
        descriptor = self.descriptor_mtp(*nblist_info[:-1])
        etot = linear(descriptor)[0].sum()
        etot.backward()
        
        coeffs_tensor_ = self.descriptor_mtp.coeffs.clone().detach()
        coeffs_tensor_[coeff_idx_modify] += self.delta
        coeffs_tensor_.requires_grad_(True)
        self.descriptor_mtp_ = DescriptorMtp(mtp_level=self.mtp_level,
                                             ntypes=self.ntypes,
                                             chebyshev_size=self.chebyshev_size,
                                             rmax=self.rmax,
                                             rmin=self.rmin,
                                             umax_num_neighs=self.umax_num_neigh).to(self.dtype)
        self.descriptor_mtp_.register_parameter("coeffs", nn.Parameter(coeffs_tensor_, requires_grad=True))
        descriptor_ = self.descriptor_mtp_(*nblist_info[:-1])
        etot_ = linear(descriptor_)[0].sum()
        
        print("---------------------------------------------------------------------------------------")
        print("6.0. Etot = {0}".format(etot.item()))
        print("6.1. Etot derivative w.r.t. coeffs[{0}][{1}][{2}][{3}] calculated by custom code = {4}".format(
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                self.descriptor_mtp.coeffs.grad[coeff_idx_modify].item())
        )
        print("6.2. Etot derivative w.r.t. coeffs[{0}][{1}][{2}][{3}] calculated by finite difference method = {4}".format(
                itype_modify, 
                jtype_modify, 
                mu_modify, 
                xi_modify,
                (etot_ - etot) / self.delta)
        )
        print("6.3. Difference between custom code and finite difference method = {0:.5f}".format(
            self.descriptor_mtp.coeffs.grad[coeff_idx_modify].item() \
            - ((etot_ - etot) / self.delta).item())
        )
        print("---------------------------------------------------------------------------------------")
    

if __name__ == "__main__":
    unittest.main()
    