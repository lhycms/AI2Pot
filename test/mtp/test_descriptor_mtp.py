import unittest
import os
from typing import List

import torch
from pymatgen.core import Structure
from ai2pot.mtp import DescriptorMtp
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
        self.umax_num_neigh: int = 40
        self.dtype = torch.float64
        self.device = torch.device("cpu")
        
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
    
    
    def tearDown(self):
        print("MtpDescriptorDerTest (TestCase) is tearing down...\n")
    
    
    def test_der2xyz(self):
        center_idx_modify: int = 4
        neigh_idx_modify: int = 7
        direction_idx_modify: int = 2
        delta: float = 1e-5
        
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        nblist_info[4].requires_grad_(True)
        descriptor = self.descriptor_mtp(*nblist_info[1:-1])
        result: torch.Tensor = descriptor.sum()
        result.backward()
        print(result)
        
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4][0][center_idx_modify][neigh_idx_modify][direction_idx_modify] += delta
        nblist_info_[4].requires_grad_(True)
        descriptor_ = self.descriptor_mtp(*nblist_info_[1:-1])
        result_: torch.Tensor = descriptor_.sum()
        result_.backward()
        print(result_)
        print(nblist_info[4].grad[0][center_idx_modify][neigh_idx_modify][direction_idx_modify])
        print((result_ - result) / delta)
    
    def test_der2coeffs(self):
        pass


if __name__ == "__main__":
    unittest.main()