import unittest
import os
from typing import List

import torch
import torch.nn as nn
from pymatgen.core import Structure

from ai2pot.fromcc import forceSrOp
from ai2pot.models.mtp import DescriptorMtp
from ai2pot.utils.testpot_utils import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")
MoS2_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "MoS2", "POSCAR")


class ForceSrOpTest(unittest.TestCase):
    def setUp(self):
        print("ForceSrOp (TestCase) is setting up...")
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
        
        self.structure = Structure.from_file(ReNbSSe_POSCAR)
        self.mlff_input = MlffInput(rcut=self.rmax,
                                    umax_num_neighs=self.umax_num_neigh,
                                    pbc_xyz=[True, True, True],
                                    sort=False,
                                    dtype=self.dtype,
                                    device=self.device)

        self.descriptor_mtp = DescriptorMtp(mtp_level=self.mtp_level,
                                            ntypes=self.ntypes,
                                            chebyshev_size=self.chebyshev_size,
                                            rmax=self.rmax,
                                            rmin=self.rmin,
                                            umax_num_neighs=self.umax_num_neigh).to(self.dtype)
        self.nmus = self.descriptor_mtp.nmus
    
    
    def tearDown(self):
        print("ForceSrOp (TestCase) is tearing down...")
        
        
    def est_der2jacobian(self):
        direction_idx_modify: int = 2
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        cidx: int = nblist_info[1][0][self.center_idx_modify].item()
        nidx: int = nblist_info[3][0][self.center_idx_modify][self.neigh_idx_modify].item()
        
        dei_drij: torch.Tensor = torch.randn(1, len(self.structure), self.umax_num_neigh, 3, dtype=self.dtype, device=self.device)
        dei_drij.requires_grad_(True)
        fi: torch.Tensor = forceSrOp(nblist_info[0],
                                     nblist_info[1],
                                     nblist_info[2],
                                     nblist_info[3],
                                     nblist_info[6],
                                     self.umax_num_neigh,
                                     dei_drij)[0]
        
        # Case 1: $$ = 1
        #fi[0][cidx][direction_idx_modify].backward(retain_graph=True)
        #print(dei_drij.grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item())
        #self.assertEqual( dei_drij.grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item(), 1.0 )
        
        # Case 2: $$ = -1
        fi[0][nidx][direction_idx_modify].backward()
        print(dei_drij.grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item())
        self.assertEqual( dei_drij.grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item(), -1.0 )


    def test_der2rcs(self):
        direction_idx_modify: int = 1
        num_descriptor: int = self.descriptor_mtp.num_descriptors
        linear: nn.Module = nn.Sequential(
            nn.Linear(num_descriptor, 80),
            nn.Tanh(),
            nn.Linear(80, 1)
        ).to(self.dtype)
        
        # Case 1:
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        nblist_info[4].requires_grad_(True)
        descriptor = self.descriptor_mtp(*nblist_info[:-1])
        etot = linear(descriptor).sum()
        
        detot_rij_jacobian: torch.Tensor = torch.autograd.grad(outputs=[descriptor],
                                                            inputs=[nblist_info[4]],
                                                            grad_outputs=[torch.ones_like(descriptor)],
                                                            retain_graph=True,
                                                            create_graph=True)[0]
        detot_rij_jacobian.sum().backward()
              
        print(detot_rij_jacobian.grad_fn)
        print(descriptor.grad_fn)
        #print(nblist_info[4].grad)
        

if __name__ == "__main__":
    unittest.main()
    