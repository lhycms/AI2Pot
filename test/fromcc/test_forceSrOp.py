import unittest
import os
from typing import List

import torch
import torch.nn as nn
from pymatgen.core import Structure
from ai2pot.fromcc import forceSrOp
from ai2pot.utils.usepot import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")
MoS2_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "MoS2", "POSCAR")


class ForceSrOpTest(unittest.TestCase):
    def setUp(self):
        print("ForceSrOp (TestCase) is setting up...")
        self.rmax: float = 5.0
        self.umax_num_neigh: int = 30
        self.dtype: torch._C.device = torch.float64
        self.device: torch._C.device = torch.device("cpu")
    
        self.center_idx_modify: int = 0
        self.neigh_idx_modify: int = 3
        self.descriptor_idx_modify: int = 40
        self.delta: float = 1e-5
        
        self.structure = Structure.from_file(ReNbSSe_POSCAR)
        self.mlff_input = MlffInput(rcut=self.rmax,
                                    umax_num_neighs=self.umax_num_neigh,
                                    pbc_xyz=[True, True, True],
                                    sort=False,
                                    dtype=self.dtype,
                                    device=self.device)
    
    
    def tearDown(self):
        print("ForceSrOp (TestCase) is tearing down...")
        
        
    def test_der2jacobian(self):
        direction_idx_modify: int = 0
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        cidx: int = nblist_info[1][0][self.center_idx_modify].item()
        nidx: int = nblist_info[1][0][self.neigh_idx_modify].item()
        
        dei_drij: torch.Tensor = torch.randn(1, len(self.structure), self.umax_num_neigh, 3, dtype=self.dtype, device=self.device)
        dei_drij.requires_grad_(True)
        fi: torch.Tensor = forceSrOp(nblist_info[0],
                                     nblist_info[1],
                                     nblist_info[2],
                                     nblist_info[3],
                                     nblist_info[6],
                                     self.umax_num_neigh,
                                     dei_drij)[0]
        #fi[0][cidx][direction_idx_modify].backward()
        #print(dei_drij.grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify])
        
        fi[0][nidx][direction_idx_modify].backward()
        print(dei_drij.grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify])

    
    
    def est_der2rcs(self):
        pass


if __name__ == "__main__":
    unittest.main()
    