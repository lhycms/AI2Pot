import unittest
import os
from typing import List
from pymatgen.core import Structure
import torch
import torch.nn as nn

from ai2pot.utils import MlffInput
from ai2pot.mtp import NNMtp

TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))


class MlffInputTest(unittest.TestCase):
    def setUp(self):
        print("MlffInputTest (TestCase) is setting up...")
        self.rcut: float = 5.0
        self.umax_num_neighs: int = 200
        self.pbc_xyz: List[int] = [True, True, True]
        self.sort: bool = True
        self.structure: Structure = Structure.from_file(ReNbSSe_POSCAR_PATH)
    
    def tearDown(self):
        print("MlffInputTest (TestCase) is tearing down...")
        
    def test_find_struct_info(self):
        mlff_input: MlffInput = MlffInput(rcut=self.rcut,
                                          umax_num_neighs=self.umax_num_neighs,
                                          pbc_xyz=self.pbc_xyz,
                                          sort=self.sort)
        nblist_info4mlff = mlff_input.find_struct_info(structure=self.structure)
        print(nblist_info4mlff[0].size())
        print(nblist_info4mlff[1].size())
        print(nblist_info4mlff[2].size())
        print(nblist_info4mlff[3].size())
        print(nblist_info4mlff[4].size())
        print(nblist_info4mlff[5].size())
        print(nblist_info4mlff[6].size())
        
    def test_vertify_mtp_force(self):
        center_idx_modify: int = 0
        neigh_idx_modify: int = 0
        direction_idx_modify: int = 0
        
        mtp_level: int = 16
        ntypes: int = 4
        chebyshev_size: int = 8
        rmax: float = 5.0
        rmin: float = 2.0
        fit_sizes_list: List[int] = [30]
        fit_activation: nn.Module = nn.Tanh()
        
        nn_mtp = NNMtp(mtp_level=mtp_level,
                       ntypes=ntypes,
                       chebyshev_size=chebyshev_size,
                       rmax=rmax,
                       rmin=rmin,
                       umax_num_neighs=self.umax_num_neighs,
                       fit_sizes_list=fit_sizes_list,
                       fit_activation=fit_activation,
                       bias_mark=False,
                       energy_shift_tensor=False,
                       has_virials=False).to(torch.float64)
        
        # 1. Structure_1
        info = MlffInput(rcut=self.rcut,
                umax_num_neighs=self.umax_num_neighs,
                pbc_xyz=self.pbc_xyz,
                sort=self.sort,
                dtype=torch.float64).find_struct_info(self.structure,)
        info[4].requires_grad_(True)
        d1 = nn_mtp.descriptor_module(*info[1:-1])
        d1.sum().backward()
        print("---- Information for structure 1 ----")
        print("1.1. Descriptor for structure 1 = \n\t", d1[0][center_idx_modify][:5])
        print("1.2. Derivatives w.r.t. xyz = ", info[4].grad[0][center_idx_modify][neigh_idx_modify][direction_idx_modify])
        print("-------------------------------------")
        
        # 2. Structure_2
        delta = 1E-4
        info[4].requires_grad_(False)
        info[4][center_idx_modify][neigh_idx_modify][direction_idx_modify] += delta
        info[4].requires_grad_(True)
        d2 = nn_mtp.descriptor_module(*info[1:-1])
        print("---- Information for structure 2 ----")
        print("2.1. Descriptor for structure 1 = \n\t", d2[0][center_idx_modify][:5])
        print("2.2. Derivatives w.r.t. xyz = ", (d2.sum() - d1.sum()) / delta)
        print("-------------------------------------")


if __name__ == "__main__":
    unittest.main()
    