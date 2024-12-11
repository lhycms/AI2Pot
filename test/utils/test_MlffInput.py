import unittest
import os
from typing import List
from pymatgen.core import Structure

from ai2pot.utils import MlffInput

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
        nblist_info4mlff = mlff_input.analyse_pymatgen(structure=self.structure)
        print(nblist_info4mlff[0].size())
        print(nblist_info4mlff[1].size())
        print(nblist_info4mlff[2].size())
        print(nblist_info4mlff[3].size())
        print(nblist_info4mlff[4].size())
        print(nblist_info4mlff[5].size())
        print(nblist_info4mlff[6].size())


if __name__ == "__main__":
    unittest.main()
    