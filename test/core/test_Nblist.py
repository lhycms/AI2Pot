import os
import unittest
from typing import List
from pymatgen.core import Structure

from ai2pot.core import Nblist


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
print(TEST_FILES_DIR)

class NblistTest(unittest.TestCase):
    def setUp(self) -> None:
        print("Nblist is setting up...\n")
        self.structure: Structure = Structure.from_file(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))
        self.rcut: float = 3.2
        self.umax_num_neigh_atoms: int = 6
        self.pbc_xyz: List[bool] = [True, True, True]
        self.sort: bool = True
    
    def test_init(self) -> None:
        nblist: Nblist = Nblist(structure=self.structure,
                                rcut=self.rcut,
                                umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                pbc_xyz=self.pbc_xyz,
                                sort=self.sort)
        assert(nblist.distances.shape[0] == 108)
        assert(nblist.distances.shape[1] == self.umax_num_neigh_atoms)
        print(nblist.distances)
    
    def test_find_info4mlff(self) -> None:
        pass
    
    def tearDown(self) -> None:
        print("Nblist is tearing down...\n")
        
    
if __name__ == "__main__":
    unittest.main()
    