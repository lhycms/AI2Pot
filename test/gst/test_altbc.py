import os
import unittest
from pymatgen.core import Structure

from ai2pot.core import Nblist
from ai2pot.gst import Altbc


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
print(TEST_FILES_DIR)

class AltbcTest(unittest.TestCase):
    def setUp(self) -> None:
        print("AltbcTest is setting up...\n")
        self.structure: Structure = Structure.from_file(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))
        self.rcut: float = 3.5
        self.umax_num_neigh_atoms: int = 100
        self.angle_threshold: float = 120.0
    
    def test_analyse_structure(self):        
        altbc: Altbc = Altbc(self.rcut, self.umax_num_neigh_atoms, self.angle_threshold)
        print( altbc.analyse_structure(self.structure) )
        
    def test_find_long_short_bonds(self):
        pass
        #Altbc.find_long_short_bonds()
    
    def tearDown(self) -> None:
        print("AltbcTest is tearing down...\n")
        

if __name__ == "__main__":
    unittest.main()