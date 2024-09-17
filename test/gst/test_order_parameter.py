import os
import unittest
from typing import List
import numpy as np
from pymatgen.core import Structure

from ai2pot.core import Nblist
from ai2pot.gst import OrderParameter

TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")


class OrderParameterTest(unittest.TestCase):
    def setUp(self):
        print("OrderParameterTest is setting up...")
        self.structure = Structure.from_file(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))
        
        self.cell: np.ndarray = self.structure.lattice.matrix
        
        #self.types: np.ndarray = 
        
    def test_find_order_parameters(self):
        pass
    
    def test_analyse_structure(self):
        pass
    
    def tearDown(self):
        print("OrderParameterTest is tearing down...")


if __name__ == "__main__":
    unittest.main()
