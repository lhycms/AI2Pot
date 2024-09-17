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
        unique_species = sorted( set(self.structure.species), key=lambda sp: sp.Z )
        type_map = {element: idx for idx, element in enumerate(unique_species)}
        self.types: np.ndarray = np.array([type_map[el] for el in (self.structure.species)])
        self.coords: np.ndarray = self.structure.frac_coords
        self.rcut: float = 3.5
        self.umax_num_neigh_atoms: int = 100
        
    def test_find_order_parameters(self):
        nblist_info = Nblist.find_info4mlff(self.cell,
                                            self.types,
                                            self.coords,
                                            self.rcut,
                                            self.umax_num_neigh_atoms,
                                            False,
                                            [True, True, True],
                                            sort=True)
        result: np.ndarray = OrderParameter.find_order_parameters(nblist_info[0],
                                                                  nblist_info[1],
                                                                  nblist_info[2],
                                                                  nblist_info[3],
                                                                  nblist_info[4],
                                                                  nblist_info[5],
                                                                  nblist_info[6],
                                                                  self.umax_num_neigh_atoms)
        print(result)
    
    def test_analyse_structure(self):
        order_parameter = OrderParameter(rcut=self.rcut, umax_num_neigh_atoms=self.umax_num_neigh_atoms)
        result: np.ndarray = order_parameter.analyse_structure(structure=self.structure)
        print(result)
        
    
    def tearDown(self):
        print("OrderParameterTest is tearing down...")


if __name__ == "__main__":
    unittest.main()
