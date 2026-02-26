import os
import unittest
import numpy as np
from pymatgen.core import Structure

from ai2pot.fromcc import nblist


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_DIR = os.path.join(TEST_FILES_DIR, "POSCARs")

class NblistTest(unittest.TestCase):
    def setUp(self) -> None:
        print("NblistTest is setting up...\n")
        #
        structure_file: str = os.path.join(ReNbSSe_POSCAR_DIR, "POSCAR")
        self.structure: Structure = Structure.from_file("/data/home/liuhanyu/mycode/AI2Pot/test/test_data/POSCARs/C/POSCAR")
        # /data/home/liuhanyu/mycode/AI2Pot/test/test_data/POSCARs/C/POSCAR
        #structure_file
        self.cell: np.ndarray = self.structure.lattice.matrix
        self.atom_types: np.ndarray = [str(ii) for ii in self.structure.species]
        element_map = {'C': 0}#{'Re': 0, 'Nb': 1, 'S': 2, 'Se': 3}
        self.atom_types = np.array([element_map[el] for el in self.atom_types])
        self.coords: np.ndarray = self.structure.frac_coords
        self.rcut: float = 7.0
        self.umax_num_neigh_atoms: int = 200
        self.is_cart_coord: bool = False
        self.pbc_xyz = [True, True, True]
        self.sort: bool = True
    
    def test_find_info4lmff(self):
        info = nblist.find_info4mlff(self.cell,
                                     self.atom_types,
                                     self.coords,
                                     self.rcut,
                                     self.umax_num_neigh_atoms,
                                     self.is_cart_coord,
                                     self.pbc_xyz,
                                     self.sort)
        print(info[4])
    
    def tearDown(self) -> None:
        print("NblistTest is tearing down...\n")


if __name__ == "__main__":
    unittest.main()
