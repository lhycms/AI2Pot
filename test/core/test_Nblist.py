import os
import unittest
from typing import List, Dict
import numpy as np
from pymatgen.core import Structure

from ai2pot.core import Nblist


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
print(TEST_FILES_DIR)

class NblistTest(unittest.TestCase):
    def setUp(self) -> None:
        print("NblistTest is setting up...\n")
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
        print(nblist.types)
    
    def test_find_info4mlff(self) -> None:
        cell: np.ndarray = self.structure.lattice.matrix.astype(np.float32)
        unique_species = sorted(set(self.structure.species), key=lambda sp: sp.Z)
        type_map: Dict[str, int] = {element: idx for idx, element in enumerate(unique_species)}
        types: np.ndarray = np.array([type_map[el] for el in self.structure.species])
        coords: np.ndarray = self.structure.frac_coords.astype(np.float32)
        result = Nblist.find_info4mlff(cell,
                              types,
                              coords,
                              self.rcut,
                              self.umax_num_neigh_atoms,
                              False,
                              [True, True, True],
                              True)
        #print(np.linalg.norm(result[4], axis=2))
    
    def tearDown(self) -> None:
        print("NblistTest is tearing down...\n")
        
    
if __name__ == "__main__":
    unittest.main()
    