import os
import unittest
from typing import List
import numpy as np
from pymatgen.core import Structure
from dpdata import LabeledSystem

from ai2pot.core import Nblist
from ai2pot.gst import Altbc


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")

class AltbcTest(unittest.TestCase):
    def setUp(self) -> None:
        print("AltbcTest is setting up...\n")
        self.structure: Structure = Structure.from_file(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))
        self.rcut: float = 3.5
        self.umax_num_neigh_atoms: int = 100
        self.angle_threshold: float = 120.0
        
        self.cell: np.ndarray = self.structure.lattice.matrix
        unique_species = sorted(set(self.structure.species), key=lambda sp: sp.Z)
        type_map = {element: idx for idx, element in enumerate(unique_species)}
        self.types: np.ndarray = np.array([type_map[el] for el in self.structure.species])
        self.coords: np.ndarray = self.structure.frac_coords
        
        self.labeled_system: LabeledSystem = LabeledSystem(os.path.join(TEST_FILES_DIR, "OUTCARs", "OUTCAR"))
        
    
    def test_analyse_structure(self):        
        altbc: Altbc = Altbc(self.rcut, self.umax_num_neigh_atoms, self.angle_threshold)
        result: np.ndarray = altbc.analyse_structure(self.structure)
        #print(result)
        
    def test_find_long_short_bonds(self):
        nblist_info = Nblist.find_info4mlff(self.cell,
                                            self.types,
                                            self.coords,
                                            self.rcut,
                                            self.umax_num_neigh_atoms,
                                            False,
                                            [True, True, True],
                                            False)
        result: np.ndarray = Altbc.find_long_short_bonds(nblist_info[0],
                                                         nblist_info[1],
                                                         nblist_info[2],
                                                         nblist_info[3],
                                                         nblist_info[4],
                                                         nblist_info[5],
                                                         nblist_info[6],
                                                         self.umax_num_neigh_atoms,
                                                         self.angle_threshold)
        #print(result)
    
    def test_analyse_labeled_system(self):
        altbc: Altbc = Altbc(self.rcut, self.umax_num_neigh_atoms, self.angle_threshold)
        result: np.ndarray = altbc.analyse_labeled_system(self.labeled_system)
        print(result)

    def test_find_batch_long_short_bonds(self):
        binum: List[int] = []
        bilist: List[np.ndarray] = []
        bnumneigh: List[np.ndarray] = []
        bfirstneigh: List[np.ndarray] = []
        brcs: List[np.ndarray] = []
        btypes: List[np.ndarray] = []
        bnghost: List[int] = []
        for ii in range(len(self.labeled_system)):
            nblist_info = Nblist.find_info4mlff(self.labeled_system["cells"][ii],
                                                self.labeled_system["atom_types"].astype(np.int32),
                                                self.labeled_system["coords"][ii],
                                                self.rcut,
                                                self.umax_num_neigh_atoms,
                                                True,
                                                [True, True, True],
                                                False)
            binum.append(nblist_info[0])
            bilist.append(nblist_info[1])
            bnumneigh.append(nblist_info[2])
            bfirstneigh.append(nblist_info[3])
            brcs.append(nblist_info[4])
            btypes.append(nblist_info[5])
            bnghost.append(nblist_info[6])
        result = Altbc.find_batch_long_short_bonds(np.array(binum).astype(np.int32),
                                                   np.array(bilist),
                                                   np.array(bnumneigh),
                                                   np.array(bfirstneigh),
                                                   np.array(brcs),
                                                   np.array(btypes),
                                                   np.array(bnghost),
                                                   self.umax_num_neigh_atoms,
                                                   self.angle_threshold)
        #print(result)

    
    def tearDown(self) -> None:
        print("AltbcTest is tearing down...\n")
        

if __name__ == "__main__":
    unittest.main()