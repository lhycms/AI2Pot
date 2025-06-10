import unittest
import os
from typing import List

from pymatgen.core import Structure
from ase import Atoms
from ase.io import read as ase_read

from ai2pot.utils import (MlffInput, MlffToLossInput)

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
        self.type_map: List[int] = [75, 41, 16, 34]
        self.atoms: Atoms = ase_read(ReNbSSe_POSCAR_PATH)
        
        self.mlff_input: MlffInput = MlffInput(type_map=self.type_map,
                                               rcut=self.rcut,
                                               umax_num_neighs=self.umax_num_neighs,
                                               pbc_xyz=self.pbc_xyz,
                                               sort=self.sort)
    
    def tearDown(self):
        print("MlffInputTest (TestCase) is tearing down...")
        
    def test_analyse_pymatgen(self):
        nblist_info4mlff = self.mlff_input.analyse_pymatgen(structure=self.structure)
        
        print("Information of structure for Potential:")
        print("\t1. binum_tensor.size() = ", nblist_info4mlff[0].size())
        print("\t2. bilist_tensor.size() = ", nblist_info4mlff[1].size())
        print("\t3. bnumneigh_tensor.size() = ", nblist_info4mlff[2].size())
        print("\t4. bfirstneigh_tensor.size() = ", nblist_info4mlff[3].size())
        print("\t5. brcs_tensor.size() = ", nblist_info4mlff[4].size())
        print("\t6. btypes.size() = ", nblist_info4mlff[5].size())
        print("\t7. bnghost.size() = ", nblist_info4mlff[6].size())
        
    def test_analyse_ase(self):
        nblist_info4mlff = self.mlff_input.analyse_ase(atoms=self.atoms)
        
        print("Information of structure for Potential:")
        print("\t1. binum_tensor.size() = ", nblist_info4mlff[0].size())
        print("\t2. bilist_tensor.size() = ", nblist_info4mlff[1].size())
        print("\t3. bnumneigh_tensor.size() = ", nblist_info4mlff[2].size())
        print("\t4. bfirstneigh_tensor.size() = ", nblist_info4mlff[3].size())
        print("\t5. brcs_tensor.size() = ", nblist_info4mlff[4].size())
        print("\t6. btypes.size() = ", nblist_info4mlff[5].size())
        print("\t7. bnghost.size() = ", nblist_info4mlff[6].size())
        


class MlffToLossInputTest(unittest.TestCase):
    def setUp(self):
        print("MlffToLossInputTest (TestCase) is setting up...\n")
        self.type_map: List[int] = [75, 41, 16, 34]
        self.rcut: float = 5.0
        self.umax_num_neighs: int = 200
        self.pbc_xyz: List[bool] = [True, True, True]
        self.sort: bool = True
        self.structure: Structure = Structure.from_file(ReNbSSe_POSCAR_PATH)
        self.e_weight: float = 1.0
        self.f_weight: float = 1.0
        self.v_weight: float = 0.1
        
        self.mlff_to_loss_input: MlffToLossInput = MlffToLossInput(type_map=self.type_map,
                                                                   rcut=self.rcut,
                                                                   umax_num_neighs=self.umax_num_neighs,
                                                                   pbc_xyz=self.pbc_xyz,
                                                                   sort=self.sort)
    
    def tearDown(self):
        print("MlffToLossInputTest (TestCase) is tearing down...\n")
        
    def test_analyse_pymatgen(self):
        input_info = self.mlff_to_loss_input.analyse_pymatgen(structure=self.structure,
                                                              e_weight=self.e_weight,
                                                              f_weight=self.f_weight,
                                                              v_weight=self.v_weight)
        print("Info:")
        print("\t1. e_weight = ", input_info[0])
        print("\t2. f_weight = ", input_info[1])
        print("\t3. v_weight = ", input_info[2])
        print("\t4. betot_dft_tensor.size() = ", input_info[3].size())
        print("\t5. bforce_dft_tensor.size() = ", input_info[4].size())
        print("\t6. bvirial_dft_tensor.size() = ", input_info[5].size())
        print("\t7. binum_tensor.size() = ", input_info[6].size())
        print("\t8. bilist_tensor.size() = ", input_info[7].size())
        print("\t9. bnumneigh_tensor.size() = ", input_info[8].size())
        print("\t10. bfirstneigh_tensor.size() = ", input_info[9].size())
        print("\t11. brcs_tensor.size() = ", input_info[10].size())
        print("\t12. btypes.size() = ", input_info[11].size())
        print("\t13. bnghost_tensor.size() = ", input_info[12].size())
        print(input_info[11])


if __name__ == "__main__":
    unittest.main()
    