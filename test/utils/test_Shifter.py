from typing import List
import os
import unittest
import torch

from ai2pot.data.mlffdataset import ExtxyzDataset
from ai2pot.utils.prepot import (ExtxyzShifter,
                                 MlffDatasetShifter)


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_OUTCAR_PATH = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe", "OUTCAR")
ReNbSTe_OUTCAR_PATH = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSTe", "OUTCAR")


class ExtxyzShifterTest(unittest.TestCase):
    def setUp(self):
        print("ExtxyzShifterTest (TestCase) is setting up...")
        filename: str = os.path.join(TEST_FILES_DIR, 
                                     "XYZ", 
                                     "11_NEP_potential_PbTe",
                                     "train_m.xyz")
        self.extxyz_shifter: ExtxyzShifter = ExtxyzShifter(extxyz_path=filename)
    
    def tearDown(self):
        pass
        #print("ExtxyzShifterTest (TestCase) is tearing down...")
    
    def test_get_energy_shifts(self):
        print(self.extxyz_shifter)

    
class MlffDatasetShifterTest(unittest.TestCase):
    def setUp(self):
        print("MlffDatasetShifter (TestCase) is setting up...")
        filename: str = os.path.join(TEST_FILES_DIR, 
                                     "XYZ", 
                                     "11_NEP_potential_PbTe",
                                     "train_m.xyz")
        rcut: float = 5.0
        umax_num_neigh_atoms: int = 200
        pbc_xyz: List[bool] = [True, True, True]
        sort = True
        torch_float_dtype: torch._C.dtype = torch.float64
        self.extxyz_dataset: ExtxyzDataset = ExtxyzDataset(filename=filename,
                                                            rcut=rcut,
                                                            umax_num_neigh_atoms=umax_num_neigh_atoms,
                                                            pbc_xyz=pbc_xyz,
                                                            sort=sort,
                                                            torch_float_dtype=torch_float_dtype,
                                                            has_virial=False)
        self.mlffdataset_shifter: MlffDatasetShifter = MlffDatasetShifter(extxyz_dataset=self.extxyz_dataset)
    
    def tearDown(self):
        print("MlffDatasetShifter (TestCase) is tearing down...")
    

    def test_get_energy_shifts(self):
        shifter = self.mlffdataset_shifter
        print(shifter)

        

if __name__ == "__main__":
    unittest.main()
    