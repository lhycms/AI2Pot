import os
import unittest
from dpdata import LabeledSystem, MultiSystems

from ai2pot.utils.prepot import LsShifter, MsShifter


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_OUTCAR_PATH = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe", "OUTCAR")
ReNbSTe_OUTCAR_PATH = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSTe", "OUTCAR")


class LsShifterTest(unittest.TestCase):
    def setUp(self):
        print("LsShifterTest (TestCase) is setting up...")
        self.ls: LabeledSystem = LabeledSystem(ReNbSSe_OUTCAR_PATH)
    
    def tearDown(self):
        print("LsShifterTest (TestCase) is tearing down...")
         
    def test_shift_energy_to_zero(self):
        ls_shifter: LsShifter = LsShifter(ls=self.ls)
        print(ls_shifter)


class MsShifterTest(unittest.TestCase):
    def setUp(self):
        print("MsShifterTest (TestCase) is setting up...")
        self.ms: MultiSystems = MultiSystems()
        ls1: LabeledSystem = LabeledSystem(ReNbSSe_OUTCAR_PATH)
        ls2: LabeledSystem = LabeledSystem(ReNbSTe_OUTCAR_PATH)
        self.ms.append(ls1)
        self.ms.append(ls2)
    
    def tearDown(self):
        print("MsShifterTest (TestCase) is tearing down...")
        
    def test_shift_energy_to_zero(self):
        ms_shifter: MsShifter = MsShifter(ms=self.ms)
        print(ms_shifter)
        

if __name__ == "__main__":
    unittest.main()
    