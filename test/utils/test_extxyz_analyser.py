import os
import unittest
from typing import List

import numpy as np
from ai2pot.utils.prepot import ExtxyzAnalyser


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
PbTe_XTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"


class ExtxyzAnalyseTest(unittest.TestCase):
    def setUp(self):
        print("ExtxyzAnalyserTest (TestCase) is setting up...")
        self.rcut: float = 5.0
        self.umax_num_neigh_atoms: int = 100
        self.pbc_xyz: List[bool] = [True, True, True]
        self.sort: bool = False
        self.extxyz_analyser: ExtxyzAnalyser = ExtxyzAnalyser(extxyz_path=PbTe_XTXYZ_PATH,
                                                              rcut=self.rcut,
                                                              umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                              pbc_xyz=self.pbc_xyz,
                                                              sort=self.sort)

    def tearDown(self):
        print("ExtxyzAnalyserTest (TestCase) is teawring down...")


    def test_get_distance(self):
        print(np.sort(self.extxyz_analyser.get_distances())[:900])


    def test_get_min_distance(self):
        print(np.min(self.extxyz_analyser.get_min_distances()))


    def test_get_forces(self):
        print(np.sort(self.extxyz_analyser.get_forces().flatten())[-30:])


if __name__ == "__main__":
    unittest.main()