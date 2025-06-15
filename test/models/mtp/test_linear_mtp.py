import os
import time
import unittest
from typing import List

import numpy as np
import torch
from pymatgen.core import Structure
from ai2pot.utils import (MlffInput, MlffToLossInput)
from ai2pot.models.mtp.linear_mtp import LinearMtp


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")
torch.manual_seed(16)
torch.set_num_threads(16)

print(torch.get_num_threads())

class LinearMtpTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtpTest (TestSuite) is setting up...\n")
        self.type_map_tensor: torch.Tensor = torch.tensor([75, 41, 16, 34], dtype=torch.int32)
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 2.0
        self.umax_num_neighs = 100
        self.device: torch._C.device = torch.device("cuda")
        self.linear_mtp: LinearMtp = LinearMtp(mtp_level=16,
                                               type_map_tensor=self.type_map_tensor,
                                               chebyshev_size=self.chebyshev_size,
                                               rmax=self.rmax,
                                               rmin=self.rmin,
                                               umax_num_neighs=self.umax_num_neighs,
                                               fit_virial=True)
        self.linear_mtp.to(self.device)
        self.mlff_input: MlffInput = MlffInput(type_map=self.type_map_tensor.numpy().tolist(),
                                               rcut=self.rmax,
                                               umax_num_neighs=self.umax_num_neighs,
                                               device=self.device)
        self.mlff_to_loss_input: MlffToLossInput = MlffToLossInput(type_map=self.type_map_tensor.numpy().tolist(),
                                                                   rcut=self.rmax,
                                                                   umax_num_neighs=self.umax_num_neighs,
                                                                   device=self.device)
        self.structure: Structure = Structure.from_file(ReNbSSe_POSCAR_PATH)
    
    
    def tearDown(self):
        print("LinearMtpTest (TestSuite) is tearing down...\n")
    

    def test_predict_loss(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            loss = self.linear_mtp.predict_loss(*self.mlff_to_loss_input.analyse_pymatgen(self.structure,
                                                                                          e_weight=1.0,
                                                                                          f_weight=1.0,
                                                                                          v_weight=1.0))
            loss.sum().backward()
            t2 = time.time()

            if (ii>9):
                times_list.append(t2-t1)

        print("0.1. Average time cost by linear_mtp.predict_loss() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by linear_mtp.predict_loss() = ", np.std(times_list) / 100)
        print("1. Loss = ", loss)


    def est_predict_ef_loss(self):
        "Ignore"
        pass


    def est_predict_efv(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            e, f, v = self.linear_mtp.predict_efv(*self.mlff_input.analyse_pymatgen(structure=self.structure))
            t2 = time.time()
            if (ii>9):
                times_list.append(t2-t1)

        print("0.1. Average time cost by linear_mtp.predict_efv() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by linear_mtp.predict_efv() = ", np.std(times_list) / 100)
        print("1. Energy = ", e)
        print("2. Force.shape = ", f.shape)
        print("3. Virial.shape = ", v.shape)

    
    def est_predict_ef(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            e, f = self.linear_mtp.predict_ef(*self.mlff_input.analyse_pymatgen(structure=self.structure))
            t2 = time.time()
            if (ii>9):
                times_list.append(t2-t1)

        print("0.1. Average time cost by linear_mtp.predict_ef() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by linear_mtp.predict_ef() = ", np.std(times_list) / 100)
        print("1. Energy = ", e)
        print("2. Force.shape = ", f.shape)


if __name__ == "__main__":
    unittest.main()
