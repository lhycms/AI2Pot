import os
import time
import unittest
from typing import List

import numpy as np
import torch
from pymatgen.core import Structure
from ai2pot.utils import (MlffInput, MlffToLossInput, MlffToEFLossInput)
from ai2pot.models.mtp.linear_mtp import LinearMtp


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")
torch.manual_seed(8769)
torch.set_num_threads(16)

print("Number of omp threads = ", torch.get_num_threads())

class LinearMtpTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtpTest (TestSuite) is setting up...\n")
        self.type_map: List[int] = [75, 41, 16, 34]
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.umax_num_neigh_atoms = 200
        self.device: torch._C.device = torch.device("cuda")
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.linear_mtp: LinearMtp = LinearMtp(mtp_level=16,
                                               type_map=self.type_map,
                                               chebyshev_size=self.chebyshev_size,
                                               rmax=self.rmax,
                                               rmin=self.rmin,
                                               umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                               q_shifter=None,
                                               q_scaler=None,
                                               fit_virial=True)
        self.linear_mtp.to(self.device)
        self.linear_mtp.to(self.torch_float_dtype)
        self.alpha_scalar_moments: int = self.linear_mtp.num_descriptors
        self.mlff_input: MlffInput = MlffInput(type_map=self.type_map,
                                               rcut=self.rmax,
                                               umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                               dtype=self.torch_float_dtype,
                                               device=self.device)
        self.mlff_to_loss_input: MlffToLossInput = MlffToLossInput(type_map=self.type_map,
                                                                   rcut=self.rmax,
                                                                   umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                                   dtype=self.torch_float_dtype,
                                                                   device=self.device)
        self.mlff_to_ef_loss_input: MlffToEFLossInput = MlffToEFLossInput(type_map=self.type_map,
                                                                          rcut=self.rmax,
                                                                          umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                                          dtype=self.torch_float_dtype,
                                                                          device=self.device)
        self.structure: Structure = Structure.from_file(ReNbSSe_POSCAR_PATH)

        ### q_shifter && q_scaler
        self.q_shifter_tensor: torch.Tensor = torch.zeros(self.alpha_scalar_moments,
                                                          dtype=self.torch_float_dtype,
                                                          device=self.device)
        self.q_scaler_tensor: torch.Tensor = torch.randn(self.alpha_scalar_moments,
                                                         dtype=self.torch_float_dtype,
                                                         device=self.device)
        self.linear_mtp.q_shifter_tensor = self.q_shifter_tensor
        self.linear_mtp.q_scaler_tensor = self.q_scaler_tensor
    

    def tearDown(self):
        print("LinearMtpTest (TestSuite) is tearing down...\n")
    

    def est_predict_loss(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            loss = self.linear_mtp.predict_loss(*self.mlff_to_loss_input.analyse_pymatgen(self.structure,
                                                                                          e_weight=1.0,
                                                                                          f_weight=1.0,
                                                                                          v_weight=0.0))
            loss.sum().backward()
            t2 = time.time()
            if (ii == 0):
                print(self.linear_mtp.type_bias_tensor.grad)

            if (ii>9):
                times_list.append(t2-t1)

        print("0.1. Average time cost by linear_mtp.predict_loss() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by linear_mtp.predict_loss() = ", np.std(times_list) / 100)
        print("1. Loss = ", loss)


    def test_predict_ef_loss(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            ef_loss = self.linear_mtp.predict_ef_loss(*self.mlff_to_ef_loss_input.analyse_pymatgen(self.structure,
                                                                                                   e_weight=1.0,
                                                                                                   f_weight=1.0))
            ef_loss.sum().backward()
            t2 = time.time()
            if (ii == 0):
                print(self.linear_mtp.type_bias_tensor.grad)

            if (ii>9):
                times_list.append(t2-t1)

        print("0.1. Average time cost by linear_mtp.predict_ef_loss() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by linear_mtp.predict_ef_loss() = ", np.std(times_list) / 100)
        print("1. Loss = ", ef_loss)


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
        print("2. Force.shape = \n", f[0][5])
        print("3. Virial.shape = \n", v)

    
    def est_predict_ef(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            e, f = self.linear_mtp.predict_ef(*self.mlff_input.analyse_pymatgen(structure=self.structure))
            t2 = time.time()
            if (ii>9):
                times_list.append(t2-t1)
            if (ii==0):
                print(e)
                print(f)

        print("0.1. Average time cost by linear_mtp.predict_ef() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by linear_mtp.predict_ef() = ", np.std(times_list) / 100)
        print("1. Energy = ", e)
        print("2. Force.shape = \n", f[0][5])


    def est_predict_e_sites(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            e_sites = self.linear_mtp.predict_e_sites(*self.mlff_input.analyse_pymatgen(structure=self.structure))
            t2 = time.time()
            if (ii>9):
                times_list.append(t2-t1)
                
        print("0.1. Average time cost by linear_mtp.predict_ef() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by linear_mtp.predict_ef() = ", np.std(times_list) / 100)
        print("\t1. Esites.shape = ", e_sites.shape)


    def est_predict_descriptors(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            descriptors = self.linear_mtp.predict_descriptors(*self.mlff_input.analyse_pymatgen(structure=self.structure))
            t2 = time.time()
            if (ii>9):
                times_list.append(t2-t1)
                
        print("0.1. Average time cost by linear_mtp.predict_descriptors() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by linear_mtp.predict_descriptors() = ", np.std(times_list) / 100)
        print("\t1. descriptors = ", descriptors)


if __name__ == "__main__":
    unittest.main()
