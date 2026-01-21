import os
import time
import unittest
from typing import List, Optional

import numpy as np
import torch
from pymatgen.core import Structure
from ai2pot.utils import (MlffInput, MlffToEFLossInput, MlffToLossInput)
from ai2pot.models.nep.nep import Nep


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")
torch.manual_seed(16)
torch.set_num_threads(16)

print("Number of omp threads = ", torch.get_num_threads())


class NepTest(unittest.TestCase):
    def setUp(self):
        print("NepTest (TestCase) is setting up...\n")
        ### Hyperparameters for Nep
        self.type_map: List[int] = [75, 41, 16, 34]
        self.energy_shifts: Optional[List[float]] = None
        self.umax_num_neigh_atoms: int = 200
        self.fit_virial: bool = False
        self.n_radial_basis: int = 6
        self.n_angular_basis: int = 4
        self.l_max: int = 4
        self.chebyshev_size: int = 8
        self.num_neurons: int = 30
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.zbl_rmax: float = 2.0
        self.zbl_rmin: float = 1.0
        self.zbl_cks_list: Optional[List[float]] = None
        self.zbl_dks_list: Optional[List[float]] = None

        self.device: torch._C.device = torch.device("cuda")
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.nep: Nep = Nep(type_map=self.type_map,
                            energy_shifts=self.energy_shifts,
                            umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                            fit_virial=self.fit_virial,
                            n_radial_basis=self.n_radial_basis,
                            n_angular_basis=self.n_angular_basis,
                            l_max=self.l_max,
                            chebyshev_size=self.chebyshev_size,
                            num_neurons=self.num_neurons,
                            rmax=self.rmax,
                            rmin=self.rmin,
                            zbl_rmax=self.zbl_rmax,
                            zbl_rmin=self.rmin,
                            zbl_cks_list=self.zbl_cks_list,
                            zbl_dks_list=self.zbl_dks_list)
        self.nep.to(self.device)
        self.nep.to(self.torch_float_dtype)
        
        ### mlffinput
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

    

    def tearDown(self):
        print("NepTest (TestCase) is tearing down...\n")
    

    def test_predict_ef_loss(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            ef_loss = self.nep.predict_ef_loss(*self.mlff_to_ef_loss_input.analyse_pymatgen(self.structure,
                                                                                         e_weight=1.0,
                                                                                         f_weight=1.0))
            ef_loss.sum().backward()
            t2 = time.time()
            if (ii == 0):
                print(self.nep.type_bias_tensor.grad)

            if (ii>9):
                times_list.append(t2-t1)
        print("0.1. Average time cost by nep.predict_ef_loss() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by nep.predict_ef_loss() = ", np.std(times_list) / 100)
        print("1. Loss = ", ef_loss)


    def est_predict_ef(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            e, f = self.nep.predict_ef(*self.mlff_input.analyse_pymatgen(structure=self.structure))
            t2 = time.time()
            if (ii>9):
                times_list.append(t2-t1)

        print("0.1. Average time cost by nep.predict_ef() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by nep.predict_ef() = ", np.std(times_list) / 100)
        print("1. Energy = ", e)
        print("2. Force.shape = \n", f[0][5])


    def est_predict_descriptors(self):
        times_list: List[float] = []
        for ii in range(110):
            t1 = time.time()
            descriptors = self.nep.predict_descriptors(*self.mlff_input.analyse_pymatgen(structure=self.structure))
            t2 = time.time()
            if (ii>9):
                times_list.append(t2-t1)
                
        print("0.1. Average time cost by nep.predict_descriptors() = ", np.sum(times_list) / 100)
        print("0.2. std time cost by nep.predict_descriptors() = ", np.std(times_list) / 100)
        print("\t1. descriptors = ", descriptors)


if __name__ == "__main__":
    unittest.main()
    