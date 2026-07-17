import os
import time
import unittest
from typing import List

import torch
import numpy as np

from ai2pot.optimizer.mtpr_solver import LinearMtpSolver
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.data.mlffdataset import ExtxyzDataset
from ai2pot.data.mlffdatamodule import ExtxyzDataModule


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "PbTe", "train.xyz")
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
# os.path.join(TEST_FILES_DIR, "XYZ", "PbTe", "train.xyz")

torch.manual_seed(42)
torch.set_num_threads(16)


class LinearMtpSolverTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtpSolverTest (TestCase) is setting up...\n")
        self.type_map: List[int] = ExtxyzDataset.get_type_map(filename=PbTe_EXTXYZ_PATH)
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.umax_num_neigh_atoms: int = 200
        self.device: torch._C.device = torch.device("cpu")
        self.torch_float_dtype: torch._C.dtype = torch.float64
        fit_virial: bool = False

        self.e_weight: float = 2.0
        self.f_weight: float = 3.0
        self.v_weight: float = 0.0
        self.lit_linear_mtp: LitLinearMtp = LitLinearMtp(type_map=self.type_map,
                                                         umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                         fit_virial=fit_virial,
                                                         mtp_level=16,
                                                         chebyshev_size=self.chebyshev_size,
                                                         rmax=self.rmax,
                                                         rmin=self.rmin,
                                                         zbl_rmax=0.0,
                                                         zbl_rmin=0.0,
                                                         zbl_cks_list=None,
                                                         zbl_dks_list=None,
                                                         e_wgt_start=self.e_weight,
                                                         e_wgt_end=self.e_weight,
                                                         f_wgt_start=self.f_weight,
                                                         f_wgt_end=self.f_weight,
                                                         v_wgt_start=self.v_weight,
                                                         v_wgt_end=self.v_weight).to(device=self.device).to(self.torch_float_dtype)
        self.linear_mtp: LinearMtp = self.lit_linear_mtp.model
        #self.linear_mtp._init_as_mlip()
        self.trainset: ExtxyzDataset = ExtxyzDataset(filename=PbTe_EXTXYZ_PATH,
                                                     rcut=self.rmax,
                                                     umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                     pbc_xyz=[True, True, True],
                                                     sort=False,
                                                     torch_float_dtype=self.torch_float_dtype,
                                                     has_virial=fit_virial)
        ridge_lambda: float = 1e-2
        self.solver: LinearMtpSolver = LinearMtpSolver(lit_linear_mtp=self.lit_linear_mtp,
                                                       trainset=self.trainset,
                                                       ridge_lambda=ridge_lambda)


    def tearDown(self):
        print("LinearMtpSolverTest (TestCase) is tearing down...\n")
    

    def test_orthogonalize(self):
        ntypes: int = self.linear_mtp.ntypes
        nmus: int = self.linear_mtp.nmus
        chebyshev_size: int = self.linear_mtp.chebyshev_size

        self.solver.orthogonalize()
        orthogonal_coeffs_tensor: torch.Tensor = self.linear_mtp.coeffs_tensor.reshape(ntypes, ntypes, nmus, chebyshev_size)

        ### 
        vertification_matrix: np.ndarray = np.zeros((nmus, nmus))
        for k1 in range(nmus):
            for k2 in range(nmus):
                a = orthogonal_coeffs_tensor[:, :, k1, :].flatten()
                b = orthogonal_coeffs_tensor[:, :, k2, :].flatten()
                vertification_matrix[k1][k2] = torch.matmul(a, b)
        print("1. Vertification Matrix for orthogonalize : ")
        print(vertification_matrix.round(4))

    
    def test_solve_linear_equation(self):
        #print(self.linear_mtp.linear_coeffs_tensor)
        self.solver.solve_linear_equation()
        #print(self.linear_mtp.linear_coeffs_tensor)


    def test_rescale_coeffs(self):
        self.solver.rescale_coeffs()
        #print(self.solver.linear_mtp.coeffs_tensor)


if __name__ == "__main__":
    unittest.main()
