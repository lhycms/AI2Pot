import os
import time
import unittest
from typing import List

import torch
import numpy as np

from ai2pot.optimizer.mtpr_solver import LinearMtpSolver
from ai2pot.models.mtp.linear_mtp import LinearMtp



class LinearMtpSolverTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtpSolverTest (TestCase) is setting up...\n")
        self.type_map: List[int] = [75, 41, 16, 34]
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.umax_num_neigh_atoms: int = 200
        self.device: torch._C.device = torch.device("cpu")
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.linear_mtp: LinearMtp = LinearMtp(type_map=self.type_map,
                                               umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                               fit_virial=True,
                                               mtp_level=18,
                                               chebyshev_size=self.chebyshev_size,
                                               rmax=self.rmax,
                                               rmin=self.rmin,
                                               zbl_rmax=0.0,
                                               zbl_rmin=0.0,
                                               zbl_cks_list=None,
                                               zbl_dks_list=None).to(device=self.device).to(self.torch_float_dtype)

        self.solver: LinearMtpSolver = LinearMtpSolver()


    def tearDown(self):
        print("LinearMtpSolverTest (TestCase) is tearing down...\n")
    

    def test_orthogonalize(self):
        ntypes: int = self.linear_mtp.ntypes
        nmus: int = self.linear_mtp.nmus
        chebyshev_size: int = self.linear_mtp.chebyshev_size

        self.solver.orthogonalize(linear_mtp=self.linear_mtp)
        orthogonal_coeffs_tensor: torch.Tensor = self.linear_mtp.coeffs_tensor.reshape(ntypes, ntypes, nmus, chebyshev_size)

        ### 
        vertification_matrix: np.ndarray = np.zeros((nmus, nmus))
        for k1 in range(nmus):
            for k2 in range(nmus):
                a = orthogonal_coeffs_tensor[:, :, k1, :].flatten()
                b = orthogonal_coeffs_tensor[:, :, k2, :].flatten()
                vertification_matrix[k1][k2] = torch.matmul(a, b)
        print(vertification_matrix.round(4))


if __name__ == "__main__":
    unittest.main()
