import os
import unittest
from typing import (List, Optional, Tuple)

import torch

from ai2pot.fromcc import coeffsSchmidtOrthOp


torch.set_num_threads(16)
torch.manual_seed(2143)


class CoeffsSchmidtOrthOpTest(unittest.TestCase):
    def setUp(self):
        print("CoeffsSchmidtOrthOpTest (TestCase) is setting up...")
        self.ntypes: int = 2
        self.nmus: int = 4
        self.chebyshev_size: int = 8
        self.num_coeffs: int = self.ntypes * self.ntypes * self.nmus * self.chebyshev_size
        self.device: torch._C.device = torch.device("cuda")
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.coeffs_tensor: torch.Tensor = torch.rand(self.num_coeffs, dtype=self.torch_float_dtype, device=self.device)
    

    def tearDown(self):
        print("CoeffsSchmidtOrthOpTest (TestCase) is tearing down...")


    def check_orthogonality(self,
                            orth_coeffs: torch.Tensor,
                            ntypes: int,
                            nmus: int,
                            chebyshev_size: int,
                            mu1: int,
                            mu2: int):
        scal: float = 0.0
        for type_central in range(ntypes):
            for type_outer in range(ntypes):
                for xi in range(chebyshev_size):
                    idx1 = (type_central*ntypes+type_outer)*nmus*chebyshev_size \
                           + mu1*chebyshev_size + xi
                    idx2 = (type_central*ntypes+type_outer)*nmus*chebyshev_size \
                           + mu2*chebyshev_size + xi
                    scal += orth_coeffs[idx1] * orth_coeffs[idx2]        
        print("1. scal of {0} and {1}: = {2:.10f}\n".format(mu1, mu2, scal))


    def test_coeffs_schmidt_orth_op(self):
        orth_coeffs_tensor: torch.Tensor = coeffsSchmidtOrthOp(
            self.ntypes,
            self.nmus,
            self.chebyshev_size,
            self.coeffs_tensor)[0]
        
        for k1 in range(self.nmus):
            for k2 in range(self.nmus):
                self.check_orthogonality(orth_coeffs_tensor,
                                         self.ntypes,
                                         self.nmus,
                                         self.chebyshev_size,
                                         k1,
                                         k2)

if __name__ == "__main__":
    unittest.main()
    