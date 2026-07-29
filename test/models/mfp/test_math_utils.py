import unittest
import os

import torch

from ai2pot.models.mfp.math_utils import (
    build_cart_basis
)



class MathUtilsTest(unittest.TestCase):
    def setUp(self):
        print("MathUtilsTest (TestCase) is setting up...")
        self.num_edges: int = 208

    def tearDown(self):
        print("MathUtilsTest (TestCase) is tearing down...")

    def test_build_cart_basis(self):
        x: torch.Tensor = torch.randn(self.num_edges, 3)
        direction_tensor: torch.Tensor = x / torch.norm(x, dim=1, keepdim=True)
        cart_basis_tensor: torch.Tensor = build_cart_basis(direction_tensor=direction_tensor)

        B0 = cart_basis_tensor[:,0:1]
        B1 = cart_basis_tensor[:,1:4]
        B2 = cart_basis_tensor[:,4:9]

        print(" 1.1. Norm of l0 basis = ", torch.sum(B0*B0, dim=1).mean())
        print(" 1.2. Norm of l1 basis = ", torch.sum(B1*B1, dim=1).mean())
        print(" 1.3. Norm of l2 basis = ", torch.sum(B2*B2, dim=1).mean())
        


if __name__ == "__main__":
    unittest.main()
