import unittest
from typing import List

import torch
import torch.nn as nn

from ai2pot.models.potential_loss import (ERmse,
                                          FRmse,
                                          VRmse)


class PotentialLossTest(unittest.TestCase):
    def setUp(self):
        print("PotentialLossTest (TestCase) is setting up...")
        self.binum = torch.tensor([1, 2, 3, 4, 5]).view(5, 1)
        self.e_ml: torch.Tensor = torch.ones([5, 1])  # [batch_size, total_energy]
        self.e_dft: torch.Tensor = torch.zeros([5, 1])
        self.f_ml: torch.Tensor = torch.ones([5, 108, 3])   # [batch_size, num_atoms, 3]
        self.f_dft: torch.Tensor = torch.zeros([5, 108, 3])
        self.v_ml: torch.Tensor = torch.ones([5, 3, 3])
        self.v_dft: torch.Tensor = torch.zeros([5, 3, 3])
        self.e_criterion: nn.Module = ERmse()
        self.f_criterion: nn.Module = FRmse()
        self.v_criterion: nn.Module = VRmse()
        
    
    def tearDown(self):
        print("PotentialLossTest (TestCase) is tearing down...")
    
    def test_ermse(self):
        e_criterion_tensor: torch.Tensor = self.e_criterion(binum=self.binum,
                                                            input_benergies=self.e_dft,
                                                            target_benergies=self.e_ml)
        print("e_rmse = ", e_criterion_tensor)
    
    def test_frmse(self):
        f_criterion_tensor: torch.Tensor = self.f_criterion(binum=self.binum,
                                                            input_bforces=self.f_ml,
                                                            target_bforces=self.f_dft)
        print("f_rmse = ", f_criterion_tensor)
    
    def test_vrmse(self):
        v_criterion_tensor: torch.Tensor = self.v_criterion(binum=self.binum,
                                                            input_bvirials=self.v_ml,
                                                            target_bvirials=self.v_dft)
        print("v_rmse = ", v_criterion_tensor)


if __name__ == "__main__":
    unittest.main()
    