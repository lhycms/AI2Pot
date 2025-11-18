import os
import unittest
from typing import (List, Optional, Tuple)

import torch

from ai2pot.fromcc import allTypeDescriptorsStatisticsOp


#torch.use_deterministic_algorithms(True)
torch.set_num_threads(16)
torch.manual_seed(2143)


class AllTypeDescriptorsStatisticsOp(unittest.TestCase):
    def setUp(self):
        print("AllTypeDescriptorsStatisticsOp (TestCase) is setting up...")

        device: torch._C.device = torch.device('cuda')
        torch_float_dtype: torch._C.device = torch.float64

        batch_size: int = 31
        natoms_pad: int = 20
        descriptor_dim: int = 9
        self.binum_tensor: torch.Tensor = torch.zeros(batch_size, 
                                                 dtype=torch.int32, 
                                                 device=device) + natoms_pad
        self.bdescriptors_tensor: torch.Tensor = torch.rand(size=(batch_size, natoms_pad, descriptor_dim),
                                                       dtype=torch_float_dtype,
                                                       device=device)
        
    
    def tearDown(self):
        print("AllTypeDescriptorsStatisticsOp (TestCase) is tearing down,,,")
    

    def test_allTypeDescriptorsStatisticsOp(self):
        natoms_in_batch_tensor, descriptors_mean_tensor, descriptors_M2_tensor = allTypeDescriptorsStatisticsOp(self.binum_tensor,
                                                                                                                self.bdescriptors_tensor)
        print("1. Calculated by allTypeDescriptorsStatisticsOp: ")
        print("\t1.1. natoms_in_batch = ", natoms_in_batch_tensor.item())
        print("\t1.2. descriptors_mean_tensor:\n\t\t", descriptors_mean_tensor)
        print("\t1.3. descriptors_M2_tensor:\n\t\t", (descriptors_M2_tensor/(natoms_in_batch_tensor.item()-1))**0.5)

        print("2. Calculated by PyTorch: ")
        print("\t2.2. descriptors_mean_tensor:\n\t\t", torch.mean(self.bdescriptors_tensor, dim=(0, 1)))
        print("\t2.3. descriptors_M2_tensor:\n\t\t", torch.std(self.bdescriptors_tensor, dim=(0, 1)))


if __name__ == "__main__":
    unittest.main()
