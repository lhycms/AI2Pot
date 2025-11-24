import os
import unittest
from typing import (List, Optional, Tuple)

import torch

from ai2pot.fromcc import allTypeDescriptorsMaxminOp


#torch.use_deterministic_algorithms(True)
torch.set_num_threads(16)
torch.manual_seed(2143)


class AllTypeDescriptorsMaxminOp(unittest.TestCase):
    def setUp(self):
        print("AllTypeDescriptorsMaxminOp (TestCase) is setting up...")

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
        print("AllTypeDescriptorsMaxminOp (TestCase) is tearing down,,,")
    

    def test_allTypeDescriptorsMaxminOp(self):
        descriptors_max_tensor, descriptors_min_tensor = allTypeDescriptorsMaxminOp(self.binum_tensor,
                                                                                    self.bdescriptors_tensor)
        print("1. Calculated by allTypeDescriptorsMaxminOp: ")
        print("\t1.1. descriptors_max_tensor = \n", descriptors_max_tensor)
        print("\t1.2. descriptors_min_tensor:\n", descriptors_min_tensor)


if __name__ == "__main__":
    unittest.main()
