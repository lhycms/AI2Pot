import unittest
from typing import List

import torch
import torch.nn as nn

from ai2pot.models.potential_submodules import FittingNet


class FittingNetTest(unittest.TestCase):
    def setUp(self) -> None:
        print("FittingNetTest (TestSuite) is setting up...\n")
        num_descriptor: int = 16
        fit_sizes_list: List[int] = [20, 10, 5]
        self.fitting_net: nn.Module = FittingNet(num_descriptor=num_descriptor,
                                                 fit_sizes_list=fit_sizes_list,
                                                 fit_activation=nn.Tanh(),
                                                 bias_mark=False,
                                                 energy_shift_tensor=torch.tensor(0))
        self.fitting_net.to(torch.float32)
        self.descriptor_tensor: torch.Tensor = torch.ones((2, num_descriptor), dtype=torch.float32)
    
    def test_forward(self):
        e_i = self.fitting_net(self.descriptor_tensor)
        print(e_i.size())
    
    def test_info(self):
        self.fitting_net.info()
        
    def tearDown(self) -> None:
        print("FittingNetTest (TestCase) is tearing down...\n")
        
        
if __name__ == "__main__":
    unittest.main()
