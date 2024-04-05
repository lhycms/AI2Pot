import unittest

from typing import List
import torch
import torch.nn as nn

from ai2pot.deepmd.se_r import (
    EmbeddingNet,
    FittingNet,
    DpSeR
)


class EmbeddingNetTest(unittest.TestCase):
    def setUp(self):
        print("EmbeddingNetTest (TestCase) is setting up...\n")
        self.layer_size_list: List[int] = [2, 4, 8]
        self.embed: nn.Module = EmbeddingNet(layers_size_list=self.layer_size_list)
    
    def test_info(self):
        self.embed.info()
    
    def test_forward(self):
        x: torch.Tensor = torch.ones((27, 1))
        y: torch.Tensor = self.embed(x)
        print("1. When layer_size_list = [2, 4, 8],")
        print("\t - input : x.size() = ", x.size())
        print("\t - output : y.size() = ", y.size())
        
    def tearDown(self):
        print("EmbeddingNetTest (TestCase) is tearing down...\n")


class FittingNetTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_info(self):
        pass
    
    def tearDown(self):
        pass
        
        
if __name__ == "__main__":
    unittest.main()
    