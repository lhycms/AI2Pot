import unittest

from typing import List
from dpdata import LabeledSystem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai2pot.dataset.mlffdataset import MlffDataset
from ai2pot.deepmd.se_r import (
    EmbeddingNet,
    FittingNet,
    DpSeR
)


class EmbeddingNetTest(unittest.TestCase):
    def setUp(self):
        print("EmbeddingNetTest (TestCase) is setting up...\n")
        self.layer_size_list: List[int] = [2, 4, 8]
        self.embed: nn.Module = EmbeddingNet(
            layers_size_list=self.layer_size_list, 
            resnet_mark=True)
    
    def test_info(self):
        self.embed.info()
    
    def test_forward(self):
        x: torch.Tensor = torch.ones((3, 27, 1))
        y: torch.Tensor = self.embed(x)
        print("1. When layer_size_list = [2, 4, 8],")
        print("\t - input : x.size() = ", x.size())
        print("\t - output : y.size() = ", y.size())
        
    def tearDown(self):
        print("EmbeddingNetTest (TestCase) is tearing down...\n")


class FittingNetTest(unittest.TestCase):
    def setUp(self):
        print("FittingNetTest (TestCase) is setting up...\n")
        self.layer_size_list: List[int] = [2, 4, 8]
        self.fit = FittingNet(
            layer_size_list=self.layer_size_list,
            resnet_mark=True)
    
    def test_info(self):
        self.fit.info()
    
    def test_forward(self):
        x: torch.Tensor = torch.ones((3, 27, 2))
        y: torch.Tensor = self.fit(x)
        print("1. When layer_size_list = [2, 4, 8],")
        print("\t - input : x.size() = ", x.size())
        print("\t - output : y.size() = ", y.size())
    
    def tearDown(self):
        print("FittingNetTest (TestCase) is tearing down...\n")


class DpSeRTest(unittest.TestCase):
    def setUp(self):
        print("DpSeRTest (TestCase) is setting up...\n")
        ntypes: int = 4 # ReNbSSe
        rcut: float = 3.2
        rcut_smooth: float = 3.0
        # for nblist (equal)
        umax_num_neighs: int = 40
        # for envMatrixOp (equal)
        umax_num_neighs_tensor: torch.Tensor = torch.tensor([10, 10, 10, 10], dtype=torch.int32)
        embed_sizes_list: List[int] = [2, 4, 8]
        fit_sizes_list: List[int] = [8, 4, 2]
        self.dp_se_r = DpSeR(
            ntypes=ntypes,
            rcut=rcut,
            rcut_smooth=rcut_smooth,
            umax_num_neighs=umax_num_neighs_tensor,
            embed_sizes_list=embed_sizes_list,
            fit_sizes_list=fit_sizes_list)

        self.outcar_path: str = "/data/home/liuhanyu/hyliu/code/AI2Pot/test_data/ReNbSSe/OUTCAR"
        self.labeled_system: LabeledSystem = LabeledSystem(self.outcar_path)
        self.mlff_dataset: MlffDataset = MlffDataset(
            labeled_system=self.labeled_system,
            rcut=3.2,
            umax_num_neigh_atoms=umax_num_neighs)
        self.mlff_dataloader: DataLoader = DataLoader(
            dataset=self.mlff_dataset,
            batch_size=5,
            shuffle=True)
        
    def test_info(self):
        print("len(mlff_dataset) = {0}".format(len(self.mlff_dataset)))
        print("len(mlff_dataloader) = {0}".format(len(self.mlff_dataloader)))
        for ii, batch_data in enumerate(self.mlff_dataloader):
            tilde_r: torch.Tensor = self.dp_se_r(
                batch_data[1],
                batch_data[2],
                batch_data[3],
                batch_data[4],
                batch_data[5],
                batch_data[6])
            print("\t{0}. In Batch#{1}, tilde_r.size() = ".format(ii+1, ii), tilde_r.size())
    
    def test_forward(self):
        pass
    
    def tearDown(self):
        print("DpSeRTest (TestCase) is tearing down...\n")
        

if __name__ == "__main__":
    unittest.main()
    