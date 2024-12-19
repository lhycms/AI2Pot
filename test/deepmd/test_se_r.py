import unittest
import os
import time

from typing import List
from dpdata import LabeledSystem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai2pot.data.mlffdataset import ScDataset
from ai2pot.deepmd.se_r import (EmbeddingNet,
                                FittingNet,
                                DescripSeR,
                                DpSeR)


TEST_FILES_DIR: str = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_OUTCAR_PATH: str = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe", "OUTCAR")


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


class DescripSeRTes(unittest.TestCase):
    def setUp(self):
        print("DescripTest (TestCase) is setting up...\n")
        ntypes: int = 4 # ReNbSSe
        rcut: float = 3.2
        rcut_smooth: float = 3.0
        # for nblist (equal)
        umax_num_neighs: int = 40
        # for envMatrixOp (equal)
        umax_num_neighs_tensor: torch.Tensor = torch.tensor([10, 10, 10, 10], dtype=torch.int32)
        embed_sizes_list: List[int] = [2, 4, 8]
        self.descrip_se_r: nn.Module = DescripSeR(
            ntypes=ntypes,
            rcut=rcut,
            rcut_smooth=rcut_smooth,
            umax_num_neighs=umax_num_neighs_tensor,
            embed_sizes_list=embed_sizes_list,
            M2=4)
        self.descrip_se_r.to(torch.float64)

        self.labeled_system: LabeledSystem = LabeledSystem(ReNbSSe_OUTCAR_PATH)
        self.mlff_dataset: ScDataset = ScDataset(
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
            descrip: torch.Tensor = self.descrip_se_r(
                batch_data[1],
                batch_data[2],
                batch_data[3],
                batch_data[4],
                batch_data[5])
            print("\t{0}. In Batch#{1}, descrip.size() = ".format(ii+1, ii), descrip.size())
    
    def test_forward(self):
        pass
    
    def tearDown(self):
        print("DescripTest (TestCase) is tearing down...\n")


class DpSeRTest(unittest.TestCase):
    def setUp(self):
        print("DpSeRTest (TestCase) is setting up...\n")
        ntypes: int = 4
        rcut: float = 3.2
        rcut_smooth: float= 3.0
        umax_num_neighs: int = 40   # for envMatrixOp (equal)
        umax_num_neighs_tensor: torch.Tensor = torch.tensor([10, 10, 10, 10], dtype=torch.int32)    # for envMatrixOp (equal)
        embed_sizes_list: List[int] = [2, 4, 8]
        fit_sizes_list: List[int] = [16, 16, 16]
        self.dp_se_r: nn.Module = DpSeR(ntypes=ntypes,
                                        rcut=rcut,
                                        rcut_smooth=rcut_smooth,
                                        umax_num_neighs=umax_num_neighs_tensor,
                                        embed_sizes_list=embed_sizes_list,
                                        fit_sizes_list=fit_sizes_list,
                                        M2=4,
                                        energy_shift_tensor=False)
        self.dp_se_r.to(torch.float64)

        self.labeled_system: LabeledSystem = LabeledSystem(ReNbSSe_OUTCAR_PATH)
        self.mlff_dataset: ScDataset = ScDataset(labeled_system=self.labeled_system,
                                                     rcut=rcut,
                                                     umax_num_neigh_atoms=umax_num_neighs)
        self.mlff_dataloader: DataLoader = DataLoader(self.mlff_dataset,
                                                      batch_size=5,
                                                      shuffle=True)
        
    def test_forward(self):
        t1 = time.time()
        for ii, batch_data in enumerate(self.mlff_dataloader):
            ei, fi, v = self.dp_se_r(batch_data[1],
                                     batch_data[2],
                                     batch_data[3],
                                     batch_data[4].requires_grad_(True),
                                     batch_data[5],
                                     batch_data[6])
            print("\t{0}. In Batch#{1}, descrip.size() = ".format(ii+1, ii), 
                  ei.size(), 
                  fi.size(),
                  v.size())
        t2 = time.time()
        print("Time = ", t2 - t1)
    
    def tearDown(self):
        print("DpSeRTest (TestCase) is tearing down...\n")


if __name__ == "__main__":
    unittest.main()
    