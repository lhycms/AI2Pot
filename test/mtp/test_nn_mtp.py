import unittest
import os
import time

from typing import List
from dpdata import LabeledSystem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L

from ai2pot.data.mlffdataset import ScDataset
from ai2pot.mtp.nn_mtp import DescriptorMtp, FittingNet, NNMtp, LitNNMtp


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_OUTCAR_DIR = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe")


class DescriptorMtpTest(unittest.TestCase):
    def setUp(self) -> None:
        print("DescriptorMtpTest (TestCase) is setting up...\n")
        mtp_level: int = 16
        ntypes: int = 4
        chebyshev_size: int = 8
        rmax: float = 3.2
        rmin: float = 1.0
        umax_num_neighs: int = 40
        
        self.descriptor_mtp: nn.Module = DescriptorMtp(mtp_level=mtp_level,
                                                       ntypes=ntypes,
                                                       chebyshev_size=chebyshev_size,
                                                       rmax=rmax,
                                                       rmin=rmin,
                                                       umax_num_neighs=umax_num_neighs)
        self.descriptor_mtp.to(torch.float64)
        
        self.outcar_path: str = f"{ReNbSSe_OUTCAR_DIR}/OUTCAR"
        self.labeled_system: LabeledSystem = LabeledSystem(self.outcar_path)
        self.mlff_dataset: ScDataset = ScDataset(labeled_system=self.labeled_system,
                                                     rcut=rmax,
                                                     umax_num_neigh_atoms=umax_num_neighs)
        self.mlff_dataloader: DataLoader = DataLoader(self.mlff_dataset,
                                                      batch_size=5,
                                                      shuffle=True)
        
    def test_forward(self):
        for ii, batch_data in enumerate(self.mlff_dataloader):
            descriptor: torch.Tensor = self.descriptor_mtp(batch_data[1],
                                                           batch_data[2],
                                                           batch_data[3],
                                                           batch_data[4],
                                                           batch_data[5])
            print("\t{0}. In Batch#{1}, descrip.size() = ".format(ii+1, ii), descriptor.size())
        
    def tearDown(self) -> None:
        print("DescriptorMtpTest (TestCase) is tearing down...\n")
        
        
class FittingNetTest(unittest.TestCase):
    def setUp(self) -> None:
        print("FittingNetTest (TestSuite) is setting up...\n")
        num_descriptor: int = 16
        fit_sizes_list: List[int] = [20, 10, 5]
        self.fitting_net: nn.Module = FittingNet(num_descriptor=num_descriptor,
                                                 fit_sizes_list=fit_sizes_list,
                                                 fit_activation=nn.Tanh(),
                                                 bias_mark=False,
                                                 energy_shift_tensor=False)
        self.fitting_net.to(torch.float64)
        self.descriptor_tensor: torch.Tensor = torch.ones((2, num_descriptor), dtype=torch.float64)
    
    def test_forward(self):
        e_i = self.fitting_net(self.descriptor_tensor)
        print(e_i.size())
    
    def test_info(self):
        self.fitting_net.info()
        
    def tearDown(self) -> None:
        print("FittingNetTest (TestCase) is tearing down...\n")
  

class NNMtpTest(object):
    def setUp(self) -> None:
        print("NNMtpTest (TestCase) is setting up...\n")
        mtp_level: int = 16
        ntypes: int = 4
        chebyshev_size: int = 8
        rmax: float = 5.0
        rmin: float = 2.0
        umax_num_neighs: int = 40
        fit_sizes_list: List[int] = []
        fit_activation: nn.Module = nn.Tanh()
        self.nn_mtp = NNMtp(mtp_level=mtp_level,
                            ntypes=ntypes,
                            chebyshev_size=chebyshev_size,
                            rmax=rmax,
                            rmin=rmin,
                            umax_num_neighs=umax_num_neighs,
                            fit_sizes_list=fit_sizes_list,
                            fit_activation=fit_activation)
        outcar_path: str = f"{ReNbSSe_OUTCAR_DIR}/OUTCAR"
        labeled_system: LabeledSystem = LabeledSystem(outcar_path)
        self.mlff_dataset: ScDataset = ScDataset(labeled_system=labeled_system,
                                                 rcut=rmax,
                                                 umax_num_neigh_atoms=umax_num_neighs,
                                                 pbc_xyz=[True, True, True],
                                                 sort=False,
                                                 torch_float_dtype=torch.float64)
        self.mlff_dataloader: DataLoader = DataLoader(dataset=self.mlff_dataset,
                                                      batch_size=5,
                                                      shuffle=True)
        self.trainer = L.Trainer(max_epochs=1,
                                 accelerator="cpu",
                                 devices=4)
        
    def test_forward(self):
        self.nn_mtp.to(torch.float64)
        t1 = time.time()
        for ii, batch_data in enumerate(self.mlff_dataloader):
            ei, fi, v = self.nn_mtp(batch_data[1],
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
        print("Cost time: ", t2 - t1)
    
    def test_train(self):
        self.nn_mtp.to(torch.float64)
        
        print("+++***")
        for k, v in self.nn_mtp.named_parameters():
            print(k, v.size(), v.requires_grad)
        
        self.nn_mtp.train()
        lit_nn_mtp: L.LightningModule = LitNNMtp(model=self.nn_mtp)
        self.trainer.fit(model=lit_nn_mtp,
                         train_dataloaders=self.mlff_dataloader)
        
    
    def tearDown(self) -> None:
        print("NNMtpTest (TestCase) is tearing down...\n")

    
if __name__ == "__main__":
    unittest.main()
