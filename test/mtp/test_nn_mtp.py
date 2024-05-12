import unittest

from typing import List
from dpdata import LabeledSystem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai2pot.dataset.mlffdataset import MlffDataset
from ai2pot.mtp.nn_mtp import DescriptorMtp, FittingNet


"""
class DescriptorMtpTest(unittest.TestCase):
    def setUp(self) -> None:
        print("DescriptorMtpTest (TestCase) is setting up...\n")
        mtp_level: int = 10
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
        
        self.outcar_path: str = "/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test_data/ReNbSSe/OUTCAR"
        self.labeled_system: LabeledSystem = LabeledSystem(self.outcar_path)
        self.mlff_dataset: MlffDataset = MlffDataset(labeled_system=self.labeled_system,
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
"""
        
        
class FittingNetTest(unittest.TestCase):
    def setUp(self) -> None:
        print("FittingNetTest (TestSuite) is setting up...\n")
        ntypes: int = 4
        num_descriptor: int = 16
        self.btypes = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        fit_sizes_list: List[int] = [20, 10, 5]
        self.fitting_net: nn.Module = FittingNet(ntypes,
                                                 num_descriptor,
                                                 fit_sizes_list)
        self.fitting_net.to(torch.float64)
        self.input_tensor: torch.Tensor = torch.ones((2, num_descriptor), dtype=torch.float64)
    
    def test_forward(self):
        output_tensor = self.fitting_net(self.btypes, self.input_tensor)
        print(output_tensor.size())
    
    def test_info(self):
        print(self.fitting_net.linears_list)
        self.fitting_net.info()
        
    def tearDown(self) -> None:
        print("FittingNetTest (TestSutie) is tearing down...\n")


if __name__ == "__main__":
    unittest.main()
