import unittest
import os
import time

from typing import List
from dpdata import LabeledSystem
from pymatgen.core import Structure
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L

from ai2pot.data.mlffdataset import ScDataset, ExtxyzDataset
from ai2pot.mtp.nn_mtp import DescriptorMtp, FittingNet, NNMtp, LitNNMtp
from ai2pot.utils.prepot import LsShifter
from ai2pot.utils.usepot import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")
ReNbSSe_OUTCAR_PATH = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe", "OUTCAR")
Cu_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "Cu", "Cu.extxyz")

class DescriptorMtpTest(object):
    def setUp(self) -> None:
        print("DescriptorMtpTest (TestCase) is setting up...\n")
        mtp_level: int = 16
        ntypes: int = 4
        chebyshev_size: int = 8
        rmax: float = 5.0
        rmin: float = 0.5
        umax_num_neighs: int = 40
        dtype: torch._C.dtype = torch.float32
        
        self.descriptor_mtp: nn.Module = DescriptorMtp(mtp_level=mtp_level,
                                                       ntypes=ntypes,
                                                       chebyshev_size=chebyshev_size,
                                                       rmax=rmax,
                                                       rmin=rmin,
                                                       umax_num_neighs=umax_num_neighs)
        self.descriptor_mtp.to(dtype=dtype)
        
        self.labeled_system: LabeledSystem = LabeledSystem(ReNbSSe_OUTCAR_PATH)
        self.mlff_dataset: ScDataset = ScDataset(labeled_system=self.labeled_system,
                                                     rcut=rmax,
                                                     umax_num_neigh_atoms=umax_num_neighs,
                                                     torch_float_dtype=dtype)
        self.mlff_dataloader: DataLoader = DataLoader(self.mlff_dataset,
                                                      batch_size=20,
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
        
        
class FittingNetTest(object):
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
  

class NNMtpTest(unittest.TestCase):
    def setUp(self) -> None:
        print("NNMtpTest (TestCase) is setting up...\n")
        mtp_level: int = 16
        ntypes: int = 4
        chebyshev_size: int = 8
        rmax: float = 5.0
        rmin: float = 0.5
        umax_num_neighs: int = 100
        dtype: torch._C.dtype = torch.float64
        device = torch.device("cpu")

        batch_norm_mark = False
        fit_sizes_list = [30]
        fit_activation = nn.Tanh()
        
        # delta
        self.delta: float = 1e-5
        self.center_idx_modify: int = 0
        self.neigh_idx_modify: int = 6
        
        self.mlff_dataset: ExtxyzDataset = ExtxyzDataset(filename=Cu_EXTXYZ_PATH,
                                                 rcut=rmax,
                                                 umax_num_neigh_atoms=umax_num_neighs,
                                                 pbc_xyz=[True, True, True],
                                                 sort=False,
                                                 torch_float_dtype=dtype,
                                                 has_virials=True)
        self.mlff_dataloader: DataLoader = DataLoader(dataset=self.mlff_dataset,
                                                      batch_size=32,
                                                      shuffle=True,
                                                      num_workers=2)
         
        self.nn_mtp = NNMtp(mtp_level=mtp_level,
                            ntypes=ntypes,
                            chebyshev_size=chebyshev_size,
                            rmax=rmax,
                            rmin=rmin,
                            umax_num_neighs=umax_num_neighs,
                            batch_norm_mark=batch_norm_mark,
                            fit_sizes_list=fit_sizes_list,
                            fit_activation=fit_activation,
                            bias_mark=False,
                            energy_shift_tensor=False,
                            has_forces=True,
                            has_virials=True).to(dtype)
        
        self.mlff_input: MlffInput = MlffInput(rcut=rmax,
                                            umax_num_neighs=umax_num_neighs,
                                            pbc_xyz=[True, True, True],
                                            sort=False,
                                            dtype=dtype,
                                            device=device)


    def tearDown(self) -> None:
        print("NNMtpTest (TestCase) is tearing down...\n")
        

    def est_forward(self):
        self.nn_mtp.to(torch.float32)
        t1 = time.time()
        print("NNMtpTest.test_forward:")
        print("-----------------------")
        for ii, batch_data in enumerate(self.mlff_dataloader):
            etot, fi, v = self.nn_mtp(batch_data[1],
                                      batch_data[2],
                                      batch_data[3],
                                      batch_data[4].requires_grad_(True),
                                      batch_data[5],
                                      batch_data[6])
            print("\t{0}. In Batch#{1}, e.size(), fi.size(), v.size() = ".format(ii+1, ii), 
                  etot.size(),
                  fi.size(),
                  v.size())
        t2 = time.time()
        print("Cost time: ", t2 - t1)


    def est_train(self):
        print("NNMtpTest.test_train:")
        print("---------------------")
        self.nn_mtp.train()
        self.trainer = L.Trainer(max_epochs=100,
                                 accelerator="cpu",
                                 devices=1)
        lit_nn_mtp: L.LightningModule = LitNNMtp(model=self.nn_mtp,
                                                 lr_start=1E-3,
                                                 e_wgt_start=1.0,
                                                 e_wgt_end=1.0,
                                                 f_wgt_start=1.0,
                                                 f_wgt_end=1.0,
                                                 v_wgt_start=0.0,
                                                 v_wgt_end=0.0)
        self.trainer.fit(model=lit_nn_mtp,
                         train_dataloaders=self.mlff_dataloader)
        
    
    def test_etot_der2xyz(self):
        direction_idx_modify: int = 1
        
        structure: Structure = Structure.from_file(ReNbSSe_POSCAR_PATH)
        
        # 1. Structure 1
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(structure)
        itype_idx_modify: int = nblist_info[5][0, self.center_idx_modify]
        jtype_idx_modify: int = nblist_info[5][0, self.neigh_idx_modify]
        nblist_info[4].requires_grad_(True)
        etot, fi, v = self.nn_mtp(*nblist_info[1:])
        etot.backward()
        
        # 2. Structure 2
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4].requires_grad_(False)
        nblist_info_[4][0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify] += self.delta
        nblist_info_[4].requires_grad_(True)
        etot_, fi_, v_ = self.nn_mtp(*nblist_info_[1:])
        
        print("--------------------------------------")
        print("1.0. Etot = {0:.5f}".format(etot.item()))
        print("1.1. Etot derivative w.r.t. rcs[{0}][{1}][{2}] calculated by custom code = {3:.5f}".format(
            self.center_idx_modify,
            self.neigh_idx_modify,
            direction_idx_modify,
            nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item())
        )
        print("1.2. Etot derivative w.r.t. rcs[{0}][{1}][{2}] calculated by finite difference method = ".format(
            self.center_idx_modify,
            self.neigh_idx_modify,
            direction_idx_modify),
            ((etot_ - etot) / self.delta).item()
        )
        print("1.3. Difference between custom code and finite difference method = {0:.7f}".format(
            ((etot_ - etot) / self.delta).item() - nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item()
            )
        )
        print("--------------------------------------")
    
    
    def test_etot_der2coeffs(self):
        pass
    
    
    def test_fi_der2xyz(self):
        pass
    
    
    def test_fi_der2coeffs(self):
        pass
    


if __name__ == "__main__":
    unittest.main()
