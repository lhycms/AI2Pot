import unittest
import os
import time
import copy

from typing import List
from dpdata import LabeledSystem
from pymatgen.core import Structure
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L

from ai2pot.data.mlffdataset import ScDataset, ExtxyzDataset
from ai2pot.models.mtp.nn_mtp import DescriptorMtp, NNMtp
from ai2pot.models.potential_train import LitPotential
from ai2pot.utils.prepot import LsShifter
from ai2pot.utils.usepot import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")
ReNbSSe_OUTCAR_PATH = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe", "OUTCAR")
Cu_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "Cu", "Cu.extxyz")
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "PbTe", "PbTe.extxyz")


class DescriptorMtpTes(object):
    def setUp(self) -> None:
        print("DescriptorMtpTest (TestCase) is setting up...\n")
        mtp_level: int = 18
        ntypes: int = 1
        chebyshev_size: int = 8
        rmax: float = 5.0
        rmin: float = 0.5
        umax_num_neighs: int = 100
        dtype: torch._C.dtype = torch.float64
        
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
                                                      batch_size=1,
                                                      shuffle=True)
        
    def test_forward(self):
        for ii, batch_data in enumerate(self.mlff_dataloader):
            descriptor: torch.Tensor = self.descriptor_mtp(batch_data[0],
                                                           batch_data[1],
                                                           batch_data[2],
                                                           batch_data[3],
                                                           batch_data[4],
                                                           batch_data[5])
            print("\t{0}. In Batch#{1}, descrip.size() = ".format(ii+1, ii), descriptor.size())
    
    def tearDown(self) -> None:
        print("DescriptorMtpTest (TestCase) is tearing down...\n")
          

class NNMtpTest(unittest.TestCase):
    def setUp(self) -> None:
        print("NNMtpTest (TestCase) is setting up...\n")
        self.mtp_level: int = 20
        self.ntypes: int = 2
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.5
        self.umax_num_neighs: int = 100
        self.dtype: torch._C.dtype = torch.float64
        self.device = torch.device("cpu")

        self.batch_norm_mark = False
        self.fit_sizes_list: List[int] = [60, 15]
        self.fit_activation: nn.Module = nn.Tanh()
        self.bias_mark: bool = False
        self.has_forces: bool = False
        self.has_virial: bool = False
        
        # delta
        self.delta: float = 1e-5
        self.center_idx_modify: int = 0
        self.neigh_idx_modify: int = 6
        
        self.mlff_dataset: ExtxyzDataset = ExtxyzDataset(filename=Cu_EXTXYZ_PATH,
                                                 rcut=self.rmax,
                                                 umax_num_neigh_atoms=self.umax_num_neighs,
                                                 pbc_xyz=[True, True, True],
                                                 sort=False,
                                                 torch_float_dtype=self.dtype,
                                                 has_virial=self.has_virial)
        self.mlff_dataloader: DataLoader = DataLoader(dataset=self.mlff_dataset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      num_workers=1)
        self.structure: Structure = Structure.from_file(ReNbSSe_POSCAR_PATH)
        
        self.nn_mtp = NNMtp(mtp_level=self.mtp_level,
                            ntypes=self.ntypes,
                            chebyshev_size=self.chebyshev_size,
                            rmax=self.rmax,
                            rmin=self.rmin,
                            umax_num_neighs=self.umax_num_neighs,
                            batch_norm_mark=self.batch_norm_mark,
                            fit_sizes_list=self.fit_sizes_list,
                            fit_activation=self.fit_activation,
                            bias_mark=self.bias_mark,
                            energy_shift_tensor=False,
                            has_forces=self.has_forces,
                            has_virial=self.has_virial).to(self.dtype)
        
        self.mlff_input: MlffInput = MlffInput(rcut=self.rmax,
                                            umax_num_neighs=self.umax_num_neighs,
                                            pbc_xyz=[True, True, True],
                                            sort=False,
                                            dtype=self.dtype,
                                            device=self.device)
        self.nmus = self.nn_mtp.descriptor_module.nmus


    def tearDown(self) -> None:
        print("NNMtpTest (TestCase) is tearing down...\n")


    def test_train(self):
        print("NNMtpTest.test_train:")
        print("---------------------")
        self.nn_mtp.train()
        self.trainer = L.Trainer(max_epochs=1000,
                                 accelerator="cpu",
                                 devices=1)
        lit_nn_mtp: L.LightningModule = LitPotential(model=self.nn_mtp,
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
        
        # 1. Structure 1
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        nblist_info[4].requires_grad_(True)
        etot = self.nn_mtp(*nblist_info)[0]
        etot.backward()
        
        # 2. Structure 2
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4].requires_grad_(False)
        nblist_info_[4][0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify] += self.delta
        nblist_info_[4].requires_grad_(True)
        etot_ = self.nn_mtp(*nblist_info_)[0]
        etot_.backward()
        
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
            ((etot_ - etot) / self.delta).item() - nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item() )
        )
        print("--------------------------------------")
    
    
    def test_etot_der2coeffs(self):
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        itype_modify: int = nblist_info[5][0][self.center_idx_modify].item()
        jtype_modify: int = nblist_info[5][0][self.neigh_idx_modify].item()
        mu_modify: int = 0
        xi_modify: int = 1
        coeff_idx_modify: int = (itype_modify*self.ntypes + jtype_modify)*self.nmus*self.chebyshev_size + mu_modify*self.chebyshev_size + xi_modify
        
        # Case 1
        etot = self.nn_mtp(*nblist_info)[0]
        etot.backward()
        
        # Case 2
        self.nn_mtp_ = copy.deepcopy(self.nn_mtp)
        self.nn_mtp_.descriptor_module.coeffs.requires_grad_(False)
        self.nn_mtp_.descriptor_module.coeffs[coeff_idx_modify] += self.delta
        self.nn_mtp_.descriptor_module.coeffs.requires_grad_(True)
        etot_ = self.nn_mtp_(*nblist_info)[0]
        etot_.backward()
        
        print("--------------------------------------")
        print("2.0. Etot = {0:.5f}".format(etot_.item()))
        print("2.1. Etot derivative w.r.t. coeffs[{0}][{1}][{2}][{3}] calculated by custom code = {4:.5f}".format(
            itype_modify,
            jtype_modify,
            mu_modify,
            xi_modify,
            self.nn_mtp.descriptor_module.coeffs.grad[coeff_idx_modify].item())
        )
        print("2.2. Etot derivative w.r.t. coeffs[{0}][{1}][{2}][{3}] calculated by finite difference method = {4:.5f}".format(
            itype_modify,
            jtype_modify,
            mu_modify,
            xi_modify,
            ((etot_ - etot) / self.delta).item())
        )
        print("2.3. Difference between custom code and finite difference method = {0:.7f}".format(
            ((etot_ - etot) / self.delta).item() - self.nn_mtp.descriptor_module.coeffs.grad[coeff_idx_modify].item() )
        )
        print("--------------------------------------")
    
    
    def est_fi_der2xyz(self):
        direction_idx_modify: int = 1
        
        # 1. Structure 1
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        nblist_info[4].requires_grad_(True)
        etot, fi = self.nn_mtp(*nblist_info)
        single_fi = fi[0][self.center_idx_modify][direction_idx_modify]
        single_fi.backward()
        
        # 2. Structure 2
        nblist_info_: List[torch.Tensor] = [tensor.clone().detach() for tensor in nblist_info]
        nblist_info_[4].requires_grad_(False)
        nblist_info_[4][0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify] += self.delta
        nblist_info_[4].requires_grad_(True)
        etot_, fi_ = self.nn_mtp(*nblist_info_)
        single_fi_ = fi_[0][self.center_idx_modify][direction_idx_modify]
        single_fi_.backward()
        
        print("--------------------------------------")
        print("3.0. fi[0][{0}][{1}] = {2:.5f}".format(self.center_idx_modify, direction_idx_modify, single_fi.item()))
        print( single_fi, fi.grad_fn )
        print(fi)
        print( nblist_info[4].grad )
        print("3.1. Etot derivative w.r.t. rcs[{0}][{1}][{2}] calculated by custom code = {3:.5f}".format(
            self.center_idx_modify,
            self.neigh_idx_modify,
            direction_idx_modify,
            nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item())
        )
        print("3.2. Etot derivative w.r.t. rcs[{0}][{1}][{2}] calculated by finite difference method = ".format(
            self.center_idx_modify,
            self.neigh_idx_modify,
            direction_idx_modify),
            ((single_fi_ - single_fi) / self.delta).item()
        )
        #print("3.3. Difference between custom code and finite difference method = {0:.7f}".format(
        #    ((etot_ - etot) / self.delta).item() - nblist_info[4].grad[0][self.center_idx_modify][self.neigh_idx_modify][direction_idx_modify].item()
        #    )
        #)
        print("--------------------------------------")
    
    
    def test_fi_der2coeffs(self):
        pass


    def test_v_der2xyz(self):
        pass
    
    
    def test_v_der2coeffs(self):
        pass


if __name__ == "__main__":
    unittest.main()
