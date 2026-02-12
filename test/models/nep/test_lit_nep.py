from typing import List
import unittest
import os
from pymatgen.core import Structure

import torch
import lightning as L

from ai2pot.data import ExtxyzDataset, ExtxyzDataModule
from ai2pot.models.nep.nep import Nep
from ai2pot.models.potential_train import LitNep
from ai2pot.utils.prepot import ExtxyzShifter
from ai2pot.models.nep.nep_train_utils import NepDescriptorNormCallback



TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
PbTe_EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"#os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")


class LitNepTest(unittest.TestCase):
    def setUp(self):
        print("LitNep (TestCase) is setting up...\n")

        extxyz_shifter: ExtxyzShifter = ExtxyzShifter(extxyz_path=PbTe_EXTXYZ_PATH)
        energy_shifts: List[float] = extxyz_shifter.types_energy_shifts

        type_map: List[int] = ExtxyzDataset.get_type_map(filename=PbTe_EXTXYZ_PATH)
        umax_num_neigh_atoms: int = 200
        fit_virial: bool = False

        # Nep hyperparameters
        rmax_radial: float = 6.0
        rmax_angular: float = 6.0
        n_radial_basis: int = 6 + 1
        n_angular_basis: int = 4 + 1
        l_max: int = 4
        chebyshev_size: int = 8 + 1
        num_neurons: int = 30


        max_epochs: int = 200
        # Lr hyperparameters
        lr_start: float = 1e-2
        lr_end: float = 5e-4
        e_wgt_start: float = 1.0
        e_wgt_end: float = 1.0
        f_wgt_start: float = 1.0
        f_wgt_end: float = 1.0
        v_wgt_start: float = 0.00
        v_wgt_end: float = 0.00
        lr_decay_step: int = max_epochs * 25

        self.lit_nep: LitNep = LitNep(type_map=type_map,
                                      energy_shifts=energy_shifts,
                                      umax_num_neigh_atoms=umax_num_neigh_atoms,
                                      fit_virial=fit_virial,
                                      n_radial_basis=n_radial_basis,
                                      n_angular_basis=n_angular_basis,
                                      l_max=l_max,
                                      chebyshev_size=chebyshev_size,
                                      num_neurons=num_neurons,
                                      rmax_radial=rmax_radial,
                                      rmax_angular=rmax_angular,
                                      zbl_rmax=0.0,
                                      zbl_rmin=0.0,
                                      lr_start=lr_start,
                                      lr_end=lr_end,
                                      e_wgt_start=e_wgt_start,
                                      e_wgt_end=e_wgt_end,
                                      f_wgt_start=f_wgt_start,
                                      f_wgt_end=f_wgt_end,
                                      v_wgt_start=v_wgt_start,
                                      v_wgt_end=v_wgt_end,
                                      lr_decay_step=lr_decay_step)

        ### DataModule hyperparameters
        rcut: float = rmax_radial
        pbc_xyz=[True, True, True]
        sort=False
        torch_float_dtype=torch.float32
        self.extxyz_datamodule: ExtxyzDataModule = ExtxyzDataModule(trainset_path=PbTe_EXTXYZ_PATH,
                                                               validset_path=PbTe_EXTXYZ_PATH,
                                                               testset_path=None,
                                                               predict_path=None,
                                                               batch_size=1,
                                                               rcut=rcut,
                                                               umax_num_neigh_atoms=umax_num_neigh_atoms,
                                                               pbc_xyz=pbc_xyz,
                                                               sort=sort,
                                                               torch_float_dtype=torch_float_dtype,
                                                               has_virial=fit_virial)

        ### NepDescriptorNormCallback initialization
        self.nep_descriptor_norm_callback: NepDescriptorNormCallback = NepDescriptorNormCallback()
    
        self.trainer: L.Trainer = L.Trainer(max_epochs=max_epochs,
                                            accelerator="cuda",
                                            devices=1,
                                            limit_val_batches=0,
                                            log_every_n_steps=1,
                                            callbacks=[self.nep_descriptor_norm_callback])


    def tearDown(self):
        print("LitNep (TestCase) is tearing down...\n")
    

    def test_train(self):
        self.trainer.fit(model=self.lit_nep,
                         datamodule=self.extxyz_datamodule)


if __name__ == "__main__":
    unittest.main()
