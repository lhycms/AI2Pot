from typing import List
import unittest
import os
from pymatgen.core import Structure

import torch
import lightning as L
from lightning.pytorch.loggers import CSVLogger

from ai2pot.data import ExtxyzDataset, ExtxyzDataModule
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.utils.prepot import ExtxyzShifter
from ai2pot.utils.usepot import MlffInput
from ai2pot.models.mtp.linear_mtp_train_utils import LinearMtpDescriptorNormCallback
from ai2pot.models.potential_train_utils import EnergyShiftCallback


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
PbTe_EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
# os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")

torch.manual_seed(42)
torch.set_num_threads(16)


class LitLinearMtpTest(unittest.TestCase):
    def setUp(self):
        print("LitLinearMtp (TestCase) is setting up...\n")
        accelerator: str = "cuda"
        torch_float_dtype: torch._C.dtype = torch.float32

        type_map: List[int] = ExtxyzDataset.get_type_map(filename=PbTe_EXTXYZ_PATH)
        umax_num_neigh_atoms: int = 200
        fit_virial: bool = False

        # LinearMtp hyperparameters
        mtp_level: int = 18
        chebyshev_size: int = 8
        rmax: float = 5.0
        rmin: float = 0.0
        
        # Lr hyperparameters
        max_epochs: int = 200
        lr_start: float = 1e-3
        lr_end: float = 1e-5
        e_wgt_start: float = 1.0
        e_wgt_end: float = 1.0
        f_wgt_start: float = 2.0
        f_wgt_end: float = 2.0
        v_wgt_start: float = 0.00
        v_wgt_end: float = 0.00
        max_clip_norm: float = 10.0

        ### LitghtingModule hyperparameters
        self.lit_linear_mtp: LitLinearMtp = LitLinearMtp(
            type_map=type_map,
            umax_num_neigh_atoms=umax_num_neigh_atoms,
            fit_virial=fit_virial,
            mtp_level=mtp_level,
            chebyshev_size=chebyshev_size,
            rmax=rmax,
            rmin=rmin,
            lr_start=lr_start,
            lr_end=lr_end,
            e_wgt_start=e_wgt_start,
            e_wgt_end=e_wgt_end,
            f_wgt_start=f_wgt_start,
            f_wgt_end=f_wgt_end,
            v_wgt_start=v_wgt_start,
            v_wgt_end=v_wgt_end,
            max_clip_norm=max_clip_norm).to(torch_float_dtype)
        
        ### DataModule hyperparameters
        rcut: float = rmax
        pbc_xyz: List[bool] = [True, True, True]
        sort: bool = False
        self.extxyz_datamodule: ExtxyzDataModule = ExtxyzDataModule(
            trainset_path=PbTe_EXTXYZ_PATH,
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
        
        ### Trainer hyperparameters
        csv_logger: CSVLogger = CSVLogger(save_dir="lightning_logs")
        self.linear_mtp_descriptor_norm_callback: LinearMtpDescriptorNormCallback = LinearMtpDescriptorNormCallback()
        self.energy_shift_callback: EnergyShiftCallback = EnergyShiftCallback()
        self.trainer: L.Trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=1,
            limit_val_batches=0,
            logger=csv_logger,
            log_every_n_steps=1,
            callbacks=[
                self.linear_mtp_descriptor_norm_callback,
                self.energy_shift_callback
                ])

    
    def tearDown(self):
        print("LitLinearMtp (TestCase) is tearing down...\n")
    
    
    def test_train(self):
        self.trainer.fit(
            model=self.lit_linear_mtp,             
            datamodule=self.extxyz_datamodule)
    

if __name__ == "__main__":
    unittest.main()
    