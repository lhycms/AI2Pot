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
PbTe_EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/gst/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
# os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train.xyz")
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/gst/train.xyz"

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
        max_epochs: int = 1
        lr_start: float = 1e-2
        lr_end: float = 1e-4
        e_wgt_start: float = 0.1
        e_wgt_end: float = 2.0
        f_wgt_start: float = 10.0
        f_wgt_end: float = 1.0
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
            max_clip_norm=max_clip_norm,
            zbl_rmax=0.0,
            zbl_rmin=0.0).to(torch_float_dtype)
        
        ### DataModule hyperparameters
        batch_size: int = 16
        rcut: float = rmax
        pbc_xyz: List[bool] = [True, True, True]
        sort: bool = False
        self.extxyz_datamodule: ExtxyzDataModule = ExtxyzDataModule(
            trainset_path=PbTe_EXTXYZ_PATH,
            validset_path=PbTe_EXTXYZ_PATH,
            testset_path=None,
            predict_path=None,
            batch_size=batch_size,
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
    

    def print_cuda_memory(self, prefix: str = ""):
        if not torch.cuda.is_available():
            print(f"{prefix} CUDA is not available.")
            return

        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**2
        reserved = torch.cuda.memory_reserved(device) / 1024**2
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2

        print(
            f"\n[{prefix}] CUDA memory on device {device}:\n"
            f"  allocated      = {allocated:.2f} MiB\n"
            f"  reserved       = {reserved:.2f} MiB\n"
            f"  cached/free    = {reserved - allocated:.2f} MiB\n"
            f"  max allocated  = {max_allocated:.2f} MiB\n"
            f"  max reserved   = {max_reserved:.2f} MiB\n"
        )


    def test_train(self):
        torch.cuda.reset_peak_memory_stats()
        self.print_cuda_memory("before fit")

        ## run 
        self.trainer.fit(
            model=self.lit_linear_mtp,             
            datamodule=self.extxyz_datamodule)
        
        torch.cuda.synchronize()
        self.print_cuda_memory("after fit")
        

    

if __name__ == "__main__":
    unittest.main()
    