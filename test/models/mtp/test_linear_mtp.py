import unittest
import os

import torch
from torch.utils.data import DataLoader
import lightning as L

from ai2pot.data import ExtxyzDataset
from ai2pot.models.mtp import LinearMtp
from ai2pot.models.potential_train import LitPotentialToLoss


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")


class LinearMtpTest(unittest.TestCase):
    def setUp(self):
        print("LinearMtp (TestCase) is setting up...\n")
        mtp_level: int = 18
        ntypes: int = 2
        chebyshev_size: int = 8
        rmax: float = 5.0
        rmin: float = 0.5
        umax_num_neighs: int = 100
        fit_virial: bool = False
        torch_float_dtype: torch._C.dtype = torch.float32
        lr_decay_epoch:int = 30
        
        self.linear_mtp: LinearMtp = LinearMtp(mtp_level=mtp_level,
                                               ntypes=ntypes,
                                               chebyshev_size=chebyshev_size,
                                               rmax=rmax,
                                               rmin=rmin,
                                               umax_num_neighs=umax_num_neighs,
                                               fit_virial=fit_virial)
        self.mlff_dataset: ExtxyzDataset = ExtxyzDataset(filename=PbTe_EXTXYZ_PATH,
                                                         rcut=rmax,
                                                         umax_num_neigh_atoms=umax_num_neighs,
                                                         pbc_xyz=[True, True, True],
                                                         sort=False,
                                                         torch_float_dtype=torch_float_dtype,
                                                         has_virial=fit_virial)
        self.mlff_dataloader: DataLoader = DataLoader(dataset=self.mlff_dataset,
                                                      batch_size=1,
                                                      shuffle=True)
        self.lit_potential_to_loss: LitPotentialToLoss = LitPotentialToLoss(model=self.linear_mtp,
                                                                            lr_start=1e-3,
                                                                            lr_end=1e-5,
                                                                            e_wgt_start=1.0,
                                                                            e_wgt_end=1.0,
                                                                            f_wgt_start=0.0,
                                                                            f_wgt_end=0.0,
                                                                            v_wgt_start=0.0,
                                                                            v_wgt_end=0.0,
                                                                            lr_decay_epoch=lr_decay_epoch)
        self.trainer: L.Trainer = L.Trainer(max_epochs=30,
                                            accelerator="cpu",
                                            devices=1)
    
    def test_train(self):
        self.trainer.fit(model=self.lit_potential_to_loss,
                         train_dataloaders=self.mlff_dataloader)


if __name__ == "__main__":
    unittest.main()
