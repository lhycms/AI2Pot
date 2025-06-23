from typing import List
import unittest
import os
from pymatgen.core import Structure

import torch
import lightning as L

from ai2pot.data import ExtxyzDataset
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.utils.usepot import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")


class LitLinearMtpTest(unittest.TestCase):
    def setUp(self):
        print("LitLinearMtp (TestCase) is setting up...\n")
        mtp_level: int = 12
        ntypes: int = 2
        chebyshev_size: int = 8
        rmax: float = 5.0
        rmin: float = 0.5
        umax_num_neighs: int = 100
        fit_virial: bool = False
        batch_size: int = 1

        trainset_path: str = PbTe_EXTXYZ_PATH
        rcut: float = 6.0
        pbc_xyz: List[bool] = [True, True, True]
        sort: bool = True
        torch_float_dtype: torch._C.dtype = torch.float64
        lr_decay_epoch:int = 30
        
        type_map: torch.Tensor = ExtxyzDataset.get_type_map(filename=PbTe_EXTXYZ_PATH)

        self.lit_potential_to_loss: LitLinearMtp = LitLinearMtp(mtp_level=mtp_level,
                                                                type_map_tensor=type_map,
                                                                chebyshev_size=chebyshev_size,
                                                                rmax=rmax,
                                                                rmin=rmin,
                                                                umax_num_neigh_atoms=umax_num_neighs,
                                                                fit_virial=fit_virial,
                                                                trainset_path=trainset_path,
                                                                validset_path=None,
                                                                testset_path=None,
                                                                rcut=rcut,
                                                                umax_num_neighs=umax_num_neighs,
                                                                pbc_xyz=pbc_xyz,
                                                                sort=sort,
                                                                torch_float_dtype=torch_float_dtype,
                                                                has_virial=fit_virial,
                                                                lr_start=1e-1,
                                                                lr_end=1e-3,
                                                                e_wgt_start=1.0,
                                                                e_wgt_end=1.0,
                                                                f_wgt_start=0.1,
                                                                f_wgt_end=0.1,
                                                                v_wgt_start=0.0,
                                                                v_wgt_end=0.0,
                                                                lr_decay_epoch=lr_decay_epoch)
        self.trainer: L.Trainer = L.Trainer(max_epochs=500,
                                            accelerator="cpu",
                                            devices=1,
                                            limit_val_batches=0)

        
        ###
        self.structure: Structure = Structure.from_file("/data/home/liuhanyu/mycode/AI2Pot/test/test_data/POSCARs/POSCAR")
        self.mlff_input: MlffInput = MlffInput(type_map=type_map.numpy().tolist(),
                                               rcut=rcut,
                                               umax_num_neighs=umax_num_neighs)

    
    
    def tearDown(self):
        print("LitLinearMtp (TestCase) is tearing down...\n")
    
    
    def test_train(self):
        self.trainer.fit(model=self.lit_potential_to_loss)
    


if __name__ == "__main__":
    unittest.main()
    