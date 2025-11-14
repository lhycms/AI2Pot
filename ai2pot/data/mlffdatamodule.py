# Copyright (C) 2025 Hanyu Liu
#
# This file is part of AI2Pot.
#
# AI2Pot is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI2Pot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.


from typing import List, Union, Optional
import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from ase import Atoms

from ai2pot.data.mlffdataset import (ExtxyzDataset)


class ExtxyzDataModule(LightningDataModule):
    def __init__(self,
                 trainset_path: Union[str, None] = None,
                 validset_path: Union[str, None] = None,
                 testset_path: Union[str, None] = None,
                 predict_path: Union[str, None] = None,
                 batch_size: int = 1,
                 rcut: float = 6.0,
                 umax_num_neigh_atoms: int = 200,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = True,
                 torch_float_dtype: torch._C.dtype = torch.float32,
                 has_virial: bool = False):
        super(ExtxyzDataModule, self).__init__()
        self.save_hyperparameters()
        
        self.trainset_path: str = trainset_path
        self.validset_path: str = validset_path
        self.testset_path: str = testset_path
        self.predict_path: str = predict_path
        self.batch_size: int = batch_size
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool = sort
        self.torch_float_dtype: torch._C.dtype = torch_float_dtype
        self.has_virial: bool = has_virial


    def setup(self, stage: Optional[str] = None):
        if (stage == "fit") or (stage is None):
            self.trainset_dataset = ExtxyzDataset(filename=self.trainset_path,
                                                  rcut=self.rcut,
                                                  umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                  pbc_xyz=self.pbc_xyz,
                                                  sort=self.sort,
                                                  torch_float_dtype=self.torch_float_dtype,
                                                  has_virial=self.has_virial)
            self.validset_dataset = ExtxyzDataset(filename=self.validset_path,
                                                  rcut=self.rcut,
                                                  umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                  pbc_xyz=self.pbc_xyz,
                                                  sort=self.sort,
                                                  torch_float_dtype=self.torch_float_dtype,
                                                  has_virial=self.has_virial)
        if (stage == "test") or (stage is None):
            self.testset_dataset = ExtxyzDataset(filename=self.testset_path,
                                                 rcut=self.rcut,
                                                 umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                 pbc_xyz=self.pbc_xyz,
                                                 sort=self.sort,
                                                 torch_float_dtype=self.torch_float_dtype,
                                                 has_virial=self.has_virial)
        if (stage == "predict") or (stage is None):
            self.predictset_dataset = ExtxyzDataset(filename=self.predict_path,
                                                    rcut=self.rcut,
                                                    umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                    pbc_xyz=self.pbc_xyz,
                                                    sort=self.sort,
                                                    torch_float_dtype=self.torch_float_dtype,
                                                    has_virial=self.has_virial)


    def train_dataloader(self):
        # trainer.fit(model, datamodule=dm)
        return DataLoader(dataset=self.trainset_dataset,
                          batch_size=self.batch_size,
                          shuffle=True)
    

    def val_dataloader(self):
        # trainer.fit(model, datamodule=dm)
        return DataLoader(dataset=self.validset_dataset,
                          batch_size=self.batch_size,
                          shuffle=True)
    

    def test_dataloader(self):
        # trainer.test(model, datamodule=dm)
        return DataLoader(dataset=self.testset_dataset,
                          batch_size=self.batch_size,
                          shuffle=True)

    
    def predict_dataloader(self):
        # trainer.predict(model, datamodule=dm)
        return DataLoader(dataset=self.predictset_dataset,
                          batch_size=self.batch_size,
                          shuffle=True)
    

