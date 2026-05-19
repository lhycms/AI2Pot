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

from typing import (List, Tuple, Dict, Optional)

import copy
from pymatgen.core import Element
import torch
from torch.utils.data import DataLoader
import lightning as L

from ai2pot.data import ExtxyzDataModule, ExtxyzDataset
from ai2pot.models.nep.nep import Nep
from ai2pot.fromcc import allTypeDescriptorsMaxminOp
from ai2pot.utils.prepot import MlffDatasetShifter


class DescriptorNormCallbackBase(L.Callback):
    BATCH_SIZE_HERE: int = 500
    
    def on_fit_start(self,
                     trainer: L.Trainer,
                     pl_module: L.LightningModule):
        print("🚀🚀🚀 Performing descriptor normalization before training --- Starting...")

        datamodule: ExtxyzDataModule = trainer.datamodule
        nep: Nep = pl_module.model
        nep_stat: Nep = copy.deepcopy(nep)
        if datamodule is None:
            raise RuntimeError("No ExtxyzDataModule found!")
        trainset: ExtxyzDataset = datamodule.trainset_dataset
        train_loader: DataLoader = DataLoader(dataset=trainset,
                                              batch_size=self.BATCH_SIZE_HERE,
                                              shuffle=False)
        num_descriptors: int = nep.get_num_descriptors()
        device: torch._C.device = pl_module.device
        torch_float_dtype: torch._C.dtype = pl_module.dtype

        max_descriptor_tensor: torch.Tensor = torch.zeros(num_descriptors, 
                                                          dtype=torch_float_dtype,
                                                          device=device) - 1e3
        min_descriptor_tensor: torch.Tensor = torch.zeros(num_descriptors, 
                                                          dtype=torch_float_dtype,
                                                          device=device) + 1e3
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(train_loader):
                input_tensors: List[torch.Tensor] = batch_data
                binum = input_tensors[0].to(device)
                bilist = input_tensors[1].to(device)
                bnumneigh = input_tensors[2].to(device)
                bfirstneigh = input_tensors[3].to(device)
                brcs = input_tensors[4].to(device, torch_float_dtype)
                btypes = input_tensors[5].to(device)
                bnghost = input_tensors[6].to(device)

                bdescriptor_tensor: torch.Tensor = nep_stat.predict_descriptors(binum, 
                                                                                bilist, 
                                                                                bnumneigh, 
                                                                                bfirstneigh, 
                                                                                brcs, 
                                                                                btypes, 
                                                                                bnghost)
                
                tmp_max_descriptor_tensor, tmp_min_descriptor_tensor = allTypeDescriptorsMaxminOp(binum, bdescriptor_tensor)
                
                max_descriptor_tensor = torch.max(max_descriptor_tensor, tmp_max_descriptor_tensor)
                min_descriptor_tensor = torch.min(min_descriptor_tensor, tmp_min_descriptor_tensor)

        q_scaler_tensor: torch.Tensor = max_descriptor_tensor - min_descriptor_tensor
        pl_module.model.q_scaler_tensor.copy_(q_scaler_tensor)
        print("\t * Number of descriptor = ", q_scaler_tensor.size(0))
        print("🚀🚀🚀 Performing descriptor normalization before training --- End!")


class EnergyShiftCallback(L.Callback):
    def on_fit_start(self,
                     trainer: L.Trainer,
                     pl_module: L.LightningModule):
        print("🚀🚀🚀 Performing energy shift before training --- Starting...")
        datamodule: ExtxyzDataModule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("No ExtxyzDataModule found!")
        trainset: ExtxyzDataset = datamodule.trainset_dataset
        energy_shifter: MlffDatasetShifter = MlffDatasetShifter(extxyz_dataset=trainset)
        
        for ii in range(energy_shifter.ntypes):
            print("\t * Type {0:3d} ({1:3s}) = {2:.4f}".format(
                ii,
                Element.from_Z(energy_shifter.type_map[ii]),
                energy_shifter.types_energy_shifts[ii]))
        with torch.no_grad():
            types_energy_shifts_tensor: torch.Tensor = torch.as_tensor(energy_shifter.types_energy_shifts,
                                                                       dtype=pl_module.dtype,
                                                                       device=pl_module.device)
            noise: torch.Tensor = torch.randn_like(types_energy_shifts_tensor) * 0.01
            pl_module.model.type_bias_tensor.copy_(types_energy_shifts_tensor+noise)
        print("🚀🚀🚀 Performing energy shift before training --- End...")
