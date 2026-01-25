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
import torch
from torch.utils.data import DataLoader
import lightning as L

from ai2pot.data import ExtxyzDataModule, ExtxyzDataset
from ai2pot.models.nep.nep import Nep
from ai2pot.fromcc import allTypeDescriptorsMaxminOp


class NepDescriptorNormCallback(L.Callback):
    def on_fit_start(self,
                     trainer: L.Trainer,
                     pl_module: L.LightningModule):
        print("🚀🚀🚀 Performing nep descriptor normalization before training --- Starting...")

        datamodule: ExtxyzDataModule = trainer.datamodule
        nep: Nep = pl_module.model
        nep_stat: Nep = copy.deepcopy(nep)
        nep_stat.init_one()
        if datamodule is None:
            raise RuntimeError("No ExtxyzDataModule found!")
        trainset: ExtxyzDataset = datamodule.trainset_dataset
        train_loader: DataLoader = DataLoader(dataset=trainset,
                                              batch_size=500,
                                              shuffle=False)
        num_descriptors: int = nep.get_num_descriptors()
        device: torch._C.device = pl_module.device
        torch_float_dtype: torch.float32 = pl_module.dtype

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
        pl_module.model.register_buffer("_q_scaler_tensor", tensor=q_scaler_tensor)
        print("🚀🚀🚀 Performing nep descriptor normalization before training --- End!")
    