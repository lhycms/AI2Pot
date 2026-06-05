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


from typing import (List, Tuple, Optional)
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
from scipy.optimize import minimize

from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.data.mlffdataset import ExtxyzDataset
from ai2pot.optimizer.mtpr_solver import LinearMtpSolver
from ai2pot.optimizer.laddar_bfgs import (TorchScipyBfgs, 
                                          ParameterInheritor,
                                          LaddarTrainer)


class CallbackBase(object):
    def on_train_start(self,
                       lit_linear_mtp: LitLinearMtp,
                       trainset: ExtxyzDataset):
        pass

    def on_fit_start(self,
                     lit_linear_mtp: LitLinearMtp,
                     trainset: ExtxyzDataset):
        pass

    def on_train_step_end(self,
                          lit_linear_mtp: LitLinearMtp,
                          trainset: ExtxyzDataset):
        pass

    def on_fit_end(self,
                   lit_linear_mtp: LitLinearMtp,
                   trainset: ExtxyzDataset):
        pass


    def on_train_end(self,
                     lit_linear_mtp: LitLinearMtp,
                     trainset: ExtxyzDataset):
        pass



class CallbackManager(object):
    def __init__(self,
                 callbacks: Optional[List[CallbackBase]] = None):
        self.callbacks: Optional[List[CallbackBase]] = callbacks
    

    def on_train_start(self,
                       lit_linear_mtp: LitLinearMtp,
                       trainset: ExtxyzDataset):
        for cb in self.callbacks:
            cb.on_train_start(lit_linear_mtp=lit_linear_mtp, trainset=trainset)
    

    def on_fit_start(self,
                     lit_linear_mtp: LitLinearMtp,
                     trainset: ExtxyzDataset):
        for cb in self.callbacks:
            cb.on_fit_start(lit_linear_mtp=lit_linear_mtp, trainset=trainset)
    

    def on_train_step_end(self,
                          lit_linear_mtp: LitLinearMtp,
                          trainset: ExtxyzDataset):
        for cb in self.callbacks:
            cb.on_train_step_end(lit_linear_mtp=lit_linear_mtp, trainset=trainset)
    

    def on_fit_end(self,
                   lit_linear_mtp: LitLinearMtp,
                   trainset: ExtxyzDataset):
        for cb in self.callbacks:
            cb.on_fit_end(lit_linear_mtp=lit_linear_mtp, trainset=trainset)
    

    def on_train_end(self,
                     lit_linear_mtp: LitLinearMtp,
                     trainset: ExtxyzDataset):
        for cb in self.callbacks:
            cb.on_train_end(lit_linear_mtp=lit_linear_mtp, trainset=trainset)



class PrintEFVLossCallback(CallbackBase):
    BATCH_SIZE_HERE: int = 800

    def __init__(self):
        super(PrintEFVLossCallback, self).__init__()


    @torch.no_grad()
    def on_train_step_end(self,
                          lit_linear_mtp: LitLinearMtp,
                          trainset: ExtxyzDataset):
        #
        linear_mtp: LinearMtp = lit_linear_mtp.model
        fit_virial: bool = linear_mtp.fit_virial
        e_weight: float = lit_linear_mtp.e_wgt_start
        f_weight: float = lit_linear_mtp.f_wgt_start
        v_weight: float = lit_linear_mtp.v_wgt_start

        device: torch._C.device = lit_linear_mtp.device
        torch_float_dtype: torch._C.dtype = lit_linear_mtp.dtype
        
        dataloader: DataLoader = DataLoader(dataset=trainset,
                                            batch_size=PrintEFVLossCallback.BATCH_SIZE_HERE,
                                            shuffle=False)

        #
        total_e_rmse: float = 0.0
        total_f_rmse: float = 0.0
        total_v_rmse: float = 0.0
        total_samples: int = len(dataloader)
        for batch_idx, batch_data in enumerate(dataloader):
            if fit_virial:
                binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = \
                    [t.to(device) for t in batch_data]
                brcs = brcs.to(torch_float_dtype)
                betot_dft_tensor = betot_dft_tensor.to(torch_float_dtype)
                bforce_dft_tensor = bforce_dft_tensor.to(torch_float_dtype)
                bvirial_dft_tensor = bvirial_dft_tensor.to(torch_float_dtype)
                bmse_tensor, e_rmse_tensor, f_rmse_tensor, v_rmse = linear_mtp.predict_loss(
                    e_weight,
                    f_weight,
                    v_weight,
                    betot_dft_tensor,
                    bforce_dft_tensor,
                    bvirial_dft_tensor,
                    binum,
                    bilist,
                    bnumneigh,
                    bfirstneigh,
                    brcs,
                    btypes,
                    bnghost)
                bmse_tensor: torch.Tensor
                e_rmse_tensor: torch.Tensor
                f_rmse_tensor: torch.Tensor
                v_rmse_tensor: torch.Tensor
            else:
                binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = \
                    [t.to(device) for t in batch_data]
                brcs = brcs.to(torch_float_dtype)
                betot_dft_tensor = betot_dft_tensor.to(torch_float_dtype)
                bforce_dft_tensor = bforce_dft_tensor.to(torch_float_dtype)
                bmse_tensor, e_rmse_tensor, f_rmse_tensor = linear_mtp.predict_ef_loss(
                    e_weight,
                    f_weight,
                    betot_dft_tensor,
                    bforce_dft_tensor,
                    binum,
                    bilist,
                    bnumneigh,
                    bfirstneigh,
                    brcs,
                    btypes,
                    bnghost)
                bmse_tensor: torch.Tensor
                e_rmse_tensor: torch.Tensor
                f_rmse_tensor: torch.Tensor
            
            current_batch_size: int = betot_dft_tensor.size(0)
            weight: float = current_batch_size / total_samples
            total_e_rmse += e_rmse_tensor.mean().item() * weight
            total_f_rmse += f_rmse_tensor.mean().item() * weight
            if fit_virial:
                total_v_rmse += v_rmse_tensor.mean().item() * weight
        
        if fit_virial:
            print(f"e_rmse: {total_e_rmse:.4f}, f_rmse: {total_f_rmse}, v_rmse: {total_v_rmse}")
        else:
            print(f"e_rmse: {total_e_rmse:.4f}, f_rmse: {total_f_rmse}")
