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


from typing import (List, Tuple, Optional, Callable)

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader
from scipy.optimize import minimize

from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.data.mlffdataset import ExtxyzDataset


class TorchScipyBfgs(object):
    def __init__(self,
                 linear_mtp: LinearMtp,
                 extxyz_dataset: ExtxyzDataset,
                 e_weight: float = 1.0,
                 f_weight: float = 2.0,
                 v_weight: float = 0.0,
                 fit_virial: bool = False,
                 batch_size: int = 300,
                 maxiter: int = 500,
                 gtol: float = 1e-7,
                 disp: bool = True,
                 use_best_params: bool = True):
        super(TorchScipyBfgs, self).__init__()
        self.linear_mtp: LinearMtp = linear_mtp
        self.extxyz_dataset: ExtxyzDataset = extxyz_dataset
        self.train_loader: DataLoader = DataLoader(dataset=self.extxyz_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        self.e_weight: float = e_weight
        self.f_weight: float = f_weight
        self.v_weight: float = v_weight
        self.fit_virial: bool = fit_virial
        self.batch_size: int = batch_size

        self.maxiter: int = maxiter
        self.gtol: float = gtol
        self.disp: bool = disp

        self.use_best_params: bool = use_best_params
        self.params: List[nn.Parameter] = [p for p in linear_mtp.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("Model has no trainable parameters.")
        
        self.device: torch._C.device = self.params[0].device
        self.torch_float_dtype: torch._C.dtype = self.params[0].dtype
        self.npy_float_dtype: np.dtype = np.float32
        if (self.torch_float_dtype == torch.float64):
            self.npy_float_dtype = np.float64
        self.best_loss = np.inf
        self.best_x: Optional[np.ndarray] = None


    def _get_x0(self) -> np.ndarray:
        x0: np.ndarray = parameters_to_vector(self.params)
        return x0.detach().cpu().numpy()
    

    def _set_x(self, x: np.ndarray) -> None:
        x_tensor: torch.Tensor = torch.from_numpy(x).to(device=self.device, dtype=self.torch_float_dtype)
        with torch.no_grad():
            vector_to_parameters(vec=x_tensor,
                                 parameters=self.params)


    def _loss_and_grad(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        # 1.
        self._set_x(x=x)
        self.linear_mtp.zero_grad()

        # 2.
        total_loss: float = 0.0
        total_samples: int = len(self.train_loader.dataset)
        for batch_idx, batch_data in enumerate(self.train_loader):
            if self.fit_virial:
                binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = \
                    [t.to(self.device) for t in batch_data]
                bmse_tensor, e_rmse_tensor, f_rmse_tensor, v_rmse_tensor = self.linear_mtp.predict_loss(
                    self.e_weight,
                    self.f_weight,
                    self.v_weight,
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
                    [t.to(self.device) for t in batch_data]
                bmse_tensor, e_rmse_tensor, f_rmse_tensor = self.linear_mtp.predict_ef_loss(
                    self.e_weight,
                    self.f_weight,
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
            loss: torch.Tensor = bmse_tensor.mean() * weight
            total_loss += loss.item()
            loss.backward()

        # 
        grad_tensors_list: List[torch.Tensor] = [
            p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel(), device=p.device, dtype=self.torch_float_dtype)
            for p in self.linear_mtp.parameters()
        ]
        grad: torch.Tensor = torch.cat(grad_tensors_list).detach().cpu().numpy()

        return float(total_loss), grad
    

    def run(self):
        result = minimize(
            fun=self._loss_and_grad,
            x0=self._get_x0(),
            method="BFGS",
            jac=True,
            options={
                "maxiter": self.maxiter,
                "gtol": self.gtol,
                "disp": self.disp})

        if self.use_best_params and self.best_x is not None:
            self._set_x(self.best_x)
        else:
            self._set_x(result.x)
        
        return result
    

