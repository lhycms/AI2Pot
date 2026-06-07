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


class TorchScipyBfgs(object):
    BATCH_SIZE_HERE = 500

    def __init__(self,
                 lit_linear_mtp: LitLinearMtp,
                 trainset: ExtxyzDataset,
                 maxiter: int = 500):
        super(TorchScipyBfgs, self).__init__()
        self.lit_linear_mtp: LitLinearMtp = lit_linear_mtp
        self.linear_mtp: LinearMtp = self.lit_linear_mtp.model
        self.trainset: ExtxyzDataset = trainset
        self.train_loader: DataLoader = DataLoader(dataset=self.trainset,
                                                   batch_size=TorchScipyBfgs.BATCH_SIZE_HERE,
                                                   shuffle=False)
        self.e_weight: float = self.lit_linear_mtp.e_wgt_start
        self.f_weight: float = self.lit_linear_mtp.f_wgt_start
        self.v_weight: float = self.lit_linear_mtp.v_wgt_start
        self.fit_virial: bool = self.linear_mtp.fit_virial

        # Device and dtype
        self.params: List[nn.Parameter] = [p for p in self.linear_mtp.parameters() if p.requires_grad]
        if not self.params:
            raise ValueError("Model has no trainable parameters.")
        
        self.device: torch._C.device = self.params[0].device
        self.torch_float_dtype: torch._C.dtype = self.params[0].dtype
        self.npy_float_dtype: np.dtype = np.float32
        if (self.torch_float_dtype == torch.float64):
            self.npy_float_dtype = np.float64

        # BFGS
        self.maxiter: int = maxiter
        self.gtol: float = 1e-5 if self.torch_float_dtype == torch.float32 else 1e-7
        self.disp: bool = True

        ### Assertion
        if (self.e_weight != self.lit_linear_mtp.e_wgt_end) or \
            (self.f_weight != self.lit_linear_mtp.f_wgt_end) or \
            (self.v_weight != self.lit_linear_mtp.v_wgt_end):
            raise ValueError(
                f"BFGS loss weights must match the final weights. "
                f"Got (e_wgt_start={self.e_weight}, f_wgt_start={self.f_weight}, v_wgt_start={self.v_weight}), "
                f"but expected "
                f"(e_wgt_end={self.lit_linear_mtp.e_wgt_end}, f_wgt_end={self.lit_linear_mtp.f_wgt_end}, v_wgt_end={self.lit_linear_mtp.v_wgt_end})."
            )


    @torch.no_grad()
    def _get_x0(self) -> np.ndarray:
        x0: torch.Tensor = parameters_to_vector(self.params)
        return x0.detach().cpu().numpy()
    

    def _set_x(self, x: np.ndarray) -> None:
        x_tensor: torch.Tensor = torch.from_numpy(x).to(device=self.device, dtype=self.torch_float_dtype)
        with torch.no_grad():
            pointer: int = 0
            for param in self.params:
                num_params: int = param.numel()
                param.copy_(x_tensor[pointer: pointer+num_params].view_as(param))
                pointer += num_params


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
                brcs = brcs.to(self.torch_float_dtype)
                betot_dft_tensor = betot_dft_tensor.to(self.torch_float_dtype)
                bforce_dft_tensor = bforce_dft_tensor.to(self.torch_float_dtype)
                bvirial_dft_tensor = bvirial_dft_tensor.to(self.torch_float_dtype)
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
                brcs = brcs.to(self.torch_float_dtype)
                betot_dft_tensor = betot_dft_tensor.to(self.torch_float_dtype)
                bforce_dft_tensor = bforce_dft_tensor.to(self.torch_float_dtype)
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
            for p in self.params
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
        
        self._set_x(x=result.x)

        return result
    

class ParameterInheritor(object):
    def __init__(self,
                 old_lit_model: LitLinearMtp,
                 new_lit_model: LitLinearMtp,
                 trainset: ExtxyzDataset,
                 ridge_lambda: float = 1e-2):
        super(ParameterInheritor, self).__init__()
        self.old_lit_model: LitLinearMtp = old_lit_model
        self.new_lit_model: LitLinearMtp = new_lit_model
        self.old_model: LinearMtp = self.old_lit_model.model
        self.new_model: LinearMtp = self.new_lit_model.model
        self.trainset: ExtxyzDataset = trainset
        self.e_weight: float = self.old_lit_model.e_wgt_start
        self.f_weight: float = self.old_lit_model.f_wgt_start
        self.v_weight: float = self.old_lit_model.v_wgt_start
        self.ridge_lambda: float = ridge_lambda

        ### Assertion
        if (self.old_lit_model.e_wgt_start != self.old_lit_model.e_wgt_end) or \
            (self.old_lit_model.f_wgt_start != self.old_lit_model.f_wgt_end) or \
            (self.old_lit_model.v_wgt_start != self.old_lit_model.v_wgt_end):
            raise ValueError(
                f"ParameterInheritor loss weights of old_lit_model must match the final weights. "
                f"Got (e_wgt_start={self.old_lit_model.e_wgt_start}, f_wgt_start={self.old_lit_model.f_wgt_start}, v_weight_start={self.old_lit_model.v_wgt_start}), "
                f"but expected "
                f"(e_wgt_end={self.old_lit_model.e_wgt_end}, f_wgt_end={self.old_lit_model.f_wgt_end}, v_wgt_end={self.old_lit_model.v_wgt_end})."
            )
        if (self.new_lit_model.e_wgt_start != self.new_lit_model.e_wgt_end) or \
            (self.new_lit_model.f_wgt_start != self.new_lit_model.f_wgt_end) or \
            (self.new_lit_model.v_wgt_start != self.new_lit_model.v_wgt_end):
            raise ValueError(
                f"ParameterInheritor loss weights of new_lit_model must match the final weights. "
                f"Got (e_wgt_start={self.new_lit_model.e_wgt_start}, f_wgt_start={self.new_lit_model.f_wgt_start}, v_weight_start={self.new_lit_model.v_wgt_start}), "
                f"but expected "
                f"(e_wgt_end={self.new_lit_model.e_wgt_end}, f_wgt_end={self.new_lit_model.f_wgt_end}, v_wgt_end={self.new_lit_model.v_wgt_end})."
            )
        if (self.old_lit_model.e_wgt_start != self.new_lit_model.e_wgt_start) or \
            (self.old_lit_model.f_wgt_start != self.new_lit_model.f_wgt_start) or \
            (self.old_lit_model.v_wgt_start != self.new_lit_model.v_wgt_start):
            raise ValueError(
                f"ParameterInheritor loss weights of old_lit_model must match the new_lit_models'. "
                f"Got (old.e_weight={self.old_lit_model.e_wgt_start}, old.f_weight={self.old_lit_model.f_wgt_start}, old.v_weight={self.old_lit_model.v_wgt_start}), "
                f"but expected "
                f"(new.e_weight={self.new_lit_model.e_wgt_start}, new.f_weight={self.new_lit_model.f_wgt_start}, new.v_weight={self.new_lit_model.v_wgt_start})."
            )

    
    @torch.no_grad()
    def transfer(self):
        self.new_model._init_all_zeros()
        # 1. coeffs_tensor
        old_coeffs_tensor: torch.Tensor = self.old_model.coeffs_tensor
        new_coeffs_tensor: torch.Tensor = self.new_model.coeffs_tensor
        if (old_coeffs_tensor.numel() > new_coeffs_tensor.numel()):
            raise ValueError(
                f"old_coeffs_tensor.numel()={old_coeffs_tensor.numel()}"
                f"> new_coeffs_tensor.numel()={new_coeffs_tensor.numel()}.")
        ntypes: int = self.old_model.ntypes
        old_nmus: int = self.old_model.nmus
        new_nmus: int = self.new_model.nmus
        chebyshev_size: int = self.old_model.chebyshev_size
        num_pairs: int = ntypes * ntypes
        
        ## 1.1. 
        old_coeffs_view: torch.Tensor = old_coeffs_tensor.view(num_pairs, old_nmus, chebyshev_size)
        new_coeffs_view: torch.Tensor = new_coeffs_tensor.view(num_pairs, new_nmus, chebyshev_size)
        new_coeffs_view[:, :old_nmus, :].copy_(old_coeffs_view)

        ## 1.2.
        new_coeffs_view= new_coeffs_tensor.view(ntypes, ntypes, new_nmus, chebyshev_size)
        for tmp_mu in range(old_nmus, new_nmus):
            idx: int = tmp_mu if tmp_mu < chebyshev_size else 0
            new_coeffs_view[:, :, tmp_mu, idx] = 1.0

        # 2. linear_coeffs_tensor && type_bias_tensor
        linear_mtp_solver: LinearMtpSolver = LinearMtpSolver(lit_linear_mtp=self.new_lit_model,
                                                             trainset=self.trainset,
                                                             ridge_lambda=self.ridge_lambda)
        linear_mtp_solver.solve_linear_equation()

        # 3. buffers for normalization
        self.new_model.q_scaler_tensor[:self.old_model.get_num_descriptors()].copy_(self.old_model.q_scaler_tensor)
        self.new_model.conv_energy_tensor.copy_(self.old_model.conv_energy_tensor)
        self.new_model.conv_length_tensor.copy_(self.old_model.conv_length_tensor)


class LaddarTrainer(object):
    def __init__(self,
                 lit_linear_mtp: LitLinearMtp,
                 trainset: ExtxyzDataset,
                 laddar_start: int = 6,
                 laddar_step: int = 2,
                 maxiter: int = 500,
                 ridge_lambda: float = 1e-2):
        super(LaddarTrainer, self).__init__()
        self.lit_linear_mtp: LitLinearMtp = lit_linear_mtp
        self.linear_mtp: LinearMtp = self.lit_linear_mtp.model
        self.trainset: ExtxyzDataset = trainset
        self.e_weight: float = self.lit_linear_mtp.e_wgt_start
        self.f_weight: float = self.lit_linear_mtp.f_wgt_start
        self.v_weight: float = self.lit_linear_mtp.v_wgt_start

        # 1.
        self.laddar_start: int = laddar_start
        self.laddar_end: int = self.linear_mtp.mtp_level
        self.laddar_step: int = laddar_step
        self.mtp_levels_list: List[int] = [*range(self.laddar_start, self.laddar_end, self.laddar_step)]
        ## 1.1
        if self.laddar_start % 2 != 0:
            raise ValueError(f"laddar_start must be an even number. Current value: {self.laddar_start}")
        if self.laddar_start > self.laddar_end:
            raise ValueError(f"laddar_start must <= laddar_end ({self.laddar_end}). Current value: {self.laddar_start}")
        
        ## 1.2.
        if self.laddar_step % 2 != 0:
            raise ValueError(f"laddar_step must be a multiple of 2. Current value: {self.laddar_step}")
        if self.laddar_step > 6:
            raise ValueError(f"laddar_step must <= 6. Current value: {self.laddar_step}")
    
        ## 1.3. 
        if self.laddar_end % 2 != 0:
            raise ValueError(f"laddar_end must be an even number. Current value: {self.laddar_end}")
        if self.laddar_end > 28:
            raise ValueError(f"laddar_end must <= 28. Current value: {self.laddar_end}")
        

        # 2. Model 
        self.type_map: List[int] = self.linear_mtp.type_map_tensor.detach().cpu().numpy().tolist()
        self.umax_num_neigh_atoms: int = self.linear_mtp.umax_num_neigh_atoms
        self.fit_virial: bool = self.linear_mtp.fit_virial
        self.chebyshev_size: int = self.linear_mtp.chebyshev_size
        self.rmax: float = self.linear_mtp.rmax
        self.rmin: float = self.linear_mtp.rmin
        self.zbl_rmax: float = self.linear_mtp.zbl_rmax
        self.zbl_rmin: float = self.linear_mtp.zbl_rmin
        self.zbl_cks_list: Optional[List[List[float]]] = self.linear_mtp.zbl_cks_tensor.detach().cpu().numpy().tolist()
        self.zbl_dks_list: Optional[List[List[float]]] = self.linear_mtp.zbl_dks_tensor.detach().cpu().numpy().tolist()        

        self.device: torch._C.device = self.lit_linear_mtp.device
        self.torch_float_dtype: torch._C.dtype = self.lit_linear_mtp.dtype

        # 3. BFGS
        self.maxiter: int = maxiter
        self.gtol: float= 1e-5 if self.torch_float_dtype == torch.float32 else 1e-7
        self.disp: bool = True

        # 4. Ridge
        self.ridge_lambda: float = ridge_lambda

        ### Assertion
        if (self.e_weight != self.lit_linear_mtp.e_wgt_end) or \
            (self.f_weight != self.lit_linear_mtp.f_wgt_end) or \
            (self.v_weight != self.lit_linear_mtp.v_wgt_end):
            raise ValueError(
                f"LadderTrainer loss weights must match the final weights. "
                f"Got (e_wgt_start={self.e_weight}, f_wgt_start={self.f_weight}, v_wgt_start={self.v_weight}), "
                f"but expected "
                f"(e_wgt_end={self.lit_linear_mtp.e_wgt_end}, f_wgt_end={self.lit_linear_mtp.f_wgt_end}, v_wgt_end={self.lit_linear_mtp.v_wgt_end})."
            )
    

    def _generate_lit_model(self, mtp_level: int) -> LitLinearMtp:
        new_lit_model: LitLinearMtp = LitLinearMtp(type_map=self.type_map,
                                                   umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                   fit_virial=self.fit_virial,
                                                   mtp_level=mtp_level,
                                                   chebyshev_size=self.chebyshev_size,
                                                   rmax=self.rmax,
                                                   rmin=self.rmin,
                                                   zbl_rmax=self.zbl_rmax,
                                                   zbl_rmin=self.zbl_rmin,
                                                   zbl_cks_list=self.zbl_cks_list,
                                                   zbl_dks_list=self.zbl_dks_list,
                                                   e_wgt_start=self.e_weight,
                                                   e_wgt_end=self.e_weight,
                                                   f_wgt_start=self.f_weight,
                                                   f_wgt_end=self.f_weight,
                                                   v_wgt_start=self.v_weight,
                                                   v_wgt_end=self.v_weight).to(device=self.device, dtype=self.torch_float_dtype)
        return new_lit_model


    def _fit_sub_models(self) -> None:
        # 1.
        old_lit_model: LinearMtp = self._generate_lit_model(mtp_level=self.mtp_levels_list[0])
        old_torch_scipy_bfgs: TorchScipyBfgs = TorchScipyBfgs(lit_linear_mtp=old_lit_model,
                                                              trainset=self.trainset,
                                                              maxiter=self.maxiter)
        #old_torch_scipy_bfgs.run()
        
        # 2.
        for ii, tmp_mtp_level in enumerate(self.mtp_levels_list):
            if ii == 0:
                continue
            ## 2.1. 
            lit_model: LitLinearMtp = self._generate_lit_model(mtp_level=tmp_mtp_level)
            
            ## 2.2.
            parameter_inheritor: ParameterInheritor = ParameterInheritor(old_lit_model=old_lit_model,
                                                                         new_lit_model=lit_model,
                                                                         trainset=self.trainset,
                                                                         ridge_lambda=self.ridge_lambda)
            parameter_inheritor.transfer()

            ## 2.3.
            torch_scipy_bfgs: TorchScipyBfgs = TorchScipyBfgs(lit_linear_mtp=lit_model,
                                                              trainset=self.trainset,
                                                              maxiter=self.maxiter)
            #torch_scipy_bfgs.run()
            
            ## 2.4. 
            old_lit_model = deepcopy(lit_model)
        
        # 3.
        parameter_inheritor = ParameterInheritor(old_lit_model=old_lit_model,
                                                 new_lit_model=self.lit_linear_mtp,
                                                 trainset=self.trainset,
                                                 ridge_lambda=self.ridge_lambda)
        parameter_inheritor.transfer()
        


    def _fit_final_model(self) -> None:
        torch_scipy_bfgs: TorchScipyBfgs = TorchScipyBfgs(lit_linear_mtp=self.lit_linear_mtp,
                                                          trainset=self.trainset,
                                                          maxiter=self.maxiter)
        linear_mtp_solver: LinearMtpSolver = LinearMtpSolver(lit_linear_mtp=self.lit_linear_mtp,
                                                             trainset=self.trainset,
                                                             ridge_lambda=self.ridge_lambda)
        #torch_scipy_bfgs.run()
        linear_mtp_solver.solve_linear_equation()


    def fit(self):
        if len(self.mtp_levels_list) == 0:
            self._fit_final_model()
        else:
            self._fit_sub_models()
