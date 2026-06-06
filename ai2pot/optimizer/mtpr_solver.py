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


from typing import List

import torch
from torch.utils.data import DataLoader

from ai2pot.fromcc import (
    coeffsSchmidtOrthOp,
    linMatrixLinVectorOp)
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.data.mlffdataset import ExtxyzDataset


class LinearMtpSolver(object):
    BATCH_SIZE_HERE: int = 500

    def __init__(self,
                 lit_linear_mtp: LitLinearMtp,
                 trainset: ExtxyzDataset,
                 ridge_lambda: float = 1e-2):
        self.lit_linear_mtp: LitLinearMtp = lit_linear_mtp
        self.linear_mtp: LinearMtp = self.lit_linear_mtp.model
        self.e_weight: float = self.lit_linear_mtp.e_wgt_start
        self.f_weight: float = self.lit_linear_mtp.f_wgt_start
        self.v_weight: float = self.lit_linear_mtp.v_wgt_start
        self.trainset: ExtxyzDataset = trainset

        self.num_parameters: int = self.linear_mtp.get_num_descriptors() + self.linear_mtp.ntypes

        # Regularization
        self.ridge_lambda: float = ridge_lambda

        ### Assertion
        if (self.e_weight != self.lit_linear_mtp.e_wgt_end) or \
            (self.f_weight != self.lit_linear_mtp.f_wgt_end) or \
            (self.v_weight != self.lit_linear_mtp.v_wgt_end):
            raise ValueError(
                f"LSTSQ loss weights must match the final weights. "
                f"Got (e_wgt_start={self.e_weight}, f_wgt_start={self.f_weight}, v_wgt_start={self.v_weight}), "
                f"but expected "
                f"(e_wgt_end={self.lit_linear_mtp.e_wgt_end}, f_wgt_end={self.lit_linear_mtp.f_wgt_end}, v_wgt_end={self.lit_linear_mtp.v_wgt_end})."
            )


    @torch.no_grad()
    def orthogonalize(self):
        orthogonal_coeffs_tensor: torch.Tensor = coeffsSchmidtOrthOp(self.linear_mtp.ntypes,
                                                                        self.linear_mtp.nmus,
                                                                        self.linear_mtp.chebyshev_size,
                                                                        self.linear_mtp.coeffs_tensor)[0]
        # Write Back
        self.linear_mtp.coeffs_tensor.copy_(orthogonal_coeffs_tensor)
    

    @torch.no_grad()
    def solve_linear_equation(self):
        fit_virial: bool = self.linear_mtp.fit_virial
        param = next(self.linear_mtp.parameters())
        device: torch._C.device = param.device
        torch_float_dtype: torch._C.dtype = param.dtype

        train_loader: DataLoader = DataLoader(dataset=self.trainset,
                                            batch_size=self.BATCH_SIZE_HERE,
                                            shuffle=False)
        # 2. Construct Equation
        ## 2.1. 
        num_parameters: int = self.linear_mtp.get_num_descriptors() + self.linear_mtp.ntypes
        lin_matrix_tensor: torch.Tensor = torch.zeros((num_parameters, num_parameters),
                                                    dtype=torch_float_dtype,
                                                    device=device)
        lin_vector_tensor: torch.Tensor = torch.zeros(num_parameters,
                                                    dtype=torch_float_dtype,
                                                    device=device)
        
        for batch_idx, batch_data in enumerate(train_loader):
            if (fit_virial):
                binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                    brcs_tensor, btypes_tensor, bnghost_tensor, \
                    betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = \
                    [t.to(device) for t in batch_data]
                brcs_tensor = brcs_tensor.to(torch_float_dtype)
                betot_dft_tensor = betot_dft_tensor.to(torch_float_dtype)
                bforce_dft_tensor = bforce_dft_tensor.to(torch_float_dtype)
                bvirial_dft_tensor = bvirial_dft_tensor.to(torch_float_dtype)

                tmp_lin_matrix, tmp_lin_vector = linMatrixLinVectorOp(
                    self.e_weight,
                    self.f_weight,
                    self.v_weight,
                    betot_dft_tensor,
                    bforce_dft_tensor,
                    bvirial_dft_tensor,
                    self.linear_mtp.chebyshev_size,
                    self.linear_mtp.scaling,
                    self.linear_mtp.coeffs_tensor,
                    self.linear_mtp.linear_coeffs_tensor,
                    self.linear_mtp.type_bias_tensor,
                    self.linear_mtp.alpha_moments_count,
                    self.linear_mtp.alpha_index_basic_tensor,
                    self.linear_mtp.alpha_index_times_tensor,
                    self.linear_mtp.alpha_moment_mapping_tensor,
                    self.linear_mtp.nmus,
                    binum_tensor,
                    bilist_tensor,
                    bnumneigh_tensor,
                    bfirstneigh_tensor,
                    brcs_tensor,
                    btypes_tensor,
                    self.linear_mtp.type_map_tensor,
                    bnghost_tensor[0].item(),
                    self.linear_mtp.rmax,
                    self.linear_mtp.rmin,
                    self.linear_mtp.q_scaler_tensor,
                    self.linear_mtp.zbl_rmax,
                    self.linear_mtp.zbl_rmin,
                    self.linear_mtp.zbl_cks_tensor,
                    self.linear_mtp.zbl_dks_tensor)
                
                lin_matrix_tensor.add_(tmp_lin_matrix)
                lin_vector_tensor.add_(tmp_lin_vector)
            else:
                binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                    brcs_tensor, btypes_tensor, bnghost_tensor, \
                    betot_dft_tensor, bforce_dft_tensor = \
                    [t.to(device) for t in batch_data]
                brcs_tensor = brcs_tensor.to(torch_float_dtype)
                betot_dft_tensor = betot_dft_tensor.to(torch_float_dtype)
                bforce_dft_tensor = bforce_dft_tensor.to(torch_float_dtype)
                batch_size = bfirstneigh_tensor.size(0)
                bvirial_dft_tensor = torch.zeros((batch_size, 9), dtype=torch_float_dtype, device=device)

                tmp_lin_matrix, tmp_lin_vector = linMatrixLinVectorOp(
                    self.e_weight,
                    self.f_weight,
                    0.0,
                    betot_dft_tensor,
                    bforce_dft_tensor,
                    bvirial_dft_tensor,
                    self.linear_mtp.chebyshev_size,
                    self.linear_mtp.scaling,
                    self.linear_mtp.coeffs_tensor,
                    self.linear_mtp.linear_coeffs_tensor,
                    self.linear_mtp.type_bias_tensor,
                    self.linear_mtp.alpha_moments_count,
                    self.linear_mtp.alpha_index_basic_tensor,
                    self.linear_mtp.alpha_index_times_tensor,
                    self.linear_mtp.alpha_moment_mapping_tensor,
                    self.linear_mtp.nmus,
                    binum_tensor,
                    bilist_tensor,
                    bnumneigh_tensor,
                    bfirstneigh_tensor,
                    brcs_tensor,
                    btypes_tensor,
                    self.linear_mtp.type_map_tensor,
                    bnghost_tensor[0].item(),
                    self.linear_mtp.rmax,
                    self.linear_mtp.rmin,
                    self.linear_mtp.q_scaler_tensor,
                    self.linear_mtp.zbl_rmax,
                    self.linear_mtp.zbl_rmin,
                    self.linear_mtp.zbl_cks_tensor,
                    self.linear_mtp.zbl_dks_tensor)
                
                lin_matrix_tensor.add_(tmp_lin_matrix)
                lin_vector_tensor.add_(tmp_lin_vector)
        
        ## 2.2. Ridge regression
        lin_matrix_diag_tensor: torch.Tensor = lin_matrix_tensor.diagonal()
        proposed_rg_tensor: torch.Tensor = self.ridge_lambda * torch.clamp(lin_matrix_diag_tensor, min=1.0)
        lin_matrix_diag_tensor.add_(proposed_rg_tensor)
        

        # 3. Gaussian Elimination
        print("Solving normal equations...")
        #print(torch.linalg.cond(lin_matrix_tensor))
        try:
            solution = torch.linalg.solve(lin_matrix_tensor, lin_vector_tensor)
        except RuntimeError as e:
            print("Matrix is singular, falling back to lstsq solver...")
            solution = torch.linalg.lstsq(lin_matrix_tensor, lin_vector_tensor).solution
        
        # 4. Write back
        alpha_scalar_moments: int = self.linear_mtp.get_num_descriptors()
        self.linear_mtp.linear_coeffs_tensor.copy_(solution[:alpha_scalar_moments])
        self.linear_mtp.type_bias_tensor.copy_(solution[alpha_scalar_moments:])


    @torch.no_grad()
    def rescale_coeffs(self):
        # The leftmost and rightmost boundaries explored
        max_scaling: float = self.linear_mtp.scaling
        min_scaling: float = self.linear_mtp.scaling

        print("Starting rescaling process...")
        while (True):
            #
            now_scaling: float = self.linear_mtp.scaling
            candidate_scalings_list: List[float] = [now_scaling / 1.2,
                                                    now_scaling / 1.1,
                                                    now_scaling,
                                                    now_scaling * 1.1,
                                                    now_scaling * 1.2]
            condition_numbers_list: List[float] = []

            #
            for tmp_check_idx in range(5):
                self.linear_mtp.scaling = candidate_scalings_list[tmp_check_idx]
                self.solve_linear_equation()
                all_coeffs: torch.Tensor = torch.cat([self.linear_mtp.linear_coeffs_tensor,
                                                      self.linear_mtp.type_bias_tensor])
                abs_all_coeffs: torch.Tensor = torch.abs(all_coeffs)
                l2_norm: float = torch.sqrt(torch.sum(torch.pow(abs_all_coeffs, 2))).item()
                #median: float = torch.median(abs_all_coeffs).item()
                sorted_all_coeffs, _ = torch.sort(abs_all_coeffs)
                median = sorted_all_coeffs[len(sorted_all_coeffs) // 2].item()
                cond_num: float = l2_norm / (median + 1e-15)
                condition_numbers_list.append(cond_num)

            #
            best_idx = 2
            for tmp_check_idx in range(5):
                if (condition_numbers_list[tmp_check_idx] < condition_numbers_list[best_idx]):
                    best_idx = tmp_check_idx
            self.linear_mtp.scaling = candidate_scalings_list[best_idx]
            self.solve_linear_equation()
            #print(f"-> Selected scaling {self.linear_mtp.scaling} for next iteration.\n")

            # Check end : Two termination conditions.
            if best_idx == 2:
                print(f"Local minimum reached. Stopping rescaling. Final scaling: {self.linear_mtp.scaling:.5f}")
                break
            if ((self.linear_mtp.scaling > min_scaling) and (self.linear_mtp.scaling < max_scaling)):
                print(f"Oscillation detected. Stopping rescaling. Final scaling: {self.linear_mtp.scaling:.5f}")
                break
            
            min_scaling = min(min_scaling, self.linear_mtp.scaling)
            max_scaling = max(max_scaling, self.linear_mtp.scaling)
    