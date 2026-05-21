from typing import (Optional, List, Union)

import torch
from torch.utils.data import (Dataset, DataLoader)
import lightning as L

from ai2pot.fromcc import (
    coeffsSchmidtOrthOp,
    linMatrixLinVectorOp)
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.data.mlffdataset import ExtxyzDataset


class LinearMtpSolver(object):
    BATCH_SIZE_HERE: int = 500

    def __init__(self,
                 e_weight: float,
                 f_weight: float,
                 v_weight: float,
                 linear_mtp: LinearMtp,
                 trainset: ExtxyzDataset):
        self.e_weight: float = e_weight
        self.f_weight: float = f_weight
        self.v_weight: float = v_weight
        self.linear_mtp: LinearMtp = linear_mtp
        self.trainset: ExtxyzDataset = trainset

        param = next(self.linear_mtp.parameters())
        device: torch._C.device = param.device
        torch_float_dtype: torch._C.dtype = param.dtype

        self.num_parameters: int = self.linear_mtp.get_num_descriptors() + self.linear_mtp.ntypes

        # Regularization
        # Whether recalculate self.reg_vector_tensor. If True, we should reset Hessian matrix.
        self.reg_init: bool = True
        self.reg_param: float = 1e-10 
        # Both for ridge regression and BFGS.
        self.reg_vector_tenosr: torch.Tensor = torch.zeros(self.num_parameters,
                                                           device=device,
                                                           dtype=torch_float_dtype) + self.reg_param


    def orthogonalize(self):
        with torch.no_grad():
            orthogonal_coeffs_tensor: torch.Tensor = coeffsSchmidtOrthOp(self.linear_mtp.ntypes,
                                                                         self.linear_mtp.nmus,
                                                                         self.linear_mtp.chebyshev_size,
                                                                         self.linear_mtp.coeffs_tensor)[0]
            # Write Back
            self.linear_mtp.coeffs_tensor.copy_(orthogonal_coeffs_tensor)
        
    
    def solve_linear_equation(self):
        with torch.no_grad():
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
            
            if (fit_virial):
                for batch_idx, batch_data in enumerate(train_loader):
                    binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                        brcs_tensor, btypes_tensor, bnghost_tensor, \
                        betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch_data
                    binum_tensor = binum_tensor.to(device)
                    bilist_tensor = bilist_tensor.to(device)
                    bnumneigh_tensor = bnumneigh_tensor.to(device)
                    bfirstneigh_tensor = bfirstneigh_tensor.to(device)
                    brcs_tensor = brcs_tensor.to(device).to(torch_float_dtype)
                    btypes_tensor = btypes_tensor.to(device)
                    bnghost_tensor = bnghost_tensor.to(device)
                    betot_dft_tensor = betot_dft_tensor.to(device).to(torch_float_dtype)
                    bforce_dft_tensor = bforce_dft_tensor.to(device).to(torch_float_dtype)
                    bvirial_dft_tensor = bvirial_dft_tensor.to(device).to(torch_float_dtype)
                    tmp_lin_matrix, tmp_lin_vector = linMatrixLinVectorOp(
                        self.e_weight,
                        self.f_weight,
                        self.v_weight,
                        betot_dft_tensor,
                        bforce_dft_tensor,
                        bvirial_dft_tensor,
                        self.linear_mtp.chebyshev_size,
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
                for batch_idx, batch_data in enumerate(train_loader):
                    binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                        brcs_tensor, btypes_tensor, bnghost_tensor, \
                        betot_dft_tensor, bforce_dft_tensor = batch_data
                    binum_tensor = binum_tensor.to(device)
                    bilist_tensor = bilist_tensor.to(device)
                    bnumneigh_tensor = bnumneigh_tensor.to(device)
                    bfirstneigh_tensor = bfirstneigh_tensor.to(device)
                    brcs_tensor = brcs_tensor.to(device).to(torch_float_dtype)
                    btypes_tensor = btypes_tensor.to(device)
                    bnghost_tensor = bnghost_tensor.to(device)
                    betot_dft_tensor = betot_dft_tensor.to(device).to(torch_float_dtype)
                    bforce_dft_tensor = bforce_dft_tensor.to(device).to(torch_float_dtype)
                    batch_size = bfirstneigh_tensor.size(0)
                    bvirial_dft_tensor = torch.zeros((batch_size, 9), dtype=torch_float_dtype, device=device)
                    tmp_lin_matrix, tmp_lin_vector = linMatrixLinVectorOp(
                        self.e_weight,
                        self.f_weight,
                        self.v_weight,
                        betot_dft_tensor,
                        bforce_dft_tensor,
                        bvirial_dft_tensor,
                        self.linear_mtp.chebyshev_size,
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
            TS_size: int = len(self.trainset)
            lin_matrix_diag_tensor: torch.Tensor = lin_matrix_tensor.diagonal()
            proposed_reg_tensor: torch.Tensor = self.reg_param * torch.clamp(lin_matrix_diag_tensor, min=1.0) / TS_size
            
            if not self.reg_init:
                lower_bound_tensor: torch.Tensor = 1e-2 * proposed_reg_tensor
                upper_bound_tensor: torch.Tensor = 1e2 * proposed_reg_tensor
                update_reg_mark: bool = torch.any(self.reg_vector_tenosr < lower_bound_tensor) or \
                                        torch.any(self.reg_vector_tenosr > upper_bound_tensor)
                if (update_reg_mark):
                    self.reg_vector_tenosr = proposed_reg_tensor.clone()
                    self.reg_init = True
                    print("Regularization parameters updated. Hessian in BFGS need to reset.")
                
            lin_matrix_diag_tensor.add_(self.reg_vector_tenosr * TS_size)

            # 3. Gaussian Elimination
            print("Solving normal equations...")
            try:
                solution = torch.linalg.solve(lin_matrix_tensor, lin_vector_tensor)
            except RuntimeError as e:
                print("Matrix is singular, falling back to lstsq solver...")
                solution = torch.linalg.lstsq(lin_matrix_tensor, lin_vector_tensor).solution
            
            # 4. Write back
            alpha_scalar_moments: int = self.linear_mtp.get_num_descriptors()
            self.linear_mtp.linear_coeffs_tensor.copy_(solution[:alpha_scalar_moments])
            self.linear_mtp.type_bias_tensor.copy_(solution[alpha_scalar_moments:])


    def rescale_coeffs(self):
        pass
        