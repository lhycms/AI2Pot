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


from typing import List, Optional

import torch
import torch.nn as nn

from ai2pot.fromcc import (mtpParamOp,
                           linearMtpToLossOp,
                           linearMtpToEFLossOp,
                           linearMtpToEFVOp,
                           linearMtpToEFOp,
                           linearMtpToEsitesOp,
                           linearMtpToDescriptorsOp)


class LinearMtp(nn.Module):
    def __init__(self,
                 type_map: List[int],
                 umax_num_neigh_atoms: int = 200,
                 fit_virial: bool = False,
                 mtp_level: int = 16,
                 chebyshev_size: int = 8,
                 rmax: float = 5.0,
                 rmin: float = 0.0,
                 zbl_rmax: float = 0.0,
                 zbl_rmin: float = 0.0,
                 zbl_cks_list: Optional[List[float]] = None,
                 zbl_dks_list: Optional[List[float]] = None):
        super(LinearMtp, self).__init__()
        self.scaling: float = 1.0
        self.mtp_level: int = mtp_level
        self.register_buffer(
            name="type_map_tensor", 
            tensor=torch.as_tensor(type_map, dtype=torch.int32))
        self.ntypes: int = len(type_map)
        self.chebyshev_size: int = chebyshev_size
        self.rmax: float = rmax
        self.rmin: float = rmin
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.fit_virial: bool = fit_virial
        self.zbl_rmax: float = zbl_rmax
        self.zbl_rmin: float = zbl_rmin
        self._init_zbl_params(zbl_cks_list=zbl_cks_list,
                              zbl_dks_list=zbl_dks_list)
        
        mtp_param_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        self.register_buffer(name="alpha_moments_count_tensor", tensor=mtp_param_info[0])
        self.alpha_moments_count: int = mtp_param_info[0].item()
        self.register_buffer(name="alpha_index_basic_tensor", tensor=mtp_param_info[1])
        self.register_buffer(name="alpha_index_times_tensor", tensor=mtp_param_info[2])
        self.register_buffer(name="alpha_moment_mapping_tensor", tensor=mtp_param_info[3])
        self.register_buffer(name="nmus_tensor", tensor=mtp_param_info[6])
        self.nmus: int = mtp_param_info[6].item()
        self.num_descriptors: int = self.alpha_moment_mapping_tensor.size()[0]
        self.num_coeffs: int = self.ntypes * self.ntypes * self.nmus * self.chebyshev_size
        
        ### Init ###
        coeffs_tensor: torch.Tensor = torch.empty(self.ntypes*self.ntypes*self.nmus*self.chebyshev_size)
        r1: torch.Tensor = 0.1 * torch.randn(self.num_coeffs, device=coeffs_tensor.device)
        r2: torch.Tensor = torch.rand(self.num_coeffs, device=coeffs_tensor.device)
        coeffs_tensor.copy_(r1 + r2)
        self.register_parameter(name="coeffs_tensor", param=nn.Parameter(data=coeffs_tensor))
        
        linear_coeffs_tensor: torch.Tensor = torch.empty(self.num_descriptors)
        init_linear_coeffs_std: float = (2.0 / (self.num_descriptors + 1)) ** 0.5
        nn.init.normal_(linear_coeffs_tensor, mean=0.0, std=init_linear_coeffs_std)
        self.register_parameter(name="linear_coeffs_tensor", param=nn.Parameter(data=linear_coeffs_tensor))
        
        type_bias_tensor: torch.Tensor = torch.empty(self.ntypes)
        nn.init.normal_(type_bias_tensor, mean=0.0, std=1.0)
        self.register_parameter(name="type_bias_tensor", param=nn.Parameter(data=type_bias_tensor))
        ### Init ###
    
        q_scaler_tensor: torch.Tensor = torch.ones(self.num_descriptors, dtype=torch.float32)
        self.register_buffer("q_scaler_tensor", tensor=q_scaler_tensor)

        # Conversion
        self.register_buffer(
            name="conv_energy_tensor",
            tensor=torch.tensor(1.0, device=coeffs_tensor.device))
        self.register_buffer(
            name="conv_length_tensor",
            tensor=torch.tensor(1.0, device=coeffs_tensor.device))


    @torch.no_grad()
    def _init_as_mlip(self):
        device: torch._C.device = self.coeffs_tensor.device
        rand_tensor: torch.Tensor = torch.rand(self.ntypes*self.ntypes, self.nmus, self.chebyshev_size, device=device) * 2.0 - 1.0
        coeffs_3d_tensor: torch.Tensor = 5e-7 * rand_tensor

        for tmp_mu in range(self.nmus):
            tmp_chebyshev_idx = min(tmp_mu, self.chebyshev_size - 1)
            special_rand = torch.rand(self.ntypes * self.ntypes, device=device) * 2.0 - 1.0
            coeffs_3d_tensor[:, tmp_mu, tmp_chebyshev_idx] = 1e-7 * (1.0 + special_rand)
        
        coeffs_tensor: torch.Tensor = coeffs_3d_tensor.view(-1).contiguous()
        self.coeffs_tensor.copy_(coeffs_tensor)


    
    @torch.no_grad()
    def _init_all_zeros(self):
        self.coeffs_tensor.copy_(torch.zeros_like(self.coeffs_tensor))
        self.linear_coeffs_tensor.copy_(torch.zeros_like(self.linear_coeffs_tensor))
        self.type_bias_tensor.copy_(torch.zeros_like(self.type_bias_tensor))
    

    @torch.no_grad()
    def _init_as_ace(self):
        self.linear_coeffs_tensor.copy_(torch.randn_like(self.linear_coeffs_tensor) * 1e-4)



    def _init_zbl_params(self,
                         zbl_cks_list: Optional[List[float]],
                         zbl_dks_list: Optional[List[float]]):
        if (zbl_cks_list is None) or (zbl_dks_list is None):
            single_zbl_ck_tensor: torch.Tensor = torch.tensor([0.18175, 0.50986, 0.28022, 0.02817])
            single_zbl_dk_tensor: torch.Tensor = torch.tensor([3.1998, 0.94229, 0.4029, 0.20162])
            zbl_cks_tensor: torch.Tensor = single_zbl_ck_tensor.repeat(self.ntypes * self.ntypes)
            zbl_dks_tensor: torch.Tensor = single_zbl_dk_tensor.repeat(self.ntypes * self.ntypes)
            self.register_buffer("zbl_cks_tensor", tensor=zbl_cks_tensor)
            self.register_buffer("zbl_dks_tensor", tensor=zbl_dks_tensor)
        else:
            zbl_cks_tensor: torch.Tensor = torch.tensor(zbl_cks_list, dtype=torch.float32)
            zbl_dks_tensor: torch.Tensor = torch.tensor(zbl_dks_list, dtype=torch.float32)
            assert(zbl_cks_tensor.numel() == self.ntypes*self.ntypes*4)
            assert(zbl_dks_tensor.numel() == self.ntypes*self.ntypes*4)
            self.register_buffer("zbl_cks_tensor", tensor=zbl_cks_tensor)
            self.register_buffer("zbl_dks_tensor", tensor=zbl_dks_tensor)


    def get_num_descriptors(self):
        return self.alpha_moment_mapping_tensor.size()[0]
    

    def get_num_temp_moments(self):
        return self.alpha_moments_count

    
    def forward(self, *args, **kwargs):
        if self.fit_virial:
            bmse_tensor = self.predict_loss(*args, **kwargs)
        else:
            bmse_tensor = self.predict_ef_loss(*args, **kwargs)
        return bmse_tensor
    
    
    def predict_loss(self,
                     e_weight: float,
                     f_weight: float,
                     v_weight: float,
                     betot_dft_tensor: torch.Tensor,
                     bforce_dft_tensor: torch.Tensor,
                     bvirial_dft_tensor: torch.Tensor,
                     binum_tensor: torch.Tensor,
                     bilist_tensor: torch.Tensor,
                     bnumneigh_tensor: torch.Tensor,
                     bfirstneigh_tensor: torch.Tensor,
                     brcs_tensor: torch.Tensor,
                     btypes_tensor: torch.Tensor,
                     bnghost_tensor: torch.Tensor):
        #
        conv_energy: float = self.conv_energy_tensor.item()
        conv_length: float = self.conv_length_tensor.item()
        conv_force: float = conv_energy / conv_length
        conv_virial: float = conv_energy

        betot_dft_tensor_norm: torch.Tensor = betot_dft_tensor * conv_energy
        bforce_dft_tensor_norm: torch.Tensor = bforce_dft_tensor * conv_force
        bvirial_dft_tensor_norm: torch.Tensor = bvirial_dft_tensor * conv_virial
        brcs_tensor_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_norm: float = self.rmax * conv_length
        rmin_norm: float = self.rmin * conv_length
        zbl_rmax_norm: float = self.zbl_rmax * conv_length
        zbl_rmin_norm: float = self.zbl_rmin * conv_length

        type_bias_norm: torch.Tensor = self.type_bias_tensor * conv_energy
        zbl_cks_norm: torch.Tensor = self.zbl_cks_tensor * conv_length * conv_energy
        zbl_dks_norm: torch.Tensor = self.zbl_dks_tensor / conv_length

        #
        bmse_tensor, e_rmse_tensor, f_rmse_tensor, v_rmse_tensor = linearMtpToLossOp(
            e_weight,
            f_weight,
            v_weight,
            betot_dft_tensor_norm,
            bforce_dft_tensor_norm,
            bvirial_dft_tensor_norm,
            self.chebyshev_size,
            self.scaling,
            self.coeffs_tensor,
            self.linear_coeffs_tensor,
            type_bias_norm,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_norm,
            rmin_norm,
            self.q_scaler_tensor,
            zbl_rmax_norm,
            zbl_rmin_norm,
            zbl_cks_norm,
            zbl_dks_norm)
        bmse_tensor: torch.Tensor
        e_rmse_tensor: torch.Tensor
        f_rmse_tensor: torch.Tensor
        v_rmse_tensor: torch.Tensor

        #
        e_rmse_tensor = e_rmse_tensor / conv_energy
        f_rmse_tensor = f_rmse_tensor / conv_force
        v_rmse_tensor = v_rmse_tensor / conv_virial

        return bmse_tensor, e_rmse_tensor, f_rmse_tensor, v_rmse_tensor
    
    
    def predict_ef_loss(self,
                        e_weight: float,
                        f_weight: float,
                        betot_dft_tensor: torch.Tensor,
                        bforce_dft_tensor: torch.Tensor,
                        binum_tensor: torch.Tensor,
                        bilist_tensor: torch.Tensor,
                        bnumneigh_tensor: torch.Tensor,
                        bfirstneigh_tensor: torch.Tensor,
                        brcs_tensor: torch.Tensor,
                        btypes_tensor: torch.Tensor,
                        bnghost_tensor: torch.Tensor):
        #
        conv_energy: float = self.conv_energy_tensor.item()
        conv_length: float = self.conv_length_tensor.item()
        conv_force: float = conv_energy / conv_length
        
        betot_dft_tensor_norm: torch.Tensor = betot_dft_tensor * conv_energy
        bforce_dft_tensor_norm: torch.Tensor = bforce_dft_tensor * conv_force
        brcs_tensor_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_norm: float = self.rmax * conv_length
        rmin_norm: float = self.rmin * conv_length
        zbl_rmax_norm: float = self.zbl_rmax * conv_length
        zbl_rmin_norm: float = self.zbl_rmin * conv_length

        type_bias_norm: torch.Tensor = self.type_bias_tensor * conv_energy
        zbl_cks_norm: torch.Tensor = self.zbl_cks_tensor * conv_length * conv_energy
        zbl_dks_norm: torch.Tensor = self.zbl_dks_tensor / conv_length

        #
        bmse_tensor, e_rmse_tensor, f_rmse_tensor = linearMtpToEFLossOp(
            e_weight,
            f_weight,
            betot_dft_tensor_norm,
            bforce_dft_tensor_norm,
            self.chebyshev_size,
            self.scaling,
            self.coeffs_tensor,
            self.linear_coeffs_tensor,
            type_bias_norm,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_norm,
            rmin_norm,
            self.q_scaler_tensor,
            zbl_rmax_norm,
            zbl_rmin_norm,
            zbl_cks_norm,
            zbl_dks_norm)
        bmse_tensor: torch.Tensor
        e_rmse_tensor: torch.Tensor
        f_rmse_tensor: torch.Tensor

        #
        e_rmse_tensor = e_rmse_tensor / conv_energy
        f_rmse_tensor = f_rmse_tensor / conv_force

        return bmse_tensor, e_rmse_tensor, f_rmse_tensor
    
    
    def predict_efv(self,
                    binum_tensor: torch.Tensor,
                    bilist_tensor: torch.Tensor,
                    bnumneigh_tensor: torch.Tensor,
                    bfirstneigh_tensor: torch.Tensor,
                    brcs_tensor: torch.Tensor,
                    btypes_tensor: torch.Tensor,
                    bnghost_tensor: torch.Tensor):
        #
        conv_energy: float = self.conv_energy_tensor.item()
        conv_length: float = self.conv_length_tensor.item()
        conv_force: float = conv_energy / conv_length
        conv_virial: float = conv_energy

        brcs_tensor_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_norm: float = self.rmax * conv_length
        rmin_norm: float = self.rmin * conv_length
        zbl_rmax_norm: float = self.zbl_rmax * conv_length
        zbl_rmin_norm: float = self.zbl_rmin * conv_length

        type_bias_norm: torch.Tensor = self.type_bias_tensor * conv_energy
        zbl_cks_norm: torch.Tensor = self.zbl_cks_tensor * conv_length * conv_energy
        zbl_dks_norm: torch.Tensor = self.zbl_dks_tensor / conv_length

        #
        betot_tensor, bforce_tensor, bvirial_tensor = linearMtpToEFVOp(
            self.chebyshev_size,
            self.scaling,
            self.coeffs_tensor,
            self.linear_coeffs_tensor,
            type_bias_norm,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_norm,
            rmin_norm,
            self.q_scaler_tensor,
            zbl_rmax_norm,
            zbl_rmin_norm,
            zbl_cks_norm,
            zbl_dks_norm)
        
        #
        betot_tensor = betot_tensor / conv_energy
        bforce_tensor = bforce_tensor / conv_force
        bvirial_tensor = bvirial_tensor / conv_virial

        return betot_tensor, bforce_tensor, bvirial_tensor


    def predict_ef(self,
                   binum_tensor: torch.Tensor,
                   bilist_tensor: torch.Tensor,
                   bnumneigh_tensor: torch.Tensor,
                   bfirstneigh_tensor: torch.Tensor,
                   brcs_tensor: torch.Tensor,
                   btypes_tensor: torch.Tensor,
                   bnghost_tensor: torch.Tensor):
        #
        conv_energy: float = self.conv_energy_tensor.item()
        conv_length: float = self.conv_length_tensor.item()
        conv_force: float = conv_energy / conv_length

        brcs_tensor_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_norm: float = self.rmax * conv_length
        rmin_norm: float = self.rmin * conv_length
        zbl_rmax_norm: float = self.zbl_rmax * conv_length
        zbl_rmin_norm: float = self.zbl_rmin * conv_length

        type_bias_norm: torch.Tensor = self.type_bias_tensor * conv_energy
        zbl_cks_norm: torch.Tensor = self.zbl_cks_tensor * conv_length * conv_energy
        zbl_dks_norm: torch.Tensor = self.zbl_dks_tensor / conv_length

        #
        betot_tensor, bforce_tensor = linearMtpToEFOp(
            self.chebyshev_size,
            self.scaling,
            self.coeffs_tensor,
            self.linear_coeffs_tensor,
            type_bias_norm,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_norm,
            rmin_norm,
            self.q_scaler_tensor,
            zbl_rmax_norm,
            zbl_rmin_norm,
            zbl_cks_norm,
            zbl_dks_norm)
        
        #
        betot_tensor = betot_tensor / conv_energy
        bforce_tensor = bforce_tensor / conv_force

        return betot_tensor, bforce_tensor
    

    def predict_e_sites(self,
                        binum_tensor: torch.Tensor,
                        bilist_tensor: torch.Tensor,
                        bnumneigh_tensor: torch.Tensor,
                        bfirstneigh_tensor: torch.Tensor,
                        brcs_tensor: torch.Tensor,
                        btypes_tensor: torch.Tensor,
                        bnghost_tensor: torch.Tensor):
        assert(brcs_tensor.device == torch.device("cpu"))
        #
        conv_energy: float = self.conv_energy_tensor.item()
        conv_length: float = self.conv_length_tensor.item()

        brcs_tensor_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_norm: float = self.rmax * conv_length
        rmin_norm: float = self.rmin * conv_length
        zbl_rmax_norm: float = self.zbl_rmax * conv_length
        zbl_rmin_norm: float = self.zbl_rmin * conv_length

        type_bias_norm: torch.Tensor = self.type_bias_tensor * conv_energy
        zbl_cks_norm: torch.Tensor = self.zbl_cks_tensor * conv_length * conv_energy
        zbl_dks_norm: torch.Tensor = self.zbl_dks_tensor / conv_length

        #
        be_sites_tensor: torch.Tensor = linearMtpToEsitesOp(
            self.chebyshev_size,
            self.scaling,
            self.coeffs_tensor,
            self.linear_coeffs_tensor,
            type_bias_norm,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_norm,
            rmin_norm,
            self.q_scaler_tensor,
            zbl_rmax_norm,
            zbl_rmin_norm,
            zbl_cks_norm,
            zbl_dks_norm)[0]
        
        #
        be_sites_tensor = be_sites_tensor / conv_energy

        return be_sites_tensor


    def predict_descriptors(self,
                            binum_tensor: torch.Tensor,
                            bilist_tensor: torch.Tensor,
                            bnumneigh_tensor: torch.Tensor,
                            bfirstneigh_tensor: torch.Tensor,
                            brcs_tensor: torch.Tensor,
                            btypes_tensor: torch.Tensor,
                            bnghost_tensor: torch.Tensor):
        #
        conv_length: float = self.conv_length_tensor.item()

        brcs_tensor_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_norm: float = self.rmax * conv_length
        rmin_norm: float = self.rmin * conv_length

        #
        bdescriptors_tensor: torch.Tensor = linearMtpToDescriptorsOp(
            self.chebyshev_size,
            self.scaling,
            self.coeffs_tensor,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_norm,
            rmin_norm)[0]
        
        return bdescriptors_tensor
