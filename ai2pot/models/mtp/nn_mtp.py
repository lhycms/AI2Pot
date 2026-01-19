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

from ai2pot.fromcc import (
    mtpParamOp,
    nnMtpToLossOp,
    nnMtpToEFLossOp,
    nnMtpToEFVOp,
    nnMtpToEFOp,
    nnMtpToEsitesOp,
    nnMtpToDescriptorsOp)


class NNMtp(nn.Module):
    def __init__(self,
                 mtp_level: int,
                 type_map: List[int],
                 energy_shifts: Optional[List[float]] = None,
                 chebyshev_size: int = 8,
                 num_neurons: int = 30,
                 rmax: float = 5.0,
                 rmin: float = 0.0,
                 umax_num_neigh_atoms: int = 200,
                 fit_virial: bool = False,
                 zbl_rmax: float = 2.0,
                 zbl_rmin: float = 1.0,
                 zbl_cks_list: Optional[List[float]] = None,
                 zbl_dks_list: Optional[List[float]] = None):
        super(NNMtp, self).__init__()
        assert(mtp_level <= 28)

        self.mtp_level: int = mtp_level
        self.register_buffer(name="type_map_tensor", tensor=torch.tensor(type_map, dtype=torch.int32))
        self.ntypes: int = len(type_map)
        self.chebyshev_size: int = chebyshev_size
        self.num_neurons: int = num_neurons
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
        self.alpha_scalar_moments: int = self.num_descriptors

        coeffs_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.ntypes*self.nmus*self.chebyshev_size)
        nn.init.normal_(coeffs_tensor, mean=0.0, std=0.1)
        self.register_parameter(name="coeffs_tensor", param=nn.Parameter(data=coeffs_tensor))

        w0_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.num_neurons*self.alpha_scalar_moments)
        nn.init.normal_(w0_tensor, mean=0.0, std=0.1)
        self.register_parameter(name="w0_tensor", param=nn.Parameter(data=w0_tensor))

        w1_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.num_neurons)
        nn.init.normal_(w1_tensor, mean=0.0, std=0.1)
        self.register_parameter(name="w1_tensor", param=nn.Parameter(data=w1_tensor))

        if energy_shifts is not None:
            assert(len(energy_shifts) == self.ntypes)
            type_bias_tensor: torch.Tensor = torch.tensor(energy_shifts)
            self.register_parameter(name="type_bias_tensor", param=nn.Parameter(data=type_bias_tensor))
        else:
            type_bias_tensor: torch.Tensor = torch.Tensor(self.ntypes)
            nn.init.normal_(type_bias_tensor, mean=0.0, std=1.0)
            self.register_parameter(name="type_bias_tensor", param=nn.Parameter(data=type_bias_tensor))

        q_shifter_tensor: torch.Tensor = torch.zeros(self.num_descriptors, dtype=torch.float32)
        self.register_buffer("_q_shifter_tensor", tensor=q_shifter_tensor)
        q_scaler_tensor: torch.Tensor = torch.ones(self.num_descriptors, dtype=torch.float32)
        self.register_buffer("_q_scaler_tensor", tensor=q_scaler_tensor)
        


    def _init_zbl_params(self,
                         zbl_cks_list: Optional[List[float]],
                         zbl_dks_list: Optional[List[float]]):
        if (zbl_cks_list is None) or (zbl_dks_list is None):
            single_zbl_ck_tensor: torch.Tensor = torch.tensor([0.18175, 0.50986, 0.28022, 0.02817])
            single_zbl_dk_tensor: torch.Tensor = torch.tensor([3.1998, 0.94229, 0.4029, 0.20162])
            zbl_cks_tensor: torch.Tensor = single_zbl_ck_tensor.repeat(self.ntypes * self.ntypes)
            zbl_dks_tensor: torch.Tensor = single_zbl_dk_tensor.repeat(self.ntypes * self.ntypes)
            self.register_buffer(name="zbl_cks_tensor", tensor=zbl_cks_tensor)
            self.register_buffer(name="zbl_dks_tensor", tensor=zbl_dks_tensor)
        else:
            zbl_cks_tensor: torch.Tensor = torch.tensor(zbl_cks_list, dtype=torch.float32)
            zbl_dks_tensor: torch.Tensor = torch.tensor(zbl_dks_list, dtype=torch.float32)
            assert(zbl_cks_tensor.size() == self.ntypes*self.ntypes*4)
            assert(zbl_dks_tensor.size() == self.ntypes*self.ntypes*4)
            self.register_buffer(name="zbl_cks_tensor", tensor=zbl_cks_tensor)
            self.register_buffer(name="zbl_dks_tensor", tensor=zbl_dks_tensor)


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
                     nghost: int):
        bmse_tensor: torch.Tensor = nnMtpToLossOp(
            e_weight,
            f_weight,
            v_weight,
            betot_dft_tensor,
            bforce_dft_tensor,
            bvirial_dft_tensor,
            self.chebyshev_size,
            self.coeffs_tensor,
            self.w0_tensor,
            self.w1_tensor,
            self.type_bias_tensor,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            self.type_map_tensor,
            nghost,
            self.rmax,
            self.rmin,
            self.zbl_rmax,
            self.zbl_rmin,
            self.zbl_cks_tensor,
            self.zbl_dks_tensor)[0]
        return bmse_tensor


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
                        nghost: int):
        bmse_tensor: torch.Tensor = nnMtpToEFLossOp(
            e_weight,
            f_weight,
            betot_dft_tensor,
            bforce_dft_tensor,
            self.chebyshev_size,
            self.coeffs_tensor,
            self.w0_tensor,
            self.w1_tensor,
            self.type_bias_tensor,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            self.type_map_tensor,
            nghost,
            self.rmax,
            self.rmin,
            self._q_shifter_tensor,
            self._q_scaler_tensor,
            self.zbl_rmax,
            self.zbl_rmin,
            self.zbl_cks_tensor,
            self.zbl_dks_tensor)[0]
        return bmse_tensor
    

    def predict_efv(self,
                    binum_tensor: torch.Tensor,
                    bilist_tensor: torch.Tensor,
                    bnumneigh_tensor: torch.Tensor,
                    bfirstneigh_tensor: torch.Tensor,
                    brcs_tensor: torch.Tensor,
                    btypes_tensor: torch.Tensor,
                    bnghost_tensor: torch.Tensor):
        betot_tensor, bforce_tensor, bvirial_tensor = nnMtpToEFVOp(
            self.chebyshev_size,
            self.coeffs_tensor,
            self.w0_tensor,
            self.w1_tensor,
            self.type_bias_tensor,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            self.rmax,
            self.rmin,
            self._q_shifter_tensor,
            self._q_scaler_tensor,
            self.zbl_rmax,
            self.zbl_rmax,
            self.zbl_cks_tensor,
            self.zbl_dks_tensor)
        return betot_tensor, bforce_tensor, bvirial_tensor


    def predict_ef(self,
                   binum_tensor: torch.Tensor,
                   bilist_tensor: torch.Tensor,
                   bnumneigh_tensor: torch.Tensor,
                   bfirstneigh_tensor: torch.Tensor,
                   brcs_tensor: torch.Tensor,
                   btypes_tensor: torch.Tensor,
                   bnghost_tensor: torch.Tensor):
        betot_tensor, bforce_tensor = nnMtpToEFOp(
            self.chebyshev_size,
            self.coeffs_tensor,
            self.w0_tensor,
            self.w1_tensor,
            self.type_bias_tensor,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            self.rmax,
            self.rmin,
            self._q_shifter_tensor,
            self._q_scaler_tensor,
            self.zbl_rmax,
            self.zbl_rmax,
            self.zbl_cks_tensor,
            self.zbl_dks_tensor)
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
        be_sites_tensor: torch.Tensor = nnMtpToEsitesOp(
            self.chebyshev_size,
            self.coeffs_tensor,
            self.w0_tensor,
            self.w1_tensor,
            self.type_bias_tensor,
            self.alpha_moments_count,
            self.alpha_index_basic_tensor,
            self.alpha_index_times_tensor,
            self.alpha_moment_mapping_tensor,
            self.nmus,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            self.rmax,
            self.rmin,
            self.zbl_rmax,
            self.zbl_rmax,
            self.zbl_cks_tensor,
            self.zbl_dks_tensor)[0]
        return be_sites_tensor


    def predict_descriptors(self,
                            binum_tensor: torch.Tensor,
                            bilist_tensor: torch.Tensor,
                            bnumneigh_tensor: torch.Tensor,
                            bfirstneigh_tensor: torch.Tensor,
                            brcs_tensor: torch.Tensor,
                            btypes_tensor: torch.Tensor,
                            bnghost_tensor: torch.Tensor):
        bdescriptors_tensor: torch.Tensor = nnMtpToDescriptorsOp(
            self.chebyshev_size,
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
            brcs_tensor,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            self.rmax,
            self.rmin)[0]
        return bdescriptors_tensor