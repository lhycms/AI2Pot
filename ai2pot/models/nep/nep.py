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

from ai2pot.fromcc import (nepToEFLossOp,
                           nepToLossOp,
                           nepToEFOp,
                           nepToEFVOp,
                           nepToDescriptorsOp)


class Nep(nn.Module):
    def __init__(self,
                 type_map: List[int],
                 umax_num_neigh_atoms: int = 200,
                 fit_virial: bool = False,
                 n_radial_basis: int = 4,
                 n_angular_basis: int = 4,
                 l_max: int = 4,
                 chebyshev_size: int = 8,
                 num_neurons: int = 30,
                 rmax_radial: float = 8.0,
                 rmax_angular: float = 4.0,
                 zbl_rmax: float = 0.0,
                 zbl_rmin: float = 0.0,
                 zbl_cks_list: Optional[List[float]] = None,
                 zbl_dks_list: Optional[List[float]] = None):
        super(Nep, self).__init__()
        assert(n_radial_basis <= 20)
        assert(n_angular_basis <= 12)
        assert(l_max <= 4)

        self.register_buffer(
            name="type_map_tensor",
            tensor=torch.as_tensor(type_map, dtype=torch.int32))
        self.ntypes: int = len(type_map)
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.fit_virial: bool = fit_virial

        self.n_radial_basis: int = n_radial_basis
        self.n_angular_basis: int = n_angular_basis
        self.l_max: int = l_max
        self.chebyshev_size: int = chebyshev_size
        self.num_neurons: int = num_neurons
        self.rmax_radial: float = rmax_radial
        self.rmax_angular: float = rmax_angular
        self.rmax: float = self.rmax_radial
        self.zbl_rmax: float = zbl_rmax
        self.zbl_rmin: float = zbl_rmin
        self._init_zbl_params(zbl_cks_list=zbl_cks_list,
                              zbl_dks_list=zbl_dks_list)

        self.num_descriptors: int = n_radial_basis + n_angular_basis * l_max
        self.num_coeffs: int = self.ntypes * self.ntypes * (self.n_radial_basis+self.n_angular_basis) * self.chebyshev_size

        ### Init ###
        coeffs_tensor: torch.Tensor = torch.Tensor(self.num_coeffs)
        r1: torch.Tensor = 0.1 * torch.randn(self.num_coeffs)
        r2: torch.Tensor = torch.rand(self.num_coeffs) - 0.5
        coeffs_tensor.copy_(r1 + r2)
        self.register_parameter(name="coeffs_tensor", param=nn.Parameter(data=coeffs_tensor))

        w0_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.num_descriptors*self.num_neurons)
        init_w0_std: torch.Tensor = (2.0 / (self.num_descriptors + self.num_neurons)) ** 0.5
        nn.init.normal_(w0_tensor, mean=0.0, std=init_w0_std)
        self.register_parameter(name="w0_tensor", param=nn.Parameter(data=w0_tensor))
        
        b0_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.num_neurons)
        nn.init.normal_(b0_tensor, mean=0.0, std=0.01)
        self.register_parameter(name="b0_tensor", param=nn.Parameter(data=b0_tensor))

        w1_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.num_neurons)
        init_w1_std: torch.Tensor = (2.0 / (self.num_neurons + 1)) ** 0.5
        nn.init.normal_(w1_tensor, mean=0.0, std=init_w1_std)
        self.register_parameter(name="w1_tensor", param=nn.Parameter(data=w1_tensor))

        type_bias_tensor: torch.Tensor = torch.Tensor(self.ntypes)
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
        return self.num_descriptors
    

    def forward(self, *args, **kwargs):
        if (self.fit_virial):
            bmse_tensor: torch.Tensor = self.predict_loss(*args, **kwargs)
        else:
            bmse_tensor: torch.Tensor = self.predict_ef_loss(*args, **kwargs)
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

        betot_dft_norm: torch.Tensor = betot_dft_tensor * conv_energy
        bforce_dft_norm: torch.Tensor = bforce_dft_tensor * conv_force
        bvirial_dft_norm: torch.Tensor = bvirial_dft_tensor * conv_virial
        brcs_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_radial_norm: float = self.rmax_radial * conv_length
        rmax_angular_norm: float = self.rmax_angular * conv_length
        zbl_rmax_norm: float = self.zbl_rmax * conv_length
        zbl_rmin_norm: float = self.zbl_rmin * conv_length

        type_bias_norm: torch.Tensor = self.type_bias_tensor * conv_energy
        zbl_cks_norm: torch.Tensor = self.zbl_cks_tensor * conv_length * conv_energy
        zbl_dks_norm: torch.Tensor = self.zbl_dks_tensor / conv_length

        #
        bmse_tensor, e_rmse_tensor, f_rmse_tensor, v_rmse_tensor = nepToLossOp(
            e_weight,
            f_weight,
            v_weight,
            betot_dft_norm,
            bforce_dft_norm,
            bvirial_dft_norm,
            self.chebyshev_size,
            self.n_radial_basis,
            self.n_angular_basis,
            self.l_max,
            self.coeffs_tensor,
            self.w0_tensor,
            self.b0_tensor,
            self.w1_tensor,
            type_bias_norm,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_radial_norm,
            rmax_angular_norm,
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

        betot_dft_norm: torch.Tensor = betot_dft_tensor * conv_energy
        bforce_dft_norm: torch.Tensor = bforce_dft_tensor * conv_force
        brcs_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_radial_norm: float = self.rmax_radial * conv_length
        rmax_angular_norm: float = self.rmax_angular * conv_length
        zbl_rmax_norm: float = self.zbl_rmax * conv_length
        zbl_rmin_norm: float = self.zbl_rmin * conv_length

        type_bias_norm: torch.Tensor = self.type_bias_tensor * conv_energy
        zbl_cks_norm: torch.Tensor = self.zbl_cks_tensor * conv_length * conv_energy
        zbl_dks_norm: torch.Tensor = self.zbl_dks_tensor / conv_length

        # 
        bmse_tensor, e_rmse_tensor, f_rmse_tensor = nepToEFLossOp(
            e_weight,
            f_weight,
            betot_dft_norm,
            bforce_dft_norm,
            self.chebyshev_size,
            self.n_radial_basis,
            self.n_angular_basis,
            self.l_max,
            self.coeffs_tensor,
            self.w0_tensor,
            self.b0_tensor,
            self.w1_tensor,
            type_bias_norm,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_radial_norm,
            rmax_angular_norm,
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


    @torch.no_grad()
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

        brcs_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_radial_norm: float = self.rmax_radial * conv_length
        rmax_angular_norm: float = self.rmax_angular * conv_length
        zbl_rmax_norm: float = self.zbl_rmax * conv_length
        zbl_rmin_norm: float = self.zbl_rmin * conv_length

        type_bias_norm: torch.Tensor = self.type_bias_tensor * conv_energy
        zbl_cks_norm: torch.Tensor = self.zbl_cks_tensor * conv_length * conv_energy
        zbl_dks_norm: torch.Tensor = self.zbl_dks_tensor / conv_length

        #
        betot_tensor, bforce_tensor, bvirial_tensor = nepToEFVOp(
            self.chebyshev_size,
            self.n_radial_basis,
            self.n_angular_basis,
            self.l_max,
            self.coeffs_tensor,
            self.w0_tensor,
            self.b0_tensor,
            self.w1_tensor,
            type_bias_norm,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_radial_norm,
            rmax_angular_norm,
            self.q_scaler_tensor,
            zbl_rmax_norm,
            zbl_rmin_norm,
            zbl_cks_norm,
            zbl_dks_norm)
        betot_tensor: torch.Tensor
        bforce_tensor: torch.Tensor
        bvirial_tensor: torch.Tensor

        #
        betot_tensor = betot_tensor / conv_energy
        bforce_tensor = bforce_tensor / conv_force
        bvirial_tensor = bvirial_tensor / conv_virial
        
        return betot_tensor, bforce_tensor, bvirial_tensor


    @torch.no_grad()
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

        brcs_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_radial_norm: float = self.rmax_radial * conv_length
        rmax_angular_norm: float = self.rmax_angular * conv_length
        zbl_rmax_norm: float = self.zbl_rmax * conv_length
        zbl_rmin_norm: float = self.zbl_rmin * conv_length

        type_bias_norm: torch.Tensor = self.type_bias_tensor * conv_energy
        zbl_cks_norm: torch.Tensor = self.zbl_cks_tensor * conv_length * conv_energy
        zbl_dks_norm: torch.Tensor = self.zbl_dks_tensor / conv_length

        #
        betot_tensor, bforce_tensor = nepToEFOp(
            self.chebyshev_size,
            self.n_radial_basis,
            self.n_angular_basis,
            self.l_max,
            self.coeffs_tensor,
            self.w0_tensor,
            self.b0_tensor,
            self.w1_tensor,
            type_bias_norm,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_radial_norm,
            rmax_angular_norm,
            self.q_scaler_tensor,
            zbl_rmax_norm,
            zbl_rmin_norm,
            zbl_cks_norm,
            zbl_dks_norm)
        
        #
        betot_tensor = betot_tensor / conv_energy
        bforce_tensor = bforce_tensor / conv_force
        
        return betot_tensor, bforce_tensor
    

    @torch.no_grad()
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

        brcs_norm: torch.Tensor = brcs_tensor * conv_length

        rmax_radial_norm: float = self.rmax_radial * conv_length
        rmax_angular_norm: float = self.rmax_angular * conv_length

        #
        bdescriptors_tensor: torch.Tensor = nepToDescriptorsOp(
            self.chebyshev_size,
            self.n_radial_basis,
            self.n_angular_basis,
            self.l_max,
            self.coeffs_tensor,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_norm,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            rmax_radial_norm,
            rmax_angular_norm)[0]
                
        return bdescriptors_tensor
    