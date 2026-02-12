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
                           nepToEFOp,
                           nepToDescriptorsOp)


class Nep(nn.Module):
    def __init__(self,
                 type_map: List[int],
                 energy_shifts: Optional[List[float]] = None,
                 umax_num_neigh_atoms: int = 200,
                 fit_virial: bool = False,
                 n_radial_basis: int = 6,
                 n_angular_basis: int = 4,
                 l_max: int = 4,
                 chebyshev_size: int = 8,
                 num_neurons: int = 30,
                 rmax_radial: float = 5.0,
                 rmax_angular: float = 0.0,
                 zbl_rmax: float = 0.0,
                 zbl_rmin: float = 0.0,
                 zbl_cks_list: Optional[List[float]] = None,
                 zbl_dks_list: Optional[List[float]] = None):
        super(Nep, self).__init__()
        assert(n_radial_basis <= 20)
        assert(n_angular_basis <= 12)
        assert(l_max <= 4)

        self.register_buffer(name="type_map_tensor", tensor=torch.tensor(type_map, dtype=torch.int32))
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
        self.zbl_rmax: float = zbl_rmax
        self.zbl_rmin: float = zbl_rmin
        self._init_zbl_params(zbl_cks_list=zbl_cks_list,
                              zbl_dks_list=zbl_dks_list)

        self.num_descriptors: int = n_radial_basis + n_angular_basis * l_max
        self.num_coeffs: int = self.ntypes * self.ntypes * (self.n_radial_basis+self.n_angular_basis) * self.chebyshev_size

        coeffs_tensor: torch.Tensor = torch.Tensor(self.num_coeffs)
        nn.init.normal_(coeffs_tensor, mean=0.0, std=0.1)
        self.register_parameter(name="coeffs_tensor", param=nn.Parameter(data=coeffs_tensor))

        w0_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.num_descriptors*self.num_neurons)
        nn.init.normal_(w0_tensor, mean=0.0, std=0.1)
        self.register_parameter(name="w0_tensor", param=nn.Parameter(data=w0_tensor))
        
        b0_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.num_neurons)
        nn.init.normal_(b0_tensor, mean=0.0, std=0.1)
        self.register_parameter(name="b0_tensor", param=nn.Parameter(data=b0_tensor))

        w1_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.num_neurons)
        nn.init.normal_(w1_tensor, mean=0.0, std=0.1)
        self.register_parameter(name="w1_tensor", param=nn.Parameter(data=w1_tensor))

        if energy_shifts is not None:
            assert(len(energy_shifts) == self.ntypes)
            type_bias_tensor: torch.Tensor = torch.tensor(energy_shifts, dtype=torch.float32)
            self.register_parameter(name="type_bias_tensor", param=nn.Parameter(data=type_bias_tensor))
        else:
            type_bias_tensor: torch.Tensor = torch.Tensor(self.ntypes)
            nn.init.normal_(type_bias_tensor, mean=0.0, std=1.0)
            self.register_parameter(name="type_bias_tensor", param=nn.Parameter(data=type_bias_tensor))
    
        q_scaler_tensor: torch.Tensor = torch.ones(self.num_descriptors, dtype=torch.float32)
        self.register_buffer("_q_scaler_tensor", tensor=q_scaler_tensor)

    @property
    def q_scaler_tensor(self):
        return self._q_scaler_tensor


    @q_scaler_tensor.setter
    def q_scaler_tensor(self, tensor: torch.Tensor):
        self._q_scaler_tensor = tensor


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
            assert(zbl_cks_tensor.size() == self.ntypes*self.ntypes*4)
            assert(zbl_dks_tensor.size() == self.ntypes*self.ntypes*4)
            self.register_buffer("zbl_cks_tensor", tensor=zbl_cks_tensor)
            self.register_buffer("zbl_dks_tensor", tensor=zbl_dks_tensor)
    

    def get_num_descriptors(self):
        return self.num_descriptors
    
    
    def init_one(self):
        with torch.no_grad():
            self.coeffs_tensor.fill_(1.0)
            #self.w0_tensor.fill_(1.0)
            #self.w1_tensor.fill_(1.0)
            #self.type_bias_tensor.fill_(1.0)


    def forward(self, *args, **kwargs):
        if (self.fit_virial):
            pass
        else:
            bmse_tensor: torch.Tensor = self.predict_ef_loss(*args, **kwargs)
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
        bmse_tensor: torch.Tensor = nepToEFLossOp(
            e_weight,
            f_weight,
            betot_dft_tensor,
            bforce_dft_tensor,
            self.chebyshev_size,
            self.n_radial_basis,
            self.n_angular_basis,
            self.l_max,
            self.coeffs_tensor,
            self.w0_tensor,
            self.b0_tensor,
            self.w1_tensor,
            self.type_bias_tensor,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            self.type_map_tensor,
            nghost,
            self.rmax_radial,
            self.rmax_angular,
            self._q_scaler_tensor,
            self.zbl_rmax,
            self.zbl_rmin,
            self.zbl_cks_tensor,
            self.zbl_dks_tensor)[0]

        return bmse_tensor


    def predict_ef(self,
                   binum_tensor: torch.Tensor,
                   bilist_tensor: torch.Tensor,
                   bnumneigh_tensor: torch.Tensor,
                   bfirstneigh_tensor: torch.Tensor,
                   brcs_tensor: torch.Tensor,
                   btypes_tensor: torch.Tensor,
                   bnghost_tensor: torch.Tensor):
        betot_tensor, bforce_tensor = nepToEFOp(
            self.chebyshev_size,
            self.n_radial_basis,
            self.n_angular_basis,
            self.l_max,
            self.coeffs_tensor,
            self.w0_tensor,
            self.b0_tensor,
            self.w1_tensor,
            self.type_bias_tensor,
            binum_tensor,
            bilist_tensor,
            bnumneigh_tensor,
            bfirstneigh_tensor,
            brcs_tensor,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            self.rmax_radial,
            self.rmax_angular,
            self._q_scaler_tensor,
            self.zbl_rmax,
            self.zbl_rmin,
            self.zbl_cks_tensor,
            self.zbl_dks_tensor)
        
        return betot_tensor, bforce_tensor
    

    def predict_descriptors(self,
                            binum_tensor: torch.Tensor,
                            bilist_tensor: torch.Tensor,
                            bnumneigh_tensor: torch.Tensor,
                            bfirstneigh_tensor: torch.Tensor,
                            brcs_tensor: torch.Tensor,
                            btypes_tensor: torch.Tensor,
                            bnghost_tensor: torch.Tensor):
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
            brcs_tensor,
            btypes_tensor,
            self.type_map_tensor,
            bnghost_tensor[0].item(),
            self.rmax_radial,
            self.rmax_angular)[0]
        
        return bdescriptors_tensor
    