from typing import List, Optional

import torch
import torch.nn as nn

from ai2pot.fromcc import (mtpParamOp,
                           linearMtpToLossOp,
                           linearMtpToEFLossOp,
                           linearMtpToEFVOp,
                           linearMtpToEFOp)


class LinearMtp(nn.Module):
    def __init__(self,
                 mtp_level: int,
                 type_map_tensor: torch.Tensor,
                 chebyshev_size: int = 8,
                 rmax: float = 6.0,
                 rmin: float = 2.0,
                 umax_num_neighs: int = 200,
                 fit_virial: bool = False,
                 zbl_rmax: float = 2.0,
                 zbl_rmin: float = 1.0,
                 zbl_cks_tensor: Optional[torch.Tensor] = None,
                 zbl_dks_tensor: Optional[torch.Tensor] = None):
        super(LinearMtp, self).__init__()
        self.mtp_level: int = mtp_level
        self.register_buffer(name="type_map_tensor", tensor=type_map_tensor)
        self.ntypes: int = type_map_tensor.size(0)
        self.chebyshev_size: int = chebyshev_size
        self.rmax: float = rmax
        self.rmin: float = rmin
        self.umax_num_neighs: int = umax_num_neighs
        self.fit_virial: bool = fit_virial
        self.zbl_rmax: float = zbl_rmax
        self.zbl_rmin: float = zbl_rmin
        self._init_zbl_params(zbl_cks_tensor=zbl_cks_tensor, zbl_dks_tensor=zbl_dks_tensor)
        
        mtp_param_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        self.register_buffer(name="alpha_moments_count_tensor", tensor=mtp_param_info[0])
        self.alpha_moments_count: int = mtp_param_info[0].item()
        self.register_buffer(name="alpha_index_basic_tensor", tensor=mtp_param_info[1])
        self.register_buffer(name="alpha_index_times_tensor", tensor=mtp_param_info[2])
        self.register_buffer(name="alpha_moment_mapping_tensor", tensor=mtp_param_info[3])
        self.register_buffer(name="nmus_tensor", tensor=mtp_param_info[6])
        self.nmus: int = mtp_param_info[6].item()
        self.num_descriptors: int = self.alpha_moment_mapping_tensor.size()[0]
        
        coeffs_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.ntypes*self.nmus*self.chebyshev_size)
        nn.init.normal_(coeffs_tensor, mean=0.0, std=0.1)
        self.register_parameter(name="coeffs_tensor", param=nn.Parameter(data=coeffs_tensor))
        
        linear_coeffs_tensor: torch.Tensor = torch.Tensor(self.num_descriptors)
        nn.init.normal_(linear_coeffs_tensor, mean=0.0, std=0.1)
        self.register_parameter(name="linear_coeffs_tensor", param=nn.Parameter(data=linear_coeffs_tensor))
        
        type_bias_tensor: torch.Tensor = torch.Tensor(self.ntypes)
        nn.init.normal_(type_bias_tensor, mean=0.0, std=1.0)
        self.register_parameter(name="type_bias_tensor", param=nn.Parameter(data=type_bias_tensor))


    def _init_zbl_params(self, 
                         zbl_cks_tensor: Optional[torch.Tensor],
                         zbl_dks_tensor: Optional[torch.Tensor]):
        if (zbl_cks_tensor is None) or (zbl_dks_tensor is None):
            single_zbl_ck_tensor: torch.Tensor = torch.tensor([0.18175, 0.50986, 0.28022, 0.02817])
            single_zbl_dk_tensor: torch.Tensor = torch.tensor([3.1998, 0.94229, 0.4029, 0.20162])
            zbl_cks_tensor: torch.Tensor = single_zbl_ck_tensor.repeat(self.ntypes * self.ntypes)
            zbl_dks_tensor: torch.Tensor = single_zbl_dk_tensor.repeat(self.ntypes * self.ntypes)
            self.register_buffer("zbl_cks_tensor", tensor=zbl_cks_tensor)
            self.register_buffer("zbl_dks_tensor", tensor=zbl_dks_tensor)
        else:
            assert(zbl_cks_tensor.size() == self.ntypes*self.ntypes*4)
            assert(zbl_dks_tensor.size() == self.ntypes*self.ntypes*4)
            self.register_buffer("zbl_cks_tensor", tensor=zbl_cks_tensor)
            self.register_buffer("zbl_dks_tensor", tensor=zbl_dks_tensor)

    
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
        bmse_tensor: torch.Tensor = linearMtpToLossOp(e_weight,
                                                      f_weight,
                                                      v_weight,
                                                      betot_dft_tensor,
                                                      bforce_dft_tensor,
                                                      bvirial_dft_tensor,
                                                      self.chebyshev_size,
                                                      self.coeffs_tensor,
                                                      self.linear_coeffs_tensor,
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
        bmse_tensor: torch.Tensor = linearMtpToEFLossOp(e_weight,
                                                        f_weight,
                                                        betot_dft_tensor,
                                                        bforce_dft_tensor,
                                                        self.chebyshev_size,
                                                        self.coeffs_tensor,
                                                        self.linear_coeffs_tensor,
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
    
    
    def predict_efv(self,
                    binum_tensor: torch.Tensor,
                    bilist_tensor: torch.Tensor,
                    bnumneigh_tensor: torch.Tensor,
                    bfirstneigh_tensor: torch.Tensor,
                    brcs_tensor: torch.Tensor,
                    btypes_tensor: torch.Tensor,
                    bnghost_tensor: torch.Tensor):
        betot_tensor, bforce_tensor, bvirial_tensor = linearMtpToEFVOp(self.chebyshev_size,
                                                                       self.coeffs_tensor,
                                                                       self.linear_coeffs_tensor,
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
        betot_tensor, bforce_tensor = linearMtpToEFOp(self.chebyshev_size,
                                                      self.coeffs_tensor,
                                                      self.linear_coeffs_tensor,
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
                                                      self.zbl_dks_tensor)
        return betot_tensor, bforce_tensor


    def predict_descriptors(self):
        pass
