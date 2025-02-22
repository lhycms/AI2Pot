from typing import List

import torch
import torch.nn as nn

from ai2pot.fromcc import (mtpParamOp, 
                           linearMtpToLossOp,
                           linearMtpToEFLossOp)


class LinearMtp(nn.Module):
    def __init__(self,
                 mtp_level: int,
                 ntypes: int,
                 chebyshev_size: int,
                 rmax: float,
                 rmin: float,
                 umax_num_neighs: int,
                 fit_virial: bool = False):
        super(LinearMtp, self).__init__()
        self.mtp_level: int = mtp_level
        self.ntypes: int = ntypes
        self.chebyshev_size: int = chebyshev_size
        self.rmax: float = rmax
        self.rmin: float = rmin
        self.umax_num_neighs: int = umax_num_neighs
        self.fit_virial: bool = fit_virial
        
        mtp_param_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        self.register_buffer(name="alpha_moments_count", tensor=mtp_param_info[0])
        self.alpha_moments_count: int = mtp_param_info[0].item()
        self.register_buffer(name="alpha_index_basic_tensor", tensor=mtp_param_info[1])
        self.register_buffer(name="alpha_index_times_tensor", tensor=mtp_param_info[2])
        self.register_buffer(name="alpha_moment_mapping_tensor", tensor=mtp_param_info[3])
        self.register_buffer(name="nmus_tensor", tensor=mtp_param_info[6])
        self.nmus: int = mtp_param_info[5].item()
        self.num_descriptors: int = self.alpha_moment_mapping_tensor.size()[0]
        
        coeffs_tensor: torch.Tensor = torch.Tensor(self.ntypes*self.ntypes*self.nmus*self.chebyshev_size)
        nn.init.xavier_uniform_(coeffs_tensor, gain=1.0)
        self.register_parameter(name="coeffs_tensor", param=nn.Parameter(data=coeffs_tensor))
        
        linear_coeffs_tensor: torch.Tensor = torch.Tensor(self.num_descriptors)
        nn.init.xavier_uniform_(linear_coeffs_tensor, gain=1.0)
        self.register_parameter(name="linear_coeffs_tensor", param=nn.Parameter(data=linear_coeffs_tensor))
        
        type_bias_tensor: torch.Tensor = torch.Tensor(self.ntypes)
        nn.init.xavier_uniform_(type_bias_tensor, gain=1.0)
        self.register_parameter(name="type_bias_tensor", param=nn.Parameter(data=type_bias_tensor))
    
    
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
                                                      self.ntypes,
                                                      nghost,
                                                      self.rmax,
                                                      self.rmin)
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
                                                        self.ntypes,
                                                        nghost,
                                                        self.rmax,
                                                        self.rmin)
        return bmse_tensor
    
    
    def predict_descriptor(self):
        pass
    
    
    def predict_efv(self):
        pass
    