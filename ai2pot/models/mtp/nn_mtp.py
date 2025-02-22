from typing import List, Union, Dict, Any

import torch
import torch.nn as nn

from ai2pot.fromcc import mtpParamOp, mtpBasisOp, forceSrOp, virialSrOp
from ai2pot.models.potential_submodules import (
    FittingNet,
    ForceVirialSrModule,
    Descriptor2EtotModule
)


class DescriptorMtp(nn.Module):
    def __init__(self,
                 mtp_level: int,
                 ntypes: int,
                 chebyshev_size: int,
                 rmax: float,
                 rmin: float,
                 umax_num_neighs: int):
        super(DescriptorMtp, self).__init__()
        self.mtp_level: int = mtp_level
        self.ntypes: int = ntypes
        self.chebyshev_size: int = chebyshev_size
        self.rmax: float = rmax
        self.rmin: float = rmin
        self.umax_num_neighs: int = umax_num_neighs
        mtp_param_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)[1:]
        self.register_buffer("alpha_index_basic_tensor", mtp_param_info[0])
        self.register_buffer("alpha_index_times_tensor", mtp_param_info[1])
        self.register_buffer("alpha_moment_mapping_tensor", mtp_param_info[2])
        self.register_buffer("num_mus4moms_tensor", mtp_param_info[3])
        self.register_buffer("mus4moms_tensor", mtp_param_info[4])
        self.register_buffer("nmus_tensor", mtp_param_info[5])
        self.nmus: int = mtp_param_info[5].item()
        self.num_descriptors: int = self.alpha_moment_mapping_tensor.size()[0]
        
        coeffs: torch.Tensor = torch.Tensor(self.ntypes*self.ntypes*self.nmus*self.chebyshev_size)
        nn.init.normal_(coeffs, mean=0, std=0.1)
        self.register_parameter("coeffs", nn.Parameter(data=coeffs, requires_grad=True))
    
    def forward(self,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes) -> torch.Tensor:
        brcs.requires_grad_(True)
        descriptor: torch.Tensor = mtpBasisOp(self.alpha_index_basic_tensor,
                                              self.alpha_index_times_tensor,
                                              self.alpha_moment_mapping_tensor,
                                              self.num_mus4moms_tensor,
                                              self.mus4moms_tensor,
                                              self.nmus_tensor,
                                              self.ntypes,
                                              self.chebyshev_size,
                                              self.coeffs,
                                              binum,
                                              bilist,
                                              bnumneigh,
                                              bfirstneigh,
                                              brcs,
                                              btypes,
                                              self.umax_num_neighs,
                                              self.rmax,
                                              self.rmin)[0]
        return descriptor


class NNMtp(nn.Module):
    def __init__(self,
                 mtp_level: int,
                 ntypes: int,
                 chebyshev_size: int,
                 rmax: float,
                 rmin: float,
                 umax_num_neighs: int,
                 batch_norm_mark: bool = True,
                 fit_sizes_list: Union[bool, List[int]] = False,
                 fit_activation: nn.Module = nn.Tanh(),
                 bias_mark: bool = True,
                 energy_shift_tensor: Union[bool, torch.Tensor] = False,
                 has_forces: bool = True,
                 has_virial: bool = False):
        if energy_shift_tensor is not False:
            assert(energy_shift_tensor.size()[0] == ntypes)
        super(NNMtp, self).__init__()
        self.mtp_level: int = mtp_level
        self.ntypes: int = ntypes
        self.chebyshev_size: int = chebyshev_size
        self.rmax: float = rmax
        self.rmin: float = rmin
        self.umax_num_neighs: int = umax_num_neighs
        self.batch_norm_mark: bool = batch_norm_mark
        self.fit_activation: nn.Module = fit_activation
        self.register_module("descriptor_module",
                             DescriptorMtp(self.mtp_level,
                                           self.ntypes,
                                           self.chebyshev_size,
                                           self.rmax,
                                           self.rmin,
                                           self.umax_num_neighs))
        #if (self.batch_norm_mark):
        self.register_module("batch_norm", nn.BatchNorm1d(num_features=self.descriptor_module.num_descriptors))
        self.fitting_modules_list: nn.ModuleList = nn.ModuleList()
        if energy_shift_tensor is False:
            energy_shift_tensor = torch.zeros(self.ntypes)
        #for ii in range(ntypes):
        #    self.fitting_modules_list.append(FittingNet(num_descriptor=self.descriptor_module.num_descriptors,
        #                                                fit_sizes_list=fit_sizes_list,
        #                                                fit_activation=fit_activation,
        #                                                bias_mark=bias_mark,
        #                                                energy_shift_tensor=energy_shift_tensor[ii]))
        self.has_forces: bool = has_forces
        self.has_virial: bool = has_virial
        
        self.linear = nn.Sequential(
            nn.Linear(self.descriptor_module.num_descriptors, 30),
            nn.Tanh(),
            nn.Linear(30, 1))
    
    def forward(self,
                binum: torch.Tensor,
                bilist: torch.Tensor,
                bnumneigh: torch.Tensor,
                bfirstneigh: torch.Tensor,
                brcs: torch.Tensor,
                btypes: torch.Tensor,
                bnghost: torch.Tensor) -> torch.Tensor:
        brcs.requires_grad_(True)
        batch_size: int = bilist.size()[0]
        bdescriptor: torch.Tensor = self.descriptor_module(binum,
                                                           bilist,
                                                           bnumneigh,
                                                           bfirstneigh,
                                                           brcs,
                                                           btypes)
        e_tot_sr = self.linear(bdescriptor).sum(dim=-1)
        
        '''
        e_tot_sr: torch.Tensor = torch.zeros(batch_size)
        for itype in range(self.ntypes):
            itype_mask: torch.Tensor = (torch.gather(input=btypes, dim=1, index=bilist.to(torch.int64)) == itype)
            itype_natoms_tensor: torch.Tensor = itype_mask.count_nonzero(dim=-1)
            if self.batch_norm_mark:
                flatten_descriptor: torch.Tensor = self.batch_norm(bdescriptor[itype_mask])
            else:
                flatten_descriptor: torch.Tenspr = bdescriptor[itype_mask]
            flatten_ei: torch.Tensor = self.fitting_modules_list[itype](flatten_descriptor)
            for bidx, flatten_ei_frame in enumerate( torch.split(flatten_ei, itype_natoms_tensor.tolist()) ):
                e_tot_sr[bidx] = torch.add(e_tot_sr[bidx], flatten_ei_frame.sum())
        
        if (self.has_forces):
            eisr_rij_jacobian: torch.Tensor = torch.autograd.grad(outputs=[e_tot_sr],
                                                                  inputs=[brcs],
                                                                  grad_outputs=[torch.ones_like(e_tot_sr)],
                                                                  retain_graph=True,
                                                                  create_graph=True)[0]
            force_sr: torch.Tensor = forceSrOp(binum,
                                               bilist,
                                               bnumneigh,
                                               bfirstneigh,
                                               bnghost[0].item(),   # int32
                                               self.umax_num_neighs,
                                               eisr_rij_jacobian)[0]
            if self.has_virial:
                virial_sr: torch.Tensor = virialSrOp(bnumneigh,
                                                     brcs,
                                                     self.umax_num_neighs,
                                                     eisr_rij_jacobian)[0]
                return [e_tot_sr, force_sr, virial_sr]
            return [e_tot_sr, force_sr]
        '''
        return [e_tot_sr]
    
    
    def get_ckpt_dict(self) -> Dict[str, Any]:
        pass
    
    
    def info(self):
        pass
