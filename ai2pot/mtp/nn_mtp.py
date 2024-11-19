from typing import List, Union, Dict, Any, Optional

import torch
import torch.nn as nn
import lightning as L

from ai2pot.fromcc import mtpParamOp, mtpBasisOp, forceSrOp, virialSrOp
from ai2pot.loss import (ERmse,
                         FRmse,
                         VRmse)


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
        mtp_param_info: List[torch.Tensor] = mtpParamOp(self.mtp_level)
        self.register_buffer("alpha_index_basic_tensor", mtp_param_info[0])
        self.register_buffer("alpha_index_times_tensor", mtp_param_info[1])
        self.register_buffer("alpha_moment_mapping_tensor", mtp_param_info[2])
        self.register_buffer("num_mus4moms_tensor", mtp_param_info[3])
        self.register_buffer("mus4moms_tensor", mtp_param_info[4])
        self.register_buffer("nmus_tensor", mtp_param_info[5])
        self.nmus: int = mtp_param_info[5].item()
        self.num_descriptors: int = self.alpha_moment_mapping_tensor.size()[0]
        
        coeffs: torch.Tensor = torch.Tensor(self.ntypes*self.ntypes*self.nmus*self.chebyshev_size)
        nn.init.normal_(coeffs, mean=0, std=1.0)
        self.register_parameter("coeffs", nn.Parameter(data=coeffs, requires_grad=True))
    
    
    def forward(self,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes) -> torch.Tensor:
        descriptor: torch.Tensor = mtpBasisOp(self.alpha_index_basic_tensor,
                                              self.alpha_index_times_tensor,
                                              self.alpha_moment_mapping_tensor,
                                              self.num_mus4moms_tensor,
                                              self.mus4moms_tensor,
                                              self.nmus_tensor,
                                              self.ntypes,
                                              self.chebyshev_size,
                                              self.coeffs,
                                              bilist,
                                              bnumneigh,
                                              bfirstneigh,
                                              brcs,
                                              btypes,
                                              self.umax_num_neighs,
                                              self.rmax,
                                              self.rmin)[0]
        return descriptor
    

class FittingNet(nn.Module):
    '''FittingNet for specific element.'''
    def __init__(self,
                 num_descriptor: int,
                 fit_sizes_list: List[int] = [],
                 fit_activation: nn.Module = nn.Tanh(),
                 bias_mark: bool = True,
                 energy_shift_tensor: Union[bool, torch.Tensor] = False):
        super(FittingNet, self).__init__()
        self.layer_sizes_list: List[int] = [num_descriptor] + fit_sizes_list + [1]
        self.fit_activation: nn.Module = fit_activation
        self.bias_mark: bool = bias_mark
        
        self.linears_list: nn.Module = nn.ModuleList()
        for ii in range(len(self.layer_sizes_list) - 1):
            if (ii == len(self.layer_sizes_list) - 2):
                self.linears_list.append(
                    nn.Linear(self.layer_sizes_list[ii], self.layer_sizes_list[ii+1], bias=False))
                nn.init.normal_(self.linears_list[ii].weight, mean=0.0, std=1.0)
            else:
                self.linears_list.append(
                    nn.Linear(self.layer_sizes_list[ii], self.layer_sizes_list[ii+1], bias=self.bias_mark))
                nn.init.normal_(self.linears_list[ii].weight, mean=0.0, std=1.0)
                if (self.bias_mark):
                    nn.init.normal_(self.linears_list[ii].bias, mean=0.0, std=1.0)
        
        if not energy_shift_tensor:
            self.register_buffer("energy_shift_tensor", tensor=torch.tensor(0))
        else:
            self.register_buffer("energy_shift_tensor", tensor=energy_shift_tensor)
    
    
    def forward(self, bdescriptor_tensor: torch.Tensor):
        x: torch.Tensor = bdescriptor_tensor
        for ii, tmp_linear in enumerate(self.linears_list):
            if (ii == (len(self.linears_list) - 1)):
                hidden = tmp_linear(x)
                hidden = hidden + self.energy_shift_tensor
                x = hidden
            else:
                hidden = tmp_linear(x)
                hidden = self.fit_activation(hidden)
                x = hidden
        return x
    
    
    def info(self):
        print("FittingNet Information:")
        print("-----------------------")
        print("\t1. Linears ({0} in total):".format(len(self.linears_list)))
        for ii, tmp_linear in enumerate(self.linears_list):
            print("\t\tlinear#{0}.weight.size() = ".format(ii), tmp_linear.weight.size())
            if (self.bias_mark):
                if (ii != len(self.linears_list) - 1):
                    print("\t\t\t+ linear#{0}.bias.size() = ".format(ii), tmp_linear.bias.size())
                

class NNMtp(nn.Module):
    def __init__(self,
                 mtp_level: int,
                 ntypes: int,
                 chebyshev_size: int,
                 rmax: float,
                 rmin: float,
                 umax_num_neighs: int,
                 fit_sizes_list: Union[bool, List[int]] = False,
                 fit_activation: nn.Module = nn.Tanh(),
                 bias_mark: bool = True,
                 energy_shift_tensor: Union[bool, torch.Tensor] = False):
        if energy_shift_tensor is not False:
            assert(energy_shift_tensor.size()[0] == ntypes)
        super(NNMtp, self).__init__()
        self.mtp_level: int = mtp_level
        self.ntypes: int = ntypes
        self.chebyshev_size: int = chebyshev_size
        self.rmax: float = rmax
        self.rmin: float = rmin
        self.umax_num_neighs: int = umax_num_neighs
        self.fit_activation: nn.Module = fit_activation
        self.register_buffer("energy_shift_tensor", energy_shift_tensor)
        self.register_module("descriptor_module",
                             DescriptorMtp(self.mtp_level,
                                           self.ntypes,
                                           self.chebyshev_size,
                                           self.rmax,
                                           self.rmin,
                                           self.umax_num_neighs))
        fitting_modules_list: nn.ModuleList = []
        for ii in range(ntypes):
            fitting_modules_list.append(FittingNet(num_descriptor=self.descriptor_module.num_descriptors,
                                                   fit_sizes_list=fit_sizes_list,
                                                   fit_activation=fit_activation,
                                                   bias_mark=bias_mark,
                                                   energy_shift_tensor=energy_shift_tensor[ii]))
    
    def forward(self,
                bilist: torch.Tensor,
                bnumneigh: torch.Tensor,
                bfirstneigh: torch.Tensor,
                brcs: torch.Tensor,
                btypes: torch.Tensor,
                bnghost: torch.Tensor) -> torch.Tensor:
        brcs.requires_grad_()
        bdescriptor: torch.Tensor = self.descriptor_module(bilist,
                                                           bnumneigh,
                                                           bfirstneigh,
                                                           brcs,
                                                           btypes)
        e_i_sr: torch.Tensor = torch.zeros(bilist.size()[0], bilist.size()[1], 1)
        for itype in range(self.ntypes):
            itype_mask: torch.Tensor = (btypes[bilist].unsqueeze(dim=-1) == itype)
            e_i_sr[itype_mask] = self.fitting_modules_list[itype](btypes, bdescriptor)
        e_tot_sr: torch.Tensor = torch.sum(e_i_sr, dim=1)
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(e_tot_sr,
                                                              device=brcs.device,
                                                              dtype=brcs.dtype)]
        eisr_rij_jacobian: torch.Tensor = torch.autograd.grad([e_tot_sr],
                                                              [brcs],
                                                              grad_outputs=mask,
                                                              retain_graph=True,
                                                              create_graph=True)[0]
        force_sr: torch.Tensor = forceSrOp(bilist,
                                           bnumneigh,
                                           bfirstneigh,
                                           bnghost[0].item(),   # int32
                                           self.umax_num_neighs,
                                           eisr_rij_jacobian)[0]
        virial_sr: torch.Tensor = virialSrOp(bnumneigh,
                                             brcs,
                                             self.umax_num_neighs,
                                             eisr_rij_jacobian)[0]
        return [e_tot_sr, force_sr, virial_sr]
    
    
    def get_checkpoint_dict(self) -> Dict[str, Any]:
        pass
    
    
    def info(self):
        pass


class LitNNMtp(L.LightningModule):
    def __init__(self, 
                 model: nn.Module, 
                 has_virials: bool = True):
        super().__init__()
        self.model: nn.Module = model
        self.e_criterion: nn.Module = ERmse()
        self.f_criterion: nn.Module = FRmse()
        self.has_virials: bool = has_virials
        self.v_criterion: nn.Module = VRmse()
        
    def training_step(self, batch, batch_idx):
        if (self.has_virials):
            inum, ilist, numneigh, firstneigh, rcs, types, nghost, e_dft, fi_dft, v_dft = batch
            e_ml, fi_ml, v_ml = self.model(ilist, numneigh, firstneigh, rcs, types, nghost)
            loss = self.e_criterion(binum=inum, input_benergies=e_ml, target_benergies=e_dft)   \
                   + self.f_criterion(binum=inum, input_bforces=fi_ml, target_bforces=fi_dft)   \
                   + self.v_criterion(binum=inum, input_bvirials=v_ml, target_bvirials=v_dft)
            print("***+++ ", e_ml, e_dft)
        else:
            inum, ilist, numneigh, firstneigh, rcs, types, nghost, e_dft, fi_dft = batch
            e_ml, fi_ml, v_ml = self.model(ilist, numneigh, firstneigh, rcs, types, nghost)
            loss = self.e_criterion(binum=inum, input_benergies=e_ml, target_benergies=e_dft)   \
                   + self.f_criterion(binum=inum, input_bforces=fi_ml, target_bforces=fi_dft)
        #self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.SGD(self.model.parameters(),
                                                           lr=1e-4)
        return optimizer
