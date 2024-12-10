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
                 energy_shift_tensor: torch.Tensor = torch.tensor(0.0)):
        super(FittingNet, self).__init__()
        self.layer_sizes_list: List[int] = [num_descriptor] + fit_sizes_list + [1]
        self.register_module("fit_activation", fit_activation)
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
        
        type_bias: torch.Tensor = torch.tensor(0.0)
        self.register_parameter("type_bias", nn.Parameter(data=type_bias, requires_grad=True))
        self.register_buffer("energy_shift_tensor", tensor=energy_shift_tensor)
    
    
    def forward(self, bdescriptor_tensor: torch.Tensor):
        x: torch.Tensor = bdescriptor_tensor
        for ii, tmp_linear in enumerate(self.linears_list):
            if (ii == (len(self.linears_list) - 1)):
                hidden = tmp_linear(x)
                x = hidden
            else:
                hidden = tmp_linear(x)
                hidden = self.fit_activation(hidden)
                x = hidden
        x = x + self.type_bias + self.energy_shift_tensor
        # size = (batch_size, num_atoms,)
        return x.squeeze(dim=-1)
    
    
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
                 energy_shift_tensor: Union[bool, torch.Tensor] = False,
                 has_virials: bool = False):
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
        self.register_module("descriptor_module",
                             DescriptorMtp(self.mtp_level,
                                           self.ntypes,
                                           self.chebyshev_size,
                                           self.rmax,
                                           self.rmin,
                                           self.umax_num_neighs))
        self.register_module("batch_norm", nn.BatchNorm1d(num_features=self.descriptor_module.num_descriptors))
        self.fitting_modules_list: nn.ModuleList = nn.ModuleList()
        if energy_shift_tensor is False:
            energy_shift_tensor = torch.zeros(self.ntypes)
        for ii in range(ntypes):
            self.fitting_modules_list.append(FittingNet(num_descriptor=self.descriptor_module.num_descriptors,
                                                        fit_sizes_list=fit_sizes_list,
                                                        fit_activation=fit_activation,
                                                        bias_mark=bias_mark,
                                                        energy_shift_tensor=energy_shift_tensor[ii]))
        self.has_virials: bool = has_virials
    
    def forward(self,
                bilist: torch.Tensor,
                bnumneigh: torch.Tensor,
                bfirstneigh: torch.Tensor,
                brcs: torch.Tensor,
                btypes: torch.Tensor,
                bnghost: torch.Tensor) -> torch.Tensor:
        batch_size: int = bilist.size()[0]
        brcs.requires_grad_()
        bdescriptor: torch.Tensor = self.descriptor_module(bilist,
                                                           bnumneigh,
                                                           bfirstneigh,
                                                           brcs,
                                                           btypes)
        
        e_tot_sr: torch.Tensor = torch.zeros(batch_size)
        for itype in range(self.ntypes):
            itype_mask: torch.Tensor = (torch.take(input=btypes, index=bilist.to(torch.int64)) == itype)
            itype_natoms_tensor: torch.Tensor = itype_mask.count_nonzero(dim=1)
            flatten_descriptor: torch.Tensor = self.batch_norm(bdescriptor[itype_mask])
            flatten_ei: torch.Tensor = self.fitting_modules_list[itype](flatten_descriptor)
            for bidx, flatten_ei_frame in enumerate( torch.split(flatten_ei, itype_natoms_tensor.tolist()) ):
                e_tot_sr[bidx] += flatten_ei_frame.sum()

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
        if self.has_virials:
            virial_sr: torch.Tensor = virialSrOp(bnumneigh,
                                                 brcs,
                                                 self.umax_num_neighs,
                                                 eisr_rij_jacobian)[0]
            return [e_tot_sr, force_sr, virial_sr]
        return [e_tot_sr, force_sr]
    
    
    def get_ckpt_dict(self) -> Dict[str, Any]:
        pass
    
    
    def info(self):
        pass


class LitNNMtp(L.LightningModule):
    def __init__(self, 
                 model: nn.Module,
                 lr_start: float = 1e-4,
                 e_wgt_start: float = 0.02,
                 e_wgt_end: float = 1.0,
                 f_wgt_start: float = 1000.0,
                 f_wgt_end: float = 1.0,
                 v_wgt_start: float = 0.0,
                 v_wgt_end: float = 0.0):
        super().__init__()
        self.model: nn.Module = model
        self.has_virials: bool = model.has_virials
        self.e_criterion: nn.Module = ERmse()
        self.f_criterion: nn.Module = FRmse()
        self.v_criterion: nn.Module = VRmse()
        
        self.lr_start: float = lr_start
        self.e_wgt_start: float = e_wgt_start
        self.e_wgt_end: float = e_wgt_end
        self.f_wgt_start: float = f_wgt_start
        self.f_wgt_end: float = f_wgt_end
        self.v_wgt_start: float = v_wgt_start
        self.v_wgt_end: float = v_wgt_end
        
    def get_efv_wgts(self):
        lr_current: float = self.optimizers().param_groups[0]['lr']
        print("***+++ ", lr_current)
        rate: float = lr_current / self.lr_start
        e_wgt: float = self.e_wgt_end * (1 - rate) + self.e_wgt_start * rate
        f_wgt: float = self.f_wgt_end * (1 - rate) + self.f_wgt_start * rate
        v_wgt: float = self.v_wgt_end * (1 - rate) + self.v_wgt_start * rate
        return [e_wgt, f_wgt, v_wgt]
        
    def training_step(self, batch, batch_idx):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.has_virials):
            inum, ilist, numneigh, firstneigh, rcs, types, nghost, e_dft, fi_dft, v_dft = batch
            e_ml, fi_ml, v_ml = self.model(ilist, numneigh, firstneigh, rcs, types, nghost)
            print("---MLFF---")
            print(fi_ml[0][:3])
            print("---DFT---")
            print(fi_dft[0][:3])
            print("*********")
            e_criterion_tensor: torch.Tensor = e_wgt * self.e_criterion(binum=inum, input_benergies=e_ml, target_benergies=e_dft)
            f_criterion_tensor: torch.Tensor = f_wgt * self.f_criterion(binum=inum, input_bforces=fi_ml, target_bforces=fi_dft)
            v_criterion_tensor: torch.Tensor = v_wgt * self.v_criterion(binum=inum, input_bvirials=v_ml, target_bvirials=v_dft)
            loss = e_criterion_tensor + f_criterion_tensor + v_criterion_tensor
        else:
            inum, ilist, numneigh, firstneigh, rcs, types, nghost, e_dft, fi_dft = batch
            e_ml, fi_ml, v_ml = self.model(ilist, numneigh, firstneigh, rcs, types, nghost)
            e_criterion_tensor: torch.Tensor = e_wgt * self.e_criterion(binum=inum, input_benergies=e_ml, target_benergies=e_dft)
            f_criterion_tensor: torch.Tensor = f_wgt * self.f_criterion(binum=inum, input_bforces=fi_ml, target_bforces=fi_dft)
            loss = e_criterion_tensor + f_criterion_tensor
        return loss
    
    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                            lr=self.lr_start)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.97)
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 5
                }
        }

    #def on_epoch_end(self):
    #    current_lr: float = self.optimizers().param_groups[0]['lr']
    #    self.log('learning_rate', current_lr, prog_bar=True)
