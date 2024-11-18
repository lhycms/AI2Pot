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
        self.nmus: int = self.nmus_tensor.item()
        self.num_descriptors: int = self.alpha_moment_mapping_tensor.size()[0]
        
        coeffs: torch.Tensor = torch.Tensor(self.ntypes*self.ntypes*self.nmus*self.chebyshev_size)
        nn.init.normal_(coeffs, mean=0, std=1.0)
        self.register_parameter("coeffs", nn.Parameter(coeffs, requires_grad=True))
    
    
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
    def __init__(self,
                 ntypes: int,
                 num_descriptor: int,
                 fit_sizes_list: List[int] = [],
                 fit_activation: nn.Module = nn.Tanh(),
                 bias_mark: bool = True,
                 energy_shift_tensor: Union[bool, torch.Tensor] = False):
        if energy_shift_tensor is not False:
            assert(ntypes == energy_shift_tensor.size()[0])
        super(FittingNet, self).__init__()
        self.ntypes: int = ntypes
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
            self.register_parameter("energy_shift_tensor",
                                    nn.Parameter(data=torch.zeros(self.ntypes), 
                                                requires_grad=True))
        else:
            self.register_parameter("energy_shift_tensor",
                                    nn.Parameter(data=energy_shift_tensor,
                                                 requires_grad=True))
    
    def forward(self,
                btypes: torch.Tensor,
                bdescriptor_tensor: torch.Tensor):
        nlocal: int = bdescriptor_tensor.size()[1]
        x: torch.Tensor = bdescriptor_tensor
        for ii, tmp_linear in enumerate(self.linears_list):
            if (ii == len(self.linears_list)):
                hidden = tmp_linear(x)
                hidden = hidden + self.energy_shift_tensor[btypes[:, :nlocal].unsqueeze(dim=-1)]
                x = hidden
            else:
                hidden = tmp_linear(x)
                hidden = self.fit_activation(hidden)
                x = hidden
        return x
    
    def info(self):
        print("1. Linears ({0} in total):".format(len(self.linears_list)))
        for ii, tmp_linear in enumerate(self.linears_list):
            print("\tlinear#{0}.weight.size() = ".format(ii), tmp_linear.weight.size())
            if (ii != len(self.linears_list) - 1):
                print("\t\t+ linear#{0}.bias.size() = ".format(ii), tmp_linear.bias.size())
                

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
        self.energy_shift_tensor: torch.tensor = energy_shift_tensor
        self.register_module("descriptor_module",
                             DescriptorMtp(self.mtp_level,
                                           self.ntypes,
                                           self.chebyshev_size,
                                           self.rmax,
                                           self.rmin,
                                           self.umax_num_neighs))
        self.register_module("fitting_module",
                             FittingNet(ntypes,
                                        self.descriptor_module.num_descriptors,
                                        fit_sizes_list,
                                        fit_activation,
                                        bias_mark,
                                        energy_shift_tensor))
        
    
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
        #print(bdescriptor)
        e_i_sr: torch.Tensor = self.fitting_module(btypes, bdescriptor)
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
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.Adam(self.model.parameters(),
                                                            lr=1e-4)
        return optimizer
 

'''
class LitNNMtp(L.LightningModule):
    def __init__(self, 
                 model: nn.Module,
                 max_epoches: int, 
                 lr_start: float,
                 lr_stop: float,
                 decay_steps: int,
                 has_forces: bool = True,
                 has_virials: bool = True):
        super().__init__()
        self.model: nn.Module = model
        self.max_epoches: int = max_epoches
        self.lr_start: float = lr_start
        self.lr_stop: float = lr_stop
        self.decay_steps: int = decay_steps
        self.criterion: nn.Module = nn.MSELoss()
        self.has_forces: bool = has_forces
        self.has_virials: bool = has_virials
        
        self.decay_rate: float = (lr_stop / lr_start) ** (self.decay_steps / self.max_epoches)
        
    def training_step(self, batch, batch_idx):
        if (self.has_forces and self.has_virials):
            inum, ilist, numneigh, firstneigh, rcs, types, nghost, energies, forces, virials = batch
            e, fi, v = self.model(ilist, numneigh, firstneigh, rcs, types, nghost)
            loss: torch.tensor = self.criterion(torch.sum(e), torch.sum(energies))
            return torch.sqrt(loss)
        elif (self.has_forces):
            inum, ilist, numneigh, firstneigh, rcs, types, nghost, energies, forces, virials = batch
            e, fi = self.model(ilist, numneigh, firstneigh, rcs, types, nghost)
            loss: torch.tensor = self.criterion(torch.sum(e), torch.sum(energies))
            return torch.sqrt(loss)
        else:
            pass
    
    
    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.Adam(self.model.parameters(),
                                                            lr=self.lr_start)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                      lr_lambda=lambda epoch: self.lr_start * self.decay_rate ** (epoch / self.decay_steps))
        return [optimizer], [scheduler]
'''
