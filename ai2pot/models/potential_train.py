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
import lightning as L

from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.mtp.nn_mtp import NNMtp
from ai2pot.models.nep.nep import Nep
from ai2pot.models.potential_loss import (ERmse,
                                          FRmse,
                                          VRmse)


class LitPotential(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 lr_start: float = 1e-3,
                 e_wgt_start: float = 0.02,
                 e_wgt_end: float = 1.0,
                 f_wgt_start: float = 1000.0,
                 f_wgt_end: float = 1.0,
                 v_wgt_start: float = 0.0,
                 v_wgt_end: float = 0.0):
        super().__init__()
        self.model: nn.Module = model
        self.has_virial: bool = model.has_virial
        self.has_forces: bool = model.has_forces
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
        rate: float = lr_current / self.lr_start
        e_wgt: float = self.e_wgt_end * (1 - rate) + self.e_wgt_start * rate
        f_wgt: float = self.f_wgt_end * (1 - rate) + self.f_wgt_start * rate
        v_wgt: float = self.v_wgt_end * (1 - rate) + self.v_wgt_start * rate
        return [e_wgt, f_wgt, v_wgt]
        
    def training_step(self, batch, batch_idx):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.has_forces and self.has_virial):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, be_dft, bfi_dft, bv_dft = batch
            be_ml, bfi_ml, bv_ml = self.model(binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost)
            e_criterion_tensor: torch.Tensor = e_wgt * self.e_criterion(binum=binum, input_benergies=be_ml, target_benergies=be_dft)
            f_criterion_tensor: torch.Tensor = f_wgt * self.f_criterion(binum=binum, input_bforces=bfi_ml, target_bforces=bfi_dft)
            v_criterion_tensor: torch.Tensor = v_wgt * self.v_criterion(binum=binum, input_bvirials=bv_ml, target_bvirials=bv_dft)
            loss = e_criterion_tensor + f_criterion_tensor + v_criterion_tensor
        if (self.has_forces and (not self.has_virial)):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, be_dft, bfi_dft = batch
            be_ml, bfi_ml = self.model(binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost)
            e_criterion_tensor: torch.Tensor = e_wgt * self.e_criterion(binum=binum, input_benergies=be_ml, target_benergies=be_dft)
            f_criterion_tensor: torch.Tensor = f_wgt * self.f_criterion(binum=binum, input_bforces=bfi_ml, target_bforces=bfi_dft)
            loss = e_criterion_tensor + f_criterion_tensor
            print(bfi_ml[0][:3])
            print(bfi_dft[0][:3])
            print("-------------")
            print(be_ml, be_dft)
        if ((not self.has_forces) and (not self.has_virial)):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, be_dft, bfi_dft = batch
            be_ml = self.model(binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost)[0]
            e_criterion_tensor: torch.Tensor = e_wgt * self.e_criterion(binum=binum, input_benergies=be_ml, target_benergies=be_dft)
            loss = e_criterion_tensor
            print("-------------")
            print(self.model.descriptor_module.coeffs.requires_grad)
            print(self.model.descriptor_module.coeffs.grad)
            print(be_ml, be_dft)
            print("-------------")
        return loss
    
    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                            lr=self.lr_start)
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                           gamma=0.9)
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
        }

    #def on_epoch_end(self):
    #    current_lr: float = self.optimizers().param_groups[0]['lr']
    #    self.log('learning_rate', current_lr, prog_bar=True)


class LitLinearMtp(L.LightningModule):
    def __init__(self,
                 mtp_level: int,
                 type_map: List[int],
                 energy_shifts: Optional[List[float]] = None,
                 chebyshev_size: int = 8,
                 rmax: float = 5.0,
                 rmin: float = 0.0,
                 umax_num_neigh_atoms: int = 200,
                 fit_virial: bool = False,
                 zbl_rmax: float = 2.0,
                 zbl_rmin: float = 1.0,
                 zbl_cks_list: Optional[List[float]] = None,
                 zbl_dks_list: Optional[List[float]] = None,
                 torch_float_dtype: torch._C.dtype = torch.float32,
                 lr_start: float = 1e-1,
                 lr_end: float = 1e-4,
                 e_wgt_start: float = 0.02,
                 e_wgt_end: float = 1.0,
                 f_wgt_start: float = 1000.0,
                 f_wgt_end: float = 0.1,
                 v_wgt_start: float = 0.0,
                 v_wgt_end: float = 0.0,
                 lr_decay_step: int = 5000):
        super(LitLinearMtp, self).__init__()
        self.save_hyperparameters()
        
        self.model: nn.Module = LinearMtp(mtp_level=mtp_level,
                                          type_map=type_map,
                                          energy_shifts=energy_shifts,
                                          chebyshev_size=chebyshev_size,
                                          rmax=rmax,
                                          rmin=rmin,
                                          umax_num_neigh_atoms=umax_num_neigh_atoms,
                                          fit_virial=fit_virial,
                                          zbl_rmax=zbl_rmax,
                                          zbl_rmin=zbl_rmin,
                                          zbl_cks_list=zbl_cks_list,
                                          zbl_dks_list=zbl_dks_list).to(torch_float_dtype)
        
        self.torch_float_dtype: torch._C.dtype = torch_float_dtype

        self.lr_start: float = lr_start
        self.lr_end: float = lr_end
        self.e_wgt_start: float = e_wgt_start
        self.e_wgt_end: float = e_wgt_end
        self.f_wgt_start: float = f_wgt_start
        self.f_wgt_end: float = f_wgt_end
        self.v_wgt_start: float = v_wgt_start
        self.v_wgt_end: float = v_wgt_end
        self.lr_decay_step: int = lr_decay_step
    
    
    def get_efv_wgts(self):
        lr_current: float = self.optimizers().param_groups[0]["lr"]
        rate: float = lr_current / self.lr_start
        e_wgt: float = self.e_wgt_start * rate + self.e_wgt_end * (1 - rate)
        f_wgt: float = self.f_wgt_start * rate + self.f_wgt_end * (1 - rate)
        v_wgt: float = self.v_wgt_start * rate + self.v_wgt_end * (1 - rate)
        return [e_wgt, f_wgt, v_wgt]


    def training_step(self, batch, batch_idx):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   v_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   bvirial_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()
        self.log("train_mse", mean_bmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        
        # log : lr, e_weight, f_weight, v_weight
        current_lr: float = self.optimizers().param_groups[0]["lr"]
        self.log("lr", 
                 current_lr,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        self.log("e_wgt",
                 e_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        self.log("f_wgt",
                 f_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True) 
        self.log("v_wgt",
                 v_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True) 
        return mean_bmse_tensor


    def validation_step(self, batch_data, batch_idx: int):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch_data
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   v_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   bvirial_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch_data
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()
        self.log("valid_mse", mean_bmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        return mean_bmse_tensor


    def test_step(self, batch_data, batch_idx: int):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch_data
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   v_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   bvirial_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch_data
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()
        self.log("test_mse", mean_bmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        return mean_bmse_tensor
        


    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                             lr=self.lr_start)
        

        #work like DeepMD
        def lr_lambda(step: int):
            lr_currrent: float = self.lr_start * (self.lr_end / self.lr_start) ** (step / self.lr_decay_step)
            return max(lr_currrent, self.lr_end) / self.lr_start

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                },
                # gradient clip
                'gradient_clip_val': 1.0,
                'gradient_clip_algorithm': 'norm'
        }



class LitNNMtp(L.LightningModule):
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
                 zbl_dks_list: Optional[List[float]] = None,
                 torch_float_dtype: torch._C.dtype = torch.float32,
                 lr_start: float = 1e-2,
                 lr_end: float = 1e-4,
                 e_wgt_start: float = 0.02,
                 e_wgt_end: float = 1.0,
                 f_wgt_start: float = 1000.0,
                 f_wgt_end: float = 0.1,
                 v_wgt_start: float = 0.0,
                 v_wgt_end: float = 0.0,
                 lr_decay_step: int = 5000):
        super(LitNNMtp, self).__init__()
        self.save_hyperparameters()
        
        self.model: nn.Module = NNMtp(mtp_level=mtp_level,
                                      type_map=type_map,
                                      energy_shifts=energy_shifts,
                                      chebyshev_size=chebyshev_size,
                                      num_neurons=num_neurons,
                                      rmax=rmax,
                                      rmin=rmin,
                                      umax_num_neigh_atoms=umax_num_neigh_atoms,
                                      fit_virial=fit_virial,
                                      zbl_rmax=zbl_rmax,
                                      zbl_rmin=zbl_rmin,
                                      zbl_cks_list=zbl_cks_list,
                                      zbl_dks_list=zbl_dks_list).to(torch_float_dtype)
        
        self.torch_float_dtype: torch._C.dtype = torch_float_dtype

        self.lr_start: float = lr_start
        self.lr_end: float = lr_end
        self.e_wgt_start: float = e_wgt_start
        self.e_wgt_end: float = e_wgt_end
        self.f_wgt_start: float = f_wgt_start
        self.f_wgt_end: float = f_wgt_end
        self.v_wgt_start: float = v_wgt_start
        self.v_wgt_end: float = v_wgt_end
        self.lr_decay_step: int = lr_decay_step
    
    
    def get_efv_wgts(self):
        lr_current: float = self.optimizers().param_groups[0]["lr"]
        rate: float = lr_current / self.lr_start
        e_wgt: float = self.e_wgt_start * rate + self.e_wgt_end * (1 - rate)
        f_wgt: float = self.f_wgt_start * rate + self.f_wgt_end * (1 - rate)
        v_wgt: float = self.v_wgt_start * rate + self.v_wgt_end * (1 - rate)
        return [e_wgt, f_wgt, v_wgt]


    def training_step(self, batch, batch_idx):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   v_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   bvirial_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()
        self.log("train_mse", mean_bmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        
        # log : lr, e_weight, f_weight, v_weight
        current_lr: float = self.optimizers().param_groups[0]["lr"]
        self.log("lr", 
                 current_lr,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        self.log("e_wgt",
                 e_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        self.log("f_wgt",
                 f_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True) 
        self.log("v_wgt",
                 v_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True) 
        return mean_bmse_tensor


    def validation_step(self, batch_data, batch_idx: int):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch_data
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   v_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   bvirial_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch_data
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()
        self.log("valid_mse", mean_bmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        return mean_bmse_tensor


    def test_step(self, batch_data, batch_idx: int):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch_data
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   v_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   bvirial_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch_data
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()
        self.log("test_mse", mean_bmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        return mean_bmse_tensor
        
    
    def on_after_backward(self):
        total_norm: float = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                self.log(f"grad_norm/{name}", param_norm.item(), on_step=True, prog_bar=False)
        total_norm = total_norm ** 0.5
        self.log("grad_norm_total", total_norm, on_step=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                             lr=self.lr_start)

        '''
        #work like DeepMD
        def lr_lambda(step: int):
            lr_currrent: float = self.lr_start * (self.lr_end / self.lr_start) ** (step / self.lr_decay_step)
            return max(lr_currrent, self.lr_end) / self.lr_start

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        '''

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               T_max=self.lr_decay_step, 
                                                               eta_min=self.lr_end)

        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                },
        }


class LitNep(L.LightningModule):
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
                 rmax: float = 5.0,
                 rmin: float = 0.0,
                 zbl_rmax: float = 2.0,
                 zbl_rmin: float = 1.0,
                 zbl_cks_list: Optional[List[float]] = None,
                 zbl_dks_list: Optional[List[float]] = None,
                 torch_float_dtype: torch._C.dtype = torch.float32,
                 lr_start: float = 1e-3,
                 lr_end: float = 1e-3,
                 e_wgt_start: float = 1.0,
                 e_wgt_end: float = 1.0,
                 f_wgt_start: float = 0.1,
                 f_wgt_end: float = 0.1,
                 v_wgt_start: float = 0.01,
                 v_wgt_end: float = 0.01,
                 lr_decay_step: int = 5000):
        super(LitNep, self).__init__()
        self.save_hyperparameters()

        self.model: nn.Module = Nep(type_map=type_map,
                                    energy_shifts=energy_shifts,
                                    umax_num_neigh_atoms=umax_num_neigh_atoms,
                                    fit_virial=fit_virial,
                                    n_radial_basis=n_radial_basis,
                                    n_angular_basis=n_angular_basis,
                                    l_max=l_max,
                                    chebyshev_size=chebyshev_size,
                                    num_neurons=num_neurons,
                                    rmax=rmax,
                                    rmin=rmin,
                                    zbl_rmax=zbl_rmax,
                                    zbl_rmin=zbl_rmin,
                                    zbl_cks_list=zbl_cks_list,
                                    zbl_dks_list=zbl_dks_list)

        self.torch_float_dtype: torch._C.dtype = torch_float_dtype

        self.lr_start: float = lr_start
        self.lr_end: float = lr_end
        self.e_wgt_start: float= e_wgt_start
        self.e_wgt_end: float = e_wgt_end
        self.f_wgt_start: float = f_wgt_start
        self.f_wgt_end: float = f_wgt_end
        self.v_wgt_start: float = v_wgt_start
        self.v_wgt_end: float = v_wgt_end
        self.lr_decay_step: int = lr_decay_step
    

    def get_efv_wgts(self):
        lr_current: float = self.optimizers().param_groups[0]["lr"]
        rate: float = lr_current / self.lr_start
        e_wgt: float = self.e_wgt_start * rate + self.e_wgt_end * (1 - rate)
        f_wgt: float = self.f_wgt_start * rate + self.f_wgt_end * (1 - rate)
        v_wgt: float = self.v_wgt_start * rate + self.v_wgt_end * (1 - rate)
        return [e_wgt, f_wgt, v_wgt]


    def training_step(self, batch_data, batch_idx):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            pass
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch_data
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()

        ### Log ###
        self.log("train_mse", mean_bmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        current_lr: float = self.optimizers().param_groups[0]["lr"]
        self.log("lr", 
                 current_lr,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        self.log("e_wgt",
                 e_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        self.log("f_wgt",
                 f_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True) 
        self.log("v_wgt",
                 v_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        ### Log ###

        return mean_bmse_tensor


    def validation_step(self, batch_date, batch_idx):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            pass
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch_date
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()
        self.log("valid_mse", mean_bmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        return mean_bmse_tensor


    def test_step(self, batch_data, batch_idx):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            pass
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch_data
            bmse_tensor: torch.Tensor = self.model(e_wgt,
                                                   f_wgt,
                                                   v_wgt,
                                                   betot_dft_tensor,
                                                   bforce_dft_tensor,
                                                   binum,
                                                   bilist,
                                                   bnumneigh,
                                                   bfirstneigh,
                                                   brcs,
                                                   btypes,
                                                   bnghost[0].item())
        
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()
        self.log("test_mse", mean_bmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        return mean_bmse_tensor


    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                             lr=self.lr_start)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=self.lr_decay_step,
                                                               eta_min=self.lr_end)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    