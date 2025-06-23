from typing import List, Optional
import torch
import torch.nn as nn
import lightning as L

from ai2pot.models.mtp.linear_mtp import LinearMtp
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
                 type_map_tensor: torch.Tensor,
                 chebyshev_size: int,
                 rmax: float = 6.0,
                 rmin: float = 2.0,
                 umax_num_neighs: int = 200,
                 fit_virial: bool = False,
                 zbl_rmax: float = 2.0,
                 zbl_rmin: float = 1.0,
                 zbl_cks_tensor: Optional[torch.Tensor] = None,
                 zbl_dks_tensor: Optional[torch.Tensor] = None,
                 torch_float_dtype: torch._C.dtype = torch.float32,
                 lr_start: float = 1e-3,
                 lr_end: float = 1e-3,
                 e_wgt_start: float = 1.0,
                 e_wgt_end: float = 1.0,
                 f_wgt_start: float = 0.1,
                 f_wgt_end: float = 0.1,
                 v_wgt_start: float = 0.0,
                 v_wgt_end: float = 0.0,
                 lr_decay_epoch: int = 30):
        super(LitLinearMtp, self).__init__()
        self.save_hyperparameters(ignore=[zbl_cks_tensor, zbl_dks_tensor])

        self.model: nn.Module = LinearMtp(mtp_level=mtp_level,
                                          type_map_tensor=type_map_tensor,
                                          chebyshev_size=chebyshev_size,
                                          rmax=rmax,
                                          rmin=rmin,
                                          umax_num_neighs=umax_num_neighs,
                                          fit_virial=fit_virial,
                                          zbl_rmax=zbl_rmax,
                                          zbl_rmin=zbl_rmin,
                                          zbl_cks_tensor=zbl_cks_tensor,
                                          zbl_dks_tensor=zbl_dks_tensor).to(torch_float_dtype)
        
        self.torch_float_dtype: torch._C.dtype = torch_float_dtype

        self.lr_start: float = lr_start
        self.lr_end: float = lr_end
        self.e_wgt_start: float = e_wgt_start
        self.e_wgt_end: float = e_wgt_end
        self.f_wgt_start: float = f_wgt_start
        self.f_wgt_end: float = f_wgt_end
        self.v_wgt_start: float = v_wgt_start
        self.v_wgt_end: float = v_wgt_end
        self.lr_decay_epoch: int = lr_decay_epoch
    
    
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
        
        def lr_lambda(epoch: int):
            lr_currrent: float = self.lr_start * (self.lr_end / self.lr_start) ** (epoch / self.lr_decay_epoch)
            return max(lr_currrent, self.lr_end) / self.lr_start

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
        }
    