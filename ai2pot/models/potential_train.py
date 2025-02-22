import torch
import torch.nn as nn
import lightning as L

from ai2pot.models.potential_loss import (ERmse,
                                          FRmse,
                                          VRmse)



class LitPotential(L.LightningModule):
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



class LitPotentialToLoss(L.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 lr_start: float = 1e-3,
                 lr_end: float = 1e-5,
                 e_wgt_start: float = 1.0,
                 e_wgt_end: float = 10.0,
                 f_wgt_start: float = 1000.0,
                 f_wgt_end: float = 10.0,
                 v_wgt_start: float = 0.0,
                 v_wgt_end: float = 0.0):
        super(LitPotentialToLoss, self).__init__()
        self.model: nn.Module = model
        self.lr_start: float = lr_start
        self.lr_end: float = lr_end
        self.e_wgt_start: float = e_wgt_start
        self.e_wgt_end: float = e_wgt_end
        self.f_wgt_start: float = f_wgt_start
        self.f_wgt_end: float = f_wgt_end
        self.v_wgt_start: float = v_wgt_start
        self.v_wgt_end: float = v_wgt_end
    
    
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
            pass
        else:
            pass
    
    
    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                            lr=self.lr_start)
        scheduler_exp = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer,
                                                               gamma=0.9)
        
        def lr_lambda(epoch):
            current_lr = scheduler_exp.get_last_lr()[0]
            return max(current_lr, self.lr_end)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
        }
    