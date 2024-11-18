from typing import Optional

import torch
import torch.nn as nn


class ERmse(nn.Module):
    def __init__(self):
        super(ERmse, self).__init__()
        self.criterion: nn.Module = nn.MSELoss(reduction="none")
        
    def forward(self,
                binum: torch.TensorType,
                input_benergies: torch.TensorType,
                target_benergies: torch.TensorType) -> torch.Tensor:
        batch_size: int = binum.size()[0]
        loss_mse: torch.Tensor = self.criterion(input_benergies,
                                                target_benergies)
        loss_rmse_avg: torch.Tensor = torch.sqrt(loss_mse / binum)
        loss: torch.Tensor = torch.sum(loss_rmse_avg) / batch_size
        return loss


class FRmse(nn.Module):
    def __init__(self):
        super(FRmse, self).__init__()
        self.criterion: nn.Module = nn.MSELoss(reduction="none")
    
    def forward(self,
                binum: torch.TensorType,
                input_bforces: torch.TensorType,
                target_bforces: torch.TensorType) -> torch.Tensor:
        batch_size: int = binum.size()[0]
        loss_mse: torch.Tensor = self.criterion(input_bforces,
                                                target_bforces)
        loss_rmse_avg: torch.Tensor = torch.sqrt(loss_mse.sum(dim=(1, 2)).view(batch_size, -1) / (3*binum))
        loss: torch.Tensor = torch.sum(loss_rmse_avg) / batch_size
        return loss
        
        
class VRmse(nn.Module):
    def __init__(self):
        super(VRmse, self).__init__()
        self.criterion: nn.Module = nn.MSELoss(reduction="none")
        
    def forward(self,
                binum: torch.TensorType,
                input_bvirials: torch.TensorType,
                target_bvirials: torch.TensorType) -> torch.Tensor:
        batch_size: int = binum.size()[0]
        loss_mse: torch.Tensor = self.criterion(input_bvirials,
                                                target_bvirials)
        loss_rmse_avg: torch.Tensor = torch.sqrt(loss_mse.sum(dim=(1, 2)).view(batch_size, -1) / (9*binum))
        loss: torch.Tensor = torch.sum(loss_rmse_avg) / batch_size
        return loss


class PotRmse(nn.Module):
    def __init__(self,
                 virial_mark: bool):
        super(PotRmse, self).__init__()
        self.virial_mark = virial_mark
            
    def forward(self,
                binum: torch.Tensor,
                input_benergies: Optional[torch.Tensor],
                input_bforces: Optional[torch.Tensor],
                input_bvirials: Optional[torch.Tensor],
                target_benergies: Optional[torch.TensorType],
                target_bforces: Optional[torch.TensorType],
                target_bvirials: Optional[torch.TensorType]):
        
        if self.virial_mark and input_bvirials and target_bvirials:
            pass
