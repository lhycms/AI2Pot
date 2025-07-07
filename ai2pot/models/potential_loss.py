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
