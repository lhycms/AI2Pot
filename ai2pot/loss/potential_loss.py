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
                target_benergyies: torch.TensorType) -> torch.Tensor:
        loss: torch.TensorType = self.criterion(input_benergies,
                                                target_benergyies)
        binum = binum.view(input_benergies.size()[0], 1)
        loss = loss / binum
        return torch.sqrt(loss)


class FRmse(nn.Module):
    def __init__(self):
        super(FRmse, self).__init__()
        self.criterion: nn.Module = nn.MSELoss(reduction="none")
    
    def forward(self,
                binum: torch.TensorType,
                input_bforces: torch.TensorType,
                target_bforces: torch.TensorType) -> torch.Tensor:
        loss: torch.TensorType = self.criterion(input_bforces,
                                                target_bforces)
        binum = binum.view(input_bforces.size()[0], 1, 1)
        loss = loss / binum
        return torch.sqrt(loss)


class VRmse(nn.Module):
    def __init__(self):
        super(VRmse, self).__init__()
        self.criterion: nn.Module = nn.MSELoss(reduction="none")
        
    def forward(self,
                binum: torch.TensorType,
                input_virials: torch.TensorType,
                target_virials: torch.TensorType) -> torch.Tensor:
        loss: torch.TensorType = self.criterion(input_virials,
                                                target_virials)
        binum = binum.view(input_virials.size()[0], 1, 1)
        loss = loss / binum
        return torch.sqrt(loss)


class PotRmse(nn.Module):
    def __init__(self,
                 energy_mark: bool,
                 force_mark: bool,
                 virial_mark: bool):
        super(PotRmse, self).__init__()
        self.energy_mark = energy_mark
        self.force_mark = force_mark
        self.virial_mark = virial_mark
            
    def forward(self,
                binum: torch.Tensor,
                input_benergies: Optional[torch.Tensor],
                input_bforces: Optional[torch.Tensor],
                input_bvirials: Optional[torch.Tensor],
                target_benergies: Optional[torch.TensorType],
                target_bforces: Optional[torch.TensorType],
                target_bvirials: Optional[torch.TensorType]):
        if self.energy_mark and input_benergies and target_benergies:
            pass
        if self.force_mark and input_bforces and target_bforces:
            pass
        if self.virial_mark and input_bvirials and target_bvirials:
            pass
        