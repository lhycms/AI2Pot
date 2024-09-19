import torch
import torch.nn as nn


class ELoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, 
                energies: torch.Tensor) -> torch.Tensor:
        pass


class FLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, 
                forces: torch.Tensor) -> torch.Tensor:
        pass
    

class VLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,
                virials: torch.Tensor) -> torch.Tensor:
        pass
