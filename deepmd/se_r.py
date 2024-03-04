from typing import List

import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    def __init__(
            self,
            layer_size_list: List[int],
            activation: torch.nn.Module = nn.Tanh(),
            bias_mark: bool = True,
            resnet_mark: bool = False):
        super(EmbeddingNet, self).__init__()
        self.layer_size_list: List[int] = layer_size_list
        self.activation: nn.Module = activation
        self.bias_mark: bool = bias_mark
        self.resnet_mark: bool = resnet_mark
        self.linears_list: nn.Module = nn.ModuleList()
        self.resnet_list: nn.Module = nn.ParameterList()
        for ii in range(len(self.layer_size_list)):
            self.linears_list.append( nn.Linear(self.layer_size_list[ii], self.layer_size_list[ii+1], bias=bias_mark) )
            nn.init.normal_(
                self.linears_list[ii].weight, 
                mean=0.0, 
                std=(1.0 / (self.network_size[ii]+self.network_size[ii+1])**0.5))   # Why?
            if self.bias_mark:
                nn.init.normal_(self.linears_list[ii].bias, mean=0.0, std=1.0)
            if (self.layer_size_list[ii] == self.layer_size_list[ii+1]) or \
                (2 * self.layer_size_list[ii] == self.layer_size_list[ii+1]):
                resnet_tensor: torch.Tensor = torch.Tensor(1, self.layer_size_list[ii+1])
                nn.init.normal_(resnet_tensor, mean=0.0, std=0.001)
                self.resnet_list.append(nn.Parameter(resnet_tensor, requires_grad=True))
            else:
                self.resnet_list.append(nn.Identity())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
    

class FittingNet(nn.Module):
    def __init__(self):
        pass


class DpSeR(nn.Module):
    def __init__(self):
        pass