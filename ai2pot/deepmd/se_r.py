from typing import List

import numpy as np
import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    def __init__(
        self,
        layers_size_list: List[int],
        activation: torch.nn.Module = nn.Tanh(),
        bias_mark: bool = True,
        resnet_mark: bool = False):
        super(EmbeddingNet, self).__init__()
        self.layers_size_list: List[int] = layers_size_list
        self.layers_size_list.insert(0, 1)
        self.activation: nn.Module = activation
        self.bias_mark: bool = bias_mark
        self.resnet_mark: bool = resnet_mark
        self.linears_list: nn.Module = nn.ModuleList()
        self.resnet_dt_list: nn.Module = nn.ParameterList()
        for ii in range(len(self.layers_size_list)-1):
            self.linears_list.append( 
                nn.Linear(self.layers_size_list[ii], self.layers_size_list[ii+1], bias=self.bias_mark) )
            nn.init.normal_(
                self.linears_list[ii].weight,
                mean=0.0,
                std=1.0)
            if (self.bias_mark):
                nn.init.normal_(self.linears_list[ii].bias, mean=0.0, std=1.0)
            if (self.resnet_mark):
                resnet_tensor: torch.Tensor = torch.Tensor(1, self.layers_size_list[ii+1])
                nn.init.normal_(resnet_tensor, mean=0.0, std=0.001)
                self.resnet_dt_list.append(nn.Parameter(resnet_tensor, requires_grad=True))
            else:
                self.resnet_dt_list.append(nn.Parameter(torch.tensor(1.0), requires_grad=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ii, linear in enumerate(self.linears_list):
            hidden: torch.Tensor = linear(x)
            hidden = self.activation(hidden)
            if self.resnet_mark:
                if (self.layers_size_list[ii] == self.layers_size_list[ii+1]):
                    x = self.resnet_dt_list[ii] * hidden + x
                elif (2*self.layers_size_list[ii] == self.layers_size_list[ii+1]):
                    x = self.resnet_dt_list[ii] * hidden + torch.cat((x, x), dim=1)
                else:
                    x = self.resnet_dt_list[ii] * hidden
            else:
                x = hidden
        return x


class FittingNet(nn.Module):
    def __init__(
        self,
        layer_size_list: List[int],
        activation: nn.Module,
        bias_mark: bool = True,
        resnet_mark: bool = False,
        energy_shift: float = 0.0):
        super(FittingNet, self).__init__()
        self.layer_size_list: List[int] = layer_size_list
        self.layer_size_list.append(1)
        self.activation: nn.Module = activation
        self.bias_mark: bool = bias_mark
        self.resnet_mark: bool = resnet_mark
        self.linears_list: nn.ModuleList = nn.ModuleList()
        self.resnet_dt_list: nn.ParameterList = nn.ParameterList()
        for ii in range(len(self.layer_size_list)-1):
            if (ii == len(self.layer)-2):
                self.linears_list.append(
                    nn.Linear(self.layer_size_list[ii], self.layer_size_list[ii+1], bias=True))
                nn.init.normal_(self.linears_list[ii].weight, 
                                mean=0.0, 
                                std=(1.0 / (self.network_size[ii] + self.network_size[ii + 1]) ** 0.5))
                nn.init.normal_(self.linears_list[ii].bias, mean=energy_shift, std=1.0)
            else:
                self.linears_list.append(
                    nn.Linear(self.layer_size_list[ii], self.layer_size_list[ii+1], bias=self.bias_mark))
                nn.init.normal_(self.linears_list[ii].weight, 
                                mean=0.0, 
                                std=(1.0 / (self.network_size[ii] + self.network_size[ii + 1]) ** 0.5))
                if (self.bias_mark):
                    nn.init.normal_(self.linears_list[ii].bias, mean=0.0, std=1.0)
            if self.resnet_mark:
                resnet_dt_tensor: torch.Tensor = torch.Tensor(1, layer_size_list[ii+1])
                nn.init.normal_(resnet_dt_tensor, mean=0.1, std=0.001)
                self.resnet_dt_list.append(nn.Parameter(data=resnet_dt_tensor, required_grad=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ii, linear in enumerate(self.linears_list):
            if (ii == len(self.linears_list)-1):
                hidden: torch.Tensor = linear(x)
                x = hidden
            else:
                hidden: torch.Tensor = linear(x)
                hidden = self.activation(x)
                if self.resnet_mark:
                    if (self.layer_size_list[ii] == self.layer_size_list[ii+1]):
                        x = self.resnet_dt_list[ii] * hidden + x
                    elif (2*self.layer_size_list[ii] == self.layer_size_list[ii+1]):
                        x = self.resnet_dt_list[ii] * hidden + torch.cat([x, x], dim=1)
                    else:
                        x = self.resnet_dt_list[ii] * hidden
                else:
                    x = hidden  
        return x


class DpSeR(nn.Module):
    def __init__(self):
        super(DpSeR, self).__init__()
    
    def forward(self, x):
        pass