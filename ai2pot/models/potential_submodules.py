from typing import List

import torch
import torch.nn as nn


class FittingNet(nn.Module):
    '''FittingNet for specific element.'''
    def __init__(self,
                 num_descriptor: int,
                 fit_sizes_list: List[int] = [],
                 fit_activation: nn.Module = nn.Tanh(),
                 bias_mark: bool = True,
                 energy_shift_tensor: torch.Tensor = torch.tensor(0.0)):
        super(FittingNet, self).__init__()
        self.layer_sizes_list: List[int] = [num_descriptor] + fit_sizes_list + [1]
        self.register_module("fit_activation", fit_activation)
        self.bias_mark: bool = bias_mark
        
        self.linears_list: nn.Module = nn.ModuleList()
        for ii in range(len(self.layer_sizes_list) - 1):
            if (ii == len(self.layer_sizes_list) - 2):
                self.linears_list.append(
                    nn.Linear(self.layer_sizes_list[ii], self.layer_sizes_list[ii+1], bias=False))
                nn.init.xavier_normal_(self.linears_list[ii].weight)
            else:
                self.linears_list.append(
                    nn.Linear(self.layer_sizes_list[ii], self.layer_sizes_list[ii+1], bias=self.bias_mark))
                nn.init.xavier_normal_(self.linears_list[ii].weight)
                if (self.bias_mark):
                    nn.init.normal_(self.linears_list[ii].bias, mean=0.0, std=0.1)
        
        type_bias: torch.Tensor = torch.tensor(0.0)
        self.register_parameter("type_bias", nn.Parameter(data=type_bias, requires_grad=True))
        self.register_buffer("energy_shift_tensor", tensor=energy_shift_tensor)
    
    
    def forward(self, bdescriptor_tensor: torch.Tensor):
        x: torch.Tensor = bdescriptor_tensor
        for ii, tmp_linear in enumerate(self.linears_list):
            if (ii == (len(self.linears_list) - 1)):
                hidden = tmp_linear(x)
                x = hidden
            else:
                hidden = tmp_linear(x)
                hidden = self.fit_activation(hidden)
                x = hidden
        x = x + self.type_bias + self.energy_shift_tensor
        return x.squeeze(dim=-1)
    
    
    def info(self):
        print("FittingNet Information:")
        print("-----------------------")
        print("\t1. Linears ({0} in total):".format(len(self.linears_list)))
        for ii, tmp_linear in enumerate(self.linears_list):
            print("\t\tlinear#{0}.weight.size() = ".format(ii), tmp_linear.weight.size())
            if (self.bias_mark):
                if (ii != len(self.linears_list) - 1):
                    print("\t\t\t+ linear#{0}.bias.size() = ".format(ii), tmp_linear.bias.size())


class ForceVirialSrModule(nn.Module):
    def __init__(self):
        super(ForceVirialSrModule, self).__init__()
    
    
    def forward(self, x: torch.Tensor):
        return x


class Descriptor2EtotModule(nn.Module):
    def __init__(self):
        super(Descriptor2EtotModule, self).__init__()
        
    def forward(self, x: torch.Tensor):
        return x
