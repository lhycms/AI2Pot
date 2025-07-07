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


from typing import List

import torch
import torch.nn as nn



class DpResNetLinear(nn.Module):
    # output * resnet + input
    def __init__(self, 
                 in_features: int,
                 out_features: int,
                 bias_mark: bool = True,
                 activation_fn: nn.Module = nn.Tanh()):
        super(DpResNetLinear, self).__init__()
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.bias_mark: bool = bias_mark
        self.activation_fn: nn.Module = activation_fn
        self.linear: nn.Module = nn.Linear(in_features=in_features, 
                                           out_features=out_features, 
                                           bias=bias_mark)
        nn.init.normal_(self.linear.weight,
                        mean=0.0,
                        std=1.0 / torch.sqrt(in_features + out_features))
        if bias_mark:
            nn.init.normal_(self.linear.bias, mean=0.0, std=1.0)
        if ( (in_features == out_features) or (in_features*2 == out_features) ):
            resnet_tensor: torch.Tensor = torch.Tensor(1, out_features)
            nn.init.normal_(resnet_tensor, mean=0.1, std=0.001)
            self.resnet_tensor: nn.Parameter = nn.Parameter(data=resnet_tensor, reqires_grad=True)
    
    def foward(self, x: torch.Tensor):
        hidden: torch.TensorType = self.linear(x)
        hidden = self.activation_fn(hidden)
        if (self.in_features == self.out_features):
            x = self.resnet_tensor * hidden + x
        elif(self.in_features * 2 == self.out_features):
            x = self.resnet_tensor * hidden + torch.cat([x, x], dim=-1)
        else:
            x = hidden
        return x



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
