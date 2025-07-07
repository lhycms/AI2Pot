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


import numpy as np
import torch
import torch.nn as nn


class ResidualLinear(nn.Module):
    def __init__(
        self,
        num_in: int,
        num_out: int,
        activation: nn.Module = nn.Tanh(),
        bavg: float = 0.0,
        stddev: float = 1.0,
        resnet_dt: bool = False):
        super(ResidualLinear, self).__init__()
        self.num_in: int = num_in
        self.num_out: int = num_out
        self.activation: nn.Module = activation
        self.resnet_dt: int = resnet_dt
        
        self.matrix: torch.Tensor = torch.nn.Parameter(data=torch.Tensor(num_in, num_out), requires_grad=True)
        nn.init.normal_(self.matrix.data, std=stddev / np.sqrt(num_out + num_in))
        self.bias: torch.Tensor = torch.nn.Parameter(data=torch.Tensor(1, num_out), requires_grad=True)
        nn.init.normal_(self.bias.data, mean=bavg, std=stddev)
        if self.resnet_dt:
            self.idt = torch.nn.Parameter(data=torch.Tensor(1, num_out), requires_grad=True)
            nn.init.normal_(data=self.idt.data, mean=1.0, std=0.001)
    
    def forward(self, inputs: torch.Tensor):
        xw_plus_b: torch.Tensor = torch.matmul(inputs, self.matrix) + self.bias
        hidden: torch.Tensor = self.activation(xw_plus_b)
        if self.resnet_dt:
            hidden = hidden * self.idt
            if (self.num_in == self.num_out):
                return inputs + hidden
            elif (self.num_in * 2 == self.num_out):
                return torch.cat([inputs, inputs], dim=1) + hidden
            else:
                return hidden
        else:
            return hidden