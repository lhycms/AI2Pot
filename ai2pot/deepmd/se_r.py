from typing import List, Union

import numpy as np
import torch
import torch.nn as nn

from matersdk.fromcc import envMatrixOp


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
                    x = self.resnet_dt_list[ii] * hidden + torch.cat((x, x), dim=-1)
                else:
                    x = self.resnet_dt_list[ii] * hidden
            else:
                x = hidden
        return x

    def info(self):
        print("1. Linears ({0} in total):".format(len(self.linears_list)))
        for ii, tmp_linear in enumerate(self.linears_list):
            print("\tlinear#{0}.weight.size() = ".format(ii), tmp_linear.weight.size())
            print("\t\t+ linear#{0}.bias.size() = ".format(ii), tmp_linear.bias.size())
        print("\n")
        print("2. Resnet_dt ({0} in total):".format(len(self.resnet_dt_list)))
        for ii, tmp_resnet_dt in enumerate(self.resnet_dt_list):
            print("\tresnet_dt_{0}.size() = ".format(ii), tmp_resnet_dt.size())


class FittingNet(nn.Module):
    def __init__(
        self,
        layer_size_list: List[int],
        activation: nn.Module = nn.Tanh(),
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
            if (ii == len(self.layer_size_list)-2):
                self.linears_list.append(
                    nn.Linear(self.layer_size_list[ii], self.layer_size_list[ii+1], bias=True))
                nn.init.normal_(self.linears_list[ii].weight, 
                                mean=0.0, 
                                std=(1.0 / (self.layer_size_list[ii] + self.layer_size_list[ii + 1]) ** 0.5))
                nn.init.normal_(self.linears_list[ii].bias, mean=energy_shift, std=1.0)
            else:
                self.linears_list.append(
                    nn.Linear(self.layer_size_list[ii], self.layer_size_list[ii+1], bias=self.bias_mark))
                nn.init.normal_(self.linears_list[ii].weight, 
                                mean=0.0, 
                                std=(1.0 / (self.layer_size_list[ii] + self.layer_size_list[ii + 1]) ** 0.5))
                if (self.bias_mark):
                    nn.init.normal_(self.linears_list[ii].bias, mean=0.0, std=1.0)
                    
                if self.resnet_mark:
                    resnet_dt_tensor: torch.Tensor = torch.Tensor(1, layer_size_list[ii+1])
                    nn.init.normal_(resnet_dt_tensor, mean=0.1, std=0.001)
                    self.resnet_dt_list.append(nn.Parameter(data=resnet_dt_tensor, requires_grad=True))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for ii, linear in enumerate(self.linears_list):
            if (ii == len(self.linears_list)-1):
                hidden: torch.Tensor = linear(x)
                x = hidden
            else:
                hidden: torch.Tensor = linear(x)
                hidden = self.activation(hidden)
                if self.resnet_mark:
                    if (self.layer_size_list[ii] == self.layer_size_list[ii+1]):
                        x = self.resnet_dt_list[ii] * hidden + x
                    elif (2*self.layer_size_list[ii] == self.layer_size_list[ii+1]):
                        x = self.resnet_dt_list[ii] * hidden + torch.cat([x, x], dim=-1)
                    else:
                        x = self.resnet_dt_list[ii] * hidden
                else:
                    x = hidden  
        return x

    def info(self):
        print("1. Linears ({0} in total):".format(len(self.linears_list)))
        for ii, tmp_linear in enumerate(self.linears_list):
            print("\tlinear#{0}.weight.size() = ".format(ii), tmp_linear.weight.size())
            print("\t\t + linear#{0}.bias.size() = ".format(ii), tmp_linear.bias.size())
        print("\n")
        print("2. Resnet_dts ({0} in total):".format(len(self.resnet_dt_list)))
        for ii, tmp_resnet_dt in enumerate(self.resnet_dt_list):
            print("\tresnet_dt#{0}.size() = ".format(ii), tmp_resnet_dt.size())


class DpSeR(nn.Module):
    def __init__(
            self,
            ntypes: int,
            rcut: float,
            rcut_smooth: float,
            umax_num_neighs: torch.Tensor,
            embed_sizes_list: List[int],
            fit_sizes_list: List[int],
            embed_activation: nn.Module = nn.Tanh(),
            fit_activation: nn.Module = nn.Tanh(),
            embed_bias_mark: bool = True,
            embed_resnet_mark: bool = False,
            fit_bias_mark: bool = True,
            fit_resnet_mark: bool = False,
            energy_shift_list: Union[bool, List[float]] = False):
        super(DpSeR, self).__init__()
        self.ntypes: int = ntypes
        self.rcut: float = rcut
        self.rcut_smooth: float = rcut_smooth
        self.umax_num_neighs: torch.Tensor = umax_num_neighs
        if energy_shift_list is False:
            self.energy_shift_list = [0.0] * ntypes
        else:
            self.energy_shift_list: List[float] = energy_shift_list
        self.embed_sizes_list: List[int] = embed_sizes_list
        self.fit_sizes_list: List[int] = fit_sizes_list

        self.embeds_list: nn.Module = nn.ModuleList()
        self.fits_list: nn.Module = nn.ModuleList()
        
        for ii in range(self.ntypes):
            for _ in range(self.ntypes):
                self.embeds_list.append(
                    EmbeddingNet(
                        layers_size_list=self.embed_sizes_list,
                        activation=embed_activation,
                        bias_mark=embed_bias_mark,
                        resnet_mark=embed_resnet_mark))
            self.fits_list.append(
                FittingNet(
                    layer_size_list=self.fit_sizes_list,
                    activation=fit_activation,
                    bias_mark=fit_bias_mark,
                    resnet_mark=fit_resnet_mark,
                    energy_shift=self.energy_shift_list[ii]))
            
    def forward(
        self,
        bilist: torch.Tensor,
        bnumneigh: torch.Tensor,
        bfirstneigh: torch.Tensor,
        brcs: torch.Tensor,
        btypes: torch.Tensor,
        bnghost: torch.Tensor):
        nbatches: int = bilist.size()[0]
        #natoms: int = bilist.size()[1]
        #umax_num_neigh_atoms: int = bfirstneigh.size()[2]
        tilde_r: torch.Tensor = envMatrixOp(
            bilist,
            bnumneigh,
            bfirstneigh,
            brcs,
            btypes,
            self.umax_num_neighs.expand(nbatches, self.ntypes),
            self.rcut,
            self.rcut_smooth)[0]
        
        return tilde_r
    