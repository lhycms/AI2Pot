from typing import List

import torch
import torch.nn as nn

from ai2pot.fromcc import envMatrixOp, forceSrOp, virialSrOp
from ai2pot.models.potential_submodules import DpResNetLinear


class EmbeddingNet(nn.Module):
    def __init__(
            self,
            layer_sizes_list: List[int],
            bias_mark: bool = True,
            activation_fn: torch.nn.Module = nn.Tanh()):
        # input: [Nbatches, Natoms, umax_num_neighs, 1]
        # output: [Nbatches, Natoms, umax_num_neighs, M1]
        super(EmbeddingNet, self).__init__()
        self.layer_sizes_list: List[int] = [1] + layer_sizes_list
        self.activation_fn: nn.Module = activation_fn
        self.bias_mark: bool = bias_mark
        self.linears: nn.ModuleList = nn.ModuleList()
        for ii in range(len(layer_sizes_list) - 1):
            self.linears.append(DpResNetLinear(in_features=layer_sizes_list[ii],
                                               out_features=layer_sizes_list[ii+1],
                                               bias_mark=self.bias_mark,
                                               activation_fn=self.activation_fn))
    
    def forward(self, srij: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = srij
        for _, linear in enumerate(self.linears):
            hidden: torch.Tensor = linear(x)    # linear is DpResNetLinear
            x = hidden
        return x
    
   
    
class FittingNet(nn.Module):
    def __init__(self,
                 layer_sizes_list: List[int],
                 type_energy_shift: float,
                 bias_mark: bool = True,
                 activation_fn: nn.Module = nn.Tanh()):
        # In terms to specific type
        # input: [Nbatches ... Natoms, M1*M2]
        # output: [Nbatches ... Natoms, ]
        super(FittingNet, self).__init__()
        self.layer_sizes_list: List[int] = layer_sizes_list + [1]
        self.activation_fn: bool = activation_fn
        self.bias_mark: bool = bias_mark
        self.linears: nn.ModuleList = nn.ModuleList()
        for ii in range(len(layer_sizes_list) - 2):
            self.linears.append(DpResNetLinear(in_features=layer_sizes_list[ii],
                                               out_features=layer_sizes_list[ii+1],
                                               bias_mark=self.bias_mark,
                                               activation_fn=self.activation_fn))
        last_linear: nn.Module = nn.Linear(layer_sizes_list[-2], layer_sizes_list[-1])
        nn.init.normal_(last_linear.weight, 
                        mean=0.0, 
                        std=1.0/torch.sqrt(layer_sizes_list[-2] + layer_sizes_list[-1]))
        if (self.bias_mark):
            nn.init.normal_(last_linear.bias, mean=type_energy_shift, std=1.0)
        self.linears.append(last_linear)
    
    
    def forward(self, descriptor: torch.Tensor):
        x: torch.Tensor = descriptor
        for _, linear in enumerate(self.linears):  # last linear is nn.Linear
            x = linear(x)
        return x
    


class DeepPotSeADescriptor(nn.Module):
    def __init__(self,
                 ntypes: int,
                 umax_num_neighs_tensor: torch.Tensor,
                 embedding_sizes_list: List[int],
                 M2: int,
                 bias_mark: bool = True,
                 activation_fn: nn.Module = nn.Tanh()):
        super(DeepPotSeADescriptor, self).__init__()
        self.ntypes: int = ntypes
        self.embedding_sizes_list: List[int] = embedding_sizes_list
        self.M1: int = embedding_sizes_list[-1]
        self.M2: int = M2
        self.bias_mark: bool = bias_mark
        self.activation_fn: nn.Module = activation_fn
    
    def forward(self):
        pass
    
    

class DeepPotSeA(nn.Module):
    def __init__(self,
                 ntypes: int,
                 type_energy_shifts: torch.Tensor,
                 umax_num_neighs_tensor: torch.Tensor,   # .size() = [ntypes,]
                 embedding_sizes_list: List[int],
                 fitting_sizes_list: List[int],
                 M2: int,
                 bias_mark: bool = True,
                 activation_fn: nn.Module = nn.Tanh()):
        super.__init__(DeepPotSeA, self)
        self.ntypes: int = ntypes
        self.type_energy_shifts: torch = type_energy_shifts
        self.embedding_sizes_list: List[int] = embedding_sizes_list
        self.fitting_sizes_list: List[int] = [embedding_sizes_list[-1] * M2] + fitting_sizes_list
        self.M2: int = M2
        self.bias_mark = bias_mark
        self.activation_fn = activation_fn
        
        self.neigh_breakpoints_list: List[int] = [0]
        for ii in range(ntypes):
            self.neigh_breakpoints_list.append(umax_num_neighs_tensor[ii].item() + self.neigh_breakpoints_list[-1])
        
        # N_embeddings = ntypes * ntypes
        self.embedding_nets: nn.ModuleList = nn.ModuleList()
        for _ in range(ntypes):
            for _ in range(ntypes):
                self.embedding_nets.append(EmbeddingNet(layer_sizes_list=embedding_sizes_list,
                                                        bias_mark=self.bias_mark,
                                                        activation_fn=activation_fn))
        
        # N_fittings = ntypes
        self.fitting_nets: nn.ModuleList = nn.ModuleList()
        for _ in range(ntypes):
            self.fitting_nets.append(FittingNet(layer_sizes_list=self.fitting_sizes_list,
                                                type_energy_shift=0,
                                                bias_mark=self.bias_mark,
                                                activation_fn=self.activation_fn))
    
    def forward(self,
                binum_tensor: torch.Tensor,
                bilist_tensor: torch.Tensor,
                bnumneigh_tensor: torch.Tensor,
                bfirstneigh_tensor: torch.Tensor,
                brcs_tensor: torch.Tensor,
                btypes_tensor: torch.Tensor,
                bnghost_tensor: torch.Tensor):
        pass
        