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

from dataclasses import dataclass

import torch


@dataclass
class GraphData(object):
    node_types_tensor: torch.Tensor
    edge_index_tensor: torch.Tensor
    edge_vec_tensor: torch.Tensor
    node_frame_idx_tensor: torch.Tensor
    binum_tensor: torch.Tensor
    
    @property
    def num_nodes(self):
        return self.node_types_tensor.shape[0]
    
    @property
    def num_edges(self):
        return self.edge_index_tensor.shape[1]


def _get_frame_offsets(binum_tensor: torch.Tensor):
    return torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=binum_tensor.device),
         torch.cumsum(binum_tensor, dim=0)],
        dim=0
    )[:-1]


def convert_nblist_to_graph(binum_tensor: torch.Tensor,
                            bilist_tensor: torch.Tensor,
                            bnumneigh_tensor: torch.Tensor,
                            bfirstneigh_tensor: torch.Tensor,
                            brcs_tensor: torch.Tensor,
                            btypes_tensor: torch.Tensor,
                            bnghost_tensor: torch.Tensor):
    batch_size: int = bfirstneigh_tensor.shape[0]
    natoms_pad: int = bfirstneigh_tensor.shape[1]
    umax_num_neigh_atoms: int = bfirstneigh_tensor.shape[2]
    device: torch._C.device = brcs_tensor.device
    
    
    # 1. _ctr_tensor
    bilist_2d_safe_tensor: torch.Tensor = torch.clamp(bilist_tensor, min=0).to(torch.long)
    bilist_3d_safe_tensor: torch.Tensor = bilist_2d_safe_tensor[:, :, None].expand(-1, -1, umax_num_neigh_atoms)
    bilist_4d_safe_tensor: torch.Tensor = bilist_2d_safe_tensor[:, :, None, None].expand(-1, -1, umax_num_neigh_atoms, 3)
        
    bnumneigh_ctr_tensor: torch.Tensor = torch.gather(input=bnumneigh_tensor,
                                                      dim=1,
                                                      index=bilist_2d_safe_tensor)    
    bfirstneigh_ctr_tensor: torch.Tensor = torch.gather(input=bfirstneigh_tensor,
                                                        dim=1,
                                                        index=bilist_3d_safe_tensor)
    brcs_ctr_tensor: torch.Tensor = torch.gather(input=brcs_tensor,
                                                 dim=1,
                                                 index=bilist_4d_safe_tensor)
    
    # 2. mask
    center_mask: torch.Tensor = torch.arange(natoms_pad, device=device)[None, :] < binum_tensor[:, None]
    neigh_mask: torch.Tensor = torch.arange(umax_num_neigh_atoms, device=device)[None, None, :] < bnumneigh_ctr_tensor[:, :, None]
    mask: torch.Tensor = center_mask[:, :, None] & neigh_mask
    
    # 3. edge local_idx/vec/frame_idx
    src_tensor: torch.Tensor = bfirstneigh_ctr_tensor[mask]
    
    bilist_3d_tensor: torch.Tensor = bilist_tensor[:, :, None].expand(-1, -1, umax_num_neigh_atoms)
    dst_tensor: torch.Tensor = bilist_3d_tensor[mask]
    
    edge_vec_tensor: torch.Tensor = brcs_ctr_tensor[mask]
    
    edge_frame_idx_3d_tensor: torch.Tensor = torch.arange(batch_size, device=device)[:, None, None].expand(-1, natoms_pad, umax_num_neigh_atoms)
    edge_frame_idx_tensor: torch.Tensor = edge_frame_idx_3d_tensor[mask]
    
    # 4. global_edge_index
    base: torch.Tensor = _get_frame_offsets(binum_tensor=binum_tensor)
    global_src_tensor: torch.Tensor = src_tensor + edge_frame_idx_tensor[edge_frame_idx_tensor]
    global_dst_tensor: torch.Tensor = dst_tensor + edge_frame_idx_tensor[edge_frame_idx_tensor]
    
    # 5. node_frame_idx_tensor
    all_center_mask: torch.Tensor = torch.arange(btypes_tensor.shape[1], device=device) < binum_tensor[:, None]
    node_types_tensor: torch.Tensor = btypes_tensor[all_center_mask]
    
    node_frame_idx_tensor: torch.Tensor = torch.repeat_interleave(
        input=torch.arange(batch_size, device=device, dtype=torch.long),
        repeats=binum_tensor.to(torch.long)
    )

    return GraphData(
        node_types_tensor=node_types_tensor,
        edge_index_tensor=torch.stack([src_tensor, dst_tensor], dim=0),
        edge_vec_tensor=edge_vec_tensor,
        node_frame_idx_tensor=node_frame_idx_tensor,
        binum_tensor=binum_tensor
    )
    