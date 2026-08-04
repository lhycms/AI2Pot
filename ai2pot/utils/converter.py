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
    bnghost_tensor: torch.Tensor
    
    @property
    def num_nodes(self):
        return self.node_types_tensor.shape[0]
    
    @property
    def num_edges(self):
        return self.edge_index_tensor.shape[1]


def _get_frame_offsets(binum_tensor: torch.Tensor,
                       bnghost_tensor: torch.Tensor):
    return torch.cat(
        [torch.zeros(1, dtype=torch.int32, device=binum_tensor.device),
         torch.cumsum(binum_tensor+bnghost_tensor, dim=0)],
        dim=0
    )[:-1]


class GraphDataConverter(object):
    @staticmethod
    def _get_center_neighbor_tensors(bilist_tensor: torch.Tensor,
                                     bnumneigh_tensor: torch.Tensor,
                                     bfirstneigh_tensor: torch.Tensor,
                                     brcs_tensor: torch.Tensor,
                                     umax_num_neigh_atoms: int):
        bilist_2d_safe_tensor: torch.Tensor = torch.clamp(bilist_tensor, min=0).to(torch.long)
        bilist_3d_safe_tensor: torch.Tensor = bilist_2d_safe_tensor[:, :, None].expand(-1, -1, umax_num_neigh_atoms)
        bilist_4d_safe_tensor: torch.Tensor = bilist_2d_safe_tensor[:, :, None, None].expand(-1, -1, umax_num_neigh_atoms, 3)
        
        bnumneigh_ctr_tensor: torch.Tensor = torch.gather(
            input=bnumneigh_tensor,
            dim=1,
            index=bilist_2d_safe_tensor)
        bfirstneigh_ctr_tensor: torch.Tensor = torch.gather(
            input=bfirstneigh_tensor,
            dim=1,
            index=bilist_3d_safe_tensor
        )
        brcs_ctr_tensor: torch.Tensor = torch.gather(
            input=brcs_tensor,
            dim=1,
            index=bilist_4d_safe_tensor
        )
        
        return bnumneigh_ctr_tensor, bfirstneigh_ctr_tensor, brcs_ctr_tensor
    
    
    @staticmethod
    def _get_edge_mask(binum_tensor: torch.Tensor,
                       bnumneigh_ctr_tensor: torch.Tensor,
                       natoms_pad: int,
                       umax_num_neigh_atoms: int):
        device: torch._C.device = binum_tensor.device
        center_mask: torch.Tensor = torch.arange(natoms_pad, device=device)[None, :] < binum_tensor[:, None]
        neigh_mask: torch.Tensor = torch.arange(umax_num_neigh_atoms, device=device)[None, None, :] < bnumneigh_ctr_tensor[:, :, None]
        edge_mask: torch.Tensor = center_mask[:, :, None] & neigh_mask
        return edge_mask
    

    @staticmethod
    def _get_edge_features(bilist_tensor: torch.Tensor,
                            bfirstneigh_ctr_tensor: torch.Tensor,
                            brcs_ctr_tensor: torch.Tensor,
                            edge_mask: torch.Tensor,
                            batch_size: int,
                            natoms_pad: int,
                            umax_num_neigh_atoms: int):
        device: torch._C.device = brcs_ctr_tensor.device
        local_src_tensor: torch.Tensor = bfirstneigh_ctr_tensor[edge_mask]
        
        bilist_3d_tensor: torch.Tensor = bilist_tensor[:, :, None].expand(-1, -1, umax_num_neigh_atoms)
        local_dst_tensor: torch.Tensor = bilist_3d_tensor[edge_mask]
        
        edge_vec_tensor: torch.Tensor = brcs_ctr_tensor[edge_mask]
        
        edge_frame_idx_3d_tensor: torch.Tensor = torch.arange(batch_size, device=device)[:, None, None].expand(-1, natoms_pad, umax_num_neigh_atoms)
        edge_frame_idx_tensor: torch.Tensor = edge_frame_idx_3d_tensor[edge_mask]
        
        return local_src_tensor, local_dst_tensor, edge_vec_tensor, edge_frame_idx_tensor
    
    
    @staticmethod
    def _get_global_edge_index(binum_tensor: torch.Tensor,
                               bnghost_tensor: torch.Tensor,
                               local_src_tensor: torch.Tensor,
                               local_dst_tensor: torch.Tensor,
                               edge_frame_idx_tensor: torch.Tensor):
        base: torch.Tensor = _get_frame_offsets(
            binum_tensor=binum_tensor,
            bnghost_tensor=bnghost_tensor)
        global_src_tensor: torch.Tensor = local_src_tensor + base[edge_frame_idx_tensor]
        global_dst_tensor: torch.Tensor = local_dst_tensor + base[edge_frame_idx_tensor]
        
        return global_src_tensor, global_dst_tensor
    
    
    @staticmethod
    def _get_node_features(binum_tensor: torch.Tensor,
                           btypes_tensor: torch.Tensor,
                           bnghost_tensor: torch.Tensor,
                           batch_size: int):
        device: torch._C.device = binum_tensor.device
        node_mask = torch.arange(btypes_tensor.shape[1], device=device)[None, :] < (binum_tensor + bnghost_tensor)[:, None]
        node_types_tensor = btypes_tensor[node_mask]
        node_frame_idx_tensor = torch.repeat_interleave(
            input=torch.arange(batch_size, device=device),
            repeats=(binum_tensor + bnghost_tensor).to(torch.long)
        )
        
        return node_types_tensor, node_frame_idx_tensor
    
    
    @staticmethod
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
        
        # 1.
        bnumneigh_ctr_tensor, bfirstneigh_ctr_tensor, brcs_ctr_tensor = GraphDataConverter._get_center_neighbor_tensors(
            bilist_tensor=bilist_tensor,
            bnumneigh_tensor=bnumneigh_tensor,
            bfirstneigh_tensor=bfirstneigh_tensor,
            brcs_tensor=brcs_tensor,
            umax_num_neigh_atoms=umax_num_neigh_atoms
        )
        
        # 2.
        edge_mask: torch.Tensor = GraphDataConverter._get_edge_mask(
            binum_tensor=binum_tensor,
            bnumneigh_ctr_tensor=bnumneigh_ctr_tensor,
            natoms_pad=natoms_pad,
            umax_num_neigh_atoms=umax_num_neigh_atoms
        )
        
        # 3.
        local_src_tensor, local_dst_tensor, edge_vec_tensor, edge_frame_idx_tensor = GraphDataConverter._get_edge_features(
            bilist_tensor=bilist_tensor,
            bfirstneigh_ctr_tensor=bfirstneigh_ctr_tensor,
            brcs_ctr_tensor=brcs_ctr_tensor,
            edge_mask=edge_mask,
            batch_size=batch_size,
            natoms_pad=natoms_pad,
            umax_num_neigh_atoms=umax_num_neigh_atoms
        )
        
        # 4.
        global_src_tensor, global_dst_tensor = GraphDataConverter._get_global_edge_index(
            binum_tensor=binum_tensor,
            bnghost_tensor=bnghost_tensor,
            local_src_tensor=local_src_tensor,
            local_dst_tensor=local_dst_tensor,
            edge_frame_idx_tensor=edge_frame_idx_tensor)
        
        # 5.
        node_types_tensor, node_frame_idx_tensor = GraphDataConverter._get_node_features(
            binum_tensor=binum_tensor,
            btypes_tensor=btypes_tensor,
            bnghost_tensor=bnghost_tensor,
            batch_size=batch_size
        )
        
        return GraphData(
            node_types_tensor=node_types_tensor,
            edge_index_tensor=torch.stack([global_src_tensor, global_dst_tensor], dim=0),
            edge_vec_tensor=edge_vec_tensor,
            node_frame_idx_tensor=node_frame_idx_tensor,
            binum_tensor=binum_tensor,
            bnghost_tensor=bnghost_tensor
        )
        