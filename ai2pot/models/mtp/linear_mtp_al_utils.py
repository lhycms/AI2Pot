# Copyright (C) 2025 Hanyu Liu
#
# SPDX-License-Identifier: LGPL-3.0-only
#
# This file is part of AI2Pot.
#
# AI2Pot is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, version 3 of the License.
#
# AI2Pot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with AI2Pot. If not, see <https://www.gnu.org/licenses/>.



from typing import List, Optional

from ase import Atoms
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ai2pot.data.mlffdatamodule import ExtxyzDataModule
from ai2pot.utils.usepot import MlffInput
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.models.mtp.linear_mtp import (LinearMtp)


class LinearMtpMaxvolGenerator(object):
    BATCH_SIZE_HERE: int = 800
    
    def __init__(self,
                 checkpoint_path: str,
                 candidate_set_path: str,
                 map_location: str = "cpu",
                 pbc_xyz: List[bool] = [True, True, True]):
        self.checkpoint_path: str = checkpoint_path
        self.candidate_set_path: str = candidate_set_path
        self.lit_module: LitLinearMtp = LitLinearMtp.load_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                                          map_location=map_location)
        self.fit_virial: bool = self.lit_module.model.fit_virial
        
        # model and data
        self.model: LinearMtp = self.lit_module.model
        extxyz_datamodule: ExtxyzDataModule = ExtxyzDataModule(
            testset_path=candidate_set_path,
            batch_size=LinearMtpMaxvolGenerator.BATCH_SIZE_HERE,
            rcut=self.model.rmax,
            umax_num_neigh_atoms=self.model.umax_num_neigh_atoms,
            pbc_xyz=pbc_xyz,
            sort=False,
            torch_float_dtype=self.lit_module.dtype,
            has_virial=self.fit_virial)
        extxyz_datamodule.setup("test")
        self.candidate_set_dataloader: DataLoader = extxyz_datamodule.test_dataloader()
        
        
    @torch.no_grad()
    def _generate_candidate_set_matrix(self):
        structure_offset: int = 0
        structure_id_list: List[torch.Tensor] = []
        atom_id_list: List[torch.Tensor] = []
        candidate_set_matrix_list: List[torch.Tensor] = []
        
        for batch_idx, batch_data in enumerate(self.candidate_set_dataloader):
            if (self.fit_virial):
                binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                    brcs_tensor, btypes_tensor, bnghost_tensor, \
                    betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = \
                        [t.to(self.lit_module.device) for t in batch_data]
            else:            
                binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                    brcs_tensor, btypes_tensor, bnghost_tensor, \
                    betot_dft_tensor, bforce_dft_tensor = \
                        [t.to(self.lit_module.device) for t in batch_data]

            batch_e_sites_jacobian: torch.Tensor = self.model.predict_e_sites_jacobian(binum_tensor,
                                                                                       bilist_tensor,
                                                                                       bnumneigh_tensor,
                                                                                       bfirstneigh_tensor,
                                                                                       brcs_tensor,
                                                                                       btypes_tensor,
                                                                                       bnghost_tensor)
            
            natoms_pad: int = bilist_tensor.size(1)
            mask_for_centers: torch.Tensor = torch.arange(bilist_tensor.size(1),
                                                          device=self.lit_module.device,
                                                          dtype=self.lit_module.dtype)[None, :] < binum_tensor[:, None]
            batch_structure_id: torch.Tensor = structure_offset + torch.arange(binum_tensor.size(0),
                                                                               device=self.lit_module.device,
                                                                               dtype=torch.int32)[:, None]
            batch_atom_id: torch.Tensor = torch.arange(natoms_pad,
                                                       device=self.lit_module.device,
                                                       dtype=torch.int32)[None, :]
            candidate_set_matrix_list.append(batch_e_sites_jacobian[mask_for_centers])
            structure_id_list.append(batch_structure_id.expand(-1, natoms_pad)[mask_for_centers])
            atom_id_list.append(batch_atom_id.expand(binum_tensor.size(0), -1)[mask_for_centers])
            
            structure_offset += binum_tensor.size(0)
        
        structure_id_tensor: torch.Tensor = torch.cat(structure_id_list, dim=0)
        atom_id_tensor: torch.Tensor = torch.cat(atom_id_list, dim=0)
        candidate_set_matrix_tensor: torch.Tensor = torch.cat(candidate_set_matrix_list, dim=0)
        
        return structure_id_tensor, atom_id_tensor, candidate_set_matrix_tensor
    
    