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



from typing import List

import torch
from torch.utils.data import DataLoader

from ai2pot.data.mlffdatamodule import ExtxyzDataModule
from ai2pot.data.mlffdataset import ExtxyzDataset
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
        self.pbc_xyz: List[bool] = pbc_xyz
        
        self.type_map: List[int] = self.lit_module.model.type_map_tensor.detach().cpu().numpy().tolist()
        
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
            has_virial=self.fit_virial,
            type_map=self.type_map)
        extxyz_datamodule.setup("test")
        self.candidate_set_dataloader: DataLoader = extxyz_datamodule.test_dataloader()
        
        
    @torch.no_grad()
    def _generate_candidate_set_matrix_local(self):
        structure_offset: int = 0
        structure_id_list: List[torch.Tensor] = []
        atom_id_list: List[torch.Tensor] = []
        candidate_set_matrix_list: List[torch.Tensor] = []
        
        for _, batch_data in enumerate(self.candidate_set_dataloader):
            binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                brcs_tensor, btypes_tensor, bnghost_tensor = \
                    [t.to(self.lit_module.device) for t in batch_data[:7]]


            batch_e_sites_jacobian: torch.Tensor = self.model.predict_e_sites_jacobian(binum_tensor,
                                                                                       bilist_tensor,
                                                                                       bnumneigh_tensor,
                                                                                       bfirstneigh_tensor,
                                                                                       brcs_tensor,
                                                                                       btypes_tensor,
                                                                                       bnghost_tensor)
            
            natoms_pad: int = bilist_tensor.size(1)
            mask_for_centers: torch.Tensor = torch.arange(bilist_tensor.size(1),
                                                          device=self.lit_module.device)[None, :] < binum_tensor[:, None]
            batch_structure_id: torch.Tensor = structure_offset + torch.arange(binum_tensor.size(0),
                                                                               device=self.lit_module.device)[:, None]
            batch_atom_id: torch.Tensor = torch.arange(natoms_pad,
                                                       device=self.lit_module.device)[None, :]
            candidate_set_matrix_list.append(batch_e_sites_jacobian[mask_for_centers])
            structure_id_list.append(batch_structure_id.expand(-1, natoms_pad)[mask_for_centers])
            atom_id_list.append(batch_atom_id.expand(binum_tensor.size(0), -1)[mask_for_centers])
            
            structure_offset += binum_tensor.size(0)
        
        structure_id_tensor: torch.Tensor = torch.cat(structure_id_list, dim=0)
        atom_id_tensor: torch.Tensor = torch.cat(atom_id_list, dim=0)
        candidate_set_matrix_tensor: torch.Tensor = torch.cat(candidate_set_matrix_list, dim=0)
        
        return structure_id_tensor, atom_id_tensor, candidate_set_matrix_tensor
    
    
    @torch.no_grad()
    def generate_maxvol_matrix_local(self,
                                     maxiter: int = 20000,
                                     tol: float = 1e-4,
                                     rank_tol: float = 1e-12):
        structure_id_tensor, atom_id_tensor, candidate_set_matrix_tensor = self._generate_candidate_set_matrix_local()
        structure_id_tensor: torch.Tensor
        atom_id_tensor: torch.Tensor
        candidate_set_matrix_tensor: torch.Tensor
        
        B_matrix: torch.Tensor = candidate_set_matrix_tensor.to(torch.float64)
        num_atoms, num_parameters = B_matrix.shape
        
        if (num_atoms < num_parameters):
            raise ValueError("Number of candidates must be >= number of parameters.")
        
        if not torch.isfinite(B_matrix).all():
            raise ValueError("Candidate matrix contains NaN or Inf.")
        
        ### Step 1. Initialize a nonsingular square matrix
        ### Step 1.1.
        selected: torch.Tensor = torch.empty(num_parameters, dtype=torch.long, device=B_matrix.device)
        selected_mask: torch.Tensor = torch.zeros(num_atoms, dtype=torch.bool, device=B_matrix.device)
        Q_matrix: torch.Tensor = torch.empty((num_parameters, num_parameters), dtype=torch.float64, device=B_matrix.device)
        residual_norms: torch.Tensor = torch.sum(B_matrix * B_matrix, dim=1)
        
        max_norm: torch.Tensor = torch.sqrt(residual_norms.max())
        if (max_norm == 0):
            raise ValueError("Candidate matrix is zero.")
        
        # Step 1.2.
        for k in range(num_parameters):
            idx: int = int(
                residual_norms.masked_fill(selected_mask, -torch.inf).argmax().item()
            )
            atom_b_tensor: torch.Tensor = B_matrix[idx].clone()
            
            if k > 0:
                basis_tensor: torch.Tensor = Q_matrix[:k]
                atom_b_tensor -= (atom_b_tensor @ basis_tensor.T) @ basis_tensor
                atom_b_tensor -= (atom_b_tensor @ basis_tensor.T) @ basis_tensor
            
            atom_b_norm: torch.Tensor = torch.linalg.vector_norm(atom_b_tensor)
            if (atom_b_norm <= max_norm * rank_tol):
                raise ValueError("Candidate matrix is rank deficient.")

            selected[k] = idx
            selected_mask[idx] = True
            Q_matrix[k] = atom_b_tensor / atom_b_norm
            
            projection: torch.Tensor = B_matrix @ Q_matrix[k]
            residual_norms.sub_( projection.square() ).clamp_(min=0.0)


        ### Step 2. Initialize coefficient matrix C = B @ A^-1
        A_matrix: torch.Tensor = B_matrix[selected].clone()
        C_matrix: torch.Tensor = torch.linalg.solve(A_matrix.T, B_matrix.T).T.contiguous()

        
        ### Step 3. Standard MaxVol row replacement
        for _ in range(maxiter):
            flat_idx: int = int( torch.argmax(C_matrix.abs()).item() )
            i, j = divmod(flat_idx, num_parameters)
            pivot: float = float(C_matrix[i, j].item())
            if (abs(pivot) <= (1.0 + tol)):
                break
            
            row: torch.Tensor = C_matrix[i].clone()
            row[j] -= 1.0
            col: torch.Tensor = C_matrix[:, j].clone()
            C_matrix.addmm_(
            col[:, None],
            row[None, :],
            beta=1.0,
            alpha=-1.0 / pivot)
            
            selected[j] = i
        else:
            raise RuntimeError(
                f"MaxVol did not converge within {maxiter} iterations."
            )
        
        ### Step 4. Save MaxVol matrix and selected atom IDs
        ### Step 4.1.        
        maxvol_matrix: torch.Tensor = B_matrix[selected].clone()
        maxvol_matrix_inv: torch.Tensor = torch.linalg.inv(maxvol_matrix)
        maxvol_structure_id_tensor: torch.Tensor = structure_id_tensor[selected]
        maxvol_atom_id_tensor: torch.Tensor = atom_id_tensor[selected]
        
        ### Step 4.2. Veritifation
        # Recompute coefficients to verify convergence
        C_check: torch.Tensor = torch.linalg.solve(
            maxvol_matrix.T, B_matrix.T
        ).T
        max_gamma: float = C_check.abs().max().item()
        if max_gamma > 1.0 + tol + 1e-8:
            raise RuntimeError(
                f"MaxVol numerical verification failed: "
                f"max_gamma = {max_gamma:.6f}"
            )
                
        return (
            maxvol_structure_id_tensor, maxvol_atom_id_tensor, maxvol_matrix, maxvol_matrix_inv
        )
    
    
    @torch.no_grad()
    def get_gamma_local(self,
                        maxvol_matrix_inv: torch.Tensor,
                        explore_path: str):
        explore_datamodule: ExtxyzDataModule = ExtxyzDataModule(
            testset_path=explore_path,
            batch_size=LinearMtpMaxvolGenerator.BATCH_SIZE_HERE,
            rcut=self.model.rmax,
            umax_num_neigh_atoms=self.model.umax_num_neigh_atoms,
            pbc_xyz=self.pbc_xyz,
            sort=False,
            torch_float_dtype=self.lit_module.dtype,
            has_virial=self.fit_virial,
            type_map=self.type_map)
        explore_datamodule.setup("test")
        explore_dataloader: ExtxyzDataset = explore_datamodule.test_dataloader()
        maxvol_matrix_inv = maxvol_matrix_inv.to(self.lit_module.device)
        
        structure_offset: int = 0
        structure_id_list: List[torch.Tensor] = []
        atom_id_list: List[torch.Tensor] = []
        gamma_structures_list: List[torch.Tensor] = []
        gamma_atoms_list: List[torch.Tensor] = []
        for _, batch_data in enumerate(explore_dataloader):
            binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                brcs_tensor, btypes_tensor, bnghost_tensor = \
                    [t.to(self.lit_module.device) for t in batch_data[:7]]
            bgamma_structures_tensor, bgamma_atom_tensor = self.model.predict_gamma_local(
                binum_tensor=binum_tensor,
                bilist_tensor=bilist_tensor,
                bnumneigh_tensor=bnumneigh_tensor,
                bfirstneigh_tensor=bfirstneigh_tensor,
                brcs_tensor=brcs_tensor,
                btypes_tensor=btypes_tensor,
                bnghost_tensor=bnghost_tensor,
                maxvol_matrix_inv=maxvol_matrix_inv)
            
            batch_size: int = bfirstneigh_tensor.size(0)
            natoms_pad: int = bfirstneigh_tensor.size(1)
            bstructure_id_tensor: torch.Tensor = structure_offset + torch.arange(batch_size, device=self.lit_module.device)[:, None]
            batom_id_tensor: torch.Tensor = torch.arange(natoms_pad, device=self.lit_module.device)[None, :]

            mask_for_centers: torch.Tensor = torch.arange(natoms_pad,
                                                          device=self.lit_module.device)[None, :] < binum_tensor[:, None]
            
            gamma_structures_list.append(bgamma_structures_tensor)
            gamma_atoms_list.append(bgamma_atom_tensor[mask_for_centers])
            structure_id_list.append(bstructure_id_tensor.expand(-1, natoms_pad)[mask_for_centers])
            atom_id_list.append(batom_id_tensor.expand(batch_size, -1)[mask_for_centers])
            
            structure_offset += batch_size
        
        gamma_structures_tensor: torch.Tensor = torch.cat(gamma_structures_list, dim=0)
        gamma_atoms_tensor: torch.Tensor = torch.cat(gamma_atoms_list, dim=0)
        structure_id_tensor: torch.Tensor = torch.cat(structure_id_list, dim=0)
        atom_id_tensor: torch.Tensor = torch.cat(atom_id_list, dim=0)
        
        return structure_id_tensor, atom_id_tensor, gamma_structures_tensor, gamma_atoms_tensor
    
    
    @torch.no_grad()
    def _generate_candidate_set_matrix_global(self):
        structure_offset: int = 0
        structure_id_list: List[torch.Tensor] = []
        candidate_set_matrix_list: List[torch.Tensor] = []
        
        for batch_idx, batch_data in enumerate(self.candidate_set_dataloader):
            binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                brcs_tensor, btypes_tensor, bnghost_tensor = \
                    [t.to(self.lit_module.device) for t in batch_data[:7]]
            
            batch_e_sites_jacobian: torch.Tensor = self.model.predict_e_sites_jacobian(binum_tensor,
                                                                                       bilist_tensor,
                                                                                       bnumneigh_tensor,
                                                                                       bfirstneigh_tensor,
                                                                                       brcs_tensor,
                                                                                       btypes_tensor,
                                                                                       bnghost_tensor)
            batch_e_sites_jacobian = batch_e_sites_jacobian.to(torch.float64)
            
            batch_size: int = bfirstneigh_tensor.size(0)
            natoms_pad: int = bfirstneigh_tensor.size(1)
            mask_for_centers: torch.Tensor = torch.arange(natoms_pad,
                                                          device=self.lit_module.device)[None, :] < binum_tensor[:, None]
            
            batch_etot_jacobian: torch.Tensor = batch_e_sites_jacobian.masked_fill(
                                                    ~mask_for_centers.unsqueeze(-1), 0.0
                                                ).sum(dim=1) / binum_tensor[:, None]
            
            batch_structure_id: torch.Tensor = structure_offset + torch.arange(batch_size, device=self.lit_module.device)
            
            candidate_set_matrix_list.append(batch_etot_jacobian)
            structure_id_list.append(batch_structure_id)
            
            structure_offset += batch_size
        
        structure_id_tensor: torch.Tensor = torch.cat(structure_id_list, dim=0)
        candidate_set_matrix_tensor: torch.Tensor = torch.cat(candidate_set_matrix_list, dim=0)
        
        return structure_id_tensor, candidate_set_matrix_tensor

    
    @torch.no_grad()
    def generate_maxvol_matrix_global(self,
                                      maxiter: int = 20000,
                                      tol: float = 1e-4,
                                      rank_tol: float = 1e-12):
        structure_id_tensor, candidate_set_matrix_tensor = self._generate_candidate_set_matrix_global()
        structure_id_tensor: torch.Tensor
        candidate_set_matrix_tensor: torch.Tensor
        
        B_matrix: torch.Tensor = candidate_set_matrix_tensor.to(torch.float64)
        num_structures, num_parameters = B_matrix.shape
        
        if (num_structures < num_parameters):
            raise ValueError("Number of candidates must be >= number of parameters.")
        
        if not torch.isfinite(B_matrix).all():
            raise ValueError("Candidate matrix contains Nan or Inf.")
        
        ### Step 1. Initialize a unsingular square matrix
        ### Step 1.1.
        selected: torch.Tensor = torch.empty(num_parameters, dtype=torch.long, device=B_matrix.device)
        selected_mask: torch.Tensor = torch.zeros(num_structures, dtype=torch.bool, device=B_matrix.device)
        Q_matrix: torch.Tensor = torch.zeros((num_parameters, num_parameters), dtype=torch.float64, device=B_matrix.device)
        residual_norms: torch.Tensor = torch.sum(B_matrix*B_matrix, dim=1)
        
        max_norm: torch.Tensor = torch.sqrt(residual_norms.max())
        if (max_norm == 0):
            raise ValueError("Candidate matrix is zero.")

        ### Step 1.2.
        for k in range(num_parameters):
            idx: int = int(
                residual_norms.masked_fill(selected_mask, -torch.inf).argmax().item()
            )
            structure_b_tensor: torch.Tensor = B_matrix[idx].clone()
            if k > 0:
                basis_tensor: torch.Tensor = Q_matrix[:k]
                structure_b_tensor -= (structure_b_tensor @ basis_tensor.T) @ basis_tensor
                structure_b_tensor -= (structure_b_tensor @ basis_tensor.T) @ basis_tensor
            
            structure_b_norm: torch.Tensor = torch.linalg.vector_norm(structure_b_tensor)
            if (structure_b_norm <= max_norm * rank_tol):
                raise ValueError("Candidate matrix is rank deficient.")
            
            selected[k] = idx
            selected_mask[idx] = True
            Q_matrix[k] = structure_b_tensor / structure_b_norm
            
            projection: torch.Tensor = B_matrix @ Q_matrix[k]
            residual_norms.sub_( projection.square() ).clamp_(min=0)
        
        ### Step 2.
        A_matrix: torch.Tensor = B_matrix[selected].clone()
        C_matrix: torch.Tensor = torch.linalg.solve(A_matrix.T, B_matrix.T).T.contiguous()
        
        ### Step 3.
        for _ in range(maxiter):
            flat_idx: int = int( torch.argmax(C_matrix.abs()).item() )
            i, j = divmod(flat_idx, num_parameters)
            pivot: float = float(C_matrix[i, j].item())
            if (abs(pivot) <= (1.0 + tol)):
                break
            
            row: torch.Tensor = C_matrix[i].clone()
            row[j] -= 1.0
            col: torch.Tensor = C_matrix[:, j].clone()
            C_matrix.addmm_(
            col[:, None],
            row[None, :],
            beta=1.0,
            alpha=-1.0 / pivot)
            
            selected[j] = i
        else:
            raise RuntimeError(
                f"Maxvol did not converge within {maxiter} iterations."
            )
        
        ### Step 4.
        ### Step 4.1.
        maxvol_matrix: torch.Tensor = B_matrix[selected].clone()
        maxvol_matrix_inv: torch.Tensor = torch.linalg.inv(maxvol_matrix)
        maxvol_structure_id_tensor: torch.Tensor = structure_id_tensor[selected]
        
        ### Step 4.2. Vertifation
        # Recompute coefficients to verify convergence
        C_check: torch.Tensor = torch.linalg.solve(
            maxvol_matrix.T, B_matrix.T
        ).T
        max_gamma: float = C_check.abs().max().item()
        if max_gamma > 1.0 + tol + 1e-8:
            raise RuntimeError(
                f"MaxVol numerical verification failed: "
                f"max_gamma = {max_gamma:.6f}"
            )
        
        return (
            maxvol_structure_id_tensor, maxvol_matrix, maxvol_matrix_inv
        )
        
    
    @torch.no_grad()
    def get_gamma_global(self,
                         maxvol_matrix_inv: torch.Tensor,
                         explore_path: str):
        explore_datamodule: ExtxyzDataModule = ExtxyzDataModule(
            testset_path=explore_path,
            batch_size=LinearMtpMaxvolGenerator.BATCH_SIZE_HERE,
            rcut=self.model.rmax,
            umax_num_neigh_atoms=self.model.umax_num_neigh_atoms,
            pbc_xyz=self.pbc_xyz,
            sort=False,
            torch_float_dtype=self.lit_module.dtype,
            has_virial=self.fit_virial,
            type_map=self.type_map)
        explore_datamodule.setup("test")
        explore_dataloader: ExtxyzDataset = explore_datamodule.test_dataloader()
        maxvol_matrix_inv = maxvol_matrix_inv.to(self.lit_module.device)
        
        structure_offset: int = 0
        structure_id_list: List[torch.Tensor] = []
        gamma_structure_list: List[torch.Tensor] = []
        
        for _, batch_data in enumerate(explore_dataloader):
            binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                brcs_tensor, btypes_tensor, bnghost_tensor = \
                    [t.to(self.lit_module.device) for t in batch_data[:7]]
            bgamma_structure_tensor: torch.Tensor = self.model.predict_gamma_global(
                binum_tensor=binum_tensor,
                bilist_tensor=bilist_tensor,
                bnumneigh_tensor=bnumneigh_tensor,
                bfirstneigh_tensor=bfirstneigh_tensor,
                brcs_tensor=brcs_tensor,
                btypes_tensor=btypes_tensor,
                bnghost_tensor=bnghost_tensor,
                maxvol_matrix_inv=maxvol_matrix_inv)
            
            batch_size: int = bfirstneigh_tensor.size(0)
            batch_structure_id_tensor: torch.Tensor = structure_offset + torch.arange(batch_size,
                                                                                      device=self.lit_module.device)
            structure_id_list.append(batch_structure_id_tensor)
            gamma_structure_list.append(bgamma_structure_tensor)

            structure_offset += batch_size
        
        structure_id_tensor: torch.Tensor = torch.cat(structure_id_list, dim=0)
        gamma_structure_tensor: torch.Tensor = torch.cat(gamma_structure_list, dim=0)
        
        return structure_id_tensor, gamma_structure_tensor
    