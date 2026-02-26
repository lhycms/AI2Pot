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


from typing import List, Dict
import numpy as np
import torch
from pymatgen.core import Structure
from ase import Atoms

from ai2pot.core import Nblist


class MlffInput(object):
    def __init__(self,
                 type_map: List[int],
                 rcut: float,
                 umax_num_neigh_atoms: int = 200,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = True,
                 dtype: torch._C.dtype = torch.float32,
                 device: torch._C.device = torch.device("cpu")):
        self.type_map: List[int] = type_map
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool = sort
        if (dtype == torch.float32):
            self.torch_float_dtype: torch._C.dtype = torch.float32
            self.np_float_dtype: np.dtype = np.float32
        else:
            self.torch_float_dtype: torch._C.dtype = torch.float64
            self.np_float_dtype: np.dtype = np.float64
        self.device: torch._C.device = device
    
    
    def analyse_pymatgen(self,
                         structure: Structure):
        cell: np.ndarray = np.array(structure.lattice.matrix).astype(self.np_float_dtype)
        types: np.ndarray = np.array([self.type_map.index(el.Z) for el in structure.species])
        cart_coords: np.ndarray = np.array(structure.cart_coords).astype(self.np_float_dtype)
        nblist_info = Nblist.find_info4mlff(cell=cell,
                                            species=types,
                                            coords=cart_coords,
                                            rcut=self.rcut,
                                            umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                            is_cart_coord=True,
                                            pbc_xyz=self.pbc_xyz,
                                            sort=self.sort)
        
        return [
            torch.tensor(nblist_info[0], dtype=torch.int32, device=self.device).view(1,),
            torch.tensor(nblist_info[1], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[2], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[3], dtype=torch.int32, device=self.device).view(1, -1, self.umax_num_neigh_atoms),
            torch.tensor(nblist_info[4], dtype=self.torch_float_dtype, device=self.device).view(1, -1, self.umax_num_neigh_atoms, 3),
            torch.tensor(nblist_info[5], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[6], dtype=torch.int32, device=self.device).view(1,)
        ]
    
    
    def analyse_ase(self,
                    atoms: Atoms):
        cell: np.ndarray = atoms.cell.array.astype(self.np_float_dtype)
        types: np.ndarray = np.array([self.type_map.index(el) for el in atoms.get_atomic_numbers()])
        cart_coords: np.ndarray = atoms.positions.astype(self.np_float_dtype)
        nblist_info = Nblist.find_info4mlff(cell=cell,
                                            species=types,
                                            coords=cart_coords,
                                            rcut=self.rcut,
                                            umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                            is_cart_coord=True,
                                            pbc_xyz=self.pbc_xyz,
                                            sort=self.sort)
        return [
            torch.tensor(nblist_info[0], dtype=torch.int32, device=self.device).view(1,),
            torch.tensor(nblist_info[1], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[2], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[3], dtype=torch.int32, device=self.device).view(1, -1, self.umax_num_neigh_atoms),
            torch.tensor(nblist_info[4], dtype=self.torch_float_dtype, device=self.device).view(1, -1, self.umax_num_neigh_atoms, 3),
            torch.tensor(nblist_info[5], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[6], dtype=torch.int32, device=self.device).view(1,)
        ]


class MlffToLossInput(object):
    def __init__(self,
                 type_map: List[int],
                 rcut: float,
                 umax_num_neigh_atoms: int = 200,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = True,
                 dtype: torch._C.dtype = torch.float32,
                 device: torch._C.device = torch.device("cpu")):
        self.type_map: List[int] = type_map
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool = sort
        if (dtype == torch.float32):
            self.torch_float_dtype = torch.float32
            self.np_float_dtype = np.float32
        else:
            self.torch_float_dtype = torch.float64
            self.np_float_dtype = np.float64
        self.device: torch._C.device = device
    
    
    def analyse_pymatgen(self,
                         structure: Structure,
                         e_weight: float,
                         f_weight: float,
                         v_weight: float):
        num_atoms = len(structure.species)
        betot_dft_tensor: torch.Tensor = torch.zeros(size=(1,),
                                                     dtype=self.torch_float_dtype, 
                                                     device=self.device) + 100.0
        bforce_dft_tensor: torch.Tensor = torch.zeros(size=(1, num_atoms, 3), 
                                                      dtype=self.torch_float_dtype, 
                                                      device=self.device) + 10.0
        bvirial_dft_tensor: torch.Tensor = torch.zeros(size=(1, 3, 3),
                                                       dtype=self.torch_float_dtype, 
                                                       device=self.device)
        cell: np.ndarray = np.array(structure.lattice.matrix).astype(self.np_float_dtype)
        types: np.ndarray = np.array([self.type_map.index(el.Z) for el in structure.species])
        cart_coords: np.ndarray = np.array(structure.cart_coords).astype(self.np_float_dtype)
        nblist_info = Nblist.find_info4mlff(cell=cell,
                                            species=types,
                                            coords=cart_coords,
                                            rcut=self.rcut,
                                            umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                            is_cart_coord=True,
                                            pbc_xyz=self.pbc_xyz,
                                            sort=self.sort)
        return [
            e_weight,
            f_weight,
            v_weight,
            betot_dft_tensor,
            bforce_dft_tensor,
            bvirial_dft_tensor,
            torch.tensor(nblist_info[0], dtype=torch.int32, device=self.device).view(1,),
            torch.tensor(nblist_info[1], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[2], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[3], dtype=torch.int32, device=self.device).view(1, -1, self.umax_num_neigh_atoms),
            torch.tensor(nblist_info[4], dtype=self.torch_float_dtype, device=self.device).view(1, -1, self.umax_num_neigh_atoms, 3),
            torch.tensor(nblist_info[5], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[6], dtype=torch.int32, device=self.device).view(1,)
        ]
        

class MlffToEFLossInput(object):
    def __init__(self,
                 type_map: List[int],
                 rcut: float,
                 umax_num_neigh_atoms: int = 200,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = True,
                 dtype: torch._C.dtype = torch.float32,
                 device: torch._C.device = torch.device("cpu")):
        self.type_map: List[int] = type_map
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool = sort
        if (dtype == torch.float32):
            self.torch_float_dtype = torch.float32
            self.np_float_dtype = np.float32
        else:
            self.torch_float_dtype = torch.float64
            self.np_float_dtype = np.float64
        self.device: torch._C.device = device
    
    
    def analyse_pymatgen(self,
                         structure: Structure,
                         e_weight: float,
                         f_weight: float):
        num_atoms = len(structure.species)
        betot_dft_tensor: torch.Tensor = torch.zeros(size=(1,),
                                                     dtype=self.torch_float_dtype, 
                                                     device=self.device) + 10.0
        bforce_dft_tensor: torch.Tensor = torch.zeros(size=(1, num_atoms, 3), 
                                                      dtype=self.torch_float_dtype, 
                                                      device=self.device) + 10.0
        cell: np.ndarray = np.array(structure.lattice.matrix).astype(self.np_float_dtype)
        types: np.ndarray = np.array([self.type_map.index(el.Z) for el in structure.species])
        cart_coords: np.ndarray = np.array(structure.cart_coords).astype(self.np_float_dtype)
        nblist_info = Nblist.find_info4mlff(cell=cell,
                                            species=types,
                                            coords=cart_coords,
                                            rcut=self.rcut,
                                            umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                            is_cart_coord=True,
                                            pbc_xyz=self.pbc_xyz,
                                            sort=self.sort)
        return [
            e_weight,
            f_weight,
            betot_dft_tensor,
            bforce_dft_tensor,
            torch.tensor(nblist_info[0], dtype=torch.int32, device=self.device).view(1,),
            torch.tensor(nblist_info[1], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[2], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[3], dtype=torch.int32, device=self.device).view(1, -1, self.umax_num_neigh_atoms),
            torch.tensor(nblist_info[4], dtype=self.torch_float_dtype, device=self.device).view(1, -1, self.umax_num_neigh_atoms, 3),
            torch.tensor(nblist_info[5], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[6], dtype=torch.int32, device=self.device).view(1,)
        ]
    