from typing import List, Dict
import numpy as np
import torch
from pymatgen.core import Structure
from ase import Atoms

from ai2pot.core import Nblist


class MlffInput(object):
    def __init__(self,
                 rcut: float,
                 umax_num_neighs: int = 200,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = True,
                 dtype: torch._C.dtype = torch.float32,
                 device: torch._C.device = torch.device("cpu")):
        self.rcut: float = rcut
        self.umax_num_neighs: int = umax_num_neighs
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
        unique_species = sorted(set(structure.species), key=lambda sp: sp.Z)
        type_map: Dict[str, int] = {element: idx for idx, element in enumerate(unique_species)}
        types: np.ndarray = np.array([type_map[el] for el in structure.species])
        cart_coords: np.ndarray = np.array(structure.cart_coords).astype(self.np_float_dtype)
        nblist_info = Nblist.find_info4mlff(cell=cell,
                                            species=types,
                                            coords=cart_coords,
                                            rcut=self.rcut,
                                            umax_num_neigh_atoms=self.umax_num_neighs,
                                            is_cart_coord=True,
                                            pbc_xyz=self.pbc_xyz,
                                            sort=self.sort)
        
        return [
            torch.tensor(nblist_info[0], dtype=torch.int32, device=self.device).view(1,),
            torch.tensor(nblist_info[1], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[2], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[3], dtype=torch.int32, device=self.device).view(1, -1, self.umax_num_neighs),
            torch.tensor(nblist_info[4], dtype=self.torch_float_dtype, device=self.device).view(1, -1, self.umax_num_neighs, 3),
            torch.tensor(nblist_info[5], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[6], dtype=torch.int32, device=self.device).view(1,)
        ]
    
    
    def analyse_ase(self,
                    atoms: Atoms):
        cell: np.ndarray = atoms.cell.array.astype(self.np_float_dtype)
        unique_species = sorted(set(atoms.get_atomic_numbers()), key=lambda an: an)
        type_map: Dict[int, int] = {element: idx for idx, element in enumerate(unique_species)}
        types: np.ndarray = np.array([type_map[el] for el in atoms.get_atomic_numbers()])
        cart_coords: np.ndarray = atoms.positions.astype(self.np_float_dtype)
        nblist_info = Nblist.find_info4mlff(cell=cell,
                                            species=types,
                                            coords=cart_coords,
                                            rcut=self.rcut,
                                            umax_num_neigh_atoms=self.umax_num_neighs,
                                            is_cart_coord=True,
                                            pbc_xyz=self.pbc_xyz,
                                            sort=self.sort)
        return [
            torch.tensor(nblist_info[0], dtype=torch.int32, device=self.device).view(1,),
            torch.tensor(nblist_info[1], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[2], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[3], dtype=torch.int32, device=self.device).view(1, -1, self.umax_num_neighs),
            torch.tensor(nblist_info[4], dtype=self.torch_float_dtype, device=self.device).view(1, -1, self.umax_num_neighs, 3),
            torch.tensor(nblist_info[5], dtype=torch.int32, device=self.device).view(1, -1),
            torch.tensor(nblist_info[6], dtype=torch.int32, device=self.device).view(1,)
        ]