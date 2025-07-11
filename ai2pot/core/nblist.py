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
import numpy as np
from pymatgen.core import Structure
from ase import Atoms

from ai2pot.fromcc import nblist


class Nblist(object):
    def __init__(self, 
                 cell: np.ndarray,
                 types: np.ndarray,
                 coords: np.ndarray,
                 rcut: float,
                 umax_num_neigh_atoms: int = 100,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = False,
                 is_cart_coords: bool = False):
        nblist_info = nblist.find_info4mlff(cell,
                                            types,
                                            coords,
                                            rcut,
                                            umax_num_neigh_atoms,
                                            is_cart_coords,
                                            pbc_xyz,
                                            sort)
        setattr(self, "_rcut", rcut)
        setattr(self, "_umax_num_neigh_atoms", umax_num_neigh_atoms)
    
        setattr(self, "_inum", nblist_info[0])
        setattr(self, "_ilist", nblist_info[1])
        setattr(self, "_numneigh", nblist_info[2])
        setattr(self, "_firstneigh", nblist_info[3])
        setattr(self, "_rcs", nblist_info[4])
        setattr(self, "_types", nblist_info[5])
        setattr(self, "_nghost", nblist_info[6])
        
        setattr(self, "_distances", np.linalg.norm(self._rcs, axis=-1))

    
    @staticmethod
    def from_pymatgen(structure: Structure,
                      rcut: float,
                      umax_num_neigh_atoms: int = 100,
                      pbc_xyz: List[bool] = [True, True, True],
                      sort: bool = False):
        cell: np.ndarray = structure.lattice.matrix
        types: np.ndarray = np.array([el.Z for el in structure.species])
        frac_coords: np.ndarray = structure.frac_coords
        nblist: Nblist = Nblist(cell=cell,
                                types=types,
                                coords=frac_coords,
                                rcut=rcut,
                                umax_num_neigh_atoms=umax_num_neigh_atoms,
                                pbc_xyz=pbc_xyz,
                                sort=sort,
                                is_cart_coords=False)
        return nblist
    


    @staticmethod
    def from_ase(atoms: Atoms,
                 rcut: float,
                 umax_num_neigh_atoms: int = 100,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = False):
        cell: np.ndarray = atoms.cell.array
        types: np.ndarray = atoms.get_atomic_numbers()
        coords: np.ndarray = atoms.get_positions()
        nblist: Nblist = Nblist(cell=cell,
                                types=types,
                                coords=coords,
                                rcut=rcut,
                                umax_num_neigh_atoms=umax_num_neigh_atoms,
                                pbc_xyz=pbc_xyz,
                                sort=sort,
                                is_cart_coords=True)
        return nblist

        
    @staticmethod
    def find_info4mlff(cell: np.ndarray,
                       species: np.ndarray,
                       coords: np.ndarray,
                       rcut: float,
                       umax_num_neigh_atoms: int,
                       is_cart_coord: bool,
                       pbc_xyz: List[bool],
                       sort: bool):
        return nblist.find_info4mlff(cell,
                                     species,
                                     coords,
                                     rcut,
                                     umax_num_neigh_atoms,
                                     is_cart_coord,
                                     pbc_xyz,
                                     sort)
    
    @property
    def inum(self) -> int:
        return self._inum
    
    @inum.setter
    def inum(self, value: int) -> None:
        self._inum = value
    
    @property
    def ilist(self) -> np.ndarray:
        return self._ilist
    
    @ilist.setter
    def ilist(self, value: np.ndarray) -> None:
        self._ilist = value
    
    @property
    def numneigh(self) -> np.ndarray:
        return self._numneigh

    @numneigh.setter
    def numneigh(self, value: np.ndarray) -> None:
        self._numneigh = value
    
    @property
    def firstneigh(self) -> np.ndarray:
        return self._firstneigh
    
    @firstneigh.setter
    def firstneigh(self, value: np.ndarray) -> None:
        self._firstneigh = value
    
    @property
    def rcs(self) -> np.ndarray:
        return self._rcs
    
    @rcs.setter
    def rcs(self, value: np.ndarray) -> None:
        self._rcs = value
        
    @property
    def types(self) -> np.ndarray:
        return self._types
    
    @types.setter
    def types(self, value: np.ndarray) -> np.ndarray:
        self._types = value
        
    @property
    def nghost(self) -> int:
        return self._nghost

    @nghost.setter
    def nghost(self, value: int) -> None:
        self._ghost = value
        
    @property
    def rcut(self) -> float:
        return self._rcut
    
    @rcut.setter
    def rcut(self, value: float) -> None:
        self._rcut = value
        
    @property
    def umax_num_neigh_atoms(self) -> int:
        return self._umax_num_neigh_atoms
    
    @umax_num_neigh_atoms.setter
    def umax_num_neigh_atoms(self, value: int) -> None:
        self._umax_num_neigh_atoms = value

    @property
    def distances(self) -> np.ndarray:
        return self._distances
    
    @distances.setter
    def distances(self, value: np.ndarray) -> None:
        self._distances = value

    @distances.deleter
    def distances(self) -> None:
        del self._distances