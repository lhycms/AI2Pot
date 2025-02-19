from typing import List, Dict
import numpy as np
from pymatgen.core import Structure, Element

from ai2pot.fromcc import nblist


class Nblist(object):
    def __init__(self, 
                 structure: Structure,
                 rcut: float,
                 umax_num_neigh_atoms: int = 100,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = False):
        self.structure: Structure = structure
        cell: np.ndarray = self.structure.lattice.matrix
        unique_species = sorted(set(self.structure.species), key=lambda sp: sp.Z)
        type_map: Dict[str, int] = {element: idx for idx, element in enumerate(unique_species)}
        type_map_inv: Dict[str, int] = {idx: element.Z for idx, element in enumerate(unique_species)}
        types: np.ndarray = np.array([type_map[el] for el in self.structure.species])
        coords: np.ndarray = self.structure.frac_coords
        nblist_info = nblist.find_info4mlff(cell,
                                            types,
                                            coords,
                                            rcut,
                                            umax_num_neigh_atoms,
                                            False,
                                            pbc_xyz,
                                            sort)
        setattr(self, "_rcut", rcut)
        setattr(self, "_umax_num_neigh_atoms", umax_num_neigh_atoms)
        setattr(self, "_type_map", type_map)
        setattr(self, "_type_map_inv", type_map_inv)
    
        setattr(self, "_inum", nblist_info[0])
        setattr(self, "_ilist", nblist_info[1])
        setattr(self, "_numneigh", nblist_info[2])
        setattr(self, "_firstneigh", nblist_info[3])
        setattr(self, "_rcs", nblist_info[4])
        setattr(self, "_types", nblist_info[5])
        setattr(self, "_nghost", nblist_info[6])
        
        setattr(self, "_distances", np.linalg.norm(self._rcs, axis=-1))
        
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
        
    @property
    def type_map(self) -> Dict[Element, int]:
        return self._type_map
    
    @type_map.setter
    def type_map(self, value: Dict[Element, int]):
        self._type_map = value

    @property
    def type_map_inv(self) -> Dict[int, int]:
        return self._type_map_inv
    
    @type_map_inv.setter
    def type_map(self, value: Dict[int, int]):
        self._type_map_inv = value