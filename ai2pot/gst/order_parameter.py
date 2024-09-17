from typing import Dict
import numpy as np
from pymatgen.core import Structure, Element

from ai2pot.core import Nblist
from ai2pot.fromcc import gst


class OrderParameter(object):
    def __init__(self, 
                 rcut: float,
                 umax_num_neigh_atoms: int):
        setattr(self, "_rcut", rcut)
        setattr(self, "_umax_num_neigh_atoms", umax_num_neigh_atoms)
    
    @staticmethod
    def find_order_parameters(inum: int,
                              ilist: np.ndarray,
                              numneigh: np.ndarray,
                              firstneigh: np.ndarray,
                              rcs: np.ndarray,
                              types: np.ndarray,
                              nghost: int,
                              umax_num_neigh_atoms: int):
        return gst.find_order_parameters(inum,
                                         ilist,
                                         numneigh,
                                         firstneigh,
                                         rcs,
                                         types,
                                         nghost,
                                         umax_num_neigh_atoms)
    
    def analyse_structure(self, structure: Structure):
        cell: np.ndarray = structure.lattice.matrix
        unique_species = sorted(set(structure.species), key=lambda sp: sp.Z)
        type_map: Dict[Element, int] = {element: idx for idx, element in enumerate(unique_species)}
        types: np.ndarray = np.array([type_map[el] for el in structure.species])
        coords: np.ndarray = structure.frac_coords
        nblist_info = Nblist.find_info4mlff(cell=cell,
                                            species=types,
                                            coords=coords,
                                            rcut=self.rcut,
                                            umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                            is_cart_coord=False,
                                            pbc_xyz=[True, True, True],
                                            sort=True)
        return gst.find_order_parameters(nblist_info[0],
                                         nblist_info[1],
                                         nblist_info[2],
                                         nblist_info[3],
                                         nblist_info[4],
                                         nblist_info[5],
                                         nblist_info[6],
                                         self.umax_num_neigh_atoms)
    
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
    def umax_num_neigh_atoms(self, value: int) -> int:
        self._umax_num_neigh_atoms = value
