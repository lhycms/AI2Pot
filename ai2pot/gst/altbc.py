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
from pymatgen.core import Structure
from dpdata import LabeledSystem

from ai2pot.core import Nblist
from ai2pot.fromcc import gst


class Altbc(object):
    def __init__(self, 
                 rcut: float = 3.2,
                 umax_num_neigh_atoms: int = 100, 
                 angle_threshold: float = 155.0):
        setattr(self, "_rcut", rcut)
        setattr(self, "_umax_num_neigh_atoms", umax_num_neigh_atoms)
        setattr(self, "_angle_threshold", float(angle_threshold))
    
    @staticmethod
    def find_long_short_bonds(inum: int,
                              ilist: np.ndarray,
                              numneigh: np.ndarray,
                              firstneigh: np.ndarray,
                              rcs: np.ndarray,
                              types: np.ndarray,
                              nghost: int,
                              umax_num_neigh_atoms: int = 100,
                              angle_threshold: float = 155.0):
        return gst.find_long_short_bonds(inum,
                                         ilist,
                                         numneigh,
                                         firstneigh,
                                         rcs,
                                         types,
                                         nghost,
                                         umax_num_neigh_atoms,
                                         float(angle_threshold))
    
    @staticmethod
    def find_batch_long_short_bonds(binum: np.ndarray,
                                    bilist: np.ndarray,
                                    bnumneigh: np.ndarray,
                                    bfirstneigh: np.ndarray,
                                    brcs: np.ndarray,
                                    btypes: np.ndarray,
                                    bnghost: np.ndarray,
                                    umax_num_neigh_atoms: int = 100,
                                    angle_threshold: float = 155.0):
        return gst.find_batch_long_short_bonds(binum,
                                               bilist,
                                               bnumneigh,
                                               bfirstneigh,
                                               brcs,
                                               btypes,
                                               bnghost,
                                               umax_num_neigh_atoms,
                                               float(angle_threshold))
        
    def analyse_structure(self, structure: Structure):
        cell: np.ndarray = structure.lattice.matrix
        unique_species = sorted(set(structure.species), key=lambda sp: sp.Z)
        type_map: Dict[str, int] = {element: idx for idx, element in enumerate(unique_species)}
        types: np.ndarray = np.array([type_map[el] for el in structure.species])
        coords: np.ndarray = structure.frac_coords
        nblist_info = Nblist.find_info4mlff(cell,
                                            types,
                                            coords,
                                            self.rcut,
                                            self.umax_num_neigh_atoms,
                                            is_cart_coord=False,
                                            pbc_xyz=[True, True, True],
                                            sort=False)
        return gst.find_long_short_bonds(nblist_info[0], 
                                         nblist_info[1],
                                         nblist_info[2],
                                         nblist_info[3],
                                         nblist_info[4],
                                         nblist_info[5],
                                         nblist_info[6],
                                         self.umax_num_neigh_atoms, 
                                         self.angle_threshold)
    
    def analyse_labeled_system(self, labeled_system: LabeledSystem):
        binum: List[int] = []
        bilist: List[np.ndarray] = []
        bnumneigh: List[np.ndarray] = []
        bfirstneigh: List[np.ndarray] = []
        brcs: List[np.ndarray] = []
        btypes: List[np.ndarray] = []
        bnghost: List[np.ndarray] = []
        for ii in range(len(labeled_system)):
            nblist_info = Nblist.find_info4mlff(labeled_system["cells"][ii],
                                                labeled_system["atom_types"].astype(np.int32),
                                                labeled_system["coords"][ii],
                                                self.rcut,
                                                self.umax_num_neigh_atoms,
                                                True,
                                                [True, True, True],
                                                False)
            binum.append(nblist_info[0])
            bilist.append(nblist_info[1])
            bnumneigh.append(nblist_info[2])
            bfirstneigh.append(nblist_info[3])
            brcs.append(nblist_info[4])
            btypes.append(nblist_info[5])
            bnghost.append(nblist_info[6])
        return Altbc.find_batch_long_short_bonds(np.array(binum).astype(np.int32),
                                                 np.array(bilist),
                                                 np.array(bnumneigh),
                                                 np.array(bfirstneigh),
                                                 np.array(brcs),
                                                 np.array(btypes),
                                                 np.array(bnghost),
                                                 self.umax_num_neigh_atoms,
                                                 self.angle_threshold)
    
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
    def umax_num_neigh_atoms(self, value: float) -> None:
        self._umax_num_neigh_atoms = value
    
    @property
    def angle_threshold(self) -> float:
        return self._angle_threshold
    
    @angle_threshold.setter
    def angle_threshold(self, value: float) -> None:
        self._angle_threshold = value
