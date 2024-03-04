import sys
from typing import List

import numpy as np

from dpdata import LabeledSystem
from torch.utils.data import Dataset
sys.path.append("/data/home/liuhanyu/hyliu/code/matersdk/source/nblist/bind/gen")
import nblist   ### Need to modify


class MlffDataset(Dataset):
    def __init__(
        self,
        labeled_system: LabeledSystem,
        rcut: float,
        umax_num_neigh_atoms: int,
        pbc_xyz: List[bool] = [True, True, True],
        sort: bool = False,
        is_cart_coord: bool = True):
        #self.atom_names: List[str] = labeled_system["atom_names"]
        #self.atom_numbs: List[int] = labeled_system["atom_numbs"]
        #self.atom_types: List[int] = labeled_system["atom_types"]
        #self.cells: np.ndarray = labeled_system["cells"]
        #self.coords: np.ndarray = labeled_system["coords"]
        #self.energies: np.ndarray = labeled_system["energies"]
        #self.forces: np.ndarray = labeled_system["forces"]
        #self.virials: np.ndarray = labeled_system["virials"]
        self.labeled_system: LabeledSystem = labeled_system
        inum_list: List[int] = []
        ilist_list: List[np.ndarray] = []
        numneigh_list: List[np.ndarray] = []
        firstneigh_list: List[np.ndarray] = []
        relativae_coords_list: List[np.ndarray] = []
        types_list: List[np.ndarray] = []
        nghost_list: List[int] = []
        for ii in range(self.labeled_system.get_nframes()):
            inum, ilist, numneigh, firstneigh, relativae_coords, types, nghost = \
                nblist.find_info4mlff(
                    self.labeled_system["cells"][ii].astype(np.float64),
                    self.labeled_system["atom_types"].astype(np.int32),
                    self.labeled_system["coords"][ii].astype(np.float64),
                    rcut,
                    pbc_xyz,
                    umax_num_neigh_atoms,
                    sort,
                    is_cart_coord)
            inum_list.append(inum)
            ilist_list.append(ilist)
            numneigh_list.append(numneigh)
            firstneigh_list.append(firstneigh)
            relativae_coords_list.append(relativae_coords)
            types_list.append(types)
            nghost_list.append(nghost)
            
        self.inum: np.ndarray = np.array(inum_list)
        self.ilist: np.ndarray = np.array(ilist_list)
        self.numneigh: np.ndarray = np.array(numneigh_list)
        self.firstneigh: np.ndarray = np.array(firstneigh_list)
        self.relative_coords: np.ndarray = np.array(relativae_coords_list)
        self.types: np.ndarray = np.array(types_list)
        self.nghost: np.ndarray = np.array(nghost_list)
        self.energies: np.ndarray = self.labeled_system["energies"]
        self.forces: np.ndarray = self.labeled_system["forces"]
        if (self.labeled_system.has_virial()):
            self.virials = self.labeled_system["virials"]

    
    def __len__(self):
        return self.labeled_system.get_nframes()
    
    
    def __getitem__(self, index) -> List[np.ndarray]:
        if self.labeled_system.has_virial():
            return [
                self.inum[index],
                self.ilist[index],
                self.numneigh[index],
                self.firstneigh[index],
                self.relative_coords[index],
                self.types[index],
                self.nghost[index],
                self.energies[index],
                self.forces[index],
                self.virials[index]
            ]
        else:
            return [
                self.inum[index],
                self.ilist[index],
                self.numneigh[index],
                self.firstneigh[index],
                self.relative_coords[index],
                self.types[index],
                self.nghost[index],
                self.energies[index],
                self.forces[index],
            ]
    
    def info(self):
        print(self.labeled_system)
        print("\t+++ 1. self.inum.shape = ", self.inum.shape)
        print("\t+++ 2. self.ilist.shape = ", self.ilist.shape)
        print("\t+++ 3. self.numneigh.shape = ", self.ilist.shape)
        print("\t+++ 4. self.firstneigh.shape = ", self.firstneigh.shape)
        print("\t+++ 5. self.relative_coords.shape = ", self.relative_coords.shape)
        print("\t+++ 6. self.types.shape = ", self.types.shape)
        print("\t+++ 7. self.nghost.shape = ", self.nghost.shape)
        print("\t+++ 8. self.energies.shape = ", self.energies.shape)
        print("\t+++ 9. self.forces.shape = ", self.forces.shape)
        if (self.labeled_system.has_virial()):
            print("\t+++ 10. self.virials.shape = ", self.virials.shape)
        