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
from dpdata import LabeledSystem, MultiSystems
from ase import Atoms
from ase.io import read as ase_read

from ai2pot.core.nblist import Nblist
from ai2pot.data.mlffdataset import ExtxyzDataset


def svd_solve(A, b):
    U, S, V = np.linalg.svd(A)
    B = np.matmul(U.T, b)
    X = B[:len(S), :] / S.reshape(len(S), -1)
    x = np.matmul(V.T, X)
    return x


class LsShifter(object):
    def __init__(self, ls: LabeledSystem):
        self.nframes: int = len(ls)
        self.ntypes: List[int] = ls["atom_types"]
        self.type_names: List[str] = ls["atom_names"]
        self.types_energy_shifts, self.systems_shifted_energy = LsShifter.get_energy_shifts(ls)

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        print("LsShifter Summary:")
        print("------------------")
        print("1. nframes = ", self.nframes)
        print("2. type_name = ", self.type_names)
        print("3. types energy shifts:\n", self.types_energy_shifts)
        print("4. systems shifted energy:\n", self.systems_shifted_energy)
        return ""
    
    @staticmethod
    def get_energy_shifts(ls: LabeledSystem):
        unique_atom_types: np.ndarray = np.unique(ls["atom_types"])
        nframes: int = len(ls)
        ntypes: int = unique_atom_types.shape[0]
        coeff_mtx: np.ndarray = np.zeros((nframes, ntypes))
        energy_mtx: np.ndarray = np.zeros((nframes, 1))

        for ii in range(nframes):
            for jj in range(ntypes):
                coeff_mtx[ii][jj] = ls["atom_numbs"][jj]
            energy_mtx[ii][0] = ls["energies"][ii]
        
        if (np.linalg.matrix_rank(coeff_mtx) < ntypes):
            import itertools
            for type_pair in itertools.combinations(unique_atom_types, 2):
                add_mtx: np.ndarray = np.zeros(ntypes)
                add_mtx[type_pair[0]] = 1
                add_mtx[type_pair[1]] = -1
                add_energy = np.zeros(1)
                coeff_mtx = np.r_[coeff_mtx, [add_mtx]]
                energy_mtx = np.r_[energy_mtx, [add_energy]]
        types_energy_shifts: np.ndarray = svd_solve(coeff_mtx, energy_mtx)
        systems_shifted_energy: np.ndarray = (energy_mtx - np.matmul(coeff_mtx, types_energy_shifts)).flatten()
        return types_energy_shifts.flatten(), systems_shifted_energy


class MsShifter(object):
    def __init__(self, ms: MultiSystems):
        self.nsys: int = len(ms)
        self.nframes: int = ms.get_nframes()
        self.ntypes: int = np.unique([ls["atom_types"] for ls in ms]).shape[0]
        self.type_names: List[str] = []
        for ls in ms:
            for ele in ls["atom_names"]:
                if ele not in self.type_names:
                    self.type_names.append(ele)
        self.types_energy_shifts, self.systems_shifted_energy = MsShifter.get_energy_shifts(ms=ms)
    
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        print("LsShifter Summary:")
        print("------------------")
        print("1. nframes = ", self.nframes)
        print("2. type_name = ", self.type_names)
        print("3. types energy shifts:\n", self.types_energy_shifts)
        print("4. systems shifted energy:\n", self.systems_shifted_energy)
        return ""
            
    @staticmethod
    def get_energy_shifts(ms: MultiSystems):
        ntypes: int = np.unique([ls["atom_types"] for ls in ms]).shape[0]
        unique_atom_types: np.ndarray = np.arange(ntypes)
        nframes: int = ms.get_nframes()
        
        coeff_mtx: np.ndarray = np.zeros((nframes, ntypes))
        energy_mtx: np.ndarray = np.zeros((nframes, 1))
        accum_nframes: int = 0
        for ls in ms:
            ls_nframes: int = len(ls)
            for ii in range(ls_nframes):
                for tmp_type in np.unique(ls["atom_types"]):
                    coeff_mtx[accum_nframes+ii][tmp_type] = ls["atom_numbs"][tmp_type]
                energy_mtx[accum_nframes+ii] = ls["energies"][ii]
            accum_nframes += ls_nframes

        if (np.linalg.matrix_rank(coeff_mtx) < ntypes):
            import itertools
            for type_pair in itertools.combinations(unique_atom_types, 2):
                add_mtx: np.ndarray = np.zeros(ntypes)
                add_mtx[type_pair[0]] = 1
                add_mtx[type_pair[1]] = -1
                add_energy: np.ndarray = np.zeros(1)
                coeff_mtx = np.r_[coeff_mtx, [add_mtx]]
                energy_mtx = np.r_[energy_mtx, [add_energy]]
        types_energy_shifts: np.ndarray = svd_solve(coeff_mtx, energy_mtx)
        systems_shifted_energy: np.ndarray = (energy_mtx - np.matmul(coeff_mtx, types_energy_shifts)).flatten()
        return types_energy_shifts.flatten(), systems_shifted_energy
        

class ExtxyzShifter(object):
    def __init__(self, extxyz_path: str):
        self.extxyz_path: str = extxyz_path
        self.atoms_list: List[Atoms] = ase_read(self.extxyz_path, index=":")
        self.nframes: int = len(self.atoms_list)
        self.type_map: List[int] = self._get_type_map()
        self.ntypes: int = len(self.type_map)
        self.unique_types: List[int] = [*range(len(self.type_map))]
        self.types_energy_shifts, self.shifted_energies = self.get_energy_shifts()

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        print("LsShifter Summary:")
        print("------------------")
        print("1. nframes = ", self.nframes)
        print("2. type_map = ", self.type_map)
        print("3. types energy shifts:\n", self.types_energy_shifts)
        print("4. systems shifted energy:\n", self.shifted_energies)
        return ''
    
    def _get_type_map(self):
        type_map: List[int] = []
        for tmp_atoms in self.atoms_list:
            an_list: List[int] = np.unique(tmp_atoms.get_atomic_numbers())
            for tmp_an in an_list:
                if tmp_an not in type_map:
                    type_map.append(tmp_an)
        return np.sort(type_map).tolist()

    
    def get_energy_shifts(self):
        coeff_mtx: np.ndarray = np.zeros((self.nframes, self.ntypes))
        energy_mtx: np.ndarray = np.zeros((self.nframes, 1))

        for tmp_frame_idx in range(self.nframes):
            types: np.ndarray = np.array([self.type_map.index(tmp_an) for tmp_an in self.atoms_list[tmp_frame_idx].get_atomic_numbers()])
            energy = self.atoms_list[tmp_frame_idx].get_potential_energy()
            for tmp_type_idx in range(self.ntypes):
                coeff_mtx[tmp_frame_idx][tmp_type_idx] = np.count_nonzero(types == tmp_type_idx)
            energy_mtx[tmp_frame_idx][0] = energy
            
        if (np.linalg.matrix_rank(coeff_mtx) < self.ntypes):
            import itertools
            for type_pair in itertools.combinations(self.unique_types, 2):
                add_mtx: np.ndarray = np.zeros(self.ntypes)
                add_mtx[type_pair[0]] = 1
                add_mtx[type_pair[1]] = -1
                add_energy: np.ndarray = np.zeros(1)
                coeff_mtx = np.r_[coeff_mtx, [add_mtx]]
                energy_mtx = np.r_[energy_mtx, [add_energy]]
        
        types_energy_shifts: np.ndarray = svd_solve(coeff_mtx, energy_mtx)
        systems_shifted_energy: np.ndarray = (energy_mtx - np.matmul(coeff_mtx, types_energy_shifts)).flatten()

        return types_energy_shifts.flatten(), systems_shifted_energy
    


class MlffDatasetShifter(object):
    def __init__(self,
                 extxyz_dataset: ExtxyzDataset):
        self.extxyz_dataset: ExtxyzDataset = extxyz_dataset
        self.nframes: int = len(self.extxyz_dataset)
        self.ntypes: int = len(self.extxyz_dataset.type_map)
        self.type_map: List[int] = self.extxyz_dataset.type_map
        self.unique_types: List[int] = [*range(len(self.type_map))]
        self.types_energy_shifts, self.shifted_energies = self.get_energy_shifts()


    def __str__(self):
        return self.__repr__()
    

    def __repr__(self):
        print("LsShifter Summary:")
        print("------------------")
        print("1. nframes = ", self.nframes)
        print("2. type_map = ", self.type_map)
        print("3. types energy shifts:\n", self.types_energy_shifts)
        print("4. systems shifted energy:\n", self.shifted_energies)
        return ''
    

    def get_energy_shifts(self):
        coeff_mtx: np.ndarray = np.zeros((self.nframes, self.ntypes))
        energy_mtx: np.ndarray = np.zeros((self.nframes, 1))

        for tmp_frame_idx in range(self.nframes):
            batch = self.extxyz_dataset
            types: np.ndarray = batch[tmp_frame_idx][5].numpy()
            energy = batch[tmp_frame_idx][7].item()
            for tmp_type_idx in range(self.ntypes):
                coeff_mtx[tmp_frame_idx][tmp_type_idx] = np.count_nonzero(types == tmp_type_idx)
            energy_mtx[tmp_frame_idx][0] = energy
        
        if (np.linalg.matrix_rank(coeff_mtx) < self.ntypes):
            import itertools
            for type_pair in itertools.combinations(self.unique_types, 2):
                add_mtx: np.ndarray = np.zeros(self.ntypes)
                add_mtx[type_pair[0]] = 1
                add_mtx[type_pair[1]] = -1
                add_energy: np.ndarray = np.zeros(1)
                coeff_mtx = np.r_[coeff_mtx, [add_mtx]]
                energy_mtx = np.r_[energy_mtx, [add_energy]]
        
        types_energy_shifts: np.ndarray = svd_solve(coeff_mtx, energy_mtx)
        systems_shifted_energy: np.ndarray = (energy_mtx - np.matmul(coeff_mtx, types_energy_shifts)).flatten()

        return types_energy_shifts.flatten(), systems_shifted_energy



class ExtxyzAnalyser(object):
    def __init__(self,
                 extxyz_path: str,
                 rcut: float,
                 umax_num_neigh_atoms: int = 300,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = False):
        self.extxyz_path: str = extxyz_path
        self.atoms_list: List[Atoms] = ase_read(extxyz_path, index=":")
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool = sort
    

    def get_max_num_neigh_atoms(self):
        pass

    
    def get_all_forces(self):
        forces_list: List[np.ndarray] = []
        for tmp_atoms in self.atoms_list:
            forces_list.append(tmp_atoms.get_forces())
        forces_array: np.ndarray = np.array(forces_list)
        return forces_array


    def get_forces_subtract_zbl(self,
                                zbl_rmax: float,
                                zbl_rmin: float):
        pass


    def get_min_distance(self):
        min_distance: float = 100.0
        for tmp_atoms in self.atoms_list:
            nblist: Nblist = Nblist.from_ase(atoms=tmp_atoms,
                                             rcut=self.rcut,
                                             umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                             pbc_xyz=self.pbc_xyz,
                                             sort=self.sort)
            tmp_min_distance = np.min(nblist.distances[nblist.distances != 0])
            if (tmp_min_distance < min_distance):
                min_distance = tmp_min_distance
        return min_distance
    

    def get_all_distances(self):
        distances_list: List[float] = []
        for tmp_atoms in self.atoms_list:
            nblist: Nblist = Nblist.from_ase(atoms=tmp_atoms,
                                             rcut=self.rcut,
                                             umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                             pbc_xyz=self.pbc_xyz,
                                             sort=self.sort)
            distances_list.append(nblist.distances)
        distances_array: np.ndarray = np.array(distances_list)
        distances_array = distances_array[distances_array != 0]
        return distances_array

    
