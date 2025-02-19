import glob
from typing import List, Dict

import numpy as np
from dpdata import LabeledSystem, MultiSystems
from ase import Atoms
from ase.io import read as ase_read
import torch
from torch.utils.data import Dataset
from ai2pot.core import Nblist


class ScDataset(Dataset):
    def __init__(self,
                 labeled_system: LabeledSystem,
                 rcut: float,
                 umax_num_neigh_atoms: int,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = False,
                 torch_float_dtype: torch._C.dtype = torch.float64,
                 has_virial: bool = True):
        self.labeled_system: LabeledSystem = labeled_system
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool = sort
        self.npy_float_dtype: np.dtype = np.float64
        self.has_virial: bool = has_virial
        if (torch_float_dtype == torch.float32):
            self.torch_float_dtype = torch.float32
            self.npy_float_dtype = np.float32
        else:
            self.torch_float_dtype = torch.float64
            self.npy_float_dtype = np.float64

    def __len__(self):
        return self.labeled_system.get_nframes()    
    
    def __getitem__(self, index) -> List[np.ndarray]:
        inum, ilist, numneigh, firstneigh, relative_coords, types, nghost = \
            Nblist.find_info4mlff(
                self.labeled_system["cells"][index].astype(self.npy_float_dtype),
                self.labeled_system["atom_types"].astype(np.int32),
                self.labeled_system["coords"][index].astype(self.npy_float_dtype),
                self.rcut,
                self.umax_num_neigh_atoms,
                True,
                self.pbc_xyz,
                self.sort)
        if self.labeled_system.has_virial() and self.has_virial:
            return [
                torch.tensor(inum, dtype=torch.int32),
                torch.tensor(ilist, dtype=torch.int32),
                torch.tensor(numneigh, dtype=torch.int32),
                torch.tensor(firstneigh, dtype=torch.int32),
                torch.tensor(relative_coords, dtype=self.torch_float_dtype),
                torch.tensor(types, dtype=torch.int32),
                torch.tensor(nghost, dtype=torch.int32),
                torch.tensor(self.labeled_system["energies"][index], dtype=self.torch_float_dtype),
                torch.tensor(self.labeled_system["forces"][index], dtype=self.torch_float_dtype),
                torch.tensor(self.labeled_system["virials"][index], dtype=self.torch_float_dtype)
            ]
        else:
            return [
                torch.tensor(inum, dtype=torch.int32),
                torch.tensor(ilist, dtype=torch.int32),
                torch.tensor(numneigh, dtype=torch.int32),
                torch.tensor(firstneigh, dtype=torch.int32),
                torch.tensor(relative_coords, dtype=self.torch_float_dtype),
                torch.tensor(types, dtype=torch.int32),
                torch.tensor(nghost, dtype=torch.int32),
                torch.tensor(self.labeled_system["energies"][index], dtype=self.torch_float_dtype),
                torch.tensor(self.labeled_system["forces"][index], dtype=self.torch_float_dtype)
            ]        


class McDataset(Dataset):
    def __init__(self,
                 multi_systems: MultiSystems,
                 rcut: float,
                 umax_num_neigh_atoms: int,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = False,
                 torch_float_dtype: torch._C.dtype = torch.float64,
                 has_virial: bool = True):
        self.multi_systems: MultiSystems = multi_systems
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool = sort
        self.has_virial: bool = has_virial
        if (torch_float_dtype == torch.float32):
            self.torch_float_dtype = torch.float32
            self.npy_float_dtype = np.float32
        else:
            self.torch_float_dtype = torch.float64
            self.npy_float_dtype = np.float64
        
        natoms_list: List[int] = []
        for system in self.multi_systems:
            natoms_list.append(system.get_natoms())
        self.max_num_atoms = max(natoms_list)

        self.nframes: List[int] = [len(ls) for ls in self.multi_systems]
        
    def __len__(self) -> int:
        num_frames: int = 0
        for ls in self.multi_systems:
            num_frames += len(ls)
        return num_frames
    
    def __getitem__(self, index: int) -> int:
        ls_index: int = -1
        el_index: int = -1        
        cumsum_value: int = 0
        for ii, tmp_ls in enumerate(self.multi_systems):
            cumsum_value += len(tmp_ls)
            if (cumsum_value > index):
                ls_index = ii
                break
        el_index = int(index - np.sum(self.nframes[:ls_index]))
        ls: LabeledSystem = self.multi_systems[ls_index]
        
        num_real_atoms: int = ls.get_natoms()
        ilist: np.ndarray = np.zeros(shape=(self.max_num_atoms), 
                                     dtype=np.int32) - 1
        numneigh: np.ndarray = np.zeros(shape=(self.max_num_atoms),
                                        dtype=np.int32) - 1
        firstneigh: np.ndarray = np.zeros(shape=(self.max_num_atoms, self.umax_num_neigh_atoms),
                                          dtype=np.int32) - 1
        relative_coords: np.ndarray = np.zeros(shape=(self.max_num_atoms, self.umax_num_neigh_atoms, 3),
                                               dtype=self.npy_float_dtype) - 1
        types: np.ndarray = np.zeros(shape=(self.max_num_atoms),
                                     dtype=np.int32) - 1
        forces: np.ndarray = np.zeros(shape=(self.max_num_atoms, 3),
                                      dtype=self.npy_float_dtype) - 1
        if ls.has_virial():
            virials: np.ndarray = np.zeros(shape=(3, 3),
                                           dtype=self.npy_float_dtype)
        energies = ls["energies"][el_index]
        forces[:num_real_atoms] = ls["forces"][el_index]
        
        if ls.has_virial():
            virials[:num_real_atoms] = ls["virials"][el_index]
        
        inum, ilist[:num_real_atoms], numneigh[:num_real_atoms], firstneigh[:num_real_atoms], \
        relative_coords[:num_real_atoms], types[:num_real_atoms], nghost = \
            Nblist.find_info4mlff(
                ls["cells"][el_index].astype(self.npy_float_dtype),
                ls["atom_types"].astype(np.int32),
                ls["coords"][el_index].astype(self.npy_float_dtype),
                self.rcut,
                self.umax_num_neigh_atoms,
                True,
                self.pbc_xyz,
                self.sort)
            
        if ls.has_virial() and self.has_virial:
            return [
                torch.tensor(inum, dtype=torch.int32),
                torch.tensor(ilist, dtype=torch.int32),
                torch.tensor(numneigh, dtype=torch.int32),
                torch.tensor(firstneigh, dtype=torch.int32),
                torch.tensor(relative_coords, dtype=self.torch_float_dtype),
                torch.tensor(types, dtype=torch.int32),
                torch.tensor(nghost, dtype=torch.int32),
                torch.tensor(energies, dtype=self.torch_float_dtype),
                torch.tensor(forces, dtype=self.torch_float_dtype),
                torch.tensor(virials, dtype=self.torch_float_dtype)
            ]
        else:
            return [
                torch.tensor(inum, dtype=torch.int32),
                torch.tensor(ilist, dtype=torch.int32),
                torch.tensor(numneigh, dtype=torch.int32),
                torch.tensor(firstneigh, dtype=torch.int32),
                torch.tensor(relative_coords, dtype=self.torch_float_dtype),
                torch.tensor(types, dtype=torch.int32),
                torch.tensor(nghost, dtype=torch.int32),
                torch.tensor(energies, dtype=self.torch_float_dtype),
                torch.tensor(forces, dtype=self.torch_float_dtype)
            ]
    
    @staticmethod
    def from_dir(dir_name: str,
                 file_name: str, 
                 fmt: str = "auto",
                 torch_float_dtype: torch._C.dtype = torch.float64):
        ms: MultiSystems = MultiSystems()
        fs: List[str] = glob.glob(pathname=f"{dir_name}/*/{file_name}",
                                  recursive=False)
        for f in fs:
            try:
                ls = LabeledSystem(file_name=f, fmt=fmt)
            except Exception as _:
                print(f)
            if (len(ls) > 0):
                ms.append(ls) 
        return McDataset(multi_systems=ms, 
                         torch_float_dtype=torch_float_dtype)


class ExtxyzDataset(Dataset):
    def __init__(self,
                 filename: str,
                 rcut: float,
                 umax_num_neigh_atoms: int,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = False,
                 torch_float_dtype: torch._C.dtype = torch.float32,
                 has_virial: bool = False):
        self.atoms_list: List[Atoms] = ase_read(filename=filename, index=":")
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool= sort
        self.has_virial: bool = has_virial
        if (torch_float_dtype == torch.float32):
            self.npy_float_dtype = np.float32
            self.torch_float_dtype = torch.float32
        else:
            self.npy_float_dtype = np.float64
            self.torch_float_dtype = torch.float64
        self.max_num_atoms: int = self._get_max_num_atoms()
        self.type_map: Dict[int, int] = self._get_type_map()    # {atomic_numbers : types}

    
    def __len__(self):
        return len(self.atoms_list)
    
    
    def __getitem__(self, index: int):
        cell: np.ndarray = self.atoms_list[index].get_cell().array.astype(self.npy_float_dtype)
        atom_types: List[int] = np.array( [self.type_map[symbol] for symbol in self.atoms_list[index].get_atomic_numbers()] ).astype(np.int32)
        coords: np.ndarray = self.atoms_list[index].get_positions().astype(self.npy_float_dtype)

        num_real_atoms: int = len(self.atoms_list[index])
        ilist: np.ndarray = np.zeros(shape=(self.max_num_atoms), dtype=np.int32)
        numneigh: np.ndarray = np.zeros(shape=(self.max_num_atoms), dtype=np.int32)
        firstneigh: np.ndarray = np.zeros(shape=(self.max_num_atoms, self.umax_num_neigh_atoms), dtype=np.int32)
        relative_coords: np.ndarray = np.zeros(shape=(self.max_num_atoms, self.umax_num_neigh_atoms, 3), dtype=self.npy_float_dtype)
        types: np.ndarray = np.zeros(shape=(self.max_num_atoms), dtype=np.int32)
        forces: np.ndarray = np.zeros(shape=(self.max_num_atoms, 3), dtype=self.npy_float_dtype)
        
        inum, ilist[:num_real_atoms], numneigh[:num_real_atoms], firstneigh[:num_real_atoms], \
        relative_coords[:num_real_atoms], types[:num_real_atoms], nghost = \
            Nblist.find_info4mlff(
                cell,
                atom_types,
                coords,
                self.rcut,
                self.umax_num_neigh_atoms,
                True,
                self.pbc_xyz,
                self.sort)
        forces[:num_real_atoms] = self.atoms_list[index].get_forces()
        
        if self.has_virial:
            return [
                torch.tensor(inum, dtype=torch.int32),
                torch.tensor(ilist, dtype=torch.int32),
                torch.tensor(numneigh, dtype=torch.int32),
                torch.tensor(firstneigh, dtype=torch.int32),
                torch.tensor(relative_coords, dtype=self.torch_float_dtype),
                torch.tensor(types, dtype=torch.int32),
                torch.tensor(nghost, dtype=torch.int32),
                torch.tensor(self.atoms_list[index].get_total_energy(), dtype=self.torch_float_dtype),
                torch.tensor(forces, dtype=self.torch_float_dtype),
                # ASE calculated in units of eV/A^3, virial: eV
                torch.tensor(-1.0 * self.atoms_list[index].get_stress(voigt=False).reshape(9) * self.atoms_list[index].get_volume()).view(3, 3)
            ]
        else:
            return [
                torch.tensor(inum, dtype=torch.int32),
                torch.tensor(ilist, dtype=torch.int32),
                torch.tensor(numneigh, dtype=torch.int32),
                torch.tensor(firstneigh, dtype=torch.int32),
                torch.tensor(relative_coords, dtype=self.torch_float_dtype),
                torch.tensor(types, dtype=torch.int32),
                torch.tensor(nghost, dtype=torch.int32),
                torch.tensor(self.atoms_list[index].get_total_energy(), dtype=self.torch_float_dtype),
                torch.tensor(forces, dtype=self.torch_float_dtype)
            ]

    def _get_max_num_atoms(self):
        max_num_atoms: int = 0
        for atoms in self.atoms_list:
            if (len(atoms) > max_num_atoms):
                max_num_atoms = len(atoms)
        return max_num_atoms

    def _get_type_map(self):
        atom_symbol_list: List[int] = []
        for atoms in self.atoms_list:
            for tmp_symbol in np.unique(atoms.get_atomic_numbers()):
                if tmp_symbol not in atom_symbol_list:
                    atom_symbol_list.append(tmp_symbol)
        type_map: Dict[int, int] = {symbol: ii for ii, symbol in enumerate(np.sort(atom_symbol_list))}
        return type_map
    