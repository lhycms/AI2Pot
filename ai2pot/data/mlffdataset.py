import glob
from typing import List

import numpy as np

from dpdata import LabeledSystem, MultiSystems
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
                 torch_float_dtype: torch._C.dtype = torch.float64):
        self.labeled_system: LabeledSystem = labeled_system
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool = sort
        self.npy_float_dtype: np.dtype = np.float64
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
        if self.labeled_system.has_virial():
            return [
                inum,
                ilist,
                numneigh,
                firstneigh,
                relative_coords,
                types,
                nghost,
                self.labeled_system["energies"],
                self.labeled_system["forces"],
                self.labeled_system["virials"]
            ]
        else:
            return [
                inum,
                ilist,
                numneigh,
                firstneigh,
                relative_coords,
                types,
                nghost,
                self.labeled_system["energies"],
                self.labeled_system["forces"]
            ]
    
    def info(self):
        print(self.labeled_system)
        print("\t+++ 1. self.inum.size() = ", self.inum.size())
        print("\t+++ 2. self.ilist.size() = ", self.ilist.size())
        print("\t+++ 3. self.numneigh.size() = ", self.ilist.size())
        print("\t+++ 4. self.firstneigh.size() = ", self.firstneigh.size())
        print("\t+++ 5. self.relative_coords.size() = ", self.relative_coords.size())
        print("\t+++ 6. self.types.size() = ", self.types.size())
        print("\t+++ 7. self.nghost.size() = ", self.nghost.size())
        print("\t+++ 8. self.energies.size() = ", self.energies.size())
        print("\t+++ 9. self.forces.size() = ", self.forces.size())
        if (self.labeled_system.has_virial()):
            print("\t+++ 10. self.virials.shape = ", self.virials.size())
        

class McDataset(Dataset):
    def __init__(self,
                 multi_systems: MultiSystems,
                 rcut: float,
                 umax_num_neigh_atoms: int,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = False,
                 torch_float_dtype: torch._C.dtype = torch.float64):
        self.multi_systems: MultiSystems = multi_systems
        self.rcut: float = rcut
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.pbc_xyz: List[bool] = pbc_xyz
        self.sort: bool = sort
        if (torch_float_dtype == torch.float32):
            self.torch_float_dtype = torch.float32
            self.npy_float_dtype = np.float32
        else:
            self.torch_float_dtype = torch.float64
            self.npy_float_dtype = np.float64
        
        # Get max number of atoms
        natoms_list: List[int] = []
        for system in self.multi_systems:
            natoms_list.append(system.get_natoms())
        self.max_num_atoms = max(natoms_list)
        
    def __len__(self) -> int:
        num_frames: int = 0
        for ls in self.multi_systems:
            num_frames += len(ls)
        return num_frames
    
    def __getitem__(self, index: int) -> int:
        inum, ilist, numneigh, firstneigh, relative_coords, types, nghost = \
            Nblist.find_info4mlff()
    
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