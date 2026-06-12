import os
import glob
import unittest
from typing import List
import torch
from torch.utils.data import DataLoader
from ase import Atoms
from pymatgen.core import Structure
from ase.io import read as ase_read

from ai2pot.data import ExtxyzDataset


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_OUTCAR_DIR = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe")
Multi_Components_OUTCARS_DIR = os.path.join(TEST_FILES_DIR, "OUTCARs")
EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/gst/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
# os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train.xyz")
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/gst/train.xyz"

    

class ExtxyzDatasetTest(unittest.TestCase):
    def setUp(self):
        print("ExtxyzDatasetTest (TestCase) is setting up...\n")
        self.extxyz_path: str = EXTXYZ_PATH
        rcut: float = 5.0
        umax_num_neigh_atoms: int = 100
        pbc_xyz: List[bool] = [True, True, True]
        sort: bool = False
        torch_float_dtype: torch._C.dtype = torch.float32
        has_virial: bool = True
        self.extxyz_dataset: ExtxyzDataset = ExtxyzDataset(filename=self.extxyz_path,
                                                           rcut=rcut,
                                                           umax_num_neigh_atoms=umax_num_neigh_atoms,
                                                           pbc_xyz=pbc_xyz,
                                                           sort=sort,
                                                           torch_float_dtype=torch_float_dtype,
                                                           has_virial=has_virial)


    def test_use(self):
        dataloader: DataLoader = DataLoader(dataset=self.extxyz_dataset,
                                            batch_size=1000,
                                            shuffle=True)
        for _, batch_data in enumerate(dataloader):
            print(batch_data[9].shape)
    

    def test_analyse_pymatgen(self):
        atoms_obj: Atoms = ase_read(filename=self.extxyz_path, index=":")
        atoms: Atoms = atoms_obj[0]
        lattice = atoms.cell.array
        types = atoms.get_atomic_numbers()
        cart_coords = atoms.positions
        structure: Structure = Structure(lattice=lattice, 
                                         species=types,
                                         coords=cart_coords, 
                                         coords_are_cartesian=True)
        nblist_info = self.extxyz_dataset.analyse_pymatgen(structure=structure)
        print("test_analyse_pymatgen:")
        print("\t 1. nblist_info[0] = ", nblist_info[0].shape)
        print("\t 2. nblist_info[1] = ", nblist_info[1].shape)
        print("\t 3. nblist_info[2] = ", nblist_info[2].shape)
        print("\t 4. nblist_info[3] = ", nblist_info[3].shape)
        print("\t 5. nblist_info[4] = ", nblist_info[4].shape)
        print("\t 6. nblist_info[5] = ", nblist_info[5].shape)
        print("\t 7. nblist_info[6] = ", nblist_info[6].shape)


    def test_analyse_ase(self):
        atoms_obj: Atoms = ase_read(filename=self.extxyz_path, index=":")
        atoms = atoms_obj[0]
        nblist_info = self.extxyz_dataset.analyse_ase(atoms=atoms)
        print("test_analyse_ase:")
        print("\t 1. nblist_info[0] = ", nblist_info[0].shape)
        print("\t 2. nblist_info[1] = ", nblist_info[1].shape)
        print("\t 3. nblist_info[2] = ", nblist_info[2].shape)
        print("\t 4. nblist_info[3] = ", nblist_info[3].shape)
        print("\t 5. nblist_info[4] = ", nblist_info[4].shape)
        print("\t 6. nblist_info[5] = ", nblist_info[5].shape)
        print("\t 7. nblist_info[6] = ", nblist_info[6].shape)


    def test_get_type_map(self):
        type_map: List[int] = ExtxyzDataset.get_type_map(filename=self.extxyz_path)
        print("1. type_map : ", end="\t")
        print(type_map)


    def tearDown(self):
        print("ExtxyzDatasetTest (TestCase) is tearing down...\n")
    

if __name__ == "__main__":
    unittest.main()
    