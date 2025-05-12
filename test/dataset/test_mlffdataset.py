import os
import glob
import unittest
from typing import List
import torch
from dpdata import LabeledSystem, MultiSystems
from torch.utils.data import DataLoader
from ase import Atoms
from pymatgen.core import Structure
from ase.io import read as ase_read

from ai2pot.data import ScDataset, McDataset, ExtxyzDataset


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_OUTCAR_DIR = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe")
Multi_Components_OUTCARS_DIR = os.path.join(TEST_FILES_DIR, "OUTCARs")
EXTXYZ_DIR = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe")

class ScDatasetTest(object):
    def setUp(self) -> None:
        print("ScDatasetTest is setting up...\n")
        self.outcar_path: str = os.path.join(ReNbSSe_OUTCAR_DIR, "OUTCAR")
        self.labeled_system : LabeledSystem = LabeledSystem(self.outcar_path)
        self.mlff_dataset: ScDataset = ScDataset(
            labeled_system=self.labeled_system,
            rcut=3.2,
            umax_num_neigh_atoms=20)
        self.mlff_dataloader: DataLoader = DataLoader(
            self.mlff_dataset, 
            batch_size=1, 
            shuffle=True)
    
    def test_getitem(self):
        for ii in self.mlff_dataset[0]:
            if (type(ii) is torch.Tensor):
                print(ii.size())
            else:
                print(type(ii))
    
    def test_dataloader(self):
        for batch_idx, batch_data in enumerate(self.mlff_dataloader):
            print("\tbinum.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[0].size()))
            print("\tbilist.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[1].size()))
            print("\tbnumneigh.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[2].size()))
            print("\tbfirstneigh.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[3].size()))
            print("\tbrcs.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[4].size()))
            print("\tbtypes.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[5].size()))
            print("\tbnghost.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[6].size()))
            print("\tbenergies.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[7].size()))
            print("\tbforces.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[8].size()))
            print("\tbvirials.size() inner Batch#{0} = {1}".format(batch_idx, batch_data[9].size()))
            print("\n")

    def tearDown(self):
        print("ScDatasetTest is tearing down...")


class McDatasetTest(object):
    def setUp(self) -> None:
        print("McDatasetTest is setting up...\n")
        self.outcars_dir: str = Multi_Components_OUTCARS_DIR
        fs: List[str] = glob.glob(f"{Multi_Components_OUTCARS_DIR}/*/OUTCAR")
        self.multi_systems = MultiSystems()
        for f in fs:
            try:
                ls: LabeledSystem = LabeledSystem(file_name=f, fmt="vasp/outcar")
            except Exception as _:
                print(f)
            if (len(ls) > 0):
                self.multi_systems.append(ls)
        self.mc_dataset: McDataset = McDataset(multi_systems=self.multi_systems,
                                               rcut=3.2,
                                               umax_num_neigh_atoms=20,
                                               pbc_xyz=[True, True, True],
                                               sort=False,
                                               torch_float_dtype=torch.float64)
        self.mc_dataloader = DataLoader(dataset=self.mc_dataset,
                                        batch_size=32,
                                        shuffle=True)
    
    def test_getitem(self):
        for ii in self.mc_dataset[0]:
            if (type(ii) is torch.Tensor):
                print(ii.size())
            else:
                print(type(ii))
    
    def test_dataloader(self):
        for ii, batch in enumerate(self.mc_dataloader):
            print("\tSize of Batch#{0} = {1}".format(ii, batch[1].size()[0]))

    def tearDown(self) -> None:
        print("McDatasetTest is tearing down...")
    

class ExtxyzDatasetTest(unittest.TestCase):
    def setUp(self):
        print("ExtxyzDatasetTest (TestCase) is setting up...\n")
        self.extxyz_path: str = os.path.join(EXTXYZ_DIR, "train_m.xyz")
        rcut: float = 5.0
        umax_num_neigh_atoms: int = 100
        pbc_xyz: List[bool] = [True, True, True]
        sort: bool = False
        torch_float_dtype: torch._C.dtype = torch.float32
        has_virial: bool = False
        self.extxyz_dataset: ExtxyzDataset = ExtxyzDataset(filename=self.extxyz_path,
                                                           rcut=rcut,
                                                           umax_num_neigh_atoms=umax_num_neigh_atoms,
                                                           pbc_xyz=pbc_xyz,
                                                           sort=sort,
                                                           torch_float_dtype=torch_float_dtype,
                                                           has_virial=has_virial)


    def test_use(self):
        dataloader: DataLoader = DataLoader(dataset=self.extxyz_dataset,
                                            batch_size=1,
                                            shuffle=True)
        #for _, batch_data in enumerate(dataloader):
            #print(batch_data[9])
    

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
    