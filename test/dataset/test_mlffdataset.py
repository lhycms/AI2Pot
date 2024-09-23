import os
import glob
import unittest
from typing import List
import torch
from dpdata import LabeledSystem, MultiSystems
from torch.utils.data import DataLoader

from ai2pot.data import ScDataset, McDataset


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_OUTCAR_DIR = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe")
Multi_Components_OUTCARS_DIR = os.path.join(TEST_FILES_DIR, "OUTCARs")

class ScDatasetTest(unittest.TestCase):
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
            batch_size=5, 
            shuffle=True)
    
    def test_dataset_info(self):
        self.mlff_dataset.info()
        #print(self.mlff_dataset[0])
    
    def test_load(self):
        for ii, data in enumerate(self.mlff_dataloader):
            print("\tSize of Batch#{0} = {1}".format(ii, data[0].size()[0]))  # data[0]: batch of inum

    def tearDown(self):
        print("ScDatasetTest is tearing down...")


class McDatasetT(unittest.TestCase):
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
        print(self.multi_systems)

    def test_init(self):
        mc_dataset: McDataset = McDataset(multi_systems=self.multi_systems,
                                          torch_float_dtype=torch.float64)

    
    def tearDown(self) -> None:
        print("McDatasetTest is tearing down...")
    
        
if __name__ == "__main__":
    unittest.main()
    