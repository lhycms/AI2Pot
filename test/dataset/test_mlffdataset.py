import os
import unittest
from dpdata import LabeledSystem
from torch.utils.data import DataLoader

from ai2pot.dataset import MlffDataset


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")

class MlffDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        print("MlffDatasetTest is setting up...\n")
        self.outcar_path: str = os.path.join(TEST_FILES_DIR, "ReNbSSe", "OUTCAR")
        self.labeled_system : LabeledSystem = LabeledSystem(self.outcar_path)
        self.mlff_dataset: MlffDataset = MlffDataset(
            labeled_system=self.labeled_system,
            rcut=3.2,
            umax_num_neigh_atoms=20)
        self.mlff_dataloader: DataLoader = DataLoader(
            self.mlff_dataset, 
            batch_size=5, 
            shuffle=True)
    
    def test_dataset_info(self):
        self.mlff_dataset.info()
        #print(mlff_dataset[0])
    
    def test_load(self):
        for ii, data in enumerate(self.mlff_dataloader):
            print("\tSize of Batch#{0} = {1}".format(ii, data[0].size()[0]))  # data[0]: batch of inum

    def tearDown(self):
        print("MlffDatasetTest is tearing down...\n")
        
if __name__ == "__main__":
    unittest.main()
    