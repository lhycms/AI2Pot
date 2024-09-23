import os
import unittest
from dpdata import LabeledSystem
from torch.utils.data import DataLoader

from ai2pot.data import ScDataset


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_OUTCAR_DIR = os.path.join(TEST_FILES_DIR, "OUTCARs", "ReNbSSe")

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
        print("ScDatasetTest is tearing down...\n")
        
if __name__ == "__main__":
    unittest.main()
    