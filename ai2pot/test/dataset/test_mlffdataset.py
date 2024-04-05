import unittest
from dpdata import LabeledSystem
from torch.utils.data import DataLoader

from ai2pot.dataset import MlffDataset


class MlffDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        print("MlffDatasetTest is setting up...\n")
        self.outcar_path: str = "/data/home/liuhanyu/hyliu/code/AI2Pot/test_data/ReNbSSe/OUTCAR"
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
        for data in self.mlff_dataloader:
            print(data[0])  # inum

    def tearDown(self):
        print("MlffDatasetTest is tearing down...\n")
        
if __name__ == "__main__":
    unittest.main()