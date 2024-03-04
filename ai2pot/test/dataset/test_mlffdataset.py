import unittest
from dpdata import LabeledSystem
from ai2pot.dataset import MlffDataset


class MlffDatasetTest(unittest.TestCase):
    def test_dataset_init(self):
        outcar_path: str = "/data/home/liuhanyu/hyliu/code/AI2Pot/test_data/ReNbSSe/OUTCAR"
        labeled_system: LabeledSystem = LabeledSystem(outcar_path)
        mlff_dataset: MlffDataset = MlffDataset(
            labeled_system=labeled_system,
            rcut=3.2,
            umax_num_neigh_atoms=20)
        mlff_dataset.info()
        print(mlff_dataset[0])

        


if __name__ == "__main__":
    unittest.main()