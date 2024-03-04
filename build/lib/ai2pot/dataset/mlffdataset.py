from dpdata import LabeledSystem
from torch.utils.data import Dataset
from matersdk.fromcc import nblist


class MlffDataset(Dataset):
    def __init__(
        self,
        labeled_system: LabeledSystem):
        pass