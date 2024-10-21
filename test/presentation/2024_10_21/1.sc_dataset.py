import torch
from torch.utils.data import DataLoader
from dpdata import LabeledSystem
from ai2pot.data import ScDataset


ls: LabeledSystem = LabeledSystem(file_name="/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/test_data/OUTCARs/ReNbSSe/OUTCAR",
                                  fmt="vasp/outcar")
mlff_dataset: ScDataset = ScDataset(labeled_system=ls,
                                    rcut=3.5,
                                    umax_num_neigh_atoms=100,
                                    pbc_xyz=[True, True, True],
                                    sort=True,
                                    torch_float_dtype=torch.float64)

data_loader: DataLoader = DataLoader(dataset=mlff_dataset,
                                     batch_size=5,
                                     shuffle=True)
for ii, data in enumerate(data_loader):
    print("Batch #{0}, types: ".format(ii), data[5].size())
    