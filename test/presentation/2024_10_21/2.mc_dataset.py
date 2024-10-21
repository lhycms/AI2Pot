import torch
from torch.utils.data import DataLoader
from dpdata import LabeledSystem, MultiSystems
from ai2pot.data import McDataset


ls1: LabeledSystem = LabeledSystem(file_name="./1/OUTCAR",
                                  fmt="vasp/outcar")
ls2: LabeledSystem = LabeledSystem(file_name="./2/OUTCAR",
                                   fmt="vasp/outcar")
ms: MultiSystems = MultiSystems()
ms.append(ls1)
ms.append(ls2)
mlff_dataset: McDataset = McDataset(multi_systems=ms,
                                    rcut=3.5,
                                    umax_num_neigh_atoms=100,
                                    pbc_xyz=[True, True, True],
                                    sort=True,
                                    torch_float_dtype=torch.float64)

data_loader: DataLoader = DataLoader(dataset=mlff_dataset,
                                     batch_size=5,
                                     shuffle=True)
for ii, data in enumerate(data_loader):
    print("Batch #{0}, types: ".format(ii), data[5].size)

