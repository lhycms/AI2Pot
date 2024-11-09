import torch
from torch.utils.data import DataLoader
from dpdata import LabeledSystem, MultiSystems
from ai2pot.data import McDataset


# 1. 初始化 LabeledSystems
ls1: LabeledSystem = LabeledSystem(file_name="./ReNbSSe/OUTCAR",
                                   fmt="vasp/outcar")
ls2: LabeledSystem = LabeledSystem(file_name="./ReNbSTe/OUTCAR",
                                   fmt="vasp/outcar")
ms: MultiSystems = MultiSystems()
ms.append(ls1)
ms.append(ls2)
# 2. 初始化 Dataset
mlff_dataset: McDataset = McDataset(multi_systems=ms,
                                    rcut=3.5,
                                    umax_num_neigh_atoms=100,
                                    pbc_xyz=[True, True, True],
                                    sort=True,
                                    torch_float_dtype=torch.float64)
# 3. 初始化 DataLoader
data_loader: DataLoader = DataLoader(dataset=mlff_dataset,
                                     batch_size=32,
                                     shuffle=True)
# 4. 输出 DataLoader 信息
for ii, data in enumerate(data_loader):
    print("Batch #{0}, types: ".format(ii), data[5].size())


