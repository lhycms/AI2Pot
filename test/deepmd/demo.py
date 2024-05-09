from typing import List
from dpdata import LabeledSystem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 步骤一. 导入模型
from ai2pot.dataset.mlffdataset import MlffDataset
from ai2pot.deepmd.se_r import DpSeR

# 步骤二. 自定义参数
ntypes: int = 4
rcut: float = 3.2
rcut_smooth: float= 3.0
umax_num_neighs: int = 40   # for envMatrixOp (equal)
umax_num_neighs_tensor: torch.Tensor = torch.tensor([10, 10, 10, 10], dtype=torch.int32)    # for envMatrixOp (equal)
embed_sizes_list: List[int] = [2, 4, 8]
fit_sizes_list: List[int] = [16, 16, 16]
dp_se_r: nn.Module = DpSeR(ntypes=ntypes,
                                rcut=rcut,
                                rcut_smooth=rcut_smooth,
                                umax_num_neighs=umax_num_neighs_tensor,
                                embed_sizes_list=embed_sizes_list,
                                fit_sizes_list=fit_sizes_list,
                                M2=4,
                                energy_shift_tensor=False)
dp_se_r.to(torch.float64)

# 步骤三. 导入数据
outcar_path: str = "/data/home/liuhanyu/hyliu/code/AI2Pot/test_data/ReNbSSe/OUTCAR"
mlff_dataset: MlffDataset = MlffDataset(labeled_system=LabeledSystem(outcar_path),
                                                rcut=rcut,
                                                umax_num_neigh_atoms=umax_num_neighs)
mlff_dataloader: DataLoader = DataLoader(mlff_dataset,
                                         batch_size=5,
                                         shuffle=False)

# 步骤四. 预测能量、受力、位力张量
for ii, batch_data in enumerate(mlff_dataloader):
    ei, fi, v = dp_se_r(batch_data[1],
                                batch_data[2],
                                batch_data[3],
                                batch_data[4].requires_grad_(True),
                                batch_data[5],
                                batch_data[6])
    
    
    print("\t{0}. In Batch#{1}, descrip.size() = ".format(ii+1, ii), 
            ei.size(), 
            fi.size(),
            v.size())