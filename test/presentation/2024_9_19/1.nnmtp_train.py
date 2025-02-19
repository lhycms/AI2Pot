from dpdata import LabeledSystem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L

from ai2pot.data import ScDataset
from ai2pot.models.mtp import NNMtp
from ai2pot.models.potential_train import LitPotential


# 1. 初始化 Dataset
labeled_system = LabeledSystem("./ReNbSSe/OUTCAR")
mlff_dataset = ScDataset(labeled_system=labeled_system,
                         rcut=3.2,
                         umax_num_neigh_atoms=100,
                         pbc_xyz=[True, True, True],
                         sort=False,
                         torch_float_dtype=torch.float64)
# 2. 初始化 MTP 势函数模型
nn_mtp = NNMtp(mtp_level=16, 
               ntypes=4,
               chebyshev_size=8,
               rmax=5,
               rmin=0.5,
               umax_num_neighs=100,
               fit_sizes_list=[],
               fit_activation=nn.Tanh())
nn_mtp.to(torch.float64)
lit_nn_mtp = LitPotential(model=nn_mtp)
dataloader = DataLoader(dataset=mlff_dataset, batch_size=5, shuffle=True)
# 3. 开始训练
trainer = L.Trainer(max_epochs=3)
trainer.fit(model=lit_nn_mtp, train_dataloaders=dataloader)


