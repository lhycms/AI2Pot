from dpdata import LabeledSystem
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import lightning as L

from ai2pot.data import ScDataset
from ai2pot.mtp import NNMtp, LitNNMtp


labeled_system = LabeledSystem("/data/home/liuhanyu/hyliu/code/rubbish/test/AI2Pot/test/test_data/OUTCARs/OUTCAR")
mlff_dataset = ScDataset(labeled_system=labeled_system,
                           rcut=3.2,
                           umax_num_neigh_atoms=100,
                           pbc_xyz=[True, True, True],
                           sort=False,
                           is_cart_coord=True)
nn_mtp = NNMtp(mtp_level=10, 
               ntypes=4,
               chebyshev_size=8,
               rmax=3.2,
               rmin=0.5,
               umax_num_neighs=100,
               fit_sizes_list=[],
               fit_activation=nn.Tanh())
nn_mtp.to(torch.float64)
lit_nn_mtp = LitNNMtp(model=nn_mtp)
dataloader = DataLoader(dataset=mlff_dataset, batch_size=5, shuffle=True)

trainer = L.Trainer(max_epochs=3)
trainer.fit(model=lit_nn_mtp, train_dataloaders=dataloader)
