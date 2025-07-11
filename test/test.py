import os
os.environ["AI2POT_PATH"] = "/data/home/liuhanyu/mycode/AI2Pot/"
os.environ["AI2POT_TUTORIALS_PATH"] = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials"
os.environ["AI2POT_TEST_DATA_PATH"] = os.path.join(os.environ.get("AI2POT_TUTORIALS_PATH"),
                                                   "data",
                                                   "XYZ")
extxyz_path: str = os.path.join(os.environ.get("AI2POT_TEST_DATA_PATH"),
                                               "11_NEP_potential_PbTe",
                                               "train_m.xyz")
extxyz_path: str = "/data/home/liuhanyu/mycode/AI2Pot-use/data/refitted_GST-GAP-22_PBE_m1.xyz"

from ase.io import read as ase_read
from typing import List
import torch
from torch.utils.data import DataLoader
from ai2pot.data.mlffdataset import ExtxyzDataset
from ai2pot.data.mlffdatamodule import ExtxyzDataModule


extxyz_dataset: ExtxyzDataset = ExtxyzDataset(filename=extxyz_path,
                                              rcut=10.0,
                                              umax_num_neigh_atoms=200,
                                              pbc_xyz=[True, True, True],
                                              sort=True,
                                              torch_float_dtype=torch.float32,
                                              has_virial=False)
dataloader: DataLoader = DataLoader(dataset=extxyz_dataset,
                                    batch_size=1,
                                    shuffle=False)

for batch_idx, batch_data in enumerate(dataloader):
    print("0. Batch_idx", batch_idx)
    print("1. binum_tensor.shape = ", batch_data[0].shape)
    print("2. bilist_tensor.shape = ", batch_data[1].shape)
    print("3. bnumneigh_tensor.shape = ", batch_data[2].shape)
    print("4. bfirstneigh_tensor.shape = ", batch_data[3].shape)
    print("5. brcs_tensor.shape = ", batch_data[4].shape)
    print("6. btypes_tensor.shape = ", batch_data[5].shape)
    print("7. bnghost_tensor.shape = ", batch_data[6].shape)
    print("8. betot_tensor.shape = ", batch_data[7].shape)
    print("9. bforce_tensor.shape = ", batch_data[8].shape)
    print()