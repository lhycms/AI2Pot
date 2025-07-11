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

from ai2pot.core.nblist import Nblist
from ai2pot.utils.usepot import MlffInput


atoms_list = ase_read(extxyz_path, index=":")
atoms = atoms_list[211]
print(atoms)
mlff_input = MlffInput(type_map=[32],
                       rcut=8.0,
                       umax_num_neighs=200)
result = mlff_input.analyse_ase(atoms=atoms)

print(result)