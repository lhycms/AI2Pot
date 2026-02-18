# Copyright (C) 2025 Hanyu Liu
#
# This file is part of AI2Pot.
#
# AI2Pot is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI2Pot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.


from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt

from ase.calculators.calculator import (Calculator, 
                                        all_changes)
from ase import Atoms
import torch
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ai2pot.data.mlffdatamodule import ExtxyzDataModule
from ai2pot.models.potential_train import LitNep
from ai2pot.models.nep.nep import Nep
from ai2pot.models.potential_utils import Potential4ExtxyzBase



class Nep4Extxyz(Potential4ExtxyzBase):
    def __init__(
            self,
            checkpoint_path: str,
            testset_path: str,
            map_location: str = "cpu",
            torch_float_dtype: torch._C.dtype = torch.float32):
        super().__init__(
            checkpoint_path=checkpoint_path,
            testset_path=testset_path,
            map_location=map_location,
            torch_float_dtype=torch_float_dtype)
        self.checkpoint_path: str = checkpoint_path
        self.testset_path: str = testset_path
        self.lit_module: LitNep = LitNep.load_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                              map_location=map_location)
        self.has_virial: bool = self.lit_module.model.fit_virial

        # model and data
        self.model: Nep = self.lit_module.model.to(torch_float_dtype)
        extxyz_datamodule: ExtxyzDataModule = ExtxyzDataModule(
            testset_path=self.testset_path,
            batch_size=1,
            rcut=self.model.rmax,
            umax_num_neigh_atoms=self.model.umax_num_neigh_atoms,
            pbc_xyz=[True, True, True],
            sort=False,
            torch_float_dtype=torch_float_dtype,
            has_virial=self.has_virial)
        extxyz_datamodule.setup("test")
        self.test_dataloader: DataLoader = extxyz_datamodule.test_dataloader()
        