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

import matplotlib.pyplot as plt
from ase.calculators.calculator import (Calculator, 
                                        all_changes)
from ase import Atoms
import numpy as np
import torch
from torch.utils.data import DataLoader
import lightning as L

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ai2pot.data.mlffdatamodule import ExtxyzDataModule
from ai2pot.utils.usepot import MlffInput


class PotentialCalculatorBase(Calculator):
    implemented_properties = ['energy',
                              'forces',
                              "descriptors",
                              'e_sites',
                              "coeffs_gradients"]
    
    def __init__(
            self,
            checkpoint_path: str,
            map_location: str = "cpu",
            torch_float_dtype: torch._C.dtype = torch.float32,
            **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_path: str = checkpoint_path
        self.torch_float_dtype: torch._C.dtype = torch_float_dtype
        self.lit_moudle: Optional[L.LightningModule] = None
        self.has_virial: Optional[bool] = None

        # model and data
        self.model: Optional[torch.Module] = None
        self.mlff_input: Optional[MlffInput] = None
    

    def calculate(
            self,
            atoms: Atoms,
            properties: Optional[List[str]] = None,
            system_changes: List[str] = all_changes):
        super().calculate(atoms, properties, system_changes)
        if ("energy" in properties) or ("forces" in properties):
            if (self.has_virial == False):
                e, f = self.predict_ef(atoms = atoms)
                self.results["energy"] = e
                self.results["forces"] = f
            else:
                pass
        
        if ("e_sites" in properties):
            self.results["e_sites"] = self.predict_e_sites(atoms=atoms)

        if ("descriptors" in properties):
            self.results["descriptors"] = self.predict_descriptors(atoms=atoms)

        #if ("coeffs_gradients" in properties):
        #    self.results["coeffs_gradients"] = self.predict_coeffs_gradients(atoms=atoms)
    

    def predict_ef(self, atoms: Atoms):
        e, f = self.model.predict_ef(*self.mlff_input.analyse_ase(atoms=atoms))
        e: float = e.item()
        f: np.ndarray = f.squeeze(dim=0).detach().numpy()
        return e, f
    

    def predict_e_sites(self, atoms: Atoms):
        e_sites_tensor: torch.Tensor = self.model.predict_e_sites(*self.mlff_input.analyse_ase(atoms=atoms))
        return e_sites_tensor.squeeze(dim=0).detach().numpy()
    

    def predict_descriptors(self, atoms: Atoms):
        descriptors: np.ndarray = self.model.predict_descriptors(*self.mlff_input.analyse_ase(atoms=atoms))
        return descriptors.squeeze(dim=0).detach().numpy()
    

    def predict_coeffs_gradients(self, atoms: Atoms):
        pass


class Potential4ExtxyzBase(object):
    def __init__(self,
                 checkpoint_path: str,
                 testset_path: str,
                 map_location: str = "cuda",
                 torch_float_dtype: torch._C.dtype = torch.float32):
        self.checkpoint_path: str = checkpoint_path
        self.testset_path: str = testset_path
        self.lit_module: L.LightningModule = None
        self.has_virial: bool = self.lit_linear_mtp.model.fit_virial

        # model and data
        self.model: torch.Module = None
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

    
    def calculate_ef_diagonal(self):
        pass


    def plot_ef_diagonal(self, save: bool = True):
        pass


    def calculate_ef_rmse(self):
        pass