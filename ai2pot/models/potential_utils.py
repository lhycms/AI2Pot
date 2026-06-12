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


from typing import List, Optional, Tuple

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



class Potential4ExtxyzBase(object):
    def __init__(
            self,
            checkpoint_path: str,
            testset_path: str,
            map_location: str = "cpu",
            torch_float_dtype: torch._C.dtype = torch.float32,
            pbc_xyz: List[bool] = [True, True, True]):
        self.checkpoint_path: str = checkpoint_path
        self.testset_path: str = testset_path
        self.lit_module: Optional[L.LightningModule] = None
        self.fit_virial: Optional[bool] = None

        # model and data
        self.model: Optional[torch.Module] = None
        extxyz_datamodule: Optional[ExtxyzDataModule] = None
        self.test_dataloader: Optional[DataLoader] = None

    
    @torch.no_grad()
    def calculate_parity(self) -> Tuple[np.ndarray]:
        e_dft_list: List[torch.Tensor] = []
        f_dft_list: List[torch.Tensor] = []
        v_dft_list: List[torch.Tensor] = []
        e_ml_list: List[torch.Tensor] = []
        f_ml_list: List[torch.Tensor] = []
        v_ml_list: List[torch.Tensor] = []
       
        for batch_idx, batch_data in enumerate(self.test_dataloader):
            if (self.fit_virial):
                binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                    brcs_tensor, btypes_tensor, bnghost_tensor, \
                    betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = \
                    [t.to(self.lit_module.device) for t in batch_data]

                betot_ml_tensor, bforce_ml_tensor, bvirial_ml_tensor= self.model.predict_efv(binum_tensor,
                                                                                             bilist_tensor,
                                                                                             bnumneigh_tensor,
                                                                                             bfirstneigh_tensor,
                                                                                             brcs_tensor,
                                                                                             btypes_tensor,
                                                                                             bnghost_tensor)
                mask_for_force: torch.Tensor = torch.arange(bilist_tensor.size(1),
                                                            device=self.lit_module.device,
                                                            dtype=self.lit_module.dtype)[None, :] < binum_tensor[:, None]
                e_ml_list.append(betot_ml_tensor / binum_tensor)
                e_dft_list.append(betot_dft_tensor / binum_tensor)
                f_ml_list.append(bforce_ml_tensor[mask_for_force])
                f_dft_list.append(bforce_dft_tensor[mask_for_force])
                v_ml_list.append(bvirial_ml_tensor / binum_tensor)
                v_dft_list.append(bvirial_dft_tensor / binum_tensor)
            else:
                binum_tensor, bilist_tensor, bnumneigh_tensor, bfirstneigh_tensor, \
                    brcs_tensor, btypes_tensor, bnghost_tensor, \
                    betot_dft_tensor, bforce_dft_tensor = \
                    [t.to(self.lit_module.device) for t in batch_data]
                
                betot_ml_tensor, bforce_ml_tensor = self.model.predict_ef(binum_tensor,
                                                                 bilist_tensor,
                                                                 bnumneigh_tensor,
                                                                 bfirstneigh_tensor,
                                                                 brcs_tensor,
                                                                 btypes_tensor,
                                                                 bnghost_tensor)
                mask_for_force: torch.Tensor = torch.arange(bilist_tensor.size(1),
                                                            device=self.lit_module.device,
                                                            dtype=self.lit_module.dtype)[None, :] < binum_tensor[:, None]
                
                e_ml_list.append(betot_ml_tensor / binum_tensor)
                e_dft_list.append(betot_dft_tensor / binum_tensor)
                f_ml_list.append(bforce_ml_tensor[mask_for_force])
                f_dft_list.append(bforce_dft_tensor[mask_for_force])


        e_dft_array: np.ndarray = torch.cat(e_dft_list, dim=0).detach().cpu().numpy()
        f_dft_array: np.ndarray = torch.cat(f_dft_list, dim=0).detach().cpu().numpy()
        e_ml_array: np.ndarray = torch.cat(e_ml_list, dim=0).detach().cpu().numpy()
        f_ml_array: np.ndarray = torch.cat(f_ml_list, dim=0).detach().cpu().numpy()
        if (self.fit_virial):
            v_dft_array: np.ndarray = torch.cat(v_dft_list, dim=0).detach().cpu().numpy()
            v_ml_array: np.ndarray = torch.cat(v_ml_list, dim=0).detach().cpu().numpy()
            return e_dft_array, f_dft_array, v_dft_array, e_ml_array, f_ml_array, v_ml_array

        return e_dft_array, f_dft_array, e_ml_array, f_ml_array


    def calculate_rmse(self) -> Tuple[float]:
        if (self.fit_virial):
            e_dft_array, f_dft_array, v_dft_array, e_ml_array, f_ml_array, v_ml_array = self.calculate_parity()
        else:
            e_dft_array, f_dft_array, e_ml_array, f_ml_array = self.calculate_parity()
        e_dft_array: np.ndarray
        f_dft_array: np.ndarray
        v_dft_array: np.ndarray
        e_ml_array: np.ndarray
        f_ml_array: np.ndarray
        v_dft_array: np.ndarray

        # rmse
        e_mse = np.mean(np.power(e_ml_array-e_dft_array, 2))
        e_rmse = np.sqrt(e_mse)
        f_mse = np.mean(np.power(f_ml_array-f_dft_array, 2))
        f_rmse = np.sqrt(f_mse)
        if (self.fit_virial):
            v_mse = np.mean(np.power(v_ml_array-v_dft_array, 2))
            v_rmse = np.sqrt(v_mse)
            return e_rmse.item(), f_rmse.item(), v_rmse.item()

        return e_rmse.item(), f_rmse.item()



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
