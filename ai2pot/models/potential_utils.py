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
    def __init__(
            self,
            checkpoint_path: str,
            testset_path: str,
            map_location: str = "cpu",
            torch_float_dtype: torch._C.dtype = torch.float32):
        self.checkpoint_path: str = checkpoint_path
        self.testset_path: str = testset_path
        self.lit_module: Optional[L.LightningModule] = None
        self.has_virial: Optional[bool] = None

        # model and data
        self.model: Optional[torch.Module] = None
        extxyz_datamodule: Optional[ExtxyzDataModule] = None
        self.test_dataloader: Optional[DataLoader] = None

    
    def calculate_ef_diagonal(self):
        e_dft_list: List[torch.Tensor] = []
        f_dft_list: List[torch.Tensor] = []
        e_ml_list: List[torch.Tensor] = []
        f_ml_list: List[torch.Tensor] = []
        
        if (self.has_virial):
            pass
        else:
            for batch_idx, batch_data in enumerate(self.test_dataloader):
                input_tensors: List[torch.Tensor] = batch_data
                binum: torch.Tensor = input_tensors[0].to(self.lit_module.device)
                bilist: torch.Tensor = input_tensors[1].to(self.lit_module.device)
                bnumneigh: torch.Tensor = input_tensors[2].to(self.lit_module.device)
                bfirstneigh: torch.Tensor = input_tensors[3].to(self.lit_module.device)
                brcs: torch.Tensor = input_tensors[4].to(self.lit_module.device)
                btypes: torch.Tensor = input_tensors[5].to(self.lit_module.device)
                bnghost: torch.Tensor = input_tensors[6].to(self.lit_module.device)
                e_dft_tensor: torch.Tensor = input_tensors[7].to(self.lit_module.device)
                f_dft_tensor: torch.Tensor = input_tensors[8].to(self.lit_module.device)

                e_ml_tensor, f_ml_tensor = self.model.predict_ef(binum,
                                                                 bilist,
                                                                 bnumneigh,
                                                                 bfirstneigh,
                                                                 brcs,
                                                                 btypes,
                                                                 bnghost)
                e_ml_tensor: torch.Tensor
                f_ml_tensor: torch.Tensor
                
                # 3. append
                e_dft_list.append(e_dft_tensor / binum)
                f_dft_list.append(f_dft_tensor.flatten())
                e_ml_list.append(e_ml_tensor / binum)
                f_ml_list.append(f_ml_tensor.flatten())

        e_ml_tensor = torch.cat(e_ml_list, dim=0)
        f_ml_tensor = torch.cat(f_ml_list, dim=0)
        e_dft_tensor = torch.cat(e_dft_list, dim=0)
        f_dft_tensor = torch.cat(f_dft_list, dim=0)

        return e_ml_tensor, e_dft_tensor, f_ml_tensor, f_dft_tensor


    def calculate_ef_rmse(self):
        e_ml_tensor, e_dft_tensor, f_ml_tensor, f_dft_tensor = self.calculate_ef_diagonal()
        e_ml_tensor: torch.Tensor
        e_dft_tensor: torch.Tensor
        f_ml_tensor: torch.Tensor
        f_dft_tensor: torch.Tensor

        # rmse
        e_mse = torch.mean(torch.pow(e_ml_tensor-e_dft_tensor, 2))
        e_rmse = torch.sqrt(e_mse)
        f_mse = torch.mean(torch.pow(f_ml_tensor-f_dft_tensor, 2))
        f_rmse = torch.sqrt(f_mse)

        return e_rmse.item(), f_rmse.item()


    def plot_ef_diagonal(self, save: bool = True):
        e_ml_tensor, e_dft_tensor, f_ml_tensor, f_dft_tensor = self.calculate_ef_diagonal()
        e_ml_array: np.ndarray = e_ml_tensor.detach().cpu().numpy()
        e_dft_array: np.ndarray = e_dft_tensor.detach().cpu().numpy()
        f_ml_array: np.ndarray = f_ml_tensor.detach().cpu().numpy().flatten()
        f_dft_array: np.ndarray = f_dft_tensor.detach().cpu().numpy().flatten()
        
        ### Plot
        #plt.rcParams["font.family"] = "Times New Roman"
        #plt.rcParams['mathtext.fontset'] = 'custom'
        #plt.rcParams['mathtext.rm'] = 'Times New Roman'
        #plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        #plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
        # 0.2. 刻度线朝内
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        plt.subplots_adjust(wspace=0.3)
        ### 1. Plot Energy
        e_max: float = max(e_ml_array.max(), e_dft_array.max())
        e_min: float = min(e_ml_array.min(), e_dft_array.min())
        e_range: float = e_max - e_min
        margin: float = 0.1
        e_max = e_max + margin * e_range
        e_min = e_min - margin * e_range

        ax1.plot([e_min, e_max], [e_min, e_max],
                 color="lightsalmon",
                 zorder=0)
        ax1.scatter(e_dft_array, e_ml_array,
                    color="steelblue",
                    alpha=0.6,
                    s=15)
        ax1.set_xlim(e_min, e_max)
        ax1.set_ylim(e_min, e_max)
        ax1.set_xlabel("DFT energy (eV/atom)",
                       fontsize=16,
                       fontweight="bold")
        ax1.set_ylabel("MLP energy (eV/atom)",
                       fontsize=16,
                       fontweight="bold")
        
        # 2. Plot Force
        f_max: float = max(f_ml_array.max(), f_dft_array.max())
        f_min: float = min(f_ml_array.min(), f_dft_array.min())
        f_range: float = f_max - f_min
        margin: float = 0.1
        f_max = f_max + margin * f_range
        f_min = f_min - margin * f_range
        ax2.plot([f_min, f_max], [f_min, f_max],
                 color="lightsalmon",
                 zorder=0)
        ax2.scatter(f_dft_array, f_ml_array,
                    color="darkseagreen",
                    alpha=0.6,
                    s=15)
        ax2.set_xlim(f_min, f_max)
        ax2.set_ylim(f_min, f_max)
        ax2.set_xlabel("DFT force (eV/Å)",
                       fontsize=16,
                       fontweight="bold")
        ax2.set_ylabel("MLP force (eV/Å)",
                       fontsize=16,
                       fontweight="bold")

        # 3. Retouch the ticks of x-axis/y-axis
        for ax in [ax1, ax2]:
            ax.tick_params(
                axis='both',       # 同时设置 x 和 y 轴
                which='major',     # 只设置主刻度
                labelsize=14,      # 刻度标签字体大小
                width=2,           # 刻度线的粗细
                length=5,          # 刻度线的长短
                direction='in',    # 刻度线朝内
            )
            # 设置刻度标签的字体粗细
            for tick in ax.get_xticklabels():
                tick.set_fontweight("bold")
            for tick in ax.get_yticklabels():
                tick.set_fontweight("bold")
        
        # 4. 设置坐标轴的粗细
        ax1.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        ax1.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        ax1.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        ax1.spines['top'].set_linewidth(1.5);###设置右边坐标轴的粗细
        ax2.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        ax2.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        ax2.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        ax2.spines['top'].set_linewidth(1.5);###设置右边坐标轴的粗细

        # 5. Output RMSE
        e_rmse, f_rmse = self.calculate_ef_rmse()
        ax1.text(0.05, 0.9,
                 s="RMSE={0:.3f} meV/atom".format(e_rmse * 1E3),
                 transform=ax1.transAxes, ha='left', va='top',
                 fontsize=13,
                 fontweight="bold")
        ax2.text(0.05, 0.9,
                 s="RMSE={0:.3f} meV/Å".format(f_rmse * 1E3),
                 transform=ax2.transAxes, ha='left', va='top',
                 fontsize=13,
                 fontweight="bold")

        if (save):
            plt.savefig("./ef_diagonal.png",
                        dpi=500, 
                        bbox_inches="tight")
        else:
            plt.show()
