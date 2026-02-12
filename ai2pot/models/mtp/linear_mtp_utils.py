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
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.utils.usepot import MlffInput


class LinearMtpCalculator(Calculator):
    implemented_properties = ['energy', 
                              'e_sites',
                              'forces',
                              "descriptors",
                              "coeffs_gradients"]

    def __init__(self,
                 checkpoint_path: str,
                 map_location: str = "cpu",
                 torch_float_dtype: torch._C.dtype = torch.float32,
                 **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_path: str = checkpoint_path
        self.lit_linear_mtp: LitLinearMtp = LitLinearMtp.load_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                                              map_location=map_location)
        self.has_virial: bool = self.lit_linear_mtp.model.fit_virial
        self.torch_float_dtype: torch._C.dtype = torch_float_dtype

        # model and data
        self.linear_mtp: LinearMtp = self.lit_linear_mtp.model.to(torch_float_dtype)
        self.mlff_input: MlffInput = MlffInput(type_map=self.linear_mtp.type_map_tensor.numpy().tolist(),
                                               rcut=self.linear_mtp.rmax,
                                               umax_num_neigh_atoms=self.linear_mtp.umax_num_neigh_atoms,
                                               pbc_xyz=[True, True, True],
                                               sort=False,
                                               dtype=self.torch_float_dtype,
                                               device=torch.device(map_location))


    def calculate(self,
                  atoms: Atoms,
                  properties: Optional[List[str]] = None,
                  system_changes: List[str] = all_changes):
        super().calculate(atoms, properties, system_changes)
        if ("energy" in properties) or ("forces" in properties):
            if (self.has_virial == False):
                e, f = self.predict_atoms_ef(atoms = atoms)
                self.results["energy"] = e
                self.results["forces"] = f
            else:
                pass
        
        if ("e_sites" in properties):
            self.results["e_sites"] = self.predict_atoms_e_sites(atoms=atoms)

        if ("descriptors" in properties):
            self.results["descriptors"] = self.predict_atoms_descriptors(atoms=atoms)

        if ("coeffs_gradients" in properties):
            self.results["coeffs_gradients"] = self.predict_atoms_coeffs_gradients(atoms=atoms)

    
    def predict_atoms_ef(self, atoms: Atoms):
        e, f = self.linear_mtp.predict_ef(*self.mlff_input.analyse_ase(atoms=atoms))
        e: float = e.item()
        f: np.ndarray = f.squeeze(dim=0).detach().numpy()
        return e, f
    

    def predict_atoms_e_sites(self, atoms: Atoms):
        e_sites_tensor: torch.Tensor = self.linear_mtp.predict_e_sites(*self.mlff_input.analyse_ase(atoms=atoms))
        return e_sites_tensor.squeeze(dim=0).detach().numpy()


    def predict_atoms_descriptors(self, atoms: Atoms):
        descriptors: np.ndarray = self.linear_mtp.predict_descriptors(*self.mlff_input.analyse_ase(atoms=atoms)).squeeze(dim=0).detach().numpy()
        return descriptors
    

    def predict_atoms_coeffs_gradients(self, atoms: Atoms):
        e_sites: torch.Tensor = self.linear_mtp.predict_e_sites(*self.mlff_input.analyse_ase(atoms=atoms))    
        e_sites.sum().backward()
        coeffs_grad: np.ndarray = self.linear_mtp.coeffs_tensor.grad.detach().numpy()
        linear_coeffs_grad: np.ndarray = self.linear_mtp.linear_coeffs_tensor.grad.detach().numpy()
        type_bias_grad: np.ndarray = self.linear_mtp.type_bias_tensor.grad.detach().numpy()
        return np.concatenate([coeffs_grad, linear_coeffs_grad, type_bias_grad])

    

class LinearMtpActiveDR(object):
    "Dimension Reduction"
    def __init__(self,
                 checkpoint_path: str,
                 extxyz_path: str,
                 map_location: str = "cpu",
                 torch_float_dtype: torch._C.dtype = torch.float32):
        self.checkpoint_path: str = checkpoint_path
        self.extxyz_path: str = extxyz_path
        self.map_location: str = map_location
        self.torch_float_dtype: torch._C.dtype = torch_float_dtype
        self.lit_linear_mtp: LitLinearMtp = LitLinearMtp.load_from_checkpoint(self.checkpoint_path,
                                                                              map_location=self.map_location)
        
        # model and data
        self.linear_mtp: LinearMtp = self.lit_linear_mtp.model.to(dtype=self.torch_float_dtype).to(device=torch.device(map_location))
        self.fit_virial: bool = self.linear_mtp.fit_virial
        self.dataloader: DataLoader = ExtxyzDataModule(trainset_path=extxyz_path,
                                                       validset_path=None,
                                                       testset_path=None,
                                                       predict_path=None,
                                                       batch_size=1,
                                                       rcut=self.linear_mtp.rmax,
                                                       umax_num_neigh_atoms=self.linear_mtp.umax_num_neigh_atoms,
                                                       pbc_xyz=[True, True, True],
                                                       sort=False,
                                                       torch_float_dtype=self.torch_float_dtype,
                                                       has_virial=self.fit_virial).train_dataloader()


    def _get_batch_descriptors(self):
        descriptors_list: List[torch.Tensor] = []
        for batch_idx, batch_data in enumerate(self.dataloader):
            input_data = batch_data[:7]
            descriptors = self.linear_mtp.predict_descriptors(*input_data)
            descriptors_list.append(descriptors)
        
        descriptors_tensor: torch.Tensor = torch.cat(descriptors_list, dim=0)
        return descriptors_tensor


    def get_pca_2d(self):
        descriptors_tensor: torch.Tensor = self._get_batch_descriptors()
        descriptors_array: np.ndarray = descriptors_tensor.detach().reshape(-1, descriptors_tensor.shape[-1]).numpy()
        
        pca: PCA = PCA(n_components=2)
        pca.fit(descriptors_array)
        descriptors_pca: np.ndarray = pca.transform(descriptors_array)
        return descriptors_pca


    def get_tsne_2d(self):
        descriptors_tensor: torch.Tensor = self._get_batch_descriptors()
        descriptors_array: np.ndarray = descriptors_tensor.detach().reshape(-1, descriptors_tensor.shape[-1]).numpy()
        tsne: TSNE = TSNE(n_components=2, random_state=42)
        descriptors_tsne = tsne.fit_transform(descriptors_array)
        return descriptors_tsne


    def get_pca_fps(self):
        pass


    def get_tsne_fps(self):
        pass

    
    def plot_pca_2d(self, save=True):
        descriptors_pca: np.ndarray = self.get_pca_2d()
        self._plot_2d(array_2d=descriptors_pca, save=save)


    def plot_tsne_2d(self, save=True):
        descriptors_pca: np.ndarray = self.get_tsne_2d()
        self._plot_2d(array_2d=descriptors_pca, save=save)


    def _plot_2d(self, 
                 array_2d: np.ndarray,
                 save: bool = True):
        margin: float = 0.1
        lims: np.ndarray = np.min(array_2d, axis=0)
        maxs: np.ndarray = np.max(array_2d, axis=0)
        pca_range: np.ndarray = maxs - lims
        lims = lims - margin * pca_range
        maxs = maxs + margin * pca_range

        # 0.0. 全局设置
        # 0.0.1. 字体设置
        #plt.rcParams["font.family"] = "Times New Roman"
        #plt.rcParams['mathtext.fontset'] = 'custom'
        #plt.rcParams['mathtext.rm'] = 'Times New Roman'
        #plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
        #plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
        # 0.0.2. 刻度线朝内
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        plt.figure(figsize=(5, 5))
        plt.scatter(array_2d[:, 0], array_2d[:, 1],
                    color="skyblue",
                    alpha=0.6)


        plt.xlabel("Component 1", fontsize=16, 
                        fontweight="bold"
                        )
        plt.ylabel("Component 2", fontsize=16, 
                fontweight="bold"
                )
        # 2.2. Retouch the ticks of x-axis/y-axis
        plt.xticks(fontsize=12, 
                fontweight="bold"
                )
        plt.yticks(fontsize=12, 
                fontweight="bold"
                )
        # 2.4. 刻度线的粗细
        plt.tick_params(
                width=2,        # 刻度线的粗细
                length=5,       # 刻度线的长短
                #labelsize=28   # 刻度线的字体大小
                )
        # 2.5. 设置坐标轴的粗细
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(1.5);###设置底部坐标轴的粗细
        ax.spines['left'].set_linewidth(1.5);####设置左边坐标轴的粗细
        ax.spines['right'].set_linewidth(1.5);###设置右边坐标轴的粗细
        ax.spines['top'].set_linewidth(1.5);###设置右边坐标轴的粗细
        plt.xlim(lims[0], maxs[0])
        plt.ylim(lims[1], maxs[1])

        if (save):
            plt.savefig("./dr_descriptor.png",
                        dpi=400, 
                        bbox_inches="tight")
        else:
            plt.show()


class LinearMtpActiveEP(object):
    '''ExtraPolation'''
    def __init__(self,
                 checkpoint_path: str,
                 extxyz_path: str,
                 map_location: str = "cpu",
                 torch_float_dtype: torch._C.dtype = torch.float32):
        self.checkpoint_path: str = checkpoint_path
        self.extxyz_path: str = extxyz_path
        self.map_location: str = map_location
        self.device: torch._C.device = torch.device(self.map_location)
        self.torch_float_dtype: torch.float32 = torch_float_dtype
        self.lit_linear_mtp: LitLinearMtp = LitLinearMtp.load_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                                              map_location=self.map_location)

        # model and data
        self.linear_mtp: LinearMtp = self.lit_linear_mtp.model
        self.has_virial: bool = self.linear_mtp.fit_virial
        self.data_loader: DataLoader = ExtxyzDataModule(trainset_path=extxyz_path,
                                                        validset_path=None,
                                                        testset_path=None,
                                                        predict_path=None,
                                                        batch_size=1,
                                                        rcut=self.linear_mtp.rmax,
                                                        umax_num_neigh_atoms=self.linear_mtp.umax_num_neigh_atoms,
                                                        pbc_xyz=[True, True, True],
                                                        sort=False,
                                                        torch_float_dtype=self.torch_float_dtype,
                                                        has_virial=self.has_virial)


    #def get_num_coeffs(self):
        #return self.linear_mtp.coeffs_tensor.sizes()[0] + self.linear_mtp.linear_coeffs_tensor.sizes()[0] + self.linear_mtp.type_bias_tensor.sizes()[0]




class LinearMtp4Extxyz(object):
    def __init__(self,
                 checkpoint_path: str,
                 testset_path: str,
                 map_location: str = "cuda",
                 torch_float_dtype: torch._C.dtype = torch.float32):
        self.checkpoint_path: str = checkpoint_path
        self.testset_path: str = testset_path
        self.lit_linear_mtp: LitLinearMtp = LitLinearMtp.load_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                                              map_location=map_location)
        self.has_virial: bool = self.lit_linear_mtp.model.fit_virial

        # model and data
        self.linear_mtp: LinearMtp = self.lit_linear_mtp.model.to(torch_float_dtype)
        extxyz_datamodule: ExtxyzDataModule = ExtxyzDataModule(testset_path=self.testset_path,
                                                               batch_size=1,
                                                               rcut=self.linear_mtp.rmax,
                                                               umax_num_neigh_atoms=self.linear_mtp.umax_num_neigh_atoms,
                                                               pbc_xyz=[True, True, True],
                                                               sort=False,
                                                               torch_float_dtype=torch_float_dtype,
                                                               has_virial=self.has_virial)
        extxyz_datamodule.setup("test")
        self.test_dataloader: DataLoader = extxyz_datamodule.test_dataloader()
    
    def calculate_ef_diagonal(self):
        e_dft_list: List[torch.Tensor] = []
        f_dft_list: List[torch.Tensor] = []
        e_ml_list: List[torch.Tensor] = []
        f_ml_list: List[torch.Tensor] = []
        for batch_idx, batch_data in enumerate(self.test_dataloader):
            # 1. dft
            binum = batch_data[0]
            e_dft = batch_data[7]
            f_dft = batch_data[8]

            # 2. ml
            input_data = batch_data[:7]
            e_ml, f_ml = self.linear_mtp.predict_ef(*input_data)
            
            # 3. append
            e_dft_list.append(e_dft / binum)
            f_dft_list.append(f_dft)
            e_ml_list.append(e_ml / binum)
            f_ml_list.append(f_ml)

        e_ml_tensor = torch.cat(e_ml_list, dim=0)
        f_ml_tensor = torch.cat(f_ml_list, dim=0)
        e_dft_tensor = torch.cat(e_dft_list, dim=0)
        f_dft_tensor = torch.cat(f_dft_list, dim=0)

        return e_ml_tensor, e_dft_tensor, f_ml_tensor, f_dft_tensor
    

    def plot_ef_diagonal(self, save: bool = True):
        e_ml_tensor, e_dft_tensor, f_ml_tensor, f_dft_tensor = self.calculate_ef_diagonal()
        e_ml_array: np.ndarray = e_ml_tensor.detach().numpy()
        e_dft_array: np.ndarray = e_dft_tensor.detach().numpy()
        f_ml_array: np.ndarray = f_ml_tensor.detach().numpy().flatten()
        f_dft_array: np.ndarray = f_dft_tensor.detach().numpy().flatten()
        
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
                    alpha=0.6)
        ax1.set_xlim(e_min, e_max)
        ax1.set_ylim(e_min, e_max)
        ax1.set_xlabel("DFT energy (eV/atom)",
                       fontsize=16,
                       fontweight="bold")
        ax1.set_ylabel("LinearMtp energy (eV/atom)",
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
                    alpha=0.6)
        ax2.set_xlim(f_min, f_max)
        ax2.set_ylim(f_min, f_max)
        ax2.set_xlabel("DFT force (eV/Å)",
                       fontsize=16,
                       fontweight="bold")
        ax2.set_ylabel("LinearMtp force (eV/Å)",
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
        ax2.text(1.25, 0.9,
                 s="RMSE={0:.3f} meV/Å".format(f_rmse * 1E3),
                 transform=ax1.transAxes, ha='left', va='top',
                 fontsize=13,
                 fontweight="bold")

        if (save):
            plt.savefig("./ef_diagonal.png",
                        dpi=500, 
                        bbox_inches="tight")
        else:
            plt.show()


    def calculate_ef_rmse(self):
        binum_list: List[torch.Tensor] = []
        e_dft_list: List[torch.Tensor] = []
        f_dft_list: List[torch.Tensor] = []
        e_ml_list: List[torch.Tensor] = []
        f_ml_list: List[torch.Tensor] = []
        for batch_idx, batch_data in enumerate(self.test_dataloader):
            # 1. dft
            binum_list.append(batch_data[0])
            e_dft = batch_data[7]
            f_dft = batch_data[8]

            # 2. ml
            input_data = batch_data[:7]
            e_ml, f_ml = self.linear_mtp.predict_ef(*input_data)
            
            # 3. append
            e_dft_list.append(e_dft)
            f_dft_list.append(f_dft)
            e_ml_list.append(e_ml)
            f_ml_list.append(f_ml)
        
        binum_tensor = torch.cat(binum_list, dim=0)
        e_dft_tensor = torch.cat(e_dft_list, dim=0)
        f_dft_tensor = torch.cat(f_dft_list, dim=0)
        e_ml_tensor = torch.cat(e_ml_list, dim=0)
        f_ml_tensor = torch.cat(f_ml_list, dim=0)

        # rmse
        e_mse = torch.mean(torch.pow(e_ml_tensor/binum_tensor-e_dft_tensor/binum_tensor, 2))
        e_rmse = torch.sqrt(e_mse)
        f_mse = torch.mean(torch.sum(torch.pow(f_ml_tensor-f_dft_tensor, 2), dim=(-2, -1)) / (3*binum_tensor))
        f_rmse = torch.sqrt(f_mse)

        return e_rmse.item(), f_rmse.item()
