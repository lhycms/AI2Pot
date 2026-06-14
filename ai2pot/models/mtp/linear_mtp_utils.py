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
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_utils import (
    PotentialCalculatorBase,
    Potential4ExtxyzBase)


class LinearMtp4Extxyz(Potential4ExtxyzBase):
    BATCH_SIZE_HERE: int = 800

    def __init__(
            self,
            checkpoint_path: str,
            testset_path: str,
            map_location: str = "cpu",
            pbc_xyz: List[bool] = [True, True, True]):
        super().__init__(
            checkpoint_path=checkpoint_path,
            testset_path=testset_path,
            map_location=map_location,
            pbc_xyz=pbc_xyz)
        self.checkpoint_path: str = checkpoint_path
        self.testset_path: str = testset_path
        self.lit_module: LitLinearMtp = LitLinearMtp.load_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                                          map_location=map_location)
        self.fit_virial: bool = self.lit_module.model.fit_virial

        # model and data
        self.model: LinearMtp = self.lit_module.model
        extxyz_datamodule: ExtxyzDataModule = ExtxyzDataModule(
            testset_path=self.testset_path,
            batch_size=LinearMtp4Extxyz.BATCH_SIZE_HERE,
            rcut=self.model.rmax,
            umax_num_neigh_atoms=self.model.umax_num_neigh_atoms,
            pbc_xyz=pbc_xyz,
            sort=False,
            torch_float_dtype=self.lit_module.dtype,
            has_virial=self.fit_virial)
        extxyz_datamodule.setup("test")
        self.test_dataloader: DataLoader = extxyz_datamodule.test_dataloader()


class LinearMtpCalculator(PotentialCalculatorBase):
    def __init__(self,
                 checkpoint_path: str,
                 map_location: str = "cpu",
                 pbc_xyz: List[bool] = [True, True, True],
                 **kwargs):
        super().__init__(
            checkpoint_path=checkpoint_path,
            map_location=map_location,
            pbc_xyz=pbc_xyz,
            **kwargs)
        self.checkpoint_path: str = checkpoint_path
        self.lit_module: LitLinearMtp = LitLinearMtp.load_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                                          map_location=map_location)
        self.fit_virial: bool = self.lit_module.model.fit_virial

        # model and data
        self.model: LinearMtp = self.lit_module.model
        self.mlff_input: MlffInput = MlffInput(
            type_map=self.model.type_map_tensor.detach().cpu().numpy().tolist(),
            rcut=self.model.rmax,
            umax_num_neigh_atoms=self.model.umax_num_neigh_atoms,
            pbc_xyz=pbc_xyz,
            sort=False,
            dtype=self.lit_module.dtype,
            device=torch.device(map_location))
    
    """
    def predict_atoms_coeffs_gradients(self, atoms: Atoms):
        e_sites: torch.Tensor = self.linear_mtp.predict_e_sites(*self.mlff_input.analyse_ase(atoms=atoms))    
        e_sites.sum().backward()
        coeffs_grad: np.ndarray = self.model.coeffs_tensor.grad.detach().numpy()
        linear_coeffs_grad: np.ndarray = self.model.linear_coeffs_tensor.grad.detach().numpy()
        type_bias_grad: np.ndarray = self.model.type_bias_tensor.grad.detach().numpy()
        return np.concatenate([coeffs_grad, linear_coeffs_grad, type_bias_grad])
    """


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
