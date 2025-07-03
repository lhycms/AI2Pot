from typing import List
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from ai2pot.data.mlffdatamodule import ExtxyzDataModule
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.models.mtp.linear_mtp import LinearMtp


class LinearMtp4Extxyz(object):
    def __init__(self,
                 checkpoint_path: str,
                 testset_path: str,
                 has_virial: bool = False,
                 map_location: str = "cpu",
                 torch_float_dtype: torch._C.dtype = torch.float32):
        self.checkpoint_path: str = checkpoint_path
        self.testset_path: str = testset_path
        self.lit_linear_mtp: LitLinearMtp = LitLinearMtp.load_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                                              map_location=map_location)
        self.has_virial: bool = has_virial

        # model and data
        self.linear_mtp: LinearMtp = self.lit_linear_mtp.model.to(torch_float_dtype)
        self.test_dataloader: DataLoader = ExtxyzDataModule(testset_path=self.testset_path,
                                                            batch_size=1,
                                                            rcut=self.linear_mtp.rmax,
                                                            umax_num_neigh_atoms=self.linear_mtp.umax_num_neighs,
                                                            pbc_xyz=[True, True, True],
                                                            sort=False,
                                                            torch_float_dtype=torch_float_dtype,
                                                            has_virial=self.has_virial).test_dataloader()
    
    def calculate_ef_diagonal(self):
        e_dft_list: List[torch.Tensor] = []
        f_dft_list: List[torch.Tensor] = []
        e_ml_list: List[torch.Tensor] = []
        f_ml_list: List[torch.Tensor] = []
        for batch_idx, batch_data in enumerate(self.test_dataloader):
            # 1. dft
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
        ax1.set_xlabel("DFT energy (eV)",
                       fontsize=16,
                       fontweight="bold")
        ax1.set_ylabel("LinearMtp energy (eV)",
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
                    color="brown",
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
            plt.savefig("./ef_diagonal.png")
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
