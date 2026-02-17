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
import torch
import torch.nn as nn
import lightning as L

from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.mtp.nn_mtp import NNMtp
from ai2pot.models.nep.nep import Nep


class LitPotentialBase(L.LightningModule):
    def __init__(
            self,
            type_map: List[int],
            energy_shifts: Optional[List[float]] = None,
            umax_num_neigh_atoms: int = 200,
            fit_virial: bool = False,
            zbl_rmax: float = 0.0,
            zbl_rmin: float = 0.0,
            zbl_cks_list: Optional[List[float]] = None,
            zbl_dks_list: Optional[List[float]] = None,
            lr_start: float = 1e-3,
            lr_end: float = 1e-5,
            e_wgt_start: float = 1.0,
            e_wgt_end: float = 1.0,
            f_wgt_start: float = 1.0,
            f_wgt_end: float = 1.0,
            v_wgt_start: float = 0.1,
            v_wgt_end: float = 0.1,
            lr_decay_step: int = 5000):
        super(LitPotentialBase, self).__init__()

        self.type_map: List[int] = type_map
        self.energy_shifts: List[int] = energy_shifts
        self.umax_num_neigh_atoms: int = umax_num_neigh_atoms
        self.fit_virial: bool = fit_virial
        self.zbl_rmax: float = zbl_rmax
        self.zbl_rmin: float = zbl_rmin
        self.zbl_cks_list: List[float] = zbl_cks_list
        self.zbl_dks_list: List[float] = zbl_dks_list

        self.model: nn.Module = None
        
        self.lr_start: float = lr_start
        self.lr_end: float = lr_end
        self.e_wgt_start: float= e_wgt_start
        self.e_wgt_end: float = e_wgt_end
        self.f_wgt_start: float = f_wgt_start
        self.f_wgt_end: float = f_wgt_end
        self.v_wgt_start: float = v_wgt_start
        self.v_wgt_end: float = v_wgt_end
        self.lr_decay_step: int = lr_decay_step

        self.save_hyperparameters()

        
    def get_efv_wgts(self):
        lr_current: float = self.optimizers().param_groups[0]["lr"]
        rate: float = lr_current / self.lr_start
        e_wgt: float = self.e_wgt_start * rate + self.e_wgt_end * (1 - rate)
        f_wgt: float = self.f_wgt_start * rate + self.f_wgt_end * (1 - rate)
        v_wgt: float = self.v_wgt_start * rate + self.v_wgt_end * (1 - rate)
        return [e_wgt, f_wgt, v_wgt]
    
        
    def training_step(self, batch_data, batch_idx):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch_data
            bmse_tensor, e_rmse_tensor, f_rmse_tensor, v_rmse_tensor = self.model(
                e_wgt,
                f_wgt,
                v_wgt,
                betot_dft_tensor,
                bforce_dft_tensor,
                bvirial_dft_tensor,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                bnghost[0].item())
            bmse_tensor: torch.Tensor
            e_rmse_tensor: torch.Tensor
            f_rmse_tensor: torch.Tensor
            v_rmse_tensor: torch.Tensor
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch_data
            bmse_tensor, e_rmse_tensor, f_rmse_tensor = self.model(
                e_wgt,
                f_wgt,
                betot_dft_tensor,
                bforce_dft_tensor,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                bnghost[0].item())
            bmse_tensor: torch.Tensor
            e_rmse_tensor: torch.Tensor
            f_rmse_tensor: torch.Tensor      
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()

        ### Log ###
        self.log("e_rmse_train", e_rmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        self.log("f_rmse_train", f_rmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        if (self.model.fit_virial):
            self.log("v_rmse_train", v_rmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        current_lr: float = self.optimizers().param_groups[0]["lr"]
        self.log("lr", 
                 current_lr,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        self.log("e_wgt",
                 e_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        self.log("f_wgt",
                 f_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True) 
        self.log("v_wgt",
                 v_wgt,
                 on_step=True,
                 on_epoch=False,
                 prog_bar=False,
                 sync_dist=True)
        ### Log ###

        return mean_bmse_tensor


    def validation_step(self, batch_data, batch_idx):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch_data
            bmse_tensor, e_rmse_tensor, f_rmse_tensor, v_rmse_tensor = self.model(
                e_wgt,
                f_wgt,
                v_wgt,
                betot_dft_tensor,
                bforce_dft_tensor,
                bvirial_dft_tensor,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                bnghost[0].item())
            bmse_tensor: torch.Tensor
            e_rmse_tensor: torch.Tensor
            f_rmse_tensor: torch.Tensor
            v_rmse_tensor: torch.Tensor
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch_data
            bmse_tensor, e_rmse_tensor, f_rmse_tensor = self.model(
                e_wgt,
                f_wgt,
                betot_dft_tensor,
                bforce_dft_tensor,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                bnghost[0].item())
            bmse_tensor: torch.Tensor
            e_rmse_tensor: torch.Tensor
            f_rmse_tensor: torch.Tensor      
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()

        ### Log ###
        self.log("e_rmse_valid", e_rmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        self.log("f_rmse_valid", f_rmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        if (self.model.fit_virial):
            self.log("v_rmse_valid", v_rmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        ### Log ###

        return mean_bmse_tensor


    def test_step(self, batch_data, batch_idx):
        e_wgt, f_wgt, v_wgt = self.get_efv_wgts()
        if (self.model.fit_virial):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor, bvirial_dft_tensor = batch_data
            bmse_tensor, e_rmse_tensor, f_rmse_tensor, v_rmse_tensor = self.model(
                e_wgt,
                f_wgt,
                v_wgt,
                betot_dft_tensor,
                bforce_dft_tensor,
                bvirial_dft_tensor,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                bnghost[0].item())
            bmse_tensor: torch.Tensor
            e_rmse_tensor: torch.Tensor
            f_rmse_tensor: torch.Tensor
            v_rmse_tensor: torch.Tensor
        else:
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch_data
            bmse_tensor, e_rmse_tensor, f_rmse_tensor = self.model(
                e_wgt,
                f_wgt,
                betot_dft_tensor,
                bforce_dft_tensor,
                binum,
                bilist,
                bnumneigh,
                bfirstneigh,
                brcs,
                btypes,
                bnghost[0].item())
            bmse_tensor: torch.Tensor
            e_rmse_tensor: torch.Tensor
            f_rmse_tensor: torch.Tensor      
        mean_bmse_tensor: torch.Tensor = bmse_tensor.mean()

        ### Log ###
        self.log("e_rmse_test", e_rmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        self.log("f_rmse_test", f_rmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        if (self.model.fit_virial):
            self.log("v_rmse_test", v_rmse_tensor,
                 on_epoch=True,
                 on_step=True,
                 prog_bar=True,
                 sync_dist=True)
        ### Log ###

        return mean_bmse_tensor
    

    def configure_optimizers(self):
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                             lr=self.lr_start)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=self.lr_decay_step,
                                                               eta_min=self.lr_end)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }


class LitLinearMtp(LitPotentialBase):
    def __init__(
            self,
            type_map: List[int],
            energy_shifts: Optional[List[float]] = None,
            umax_num_neigh_atoms: int = 200,
            fit_virial: bool = False,
            mtp_level: int = 16,
            chebyshev_size: int = 8,
            rmax: float = 5.0,
            rmin: float = 0.0,
            zbl_rmax: float = 0.0,
            zbl_rmin: float = 0.0,
            zbl_cks_list: Optional[List[float]] = None,
            zbl_dks_list: Optional[List[float]] = None,
            lr_start: float = 1e-3,
            lr_end: float = 1e-5,
            e_wgt_start: float = 1.0,
            e_wgt_end: float = 1.0,
            f_wgt_start: float = 1.0,
            f_wgt_end: float = 1.0,
            v_wgt_start: float = 0.1,
            v_wgt_end: float = 0.1,
            lr_decay_step: int = 5000):
        super().__init__(
            type_map=type_map,
            energy_shifts=energy_shifts,
            umax_num_neigh_atoms=umax_num_neigh_atoms,
            fit_virial=fit_virial,
            zbl_rmax=zbl_rmax,
            zbl_rmin=zbl_rmin,
            zbl_cks_list=zbl_cks_list,
            zbl_dks_list=zbl_dks_list,
            lr_start=lr_start,
            lr_end=lr_end,
            e_wgt_start=e_wgt_start,
            e_wgt_end=e_wgt_end,
            f_wgt_start=f_wgt_start,
            f_wgt_end=f_wgt_end,
            v_wgt_start=v_wgt_start,
            v_wgt_end=v_wgt_end,
            lr_decay_step=lr_decay_step)
        
        self.model: nn.Module = LinearMtp(
            type_map=type_map,
            energy_shifts=energy_shifts,
            umax_num_neigh_atoms=umax_num_neigh_atoms,
            fit_virial=fit_virial,
            mtp_level=mtp_level,
            chebyshev_size=chebyshev_size,
            rmax=rmax,
            rmin=rmin,
            zbl_rmax=zbl_rmax,
            zbl_rmin=zbl_rmin,
            zbl_cks_list=zbl_cks_list,
            zbl_dks_list=zbl_dks_list)
        
        self.save_hyperparameters()


class LitNep(LitPotentialBase):
    def __init__(
            self,
            type_map: List[int],
            energy_shifts: Optional[List[float]] = None,
            umax_num_neigh_atoms: int = 200,
            fit_virial: bool = False,
            n_radial_basis: int = 6,
            n_angular_basis: int = 4,
            l_max: int = 4,
            chebyshev_size: int = 8,
            num_neurons: int = 30,
            rmax_radial: float = 6.0,
            rmax_angular: float = 4.0,
            zbl_rmax: float = 0.0,
            zbl_rmin: float = 0.0,
            zbl_cks_list: Optional[List[float]] = None,
            zbl_dks_list: Optional[List[float]] = None,
            lr_start: float = 1e-3,
            lr_end: float = 1e-5,
            e_wgt_start: float = 1.0,
            e_wgt_end: float = 1.0,
            f_wgt_start: float = 1.0,
            f_wgt_end: float = 1.0,
            v_wgt_start: float = 0.1,
            v_wgt_end: float = 0.1,
            lr_decay_step: int = 5000):
        super(LitNep, self).__init__(
            type_map=type_map,
            energy_shifts=energy_shifts,
            umax_num_neigh_atoms=umax_num_neigh_atoms,
            fit_virial=fit_virial,
            zbl_rmax=zbl_rmax,
            zbl_rmin=zbl_rmin,
            zbl_cks_list=zbl_cks_list,
            zbl_dks_list=zbl_dks_list,
            lr_start=lr_start,
            lr_end=lr_end,
            e_wgt_start=e_wgt_start,
            e_wgt_end=e_wgt_end,
            f_wgt_start=f_wgt_start,
            f_wgt_end=f_wgt_end,
            v_wgt_start=v_wgt_start,
            v_wgt_end=v_wgt_end,
            lr_decay_step=lr_decay_step)

        self.model: nn.Module = Nep(
            type_map=type_map,
            energy_shifts=energy_shifts,
            umax_num_neigh_atoms=umax_num_neigh_atoms,
            fit_virial=fit_virial,
            n_radial_basis=n_radial_basis,
            n_angular_basis=n_angular_basis,
            l_max=l_max,
            chebyshev_size=chebyshev_size,
            num_neurons=num_neurons,
            rmax_radial=rmax_radial,
            rmax_angular=rmax_angular,
            zbl_rmax=zbl_rmax,
            zbl_rmin=zbl_rmin,
            zbl_cks_list=zbl_cks_list,
            zbl_dks_list=zbl_dks_list)
        
        self.save_hyperparameters()


class LitNNMtp(LitPotentialBase):
    def __init__(
            self,
            mtp_level: int,
            type_map: List[int],
            energy_shifts: Optional[List[float]] = None,
            chebyshev_size: int = 8,
            num_neurons: int = 30,
            rmax: float = 5.0,
            rmin: float = 0.0,
            umax_num_neigh_atoms: int = 200,
            fit_virial: bool = False,
            zbl_rmax: float = 2.0,
            zbl_rmin: float = 1.0,
            zbl_cks_list: Optional[List[float]] = None,
            zbl_dks_list: Optional[List[float]] = None,
            lr_start: float = 1e-2,
            lr_end: float = 1e-4,
            e_wgt_start: float = 0.02,
            e_wgt_end: float = 1.0,
            f_wgt_start: float = 1000.0,
            f_wgt_end: float = 0.1,
            v_wgt_start: float = 0.0,
            v_wgt_end: float = 0.0,
            lr_decay_step: int = 5000):
        super().__init__()
        self.save_hyperparameters()
        
        self.model: nn.Module = NNMtp(mtp_level=mtp_level,
                                      type_map=type_map,
                                      energy_shifts=energy_shifts,
                                      chebyshev_size=chebyshev_size,
                                      num_neurons=num_neurons,
                                      rmax=rmax,
                                      rmin=rmin,
                                      umax_num_neigh_atoms=umax_num_neigh_atoms,
                                      fit_virial=fit_virial,
                                      zbl_rmax=zbl_rmax,
                                      zbl_rmin=zbl_rmin,
                                      zbl_cks_list=zbl_cks_list,
                                      zbl_dks_list=zbl_dks_list)
        
        self.lr_start: float = lr_start
        self.lr_end: float = lr_end
        self.e_wgt_start: float = e_wgt_start
        self.e_wgt_end: float = e_wgt_end
        self.f_wgt_start: float = f_wgt_start
        self.f_wgt_end: float = f_wgt_end
        self.v_wgt_start: float = v_wgt_start
        self.v_wgt_end: float = v_wgt_end
        self.lr_decay_step: int = lr_decay_step
