from typing import Optional, List
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.optimize import minimize

import lightning as L

from ai2pot.data.mlffdataset import ExtxyzDataset
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_train import LitLinearMtp


class ParametersOptimizer(object):
    def __init__(self):
        pass


    def get_params(self):
        pass
    

    def set_params(self):
        pass


class FullParametersOptimizer(ParametersOptimizer):
    def __init__(self,
                 lit_linear_mtp: LitLinearMtp,
                 extxyz_path: str,
                 rcut: float = 5.0,
                 umax_num_neigh_atoms: int = 200,
                 pbc_xyz: List[bool] = [True, True, True],
                 sort: bool = False,
                 has_virial: bool = False,
                 torch_float_dtype: torch._C.dtype = torch.float32,
                 device: torch._C.device = torch.device("cpu")):
        super(FullParametersOptimizer, self).__init__()
        self.lit_linear_mtp: LitLinearMtp = lit_linear_mtp
        self.linear_mtp: LinearMtp = self.lit_linear_mtp.model
        self.num_coeffs: int = self.linear_mtp.num_coeffs
        self.num_linear_coeffs: int = self.linear_mtp.num_descriptors
        self.num_type_bias: int = self.linear_mtp.ntypes
        
        self.torch_float_dtype: torch._C.dtype = torch_float_dtype
        if (self.torch_float_dtype == torch.float32):
            self.npy_float_dtype = np.float32
        else:
            self.npy_float_dtype = np.float64
        self.device: torch._C.device = device
        self.linear_mtp.to(self.torch_float_dtype).to(self.device)

        ### Dataset
        extxyz_dataset: ExtxyzDataset = ExtxyzDataset(filename=extxyz_path,
                                                      rcut=rcut,
                                                      umax_num_neigh_atoms=umax_num_neigh_atoms,
                                                      pbc_xyz=pbc_xyz,
                                                      sort=sort,
                                                      torch_float_dtype=self.torch_float_dtype,
                                                      has_virial=has_virial)
        self.num_structures: int = len(extxyz_dataset)
        self.dataloader: DataLoader = DataLoader(dataset=extxyz_dataset,
                                                 batch_size=1,
                                                 shuffle=True)

        ### For log
        self.last_error: Optional[float] = None
        self.current_epoch: int = 0.0
        self.log = {
            "train_mse_epoch": []
        }
    

    def fit(self):
        # 1. Get parameters
        init_params: np.ndarray = self._get_params()

        # 2. Optimize
        result = minimize(
            fun=lambda p: self._loss_and_grad(p),
            x0=init_params,
            jac=True,
            method='l-bfgs-b',
            options={'disp': True,
                     #'gtol': 1e-4,
                     #'eps': 1e-8,
                     #'maxiter': 1000
                     },
            callback=self._callback)

        # 3. Set parameters
        self._set_params(params=result.x)


    def _get_params(self) -> np.ndarray:
        coeffs_array: np.ndarray = self.linear_mtp.coeffs_tensor.cpu().detach().numpy().flatten()
        linear_coeffs_array: np.ndarray = self.linear_mtp.linear_coeffs_tensor.cpu().detach().numpy().flatten()
        type_bias_array: np.ndarray = self.linear_mtp.type_bias_tensor.cpu().detach().numpy().flatten()
        return np.concatenate([coeffs_array, linear_coeffs_array, type_bias_array])


    def _set_params(self, 
                   params: np.ndarray) -> None:
        coeffs_array: np.ndarray = params[:self.num_coeffs]
        linear_coeffs_array: np.ndarray = params[self.num_coeffs: (self.num_coeffs+self.num_linear_coeffs)]
        type_bias_array: np.ndarray = params[(self.num_coeffs+self.num_linear_coeffs):]
        self.linear_mtp.coeffs_tensor.data = torch.tensor(coeffs_array, 
                                                          dtype=self.torch_float_dtype,
                                                          device=self.device)
        self.linear_mtp.linear_coeffs_tensor.data = torch.tensor(linear_coeffs_array,
                                                                 dtype=self.torch_float_dtype,
                                                                 device=self.device)
        self.linear_mtp.type_bias_tensor.data = torch.tensor(type_bias_array,
                                                             dtype=self.torch_float_dtype,
                                                             device=self.device)
    
    def _loss_and_grad(self, params: np.ndarray):
        # 1.
        self._set_params(params=params)

        # 2. forward and backward
        loss: torch.Tensor = torch.tensor(0.0, 
                                          dtype=self.torch_float_dtype, 
                                          device=self.device)
        self.linear_mtp.zero_grad()
        for batch_idx, batch_data in enumerate(self.dataloader):
            binum, bilist, bnumneigh, bfirstneigh, brcs, btypes, bnghost, betot_dft_tensor, bforce_dft_tensor = batch_data
            bmse_loss: torch.Tensor = self.linear_mtp(1.0, 
                                                      1.0,
                                                      betot_dft_tensor,
                                                      bforce_dft_tensor,
                                                      binum,
                                                      bilist,
                                                      bnumneigh,
                                                      bfirstneigh,
                                                      brcs,
                                                      btypes,
                                                      bnghost[0].item())
            batch_loss: torch.Tensor = bmse_loss.sum()
            loss = loss + batch_loss
        loss.backward()
        self.last_error = loss.item()

        # 3. 
        coeffs_grad_array: np.ndarray = self.linear_mtp.coeffs_tensor.grad.cpu().detach().numpy().flatten()
        linear_coeffs_array: np.ndarray = self.linear_mtp.linear_coeffs_tensor.grad.cpu().detach().numpy().flatten()
        type_bias_array: np.ndarray = self.linear_mtp.type_bias_tensor.grad.cpu().detach().numpy().flatten()

        return loss.item(), np.concatenate([coeffs_grad_array, linear_coeffs_array, type_bias_array])


    def _callback(self, params: np.ndarray):
        self.log['train_mse_epoch'].append(self.last_error)
        print(self.current_epoch, self.last_error)
        
        # checkpoint
        adjusted_state_dict = {f"model.{k}": v for k, v in self.linear_mtp.state_dict().items()}
        checkpoint = {
            'pytorch-lightning_version': L.__version__,
            'hparams': self.lit_linear_mtp.hparams,
            'train_mse_epoch': self.last_error,
            'epoch': self.current_epoch,
            'global_step': self.current_epoch,
            'state_dict': adjusted_state_dict,    # 模型参数
        }
        torch.save(checkpoint, f="./checkpoints/checkpoint_epoch_{0}.ckpt".format(self.current_epoch))
        
        self.current_epoch += 1
        