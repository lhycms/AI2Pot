from typing import Optional
import torch
import numpy as np
from scipy.optimize import minimize

import lightning as L

from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_train import LitLinearMtp


class ParametersOptimizer(object):
    def __init__(self,
                 lit_linear_mtp: LitLinearMtp):
        pass


    def get_params(self):
        pass
    

    def set_params(self):
        pass


class FullParametersOptimizer(ParametersOptimizer):
    def __init__(self,
                 lit_linear_mtp: LitLinearMtp,
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

        ### For log
        self.last_error: Optional[float] = None
        self.current_epoch: int = 0.0
        self.log = {
            "train_mse_epoch": []
        }
    

    def fit(self):
        # 1. Get parameters
        params: np.ndarray = self._get_params()

        # 2. Optimize
        result = minimize(
            fun=None,
            x0=None,
            jac=None,
            method='BFGS',
            options={'disp': True})

        # 3. Set parameters
        self._set_params(params=result.x)


    def _get_params(self) -> np.ndarray:
        coeffs_array: np.ndarray = self.linear_mtp.coeffs_tensor.detach().numpy().flatten()
        linear_coeffs_array: np.ndarray = self.linear_mtp.linear_coeffs_tensor.detach().numpy().flatten()
        type_bias_array: np.ndarray = self.linear_mtp.type_bias_tensor.detach().numpy().flatten()
        return np.concatenate([coeffs_array, linear_coeffs_array, type_bias_array])


    def _set_params(self, 
                   params: np.ndarray) -> None:
        coeffs_array: np.ndarray = params[:self.num_coeffs]
        linear_coeffs_array: np.ndarray = params[self.num_coeffs: (self.num_coeffs+self.num_linear_coeffs)]
        type_bias_array: np.ndarray = params[(self.num_coeffs+self.num_linear_coeffs):]
        self.linear_mtp.coeffs_tensor.data = torch.tensor(coeffs_array, 
                                                          dtype=self.dtype,
                                                          device=self.device)
        self.linear_mtp.linear_coeffs_tensor.data = torch.tensor(linear_coeffs_array,
                                                                 dtype=self.dtype,
                                                                 device=self.device)
        self.linear_mtp.type_bias_tensor.data = torch.tensor(type_bias_array,
                                                             dtype=self.dtype,
                                                             device=self.device)
    
    def _loss_and_grad(self, params: np.ndarray):
        # 1.
        self._set_params(params=params)

        # 2. forward
        #bmse_tensor: torch.Tensor = self.linear_mtp()
        loss = None
        
        # 3. backward
        self.linear_mtp.zero_grad()
        #loss.backward()

        # 4. 
        coeffs_grad_array: np.ndarray = self.linear_mtp.coeffs_tensor.grad.detach().numpy().flatten()
        linear_coeffs_array: np.ndarray = self.linear_mtp.linear_tensor.grad.detach().numpy().flatten()
        type_bias_array: np.ndarray = self.linear_mtp.linear_tensor.grad.detach().numpy().flatten()

        return loss.item(), np.concatenate([coeffs_grad_array, linear_coeffs_array, type_bias_array])


    def _callback(self, params: np.ndarray):
        self.log['train_mse_epoch'].append(self.last_error)
        
        # checkpoint
        adjusted_state_dict = {f"model.{k}": v for k, v in self.model.state_dict().items()}
        checkpoint = {
            'pytorch-lightning_version': L.__version__,
            'hparams': self.lit_linear_mtp.hparams,
            'train_mse_epoch': self.last_error,
            'epoch': self.current_epoch,
            'global_step': self.current_epoch,
            'state_dict': adjusted_state_dict,    # 模型参数
        }
        torch.save(checkpoint, f="./checkpoint_epoch_{0}.ckpt".format(self.current_epoch))
        
        self.current_epoch += 1
        