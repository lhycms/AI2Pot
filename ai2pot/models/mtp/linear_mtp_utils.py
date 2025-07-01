from typing import List
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from ai2pot.data.mlffdatamodule import ExtxyzDataModule
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.models.mtp.linear_mtp import LinearMtp


class LinearMtpCalculator(object):
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

        # mse
        e_mse = torch.mean(torch.pow(e_ml_tensor/binum_tensor-e_dft_tensor/binum_tensor, 2))
        e_rmse = torch.sqrt(e_mse)
        #f_mse = f_ml_tensor - f_dft_tensor
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.plot([-2.5, 2.5], [-2.5, 2.5])
        plt.scatter(f_dft_tensor.flatten().detach().numpy(), f_ml_tensor.flatten().detach().numpy())
        plt.savefig("/data/home/liuhanyu/mycode/AI2Pot/f_test.png")

        return e_rmse


    def calculate_efv_rmse(self):
        #for batch_idx, batch_data in enumerate(self.test_dataloader):
            #print(batch_idx, batch_data[0].shape)
        pass