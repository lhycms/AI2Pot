import os
import time
import unittest
from typing import (List)

import numpy as np
import torch

from ase import Atoms
from ase.io import read as ase_read

from ai2pot.optimizer.laddar_bfgs import (TorchScipyBfgs,
                                          ParameterInheritor,
                                          LaddarTrainer)
from ai2pot.optimizer.mtpr_solver import LinearMtpSolver
from ai2pot.models.mtp.linear_mtp import LinearMtp
from ai2pot.models.potential_train import LitLinearMtp
from ai2pot.data.mlffdataset import ExtxyzDataset
from ai2pot.utils.usepot import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
# os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")

torch.manual_seed(42)
torch.set_num_threads(16)


class TorchScipyBfgsTest(unittest.TestCase):
    def setUp(self):
        print("TorchScipyBfgsTest (TestCase) is setting up...\n")
        ###
        self.type_map: List[int] = ExtxyzDataset.get_type_map(filename=PbTe_EXTXYZ_PATH)
        self.umax_num_neigh_atoms: int = 200
        self.fit_virial: bool = False
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.device: torch._C.device = torch.device("cuda")
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.linear_mtp: LinearMtp = LinearMtp(type_map=self.type_map,
                                               umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                               fit_virial=self.fit_virial,
                                               mtp_level=16,
                                               chebyshev_size=self.chebyshev_size,
                                               rmax=self.rmax,
                                               rmin=self.rmin,
                                               zbl_rmax=0.0,
                                               zbl_rmin=0.0,
                                               zbl_cks_list=None,
                                               zbl_dks_list=None).to(device=self.device).to(self.torch_float_dtype)
        
        self.trainset: ExtxyzDataset = ExtxyzDataset(filename=PbTe_EXTXYZ_PATH,
                                                     rcut=self.rmax,
                                                     umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                     pbc_xyz=[True, True, True],
                                                     sort=False,
                                                     torch_float_dtype=self.torch_float_dtype,
                                                     has_virial=self.fit_virial)

        ###
        self.maxiter: int = 500
        self.torch_scipy_bfgs: TorchScipyBfgs = TorchScipyBfgs(linear_mtp=self.linear_mtp,
                                                               trainset=self.trainset,
                                                               e_weight=1.0,
                                                               f_weight=2.0,
                                                               v_weight=0.0,
                                                               maxiter=self.maxiter,
                                                               gtol=1e-7,
                                                               disp=True)

    
    def tearDown(self):
        print("TorchScipyBfgsTest (TestCase) is tearing down...\n")
    

    def est_get_x0(self):
        x0: np.ndarray = self.torch_scipy_bfgs._get_x0()
        print("1. x0:")
        print("*" * 80)
        print(x0)

    
    def est_set_x(self):
        x0: np.ndarray = self.torch_scipy_bfgs._get_x0() + 1.0
        self.torch_scipy_bfgs._set_x(x=x0)

    
    def est_loss_and_grad(self):
        x1: np.ndarray = self.torch_scipy_bfgs._get_x0() + 1.0
        total_loss, grad = self.torch_scipy_bfgs._loss_and_grad(x=x1)
        total_loss: float
        grad: np.ndarray
        print("1. Total loss = ", total_loss)
        print("2. grad = \n", grad)



class ParameterInheritorTest(unittest.TestCase):
    def setUp(self):
        print("ParameterInheritorTest (TestCase) is setting up...\n")
        ###
        self.type_map: List[int] = ExtxyzDataset.get_type_map(filename=PbTe_EXTXYZ_PATH).detach().cpu().numpy().tolist()
        self.umax_num_neigh_atoms: int = 200
        self.fit_virial: bool = False
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.device: torch._C.device = torch.device("cuda")
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.old_linear_mtp: LinearMtp = LinearMtp(type_map=self.type_map,
                                                   umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                   fit_virial=self.fit_virial,
                                                   mtp_level=14,
                                                   chebyshev_size=self.chebyshev_size,
                                                   rmax=self.rmax,
                                                   rmin=self.rmin,
                                                   zbl_rmax=0.0,
                                                   zbl_rmin=0.0,
                                                   zbl_cks_list=None,
                                                   zbl_dks_list=None).to(device=self.device, dtype=self.torch_float_dtype)
        self.new_linear_mtp: LinearMtp = LinearMtp(type_map=self.type_map,
                                                   umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                   fit_virial=self.fit_virial,
                                                   mtp_level=16,
                                                   chebyshev_size=self.chebyshev_size,
                                                   rmax=self.rmax,
                                                   rmin=self.rmin,
                                                   zbl_rmax=0.0,
                                                   zbl_rmin=0.0,
                                                   zbl_cks_list=None,
                                                   zbl_dks_list=None).to(device=self.device, dtype=self.torch_float_dtype)
        
        self.trainset: ExtxyzDataset = ExtxyzDataset(filename=PbTe_EXTXYZ_PATH,
                                                     rcut=self.rmax,
                                                     umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                     pbc_xyz=[True, True, True],
                                                     sort=False,
                                                     torch_float_dtype=self.torch_float_dtype,
                                                     has_virial=self.fit_virial)
        
        ### Main
        self.e_weight: float = 1.0
        self.f_weight: float = 2.0
        self.v_weight: float = 0.0
        self.ridge_lambda: float = 1e-2
        self.param_inheritor: ParameterInheritor = ParameterInheritor(old_model=self.old_linear_mtp,
                                                                      new_model=self.new_linear_mtp,
                                                                      trainset=self.trainset,
                                                                      e_weight=self.e_weight,
                                                                      f_weight=self.f_weight,
                                                                      v_weight=self.v_weight,
                                                                      ridge_lambda=self.ridge_lambda)


        ### 
        self.structure: Atoms = ase_read(filename=PbTe_EXTXYZ_PATH, index=":")[0]
        self.mlff_input: MlffInput = MlffInput(type_map=self.type_map,
                                               rcut=self.rmax,
                                               umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                               pbc_xyz=[True, True, True],
                                               sort=False,
                                               dtype=self.torch_float_dtype,
                                               device=self.device)
    

    def tearDown(self):
        print("ParameterInheritorTest (TestCase) is tearing down...\n")

    
    def est_transfer(self):
        # 1. Solve the linear equation for old_model
        linear_mtp_solver: LinearMtpSolver = LinearMtpSolver(e_weight=self.e_weight,
                                                             f_weight=self.f_weight,
                                                             v_weight=self.v_weight,
                                                             linear_mtp=self.old_linear_mtp,
                                                             trainset=self.trainset,
                                                             ridge_lambda=self.ridge_lambda)
        linear_mtp_solver.solve_linear_equation()
        old_descriptors = self.old_linear_mtp.predict_descriptors(*self.mlff_input.analyse_ase(atoms=self.structure))

        # 2. Transfer parameters
        self.new_linear_mtp._init_all_zeros()
        self.param_inheritor.transfer()
        new_descriptors = self.new_linear_mtp.predict_descriptors(*self.mlff_input.analyse_ase(atoms=self.structure))
        print("1.1. Descriptors calculated by old MTP:\n", old_descriptors[0, 1, :])
        print("1.2. Descriptors calculated by new MTP:\n", new_descriptors[0, 1, :])
        print("2.1. Old linear_coeffs_tensor:\n", self.old_linear_mtp.linear_coeffs_tensor)
        print("2.2. New linear_coeffs_tensor:\n", self.new_linear_mtp.linear_coeffs_tensor)
        


class LaddarTrainerTest(unittest.TestCase):
    def setUp(self):
        print("LaddarTrainerTest (TestCase) is setting up...\n")
        ###
        self.type_map: List[int] = ExtxyzDataset.get_type_map(filename=PbTe_EXTXYZ_PATH)
        self.umax_num_neigh_atoms: int = 200
        self.fit_virial: bool = False
        self.chebyshev_size: int = 8
        self.rmax: float = 5.0
        self.rmin: float = 0.0
        self.device: torch._C.device = torch.device("cuda")
        self.torch_float_dtype: torch._C.dtype = torch.float32
        self.lit_linear_mtp: LitLinearMtp = LitLinearMtp(type_map=self.type_map,
                                                         umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                         fit_virial=self.fit_virial,
                                                         mtp_level=16,
                                                         chebyshev_size=self.chebyshev_size,
                                                         rmax=self.rmax,
                                                         rmin=self.rmin,
                                                         zbl_rmax=0.0,
                                                         zbl_rmin=0.0,
                                                         zbl_cks_list=None,
                                                         zbl_dks_list=None).to(device=self.device, dtype=self.torch_float_dtype)
        self.trainset: ExtxyzDataset = ExtxyzDataset(filename=PbTe_EXTXYZ_PATH,
                                                     rcut=self.rmax,
                                                     umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                     pbc_xyz=[True, True, True],
                                                     sort=False,
                                                     torch_float_dtype=self.torch_float_dtype,
                                                     has_virial=self.fit_virial)

        ###
        self.e_weight: float = 2.0
        self.f_weight: float = 3.0
        self.v_weight: float = 0.0
        self.laddar_start: int = 6
        self.laddar_step: int = 2
        self.laddar_trainer: LaddarTrainer = LaddarTrainer(lit_linear_mtp=self.lit_linear_mtp,
                                                           trainset=self.trainset,
                                                           e_weight=self.e_weight,
                                                           f_weight=self.f_weight,
                                                           v_weight=self.v_weight,
                                                           laddar_start=self.laddar_start,
                                                           laddar_step=self.laddar_step)


    def tearDown(self):
        print("LaddarTrainerTest (TestCase) is tearing down...\n")
    

    def test_generate_model(self):
        new_model: LinearMtp = self.laddar_trainer._generate_model(mtp_level=8)
        param: torch.nn.Parameter = next(new_model.parameters())
        print(f"1. The mtp level of new model = {new_model.mtp_level}")
        assert(param.device == self.lit_linear_mtp.device)
        assert(param.dtype == self.lit_linear_mtp.dtype)


    def test_fit(self):
        self.laddar_trainer.fit()


if __name__ == "__main__":
    unittest.main()
