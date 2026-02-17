import os
import unittest
from typing import List

from pymatgen.core import Structure
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from ai2pot.utils.usepot import MlffToEFLossInput, MlffInput
from ai2pot.fromcc import (
    nepToEFLossOp,
    nepToEFOp,
    nepToDescriptorsOp)


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))
MoS2_POSCAR_PATH = os.path.join(TEST_FILES_DIR, "POSCARs", "MoS2", "POSCAR_perturbed0.2")
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")


#torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
torch.manual_seed(214)


class NepTest(unittest.TestCase):
    def setUp(self):
        print("NepTest (TestCase) is setting up...\n")
        # 0.
        self.torch_float_dtype: torch._C.dtype = torch.float64
        self.device: torch._C.device = torch.device("cuda")

        # 1. 
        self.n_radial_basis: int = 6
        self.n_angular_basis: int = 4
        self.l_max: int = 4
        self.chebyshev_size: int = 8
        self.num_neurons: int = 30
        self.rmax_radial: float = 6.0
        self.rmax_angular: float = 5.0
        self.umax_num_neigh_atoms: int = 200
        self.fit_virial: bool = False
        
        self.num_descriptors: int = self.n_radial_basis + self.n_angular_basis * self.l_max


        self.ntypes: int = 2
        #self.type_map_tensor: torch.Tensor = torch.tensor(data=[16, 34, 41, 75], dtype=torch.int32)
        self.type_map_tensor: torch.Tensor = torch.tensor(data=[42, 16], dtype=torch.int32)
        self.structure: Structure = Structure.from_file(MoS2_POSCAR_PATH)
        """
        self.ntypes: int = 2
        self.type_map_tensor: torch.Tensor = torch.tensor(data=[1, 8], dtype=torch.int32)
        self.structure: Structure = Structure(lattice=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
                                              species=["H", "H", "O"],
                                              coords=[[0, 0, 0], 
                                                      [0, 4.0, 0],
                                                      [3.0, 0.0, 0]
                                                      ],
                                              coords_are_cartesian=True)
        """
        print(self.structure)

        # 2. ZBL
        self.zbl_rmax: float = 0.0
        self.zbl_rmin: float = 0.0
        self.zbl_cks_tensor: torch.Tensor = torch.zeros(self.ntypes*self.ntypes*4, 
                                                        dtype=self.torch_float_dtype,
                                                        device=self.device)
        self.zbl_dks_tensor: torch.Tensor = torch.zeros(self.ntypes*self.ntypes*4, 
                                                        dtype=self.torch_float_dtype,
                                                        device=self.device)
        for ii in range(self.ntypes):
            for jj in range(self.ntypes):
                idx = ii*self.ntypes + jj
                self.zbl_cks_tensor[idx*4 + 0] = 0.18175
                self.zbl_cks_tensor[idx*4 + 1] = 0.50986
                self.zbl_cks_tensor[idx*4 + 2] = 0.28022
                self.zbl_cks_tensor[idx*4 + 3] = 0.02817
                self.zbl_dks_tensor[idx*4 + 0] = 3.1998
                self.zbl_dks_tensor[idx*4 + 1] = 0.94229
                self.zbl_dks_tensor[idx*4 + 2] = 0.4029
                self.zbl_dks_tensor[idx*4 + 3] = 0.20162

        # 3. 
        self.mlff_to_loss_input: MlffToEFLossInput = MlffToEFLossInput(type_map=self.type_map_tensor.numpy().tolist(),
                                                                       rcut=self.rmax_radial,
                                                                       umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                                       dtype=self.torch_float_dtype,
                                                                       device=self.device)
        self.mlff_input: MlffInput = MlffInput(type_map=self.type_map_tensor.numpy().tolist(),
                                                rcut=self.rmax_radial,
                                                umax_num_neigh_atoms=self.umax_num_neigh_atoms,
                                                dtype=self.torch_float_dtype,
                                                device=self.device)

        # 4. coeffs, linear coeffs, type bias``
        self.coeffs_tensor: torch.Tensor = torch.zeros(self.ntypes*self.ntypes*(self.n_radial_basis+self.n_angular_basis)*self.chebyshev_size, 
                                                       dtype=self.torch_float_dtype,
                                                       device=self.device)
        nn.init.normal_(self.coeffs_tensor, mean=0.0, std=0.2)
        self.w0_tensor: torch.Tensor = torch.zeros(self.ntypes * self.num_neurons * self.num_descriptors,
                                                   dtype=self.torch_float_dtype,
                                                   device=self.device)
        nn.init.normal_(self.w0_tensor, mean=0.0, std=0.5)
        self.b0_tensor: torch.Tensor = torch.zeros(self.ntypes * self.num_neurons,
                                                   dtype=self.torch_float_dtype,
                                                   device=self.device)
        nn.init.normal_(self.b0_tensor, mean=0.0, std=0.5)
        self.w1_tensor: torch.Tensor = torch.zeros(self.ntypes * self.num_neurons,
                                                   dtype=self.torch_float_dtype,
                                                   device=self.device)
        nn.init.normal_(self.w1_tensor, mean=0.0, std=0.5)
        self.type_bias_tensor: torch.Tensor = torch.zeros(self.ntypes,
                                                          dtype=self.torch_float_dtype,
                                                          device=self.device)
        nn.init.normal_(self.type_bias_tensor, mean=0.0, std=1.0)

        # q_scaler_tensor
        self.q_scaler_tensor: torch.Tensor = torch.ones(self.num_descriptors,
                                                        dtype=self.torch_float_dtype,
                                                        device=self.device)

    
    def tearDown(self):
        print("NepTest (TestCase) is tearing down...\n")


    def test_nepToEF(self):
        input_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        e, f = nepToEFOp(self.chebyshev_size,
                         self.n_radial_basis,
                         self.n_angular_basis,
                         self.l_max,
                         self.coeffs_tensor,
                         self.w0_tensor,
                         self.b0_tensor,
                         self.w1_tensor,
                         self.type_bias_tensor,
                         input_info[0],
                         input_info[1],
                         input_info[2],
                         input_info[3],
                         input_info[4],
                         input_info[5],
                         self.type_map_tensor,
                         input_info[6],
                         self.rmax_radial,
                         self.rmax_angular,
                         self.q_scaler_tensor,
                         self.zbl_rmax,
                         self.zbl_rmin,
                         self.zbl_cks_tensor,
                         self.zbl_dks_tensor)
        e: torch.Tensor
        f: torch.Tensor
        #print(e)
        #print("force = ", f)


    def test_nepToEFLoss(self):
        # 1. Parameters
        e_weight: float = 1e1
        f_weight: float = 1e1
        self.coeffs_tensor.requires_grad_(True)
        self.w0_tensor.requires_grad_(True)
        self.b0_tensor.requires_grad_(True)
        self.w1_tensor.requires_grad_(True)
        self.type_bias_tensor.requires_grad_(True)
        
        # 2. Run
        input_info: List[torch.Tensor] = self.mlff_to_loss_input.analyse_pymatgen(self.structure,
                                                                                  e_weight=e_weight,
                                                                                  f_weight=f_weight)

        def nepToEFLossOp1(*args, **kwargs):
            return nepToEFLossOp(*args, **kwargs)[0]
        
        test = gradcheck(func=nepToEFLossOp1,
                         inputs=(e_weight,
                                 f_weight,
                                 input_info[2],
                                 input_info[3],
                                 self.chebyshev_size,
                                 self.n_radial_basis,
                                 self.n_angular_basis,
                                 self.l_max,
                                 self.coeffs_tensor,
                                 self.w0_tensor,
                                 self.b0_tensor,
                                 self.w1_tensor,
                                 self.type_bias_tensor,
                                 input_info[4],
                                 input_info[5],
                                 input_info[6],
                                 input_info[7],
                                 input_info[8],
                                 input_info[9],
                                 self.type_map_tensor,
                                 input_info[10].item(),
                                 self.rmax_radial,
                                 self.rmax_angular,
                                 self.q_scaler_tensor,
                                 self.zbl_rmax,
                                 self.zbl_rmin,
                                 self.zbl_cks_tensor,
                                 self.zbl_dks_tensor),
                            rtol=1e-4,
                            atol=1e-4,
                            eps=1e-8)
        print("-------------------------------------------------")
        print("* NepToEFLossOp Gradient pass check: ", test)
        print("-------------------------------------------------")


    def test_nepToEFLoss_print(self):
        # 1. Parameters
        e_weight: float = 1e1
        f_weight: float = 1e1
        self.coeffs_tensor.requires_grad_(True)
        self.w0_tensor.requires_grad_(True)
        self.b0_tensor.requires_grad_(True)
        self.b0_tensor.requires_grad_(True)
        self.w1_tensor.requires_grad_(True)
        self.type_bias_tensor.requires_grad_(True)
        
        # 2. Run
        input_info: List[torch.Tensor] = self.mlff_to_loss_input.analyse_pymatgen(self.structure,
                                                                                  e_weight=e_weight,
                                                                                  f_weight=f_weight)
        loss = nepToEFLossOp(e_weight,
                            f_weight,
                            input_info[2],
                            input_info[3],
                            self.chebyshev_size,
                            self.n_radial_basis,
                            self.n_angular_basis,
                            self.l_max,
                            self.coeffs_tensor,
                            self.w0_tensor,
                            self.b0_tensor,
                            self.w1_tensor,
                            self.type_bias_tensor,
                            input_info[4],
                            input_info[5],
                            input_info[6],
                            input_info[7],
                            input_info[8],
                            input_info[9],
                            self.type_map_tensor,
                            input_info[10].item(),
                            self.rmax_radial,
                            self.rmax_angular,
                            self.q_scaler_tensor,
                            self.zbl_rmax,
                            self.zbl_rmin,
                            self.zbl_cks_tensor,
                            self.zbl_dks_tensor)[0].sum()
        loss.backward()
        print("0. loss = ")
        print(loss.item())
        print("1. self.coeffs_tensor.grad:")
        print(self.coeffs_tensor.grad)
        #print("2. self.w0_tensor.grad:")
        #print(self.w0_tensor.grad)
        #print("3. self.b0_tensor.grad:")
        #print(self.b0_tensor.grad)
        #print("4. self.w1_tensor.grad:")
        #print(self.w1_tensor.grad)
        #print("5. self.type_bias_tensor.grad:")
        #print(self.type_bias_tensor.grad)


    def est_nepToDescriptors(self):
        input_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(self.structure)
        bdescriptors_tensor: torch.Tensor = nepToDescriptorsOp(self.chebyshev_size,
                                                               self.n_radial_basis,
                                                               self.n_angular_basis,
                                                               self.l_max,
                                                               self.coeffs_tensor,
                                                               input_info[0],
                                                               input_info[1],
                                                               input_info[2],
                                                               input_info[3],
                                                               input_info[4],
                                                               input_info[5],
                                                               self.type_map_tensor,
                                                               input_info[6].item(),
                                                               self.rmax_radial,
                                                               self.rmax_angular)
        print("-------------------------------------------------")
        print("* bdescriptor_tensor: ")
        print("-------------------------------------------------")
        print(bdescriptors_tensor)


if __name__ == "__main__":
    unittest.main()
