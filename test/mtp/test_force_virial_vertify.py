import unittest
import os
from typing import List

import torch
import torch.nn as nn
import numpy as np
from ai2pot.mtp.nn_mtp import NNMtp
from ai2pot.utils.usepot import MlffInput
from pymatgen.core import Structure


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")

class NNMtpForceVirialTest(unittest.TestCase):
    def setUp(self):
        print("NNMtpForceVirialTest (TestCase) is setting up...\n")
        mtp_level: int = 16
        ntypes: int = 4
        chebyshev_size: int = 8
        rmax: float = 5.0
        rmin: float = 0.5
        umax_num_neighs: int = 40
        self.dtype = torch.float64
        self.device = torch.device("cpu")
        
        self.nn_mtp: nn.Module = NNMtp(mtp_level=mtp_level,
                                       ntypes=ntypes,
                                       chebyshev_size=chebyshev_size,
                                       rmax=rmax,
                                       rmin=rmin,
                                       umax_num_neighs=umax_num_neighs,
                                       fit_sizes_list=[30],
                                       fit_activation=nn.Tanh(),
                                       bias_mark=False,
                                       energy_shift_tensor=False,
                                       has_virials=True).to(self.dtype).to(self.device)
        self.mlff_input: MlffInput = MlffInput(rcut=rmax,
                                               umax_num_neighs=umax_num_neighs,
                                               pbc_xyz=[True, True, True],
                                               sort=False,
                                               dtype=self.dtype,
                                               device=self.device)
        self.structure: Structure = Structure.from_file(ReNbSSe_POSCAR)
    
    
    def tearDown(self):
        print("NNMtpForceVirialTest (TestCase) is tearing down...\n")
        
        
    def test_force(self):
        center_idx_modify: int = 0
        direction_idx_modify: int = 1
        delta: float = 1e-6
        
        # Structure 1
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(structure=self.structure)
        etot, fi, v = self.nn_mtp(*nblist_info[1:])

        # Structure 2
        lattice_: Structure = self.structure.lattice
        species_ = self.structure.species
        cart_coords_ = self.structure.cart_coords
        cart_coords_[center_idx_modify][direction_idx_modify] += delta
        structure_ = Structure(lattice=lattice_,
                               species=species_,
                               coords=cart_coords_,
                               coords_are_cartesian=True)
        nblist_info_: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(structure=structure_)
        etot_, fi_, v_ = self.nn_mtp(*nblist_info_[1:])
        
        print("----------------------------------------------")
        print("1.1. Before perturbation, force of atom[{0}] on {1} direction = ")
        print(fi[0][center_idx_modify][direction_idx_modify])
        print("1.2. After perturbation, force of atom[{0}] on {1} direction = ")
        print(fi_[0][center_idx_modify][direction_idx_modify])
        print("1.3. Force of atom[{0}] on {1} direcotion calculated by $f_{i\alpha} = -\frac{\partial E_{tot}}{\partial r_{i\alpha}}$ = ")
        print(-(etot_ - etot) / delta)
        print("----------------------------------------------")
        
        
    def test_virial(self):
        direction1_idx_modify: int = 0
        direction2_idx_modify: int = 1
        delta: float = 1e-4
        
        # Structure 1
        nblist_info: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(structure=self.structure)
        etot, fi, v = self.nn_mtp(*nblist_info[1:])
        
        # Structure 2
        lattice_: np.ndarray = np.zeros([3, 3])
        for ii in range(3):
            for jj in range(3):
                lattice_[ii][jj] = self.structure.lattice.matrix[ii][jj]
        lattice_[direction1_idx_modify][direction2_idx_modify] += delta
        species_ = self.structure.species
        frac_coords_ = self.structure.frac_coords
        structure_ = Structure(lattice=lattice_,
                               species=species_,
                               coords=frac_coords_,
                               coords_are_cartesian=False)
        nblist_info_: List[torch.Tensor] = self.mlff_input.analyse_pymatgen(structure=structure_)
        etot_, fi_, v_ = self.nn_mtp(*nblist_info_[1:])
        
        print("----------------------------------------------")
        print("2.1. Before perturbation, force of atom[{0}] on {1} direction = ")
        print(v[0][direction1_idx_modify][direction2_idx_modify])
        print("2.2. Before perturbation, force of atom[{0}] on {1} direction = ")
        print(v_[0][direction1_idx_modify][direction2_idx_modify])
        print("2.3. Before perturbation, force of atom[{0}] on {1} direction = ")
        print(np.sum(fi[:, :, direction1_idx_modify].flatten().numpy() * self.structure.cart_coords[:, direction2_idx_modify]))
        print("----------------------------------------------")

    
if __name__ == "__main__":
    unittest.main()
