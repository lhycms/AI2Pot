import os
import unittest
from typing import (List, Optional, Tuple)

import torch
from torch.utils.data import (DataLoader)

from ai2pot.data.mlffdataset import ExtxyzDataset
from ai2pot.fromcc import (
    targetStatisticsOp
)

TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR_PATH = os.path.join(os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR"))
PbTe_EXTXYZ_PATH = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"#os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")

#torch.use_deterministic_algorithms(True)
torch.set_num_threads(16)
torch.manual_seed(2143)


class TargetStatisticsOpTest(unittest.TestCase):
    def setUp(self):
        print("TargetStatisticsOpTest (TestCase) is setting up...")

        filename: str = PbTe_EXTXYZ_PATH
        rcut: float = 5.0
        umax_num_neigh_atoms: int = 200
        pbc_xyz: List[bool] = [True, True, True]
        sort: bool = False
        torch_float_dtype: torch._C.dtype = torch.float64
        has_virial: bool = False
        self.extxyz_dataset: ExtxyzDataset = ExtxyzDataset(filename=filename,
                                                      rcut=rcut,
                                                      umax_num_neigh_atoms=umax_num_neigh_atoms,
                                                      pbc_xyz=pbc_xyz,
                                                      sort=sort,
                                                      torch_float_dtype=torch_float_dtype,
                                                      has_virial=has_virial)
        

    def tearDown(self):
        print("TargetStatisticsOpTest (TestCase) is tearing down...")

    
    def test_targetStatisticsOp_single_batch(self):
        batch_size: int = 25
        dataloader = DataLoader(dataset=self.extxyz_dataset,
                                batch_size=batch_size,
                                shuffle=False)
        
        for batch_idx, batch_data in enumerate(dataloader):
            results = targetStatisticsOp(batch_data[0],
                               batch_data[7],
                               batch_data[8])
            natoms_in_batch, ei_mean, ei_M2, force_M2 = results

            print("1. Calculate using torch:")
            print("\t1.2. ei_mean = ", (torch.sum(batch_data[7])/natoms_in_batch).item())
            print("\t1.4. force_std = ", torch.std(batch_data[8]).item())
            print("\n")

        print("2. Calculate using targetStatisticsOp:")
        print("\t2.1. natoms_in_batch = ", natoms_in_batch)
        print("\t2.2. ei_mean = ", ei_mean.item())
        print("\t2.3. ei_std = ", torch.sqrt(ei_M2/(batch_size-1)).item())
        print("\t2.4. force_std = ", torch.sqrt(force_M2/(3*natoms_in_batch-1)).item())
        


if __name__ == "__main__":
    unittest.main()
    