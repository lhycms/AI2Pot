import os
import torch
from pymatgen.core import Structure
from ai2pot.utils.usepot import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")

mlff_input = MlffInput(rcut=5.0,
                       umax_num_neighs=40,
                       pbc_xyz=[True, True, True],
                       sort=False,
                       dtype=torch.float64,
                       device=torch.device("cpu"))
structure = Structure.from_file(ReNbSSe_POSCAR)
nblist_info = mlff_input.analyse_pymatgen(structure)
bilist = nblist_info[1]
btypes = nblist_info[5]


for itype in range(4):
    result = torch.gather(input=btypes, dim=1, index=bilist.to(torch.int64))
    print((result == itype))
    