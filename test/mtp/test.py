import os
import torch
import torch.nn as nn
from pymatgen.core import Structure
from ai2pot.mtp.nn_mtp import NNMtp
from ai2pot.utils.usepot import MlffInput


TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
ReNbSSe_POSCAR = os.path.join(TEST_FILES_DIR, "POSCARs", "POSCAR")

mtp_level: int = 16
ntypes: int = 4
chebyshev_size: int = 8
rmax: float = 5.0
rmin: float = 0.5
umax_num_neighs: int = 100
dtype: torch._C.dtype = torch.float64
device = torch.device("cpu")

batch_norm_mark = False
fit_sizes_list = [30]
fit_activation = nn.Tanh()

nn_mtp = NNMtp(mtp_level=mtp_level,
                            ntypes=ntypes,
                            chebyshev_size=chebyshev_size,
                            rmax=rmax,
                            rmin=rmin,
                            umax_num_neighs=umax_num_neighs,
                            batch_norm_mark=batch_norm_mark,
                            fit_sizes_list=fit_sizes_list,
                            fit_activation=fit_activation,
                            bias_mark=False,
                            energy_shift_tensor=False,
                            has_forces=False,
                            has_virials=False).to(dtype)
mlff_input: MlffInput = MlffInput(rcut=rmax,
                                umax_num_neighs=umax_num_neighs,
                                pbc_xyz=[True, True, True],
                                sort=False,
                                dtype=dtype,
                                device=device)

# 0.
center_idx_modify: int = 0
neigh_idx_modify: int = 6
direction_idx_modify: int = 1
delta: float = 1e-5

# Structure 1
structure = Structure.from_file(ReNbSSe_POSCAR)
nblist_info = mlff_input.analyse_pymatgen(structure=structure)
etot = nn_mtp(*nblist_info[1:])[0]
etot.backward()
#print(etot.item())
print("etot = ", etot.item())
print("derivative by custom code = ", nblist_info[4].grad[0][center_idx_modify][neigh_idx_modify][direction_idx_modify].item() )

# Structure 2
nblist_info_ = [tensor.clone().detach() for tensor in nblist_info]
nblist_info_[4][0][center_idx_modify][neigh_idx_modify][direction_idx_modify] += delta
nblist_info_[4].requires_grad_(True)
etot_ = nn_mtp(*nblist_info_[1:])[0]
#print(etot_.item())
print("derivative by finite difference method = ", ((etot_ - etot) / delta).item() )
print("difference = ", nblist_info[4].grad[0][center_idx_modify][neigh_idx_modify][direction_idx_modify].item() - ((etot_ - etot) / delta).item() )
