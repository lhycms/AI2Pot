import os
os.environ["AI2POT_PATH"] = "/data/home/liuhanyu/mycode/AI2Pot/"
os.environ["AI2POT_TUTORIALS_PATH"] = "/data/home/liuhanyu/mycode/AI2Pot-Tutorials"
os.environ["AI2POT_TEST_DATA_PATH"] = os.path.join(os.environ.get("AI2POT_TUTORIALS_PATH"),
                                                   "data",
                                                   "XYZ")

from typing import List
import torch
from ai2pot.data.mlffdatamodule import ExtxyzDataModule
from ai2pot.models.potential_train import LitLinearMtp
import lightning as L
from ai2pot.models.mtp.linear_mtp_train_utils import FullParametersOptimizer


# 1. logger path
save_dir: str = os.path.join(os.environ.get("AI2POT_TUTORIALS_PATH"),
                                            "lightning_logs")
name: str = "linear_mtp"

# 2. extxyz path
TEST_FILES_DIR = os.path.join(os.getenv("AI2POT_PATH"), "test", "test_data")
PbTe_EXTXYZ_PATH = os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/C/train.xyz"
# "/data/home/liuhanyu/mycode/AI2Pot-Tutorials/data/XYZ/Li_battery/train.xyz"
# os.path.join(TEST_FILES_DIR, "XYZ", "11_NEP_potential_PbTe", "train_m.xyz")

from ai2pot.utils.prepot import ExtxyzShifter

extxyz_shifter: ExtxyzShifter = ExtxyzShifter(extxyz_path=PbTe_EXTXYZ_PATH)
energy_shifts: List[float] = extxyz_shifter.types_energy_shifts
print("energy_shifts =", energy_shifts)

max_epochs: int = 200
accelerator: str = "cpu"
devices: int = 1

trainset_path: str = PbTe_EXTXYZ_PATH
rcut: float = 5.0
umax_num_neigh_atoms: int = 200
pbc_xyz: List[bool] = [True, True, True]
sort: bool = False
torch_float_dtype: torch._C.dtype = torch.float32
has_virial: bool = False

mtp_level: int = 16
type_map: List[int] = extxyz_shifter.type_map
chebyshev_size: int = 8
rmax: float = rcut
rmin: float = 0.0
umax_num_neighs: int = umax_num_neigh_atoms
zbl_rmax: float = 0.0
zbl_rmin: float = 0.0

lr_start: float = 1e-1
lr_end: float = 1e-4
e_wgt_start: float = 0.02
e_wgt_end: float = 1.0
f_wgt_start: float = 1000.0
f_wgt_end: float = 0.1
v_wgt_start: float = 0.0
v_wgt_end: float = 0.0
lr_decay_step: int = 5000

torch.set_num_threads(16)
torch.manual_seed(4231)

lit_linear_mtp: LitLinearMtp = LitLinearMtp(mtp_level=mtp_level,
                                            type_map=type_map,
                                            #energy_shifts=energy_shifts,
                                            chebyshev_size=chebyshev_size,
                                            rmax=rmax,
                                            rmin=rmin,
                                            umax_num_neigh_atoms=umax_num_neigh_atoms,
                                            fit_virial=has_virial,
                                            zbl_rmax=zbl_rmax,
                                            zbl_rmin=zbl_rmin,
                                            zbl_cks_list=None,
                                            zbl_dks_list=None,
                                            lr_start=lr_start,
                                            lr_end=lr_end,
                                            e_wgt_start=e_wgt_start,
                                            e_wgt_end=e_wgt_end,
                                            f_wgt_start=f_wgt_start,
                                            f_wgt_end=f_wgt_end)

optimizer = FullParametersOptimizer(
    lit_linear_mtp=lit_linear_mtp,
    extxyz_path=PbTe_EXTXYZ_PATH,
    rcut=rcut,
    umax_num_neigh_atoms=umax_num_neigh_atoms,
    pbc_xyz=[True, True, True],
    sort=False,
    has_virial=False,
    torch_float_dtype=torch_float_dtype,
    device=torch.device(accelerator))

optimizer.fit()
