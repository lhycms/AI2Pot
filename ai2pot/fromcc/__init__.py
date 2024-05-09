import os
import torch
import sys

ai2pot_root_dir:str = "/data/home/liuhanyu/hyliu/code/AI2Pot"  #Path(__file__).parent.absolute().parent.absolute().parent.absolute()
ai2pot_source_build_lib_dir:str = os.path.join(ai2pot_root_dir, "source", "build", "lib")

### Part 1 . nblist
nblist_bind_dir:str = os.path.join(ai2pot_root_dir, "source", "nblist", "bind")
nblist_bind_gen_dir:str = os.path.join(nblist_bind_dir, "gen")
sys.path.append(nblist_bind_gen_dir)
import nblist


### Part 2.1. deepmd
deepmd_lib_dir:str = os.path.join(ai2pot_source_build_lib_dir, "descriptor", "deepmd")
envMatrixOp_bind_so_path:str = os.path.join(deepmd_lib_dir, "libenvMatrixOp_bind.so")
torch.ops.load_library(envMatrixOp_bind_so_path)
# name `envMatrixOp`
envMatrixOp = torch.ops.deepmd.EnvMatrixOp

### Part 2.2. mtpr
mtpr_lib_dir:str = os.path.join(ai2pot_source_build_lib_dir, "descriptor", "mtpr")
mtpr_bind_so_path: str = os.path.join(mtpr_lib_dir, "libmtpr_bind.so")
torch.ops.load_library(mtpr_bind_so_path)
# name
mtpParamOp = torch.ops.mtpr.mtpParamOp

### Part 3. fvt
fvt_sr_lib_dir: str = os.path.join(ai2pot_source_build_lib_dir, "fvt")
fvt_sr_bind_so_path: str = os.path.join(fvt_sr_lib_dir, "libfvt_sr_op_bind.so")
torch.ops.load_library(fvt_sr_bind_so_path)
# name
forceSrOp = torch.ops.fvt.ForceSrOp
virialSrOp = torch.ops.fvt.VirialSrOp