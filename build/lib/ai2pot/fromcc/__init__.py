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


### Part 2. deepmd
deepmd_lib_dir:str = os.path.join(ai2pot_source_build_lib_dir, "descriptor", "deepmd")
envMatrixOp_bind_so_path:str = os.path.join(deepmd_lib_dir, "libenvMatrixOp_bind.so")
torch.ops.load_library(envMatrixOp_bind_so_path)
# name `envMatrixOp`
envMatrixOp = torch.ops.deepmd.EnvMatrixOp

force_sr_op_bind_so_path: str = os.path.join(ai2pot_source_build_lib_dir, "fvt", "libforce_sr_op_bind.so")
torch.ops.load_library(force_sr_op_bind_so_path)
forceSrOp = torch.ops.ForceSrOp
