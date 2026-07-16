# Copyright (C) 2025 Hanyu Liu
#
# This file is part of AI2Pot.
#
# AI2Pot is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AI2Pot is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AI2Pot.  If not, see <http://www.gnu.org/licenses/>.


import os
import torch
import platform
import sys
import importlib.resources


# 1. Device
IS_DARWIN: bool = platform.system() == "Darwin"


# 2. Environment Variable
#       - AI2POT_PATH
#       - ai2pot_lib_dir
def _get_ai2pot_path():
    ai2pot_path: str = os.getenv("AI2POT_PATH")
    if (ai2pot_path):
        return ai2pot_path
    else:
        return str(importlib.resources.files("ai2pot"))

def _get_ai2pot_lib_dir():
    ai2pot_path: str = os.getenv("AI2POT_PATH")
    if (ai2pot_path):
        return os.path.join(ai2pot_path, "source", "build", "lib")
    else:
        return str(importlib.resources.files("ai2pot").joinpath("_lib"))

ai2pot_path: str = _get_ai2pot_path()
ai2pot_lib_dir:str = _get_ai2pot_lib_dir()



# 3. Register CPython operators
if os.getenv("AI2POT_PATH"):
    for lib_name in ("nblist", "gst"):
        sys.path.append(os.path.join(ai2pot_lib_dir, lib_name))
else:
    sys.path.append(ai2pot_lib_dir)
import nblist
import gst



# 4. Register Torch opeartors
if not IS_DARWIN:
    ### Part 2.1. fituils
    fitutils_lib_dir:str = os.path.join(ai2pot_lib_dir, "descriptor", "fitutils") if os.getenv("AI2POT_PATH") else ai2pot_lib_dir
    fitutils_bind_cpu_so_path: str = os.path.join(fitutils_lib_dir, "libfitutils_bind_cpu.so")
    torch.ops.load_library(fitutils_bind_cpu_so_path)
    if torch.cuda.is_available():
        fitutils_bind_cuda_so_path = os.path.join(fitutils_lib_dir, "libfitutils_bind_cuda.so")
        torch.ops.load_library(fitutils_bind_cuda_so_path)
    # name
    targetStatisticsOp = torch.ops.fitutils.targetStatisticsOp
    allTypeDescriptorsStatisticsOp = torch.ops.fitutils.allTypeDescriptorsStatisticsOp
    eachTypeDescriptorsStatisticsOp = torch.ops.fitutils.eachTypeDescriptorsStatisticsOp
    allTypeDescriptorsMaxminOp = torch.ops.fitutils.allTypeDescriptorsMaxminOp

    ### Part 2.2. deepmd
    deepmd_lib_dir:str = os.path.join(ai2pot_lib_dir, "descriptor", "deepmd") if os.getenv("AI2POT_PATH") else ai2pot_lib_dir
    envMatrixOp_bind_so_path:str = os.path.join(deepmd_lib_dir, "libenvMatrixOp_bind.so")
    torch.ops.load_library(envMatrixOp_bind_so_path)
    # name `envMatrixOp`
    envMatrixOp = torch.ops.deepmd.EnvMatrixOp


    ### Part 2.3. mtpr
    mtpr_lib_dir:str = os.path.join(ai2pot_lib_dir, "descriptor", "mtpr") if os.getenv("AI2POT_PATH") else ai2pot_lib_dir
    mtpr_bind_so_path: str = os.path.join(mtpr_lib_dir, "libmtpr_bind.so")
    torch.ops.load_library(mtpr_bind_so_path)
    # name
    coeffsSchmidtOrthOp = torch.ops.mtpr.CoeffsSchmidtOrthOp
    linMatrixLinVectorOp = torch.ops.mtpr.LinMatrixLinVectorOp
    mtpParamOp = torch.ops.mtpr.mtpParamOp
    mtpBasisOp = torch.ops.mtpr.mtpBasisOp
    linearMtpToLossOp = torch.ops.mtpr.linearMtpToLossOp
    linearMtpToEFLossOp = torch.ops.mtpr.linearMtpToEFLossOp
    linearMtpToEFVOp = torch.ops.mtpr.linearMtpToEFVOp
    linearMtpToEFOp = torch.ops.mtpr.linearMtpToEFOp
    linearMtpToEsitesOp = torch.ops.mtpr.linearMtpToEsitesOp
    linearMtpToDescriptorsOp = torch.ops.mtpr.linearMtpToDescriptorsOp
    # Action
    torch.ops.mtpr.set_ai2pot_path(ai2pot_path)


    ### Part 2.4. nnmtp
    nnmtp_lib_dir:str = os.path.join(ai2pot_lib_dir, "descriptor", "nnmtp") if os.getenv("AI2POT_PATH") else ai2pot_lib_dir
    nnmtp_bind_so_path: str = os.path.join(nnmtp_lib_dir, "libnnmtp_bind.so")
    torch.ops.load_library(nnmtp_bind_so_path)
    # name
    mtpParamOp = torch.ops.nnmtp.mtpParamOp
    nnMtpToEFLossOp = torch.ops.nnmtp.nnMtpToEFLossOp
    nnMtpToLossOp = torch.ops.nnmtp.nnMtpToLossOp
    nnMtpToEFOp = torch.ops.nnmtp.nnMtpToEFOp
    nnMtpToEFVOp = torch.ops.nnmtp.nnMtpToEFVOp
    nnMtpToDescriptorsOp = torch.ops.nnmtp.nnMtpToDescriptorsOp
    nnMtpToEsitesOp = torch.ops.nnmtp.nnMtpToEsitesOp
    # Action
    torch.ops.nnmtp.set_ai2pot_path(ai2pot_path)


    ### Part 2.5. nep
    nep_lib_dir:str = os.path.join(ai2pot_lib_dir, "descriptor", "nep") if os.getenv("AI2POT_PATH") else ai2pot_lib_dir
    nep_bind_cpu_so_path: str = os.path.join(nep_lib_dir, "libnep_bind_cpu.so")
    torch.ops.load_library(nep_bind_cpu_so_path)
    if torch.cuda.is_available():
        nep_bind_cuda_so_path: str = os.path.join(nep_lib_dir, "libnep_bind_cuda.so")
        torch.ops.load_library(nep_bind_cuda_so_path)
    # name
    nepToEFOp = torch.ops.nep.nepToEFOp
    nepToEFVOp = torch.ops.nep.nepToEFVOp
    nepToEFLossOp = torch.ops.nep.nepToEFLossOp
    nepToLossOp = torch.ops.nep.nepToLossOp
    nepToDescriptorsOp = torch.ops.nep.nepToDescriptorsOp


    ### Part 3. fvt
    fvt_sr_lib_dir: str = os.path.join(ai2pot_lib_dir, "fvt") if os.getenv("AI2POT_PATH") else ai2pot_lib_dir
    fvt_sr_bind_so_path: str = os.path.join(fvt_sr_lib_dir, "libfvt_sr_op_bind.so")
    torch.ops.load_library(fvt_sr_bind_so_path)
    # name
    forceSrOp = torch.ops.fvt.ForceSrOp
    virialSrOp = torch.ops.fvt.VirialSrOp
