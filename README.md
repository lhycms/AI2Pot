# `AI2Pot`
Copyright (2025) Hanyu Liu. This is AI2Pot software package. This software is distributed under the GNU General Public License (GPL) version 3.

# What is AI2Pot
Ab Initio and Artificial Intelligence POTential framework, applied in materials and molecular systems, with the ability to predict scalars and tensors.

# Installation
```shell
# 1. CUDA
export CUDA_HOME=/path/to/pytorch/cuda  # /usr/local/cuda-11.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# 2. 
export AI2POT_PATH=
cmake -DBUILD_TEST=yes -DTORCH_OP=yes -DUSE_CUDA=yes -DMAX_MTP_LEVEL=16 ..
```