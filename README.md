# `AI2Pot`
Copyright (2025) Hanyu Liu. This is AI2Pot software package. This software is distributed under the GNU General Public License (GPL) version 3.

# What is AI2Pot
AI2Pot (Ab Initio and Artificial Intelligence POTential) is a machine learning interatomic potential framework for materials modeling.

The framework emphasizes:
- High-performance C++/CUDA operators
- PyTorch-based training pipeline
- Interoperability with pymatgen, ASE, and LAMMPS

# Installation
```shell
# 1. CUDA
export CUDA_HOME=/path/to/pytorch/cuda  # /usr/local/cuda-11.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 2. Torch
## 2.1. CPU
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
## 2.2. GPU
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# 3. 
# 3.1.
export AI2POT_PATH=
# 3.2
# -DCMAKE_CXX_COMPILER=/data/app/gcc/11.3.0/bin/g++
# -DCMAKE_C_COMPILER=/data/app/gcc/11.3.0/bin/gcc
# -DCMAKE_CUDA_HOST_COMPILER=/data/app/gcc/11.3.0/bin/g++
# -DCMAKE_INSTALL_PREFIX="$HOME/.ai2pot"
cmake -DBUILD_TEST=yes -DTORCH_OP=yes -DUSE_CUDA=yes -DUSE_OPENMP=yes -DMAX_MTP_LEVEL=20 ..
$ make -j
$ make install

# 4.
## 4.1. User
$ pip install .
## 4.2. developer
$ pip3 install -e .
# .vscode/settings.json
#{
#  "python.analysis.extraPaths": [
#    "/data/home/liuhanyu/mycode/AI2Pot"
#  ]
#}
```

# 5. New mode
```shell
$CC=/data/app/gcc/11.3.0/bin/gcc \
 CXX=/data/app/gcc/11.3.0/bin/g++ \
 CUDAHOSTCXX=/data/app/gcc/11.3.0/bin/g++ \
 pip install -e .
```
