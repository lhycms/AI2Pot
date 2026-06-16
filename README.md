# AI2Pot

Copyright © 2025 Hanyu Liu.

AI2Pot is distributed under the GNU General Public License v3.0.


## What is AI2Pot?

**AI2Pot** (**Ab Initio and Artificial Intelligence POTential**) is a machine-learning interatomic-potential framework for materials modeling.

AI2Pot provides:

* High-performance C++ and CUDA operators
* PyTorch-based training pipelines
* Interoperability with pymatgen, ASE, and LAMMPS

## Installation

### 1. Configure CUDA

For a CUDA-enabled installation, set `CUDA_HOME` to the CUDA toolkit compatible with the installed PyTorch version:

```bash
$ export CUDA_HOME=/path/to/cuda
$ export PATH="${CUDA_HOME}/bin:${PATH}"
$ export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
```

For example:

```bash
$ export CUDA_HOME=/usr/local/cuda-12.4
```

Check the CUDA compiler:

```bash
$ nvcc --version
```

### 2. Install PyTorch

Choose either the CPU or CUDA version.

#### CPU

```bash
$ python -m pip install torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

#### CUDA 12.4

```bash
$ python -m pip install torch==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124
```

Verify the PyTorch installation:

```bash
$ python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

### 3. Install AI2Pot

AI2Pot currently needs to compile against the NumPy and PyTorch packages installed in the active Python environment. Therefore, build isolation must be disabled.

Install the build backend first:

```bash
$ python -m pip install scikit-build-core==0.12.2
```

Then install AI2Pot:

```bash
$ CMAKE_BUILD_PARALLEL_LEVEL=16 python -m pip install -v --no-build-isolation .
```

`CMAKE_BUILD_PARALLEL_LEVEL` controls the number of parallel compilation jobs. Reduce this value if compilation consumes too much memory.

To use specific C, C++, and CUDA host compilers:

```bash
$ CMAKE_BUILD_PARALLEL_LEVEL=16 \
CC=/path/to/gcc \
CXX=/path/to/g++ \
CUDAHOSTCXX=/path/to/g++ \
python -m pip install -v --no-build-isolation .
```

<details>
<summary><strong>Developer installation and maintainer notes</strong></summary>

### Set the source-tree path

```bash
$ export AI2POT_PATH=/path/to/AI2Pot
```

### Configure and build the native libraries

```bash
$ cd "${AI2POT_PATH}/source"
$ mkdir -p build
$ cd build
```

Configure the project:

```bash
$ cmake \
    -DBUILD_TEST=ON \
    -DTORCH_OP=ON \
    -DUSE_CUDA=ON \
    -DUSE_OPENMP=ON \
    -DMAX_MTP_LEVEL=20 \
    ..
```

Compile and install:

```bash
$ cmake --build . --parallel 16
$ cmake --install .
```

Equivalent Make commands:

```bash
$ make -j16
$ make install
```

Install the Python package in editable mode:

```bash
$ cd "${AI2POT_PATH}"
$ python -m pip install -e .
```

### Optional CMake arguments

```text
-DCMAKE_C_COMPILER=/path/to/gcc
-DCMAKE_CXX_COMPILER=/path/to/g++
-DCMAKE_CUDA_HOST_COMPILER=/path/to/g++
-DCMAKE_INSTALL_PREFIX=$HOME/.ai2pot
```

### Current development-server configuration

```text
C compiler:         /data/app/gcc/11.3.0/bin/gcc
C++ compiler:       /data/app/gcc/11.3.0/bin/g++
CUDA host compiler: /data/app/gcc/11.3.0/bin/g++
```

Equivalent installation command:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=16 \
CC=/data/app/gcc/11.3.0/bin/gcc \
CXX=/data/app/gcc/11.3.0/bin/g++ \
CUDAHOSTCXX=/data/app/gcc/11.3.0/bin/g++ \
python -m pip install -v --no-build-isolation .
```

### Optional VS Code configuration

Add the repository path to `.vscode/settings.json` if Pylance cannot resolve the local package:

```json
{
  "python.analysis.extraPaths": [
    "/path/to/AI2Pot"
  ]
}
```

</details>

## License

AI2Pot is licensed under the GNU General Public License v3.0. See [`LICENSE`](LICENSE) for details.
