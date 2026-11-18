[Back to README](../README.md) | [Wheel install](install-wheel.md) | [Source install](install-source.md) | **LAMMPS interface**

# Install LAMMPS with AI2Pot

## Install AI2Pot-lammps

1. Copy the AI2Pot interface files into the LAMMPS source directory:

```bash
$ cp AI2Pot/interface/lammps/AI2POT lammps/src
$ cp AI2Pot/interface/lammps/Makefile.mpi lammps/src/MAKE/
```

2. Build LAMMPS with the AI2POT package:

```bash
$ cd lammps/src
$ make yes-AI2POT
$ make -j 16 mpi \
    TORCH_ROOT=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))") \
    AI2POT_ROOT=$(python -c "import ai2pot, os; print(os.path.dirname(ai2pot.__file__))")
```

## AI2Pot pair style usage

```txt
pair_style ai2pot ./ai2pot_libtorch.pt
pair_coeff * * Ge Sb Te
```
