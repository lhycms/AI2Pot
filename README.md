# AI2Pot

Copyright © 2025-2026 Hanyu Liu.

AI2Pot is distributed under the GNU Lesser General Public License v3.0.

## What is AI2Pot?

**AI2Pot** (**Ab Initio and Artificial Intelligence POTential**) is a machine-learning interatomic-potential framework for materials modeling. It integrates multiple potential models, including the **Moment Tensor Potential** (**MTP**), **Neuroevolution Potential** (**NEP**), **Neuron Network Moment Tensor Potential** (**NNMTP**), within a unified PyTorch-based training and simulation workflow.

AI2Pot provides:

- Unified implementations of **MTP**, **NEP** and **NNMTP**
- High-performance C++ and CUDA operators
- PyTorch-based training, inference, and evaluation pipelines
- Interoperability with pymatgen, ASE, and LAMMPS
- A companion CLI tool, `AI2Pot-cli`, for data preprocessing, model training, and post-processing. (Repository: <https://github.com/lhycms/AI2Pot-cli>)

## Installation

AI2Pot is compiled against the PyTorch and NumPy versions installed in the active Python environment, so build isolation must be disabled when installing AI2Pot.

### Prerequisites

- Python 3.11 or 3.12
- PyTorch >= 2.2.0 (tested with 2.5.0; the wheel is built against the installed version)
- NumPy 2.x (baseline 2.2.6)
- CUDA toolkit with `nvcc` matching the CUDA version of PyTorch for GPU builds (tested with CUDA 12.4)
- A C/C++ toolchain (`gcc`/`g++`)
- `AI2Pot-cli` for the wheel workflow (installed from source; not on PyPI)

### Choose an installation method

| Method | When to use | Guide |
| --- | --- | --- |
| Build a wheel with `AI2Pot-cli` (recommended) | You want to build AI2Pot once and reuse the wheel in the same environment | [Install from a wheel](docs/install-wheel.md) |
| Install from source | You prefer a manual installation, or you are developing AI2Pot | [Install from source](docs/install-source.md) |
| Install the LAMMPS interface | You need `pair_style ai2pot` in LAMMPS | [Install LAMMPS with AI2Pot](docs/install-lammps.md) |

### Verify the installation

```bash
$ python -c "import ai2pot; print(ai2pot.__version__)"
$ python -c "import ai2pot.fromcc"
```

## LAMMPS Interface

AI2Pot provides a LAMMPS interface through `pair_style ai2pot`. See [Install LAMMPS with AI2Pot](docs/install-lammps.md) for build and usage instructions.

## Citation

[Liu2026] Hanyu Liu, Linggang Zhu, Xuanguang Zhang, Ning Yang, Jian Zhou, Zhimei Sun. [AI2Pot: A scalable and unified framework for machine-learning interatomic potential development and large-scale molecular dynamics simulations](https://arxiv.org/abs/2607.06969), arXiv preprint arXiv:2607.06969 (2026).
