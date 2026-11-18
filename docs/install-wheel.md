[Back to README](../README.md) | **Wheel install** | [Source install](install-source.md) | [LAMMPS interface](install-lammps.md)

# Install AI2Pot from a Wheel

`AI2Pot-cli` can build an AI2Pot wheel matched to the PyTorch and CUDA versions in the current Python environment. The wheel can then be installed directly, without recompiling AI2Pot in every environment that uses the same versions.

There is no fixed PyTorch or CUDA version requirement: the wheel is built against the versions installed in the active environment. PyTorch 2.5.0 with CUDA 12.4 is the tested baseline, and the project metadata requires PyTorch >= 2.2.0 and Python 3.11–3.12.

## When to use this method

- You want a reusable wheel for machines with the same Python, PyTorch, and CUDA versions.
- You do not need an editable or developer installation of AI2Pot.

## Prerequisites

The active Python environment must contain:

- Python 3.11 or 3.12
- PyTorch >= 2.2.0 (tested with 2.5.0); `torch` must be importable
- NumPy 2.x (baseline 2.2.6)
- Build dependencies:
  ```bash
  $ python -m pip install -U pip setuptools wheel
  $ python -m pip install scikit-build-core==0.12.2 cmake==4.3.2 pybind11==2.11.1
  $ python -m pip install numpy==2.2.6
  # Install the PyTorch build for your environment; see:
  # https://pytorch.org/get-started/previous-versions/
  ```
- For CUDA builds, a CUDA toolkit whose major version matches `torch.version.cuda` (CUDA 12.4 for the cu124 PyTorch builds), including `nvcc`.
- The AI2Pot source tree:
  ```bash
  $ git clone https://github.com/lhycms/AI2Pot.git
  ```

## Install AI2Pot-cli

`AI2Pot-cli` is not distributed on PyPI. Install it from source:

```bash
$ git clone https://github.com/lhycms/AI2Pot-cli.git
$ cd AI2Pot-cli
$ python -m pip install .
$ ai2pot-cli --version
```

## Build the wheel

Start the interactive CLI and choose `1) Build AI2Pot Wheel`:

```bash
$ ai2pot-cli
```

The build prompts are:

| Prompt | Default | Notes |
| --- | --- | --- |
| AI2Pot source dir | `$AI2POT_SOURCE` or current directory | Path to the AI2Pot source tree |
| Build mode | `auto` | `auto` uses CUDA if PyTorch has CUDA; `cpu` forces a CPU-only build |
| CUDA architectures | `auto` | Auto-detect, or enter a comma-separated list (e.g. `80-real,86-real,89-real,90-real,90-virtual`) |
| CUDA toolkit path | `auto` | Only used for CUDA builds |
| CC / CXX | detected `gcc` / `g++` | Compilers used for the build |
| Parallel jobs | `16` | Compilation parallelism |
| Output root | `dist` | Relative to the AI2Pot source tree |

The wheel is written to `<AI2Pot-source>/dist/<variant>/`, where `<variant>` records the build environment, for example `torch2.5-cu124-py311` or `torch2.5-cpu-py311`.

## Install and verify

```bash
$ python -m pip install --force-reinstall <wheel>
$ python -c "import ai2pot; print(ai2pot.__version__)"
$ python -c "import ai2pot.fromcc"
```

## Compatibility notes

- The wheel is built from the active environment; its Python, PyTorch, and CUDA combination is recorded in the variant, so rebuild it for a different environment.
- The CUDA toolkit major version must match the CUDA version used by PyTorch.
- CPU-only PyTorch or `cpu` mode produces a CPU-only wheel.
