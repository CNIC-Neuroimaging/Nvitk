# Nvitk

## Neuro-Vascular Imaging ToolKit

## Installation (conda recommended)

This project depends on a few packages that are **only available via conda** (e.g. `mrtrix3`, `freeimage`, and `libstdcxx-ng`). For that reason, the recommended install path is:

```bash
conda env create -f environment.yml
conda activate nvitk
```

The environment file will install the conda dependencies and then install this repo via pip. For more help, type the command `pyhelp` in a terminal once installed to list the available client commands.

> ⚠️ **Note:** CPU-only installation, is you want GPU-enabled skip this installation step.

### GPU / CUDA Installation

CUDA installation cannot be auto-detected from env file, so the above installation will install it cpu-only. For installing it gpu-enabled, you have two options:

- **Conda (recommended)**:
  - `environment-gpu.yml`: full GPU environment for this repo (CUDA-enabled `torch` + `cupy-cuda12x` + conda-only deps + installs `nvitk`)
  - `gpu-base.yml`: a smaller base GPU environment if `environment-gpu.yml` does not work (does not install `nvitk`, you'll have to then manually install the specific conda requirements under `environment.yml` and isntall `nvitk` as `pip install -e .`)
- **pip extras**:
  - `pip install -e ".[gpu]"` installs `cupy-cuda12x` and `torch`/`torchvision` (pip approach if you cannot configure CuPy's or PyTorch’s CUDA index with the cuda-recommended approach).
  
