# Installation

## Install via conda (recommended)

`nvitk` is published on [Anaconda.org](https://anaconda.org/cnic/nvitk) under the `cnic`
channel with the full `nvitk[all]` feature set (GUI, GPU/CUDA 13, and R-based statistics)
bundled in — there is no separate CPU/GPU profile to choose.

```bash
conda config --add channels conda-forge
conda config --add channels bioconda
conda config --add channels mrtrix3
conda config --add channels ejolly
conda config --add channels cnic
conda config --set channel_priority strict

conda create -n nvitk-env nvitk
conda activate nvitk-env
```

```{note}
This pulls in the CUDA 13 PyTorch/CuPy stack automatically as part of installation.
Packages with no conda equivalent (e.g. `totalsegmentator`, `nnunetv2`, `antspyx`) are
installed via `pip` through a post-link step — see
[`recipe/post-link.sh`](https://github.com/ignacio-ms/Nvitk/blob/main/recipe/post-link.sh)
for the exact list.
```

```{warning}
Use `conda create`/`conda install`, **not** `mamba create`/`mamba install`. `mamba` does
not run post-link scripts (a known gap across mamba/micromamba), so a mamba-based install
silently skips the entire pip-only dependency stack above — the command succeeds, but
`import nvitk` then fails. Plain `conda` (even with the libmamba solver enabled) works
correctly, since conda's own transaction executor still runs the post-link step regardless
of which solver resolved the environment.
```

Run {doc}`pyhelp <api/cli-catalog>` after activation for an interactive catalog of every
CLI tool (`pyhelp --no-interactive` for CI/scripts).

## Development install (pixi)

Contributors working from a clone should use [pixi](https://pixi.sh), which
`pyproject.toml` already configures for every environment below:

```bash
git clone https://github.com/ignacio-ms/Nvitk.git && cd Nvitk
pixi install               # core only, CPU
pixi install -e gui        # + Napari workbench
pixi install -e gpu        # + gui, + CUDA 13 torch/cupy stack
pixi install -e stats      # + gui, + R/rpy2/pymer4 stack
pixi install -e all        # gui + gpu + stats
pixi install -e dev        # + pytest, sphinx toolchain (for docs/testing)

pixi shell -e gpu           # activate, like `conda activate`
pixi run -e all pyhelp --no-interactive
```

| Environment | Includes |
|---|---|
| `default` | Core toolkit only — I/O, segmentation, registration, measurements, pipelines |
| `gui` | + Napari, magicgui, superqt, PyQt6-WebEngine |
| `gpu` | + `gui`, + CUDA 13 `torch`/`torchvision`/`cupy`/`cutensor`/`nccl` |
| `stats` | + `gui`, + R/`rpy2`/`pymer4`/mixed-model tooling |
| `all` | `gui` + `gpu` + `stats` |
| `dev` | pytest, sphinx + autoapi + sphinx-click (docs/testing only) |

### GPU stack

CUDA 13. `torch`/`torchvision` from the `cu130` PyPI wheel index, plus `cupy-cuda13x`,
`cutensor-cu13`, `nvidia-nccl-cu13`. Requires **Python 3.11**. Conda channels needed for the
pixi `gpu` feature: `nvidia`, `pytorch` (public) and `morpheme`, `mosaic`.

## External prerequisites (not installed by conda or pixi)

Separately licensed or with no package-manager distribution — install these yourself and
confirm they're on `PATH`/the relevant environment variable. nvitk's CLIs notify you when a
needed-but-missing external tool is required for the operation you're running.

| Dependency | Version | Used by | Install |
|---|---|---|---|
| **FSL** | 6.0.7.19 | `nvitk-flirt`, registration CLIs, the {doc}`QVTPy pipeline <pipelines/qvtpy>` | [fsl.fmrib.ox.ac.uk/fsldownloads](https://fsl.fmrib.ox.ac.uk/fsldownloads/) — official `fslinstaller.py`; non-commercial license required |
| **FreeSurfer** | 7.x | Desikan atlas lookups (`$FREESURFER_HOME`) | [surfer.nmr.mgh.harvard.edu](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) — free registration required |
| **SPM12** | r7771 | The QVT+ MATLAB reference pipeline | [fil.ion.ucl.ac.uk/spm/software/spm12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) |

```{seealso}
The full CLI catalog is in {doc}`api/cli-catalog`, and every command's `--help` output is
also embedded throughout the {doc}`api/index`, {doc}`gui/index`, and {doc}`pipelines/index`
pages.
```
