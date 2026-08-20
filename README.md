# Nvitk | Neuro-Vascular Imaging ToolKit

[![Conda Version](https://anaconda.org/cnic/nvitk/badges/version.svg)](https://anaconda.org/cnic/nvitk) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](pyproject.toml)

**nvitk** is a research toolkit for neurological and vascular biomedical image processing — CT, PET, and MRI/MRA — covering I/O, filtering, restoration, segmentation, registration, imaging metrics, mesh reconstruction, statistics, and full research pipelines, with an optional Napari-based GUI and a NumPy/CuPy dual backend for CPU/GPU execution.
 
Developed at CNIC for intracranial and vascular research, including specific image processing pipelines such as 4D flow MRI Hemodynamics, TOF Morphometrics, whole-body PET/CT quantification.

> Full API reference and step-by-step tutorials: **[ignacio-ms.github.io/Nvitk](https://ignacio-ms.github.io/Nvitk/)**
> *(live once the one-time GitHub Pages setup step is done — see `docs/`)*

## Contents
 
1. [Installation](#1-installation)
2. [Tool Catalog](#2-tool-catalog)
3. [Slicer Modules](#3-slicer-modules)
4. [Quickstart](#4-quickstart)
5. [Contributing · Citation · License](#5-contributing--citation--license)
---

## 1. Installation

### Install via conda (recommended)

`nvitk` is published on Anaconda.org under the `cnic` channel with the full `nvitk[all]` feature set (GUI, GPU/CUDA 13, and R-based stats) bundled in no separate CPU/GPU profile to pick. 

```bash
conda config --add channels conda-forge
conda config --add channels bioconda
conda config --add channels mrtrix3
conda config --add channels ejolly

conda config --add channels cnic
conda config --set channel_priority strict

conda create -n nvitk-env nvitk
conda activate nvitk-env

conda install nvitk
```

> **Note:** This pulls in the CUDA 13 PyTorch/CuPy stack automatically as part of installation. Packages with no conda equivalent (e.g. `totalsegmentator`, `nnunetv2`, ...) are installed via `pip` through a post-link step — see `recipe/post-link.sh` for the exact list.
>
> **Use `conda create`/`conda install`, not `mamba create`/`mamba install`.** `mamba` does not run post-link scripts (a known gap across mamba/micromamba), so a mamba-based install silently skips the entire pip-only dependency stack above — the command succeeds, but `import nvitk` then fails. Installing with plain `conda` (even with the libmamba solver enabled) works correctly, since conda's own transaction executor still runs the post-link step regardless of which solver resolved it.

Run **`pyhelp`** after activation for an interactive catalog of every CLI tool (`pyhelp --no-interactive` for CI/scripts).

### Development install (pixi)

Contributors working from a clone should use [pixi](https://pixi.sh), which `pyproject.toml` already configures every environment below:

```bash
git clone <repo-url> && cd nvitk
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
`cutensor-cu13`, `nvidia-nccl-cu13`. Requires **Python 3.11**. Conda channels needed: 
`nvidia`, `pytorch` (public) and `morpheme`, `mosaic`.
 
### External prerequisites (not installed by conda or pixi)
 
Separately licensed or with no package-manager distribution — install these yourself and
confirm they're on `PATH`/the relevant env var. nvitk's CLIs will notify for any needed-not-installed software.
 
| Dependency | Version | Used by | Install |
|---|---|---|---|
| **FSL** | 6.0.7.19 | `nvitk-flirt`, registration CLIs, `QVTPy` pipeline | [fsl.fmrib.ox.ac.uk/fsldownloads](https://fsl.fmrib.ox.ac.uk/fsldownloads/) — official `fslinstaller.py`; non-commercial license required |
| **FreeSurfer** | 7.x | Desikan atlas lookups (`$FREESURFER_HOME`) | [surfer.nmr.mgh.harvard.edu](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall) — free registration required |
| **SPM12** | r7771 | QVT+ pipeline, via MATLAB Runtime | [fil.ion.ucl.ac.uk/spm/software/spm12](https://www.fil.ion.ucl.ac.uk/spm/software/spm12/) |
<!-- | **MATLAB Runtime** | R2023a | QVT+ pipeline | [mathworks.com/products/compiler/matlab-runtime](https://www.mathworks.com/products/compiler/matlab-runtime.html) — MathWorks EULA |
| **MATLAB (full)** | R2025b (`matlabengine==25.2.2`, `pip install "nvitk[matlab]"`) | Python↔MATLAB engine bridge | requires a full MATLAB R2025b install; MathWorks EULA | -->
 
<!-- > The MATLAB Runtime (R2023a) and full MATLAB (R2025b) entries are two separate
> requirements for two separate purposes, not a typo. -->
 
---

## 2. Tool Catalog
 
Every command below is a registered entry point; run any with `--help`, or browse them
all interactively with `pyhelp`. All tools are also individually included on the main GUI `nvitk-gui`.
 
### Conversion
 
| Command | Purpose |
|---|---|
| `dcm2nii` | DICOM → NIfTI |
| `stl2nifti` | Surface mesh → labelmap/NIfTI |
| `phase2volume` | Phase-contrast MRI → velocity volume and derivates |
| `nikon2nifti` | Nikon microscopy → NIfTI |
 
### Segmentation
 
| Command | Purpose |
|---|---|
| `nvitk-totalseg` | TotalSegmentator wrapper (local / SGE, CPU / GPU) |
| `nvitk-eicab` | eICAB TOF / Circle-of-Willis segmentation (CPU) |
| `nvitk-seg` | General segmentation entry point |
 
### Registration
 
| Command | Purpose |
|---|---|
| `nvitk-flirt` | FSL FLIRT wrapper |
| `nvitk-ants` | ANTsPy registration |
| `nvitk-fireants` | FireANTs (GPU-accelerated) registration |
 
### Image module CLIs
 
| Command | Purpose |
|---|---|
| `nvitk-morph` | Morphology — dilate, centerline, siphon-correct, ... |
| `nvitk-restore` | Restoration — bilateral denoising, bias field correction, ... |
| `nvitk-filter` | Filters — sliding-threshold, vesselness, ... |
| `nvitk-measure` | Metrics — volume, SUV, Dice, surface, hemodynamics, morphometrics, ... |
| `nvitk-transform` | Resample, isotropy, oblique slice, ... |
 
```bash
nvitk-restore bilateral -i pet.nii.gz -o pet_denoised.nii.gz --backend gpu
nvitk-morph dilate -i mask.nii.gz -o mask_dil.nii.gz --footprint 2
nvitk-filter sliding-threshold -i cd.nii.gz -o mask.nii.gz --dim 3d
nvitk-measure volume -i mask.nii.gz -o vol.txt
nvitk-transform resample -i pet.nii.gz -r ct.nii.gz -o pet_on_ct.nii.gz
```
 
Module CLIs accept `-i`/`-o` and optional `--submit local|sge` — configure cluster
defaults in `.nvitk/sge.json`.

> *(To specify json-specific configuration schemas.)*
 
### Pipelines — PESA-Fat
 
| Command | Purpose |
|---|---|
| `nvitk-pesa-fat` | Batch driver |
| `nvitk-pesa-fat-ctpet` | CT/PET pipeline |
| `nvitk-pesa-fat-dixon` | Dixon pipeline |
| `nvitk-pesa-fat-qc` | QC stage |
| `nvitk-pesa-fat-qc-portal` | QC review portal |
| `nvitk-pesa-fat-sync-measurements` | Sync measurements |
 
### Pipelines — PESA-Brain (qvtpy)
 
| Command | Purpose |
|---|---|
| `nvitk-qvtpy` | 4D-flow hemodynamics pipeline |
| `nvitk-qvtpy-xnat-upload` | XNAT upload |
| `nvitk-qvtpy-autoqc` | Automated QC |
 
<!-- ### Database / Sync
 
| Command | Purpose |
|---|---|
| `nvitk-xnat-sync` | XNAT dataset sync |
| `nvitk-xnat-pipeline-sync` | XNAT pipeline-resource sync | -->
 
### GUI
 
| Command | Purpose |
|---|---|
| `nvitk-gui` | Napari workbench — Tool catalog dock (imaging | mesh), pipeline export, ... |
| `nvitk-statsmodels` | Statistical modeling workbench — Python and R models |
 
### General
 
| Command | Purpose |
|---|---|
| `pyhelp` | Interactive or flat catalog of every command above |
 
```bash
pyhelp                    # interactive tree; Enter selects a command and prints --help
pyhelp --no-interactive   # full static tree
```
 
---

## 3. Slicer Modules
 
Scripted 3D Slicer modules live under `slicer/`, loaded via Slicer's **Additional module
paths** rather than `pip`/`conda` — they don't import `nvitk` directly, so they run
without the full dependency stack.
 
| Module | Purpose |
|---|---|
| MouseTOFCoW | Semi-automatic segmentation of the main Circle of Willis artery trees for Mouse TOF MRI |
| MouseTOFMorphometrics | Morphometric characterization of the CoW arteries for Mouse/Human TOF MRI |
 
---
 
## 4. Quickstart
 
```python
from nvitk.core import setup
from nvitk.io import imread, imsave
from nvitk.measure import volume_cc, Measurer
 
setup(globals())  # backend-aware np / ndi / scipy
 
img = imread("study/pet", force_type="dicom", backend="gpu")
mask = imread("mask.nii.gz")
 
print(volume_cc(mask))
summary = Measurer(img, mask).volume() | Measurer(img, mask).suv(kinds=("bw",))
imsave("out/pet_copy.nii.gz", img)
```
 
```bash
nvitk-restore bilateral -i pet.nii.gz -o pet_denoised.nii.gz --backend gpu
nvitk-measure volume -i mask.nii.gz -o vol.txt
```
 
---
 
## 5. Contributing · Citation · License
 
**Contributing** — issues and PRs welcome; match the repo's docstring/logging/backend
conventions (see `CONTRIBUTING.md`, once added) before submitting.
 
**Citation** — no publication yet; cite this repository if you use nvitk in your research.
 
**License** — see [`LICENSE`](LICENSE).
