#!/bin/bash
# Installs every dependency from pyproject.toml that pixi itself already treats as a PyPI
# dependency (i.e. everything except the small conda-native core in recipe.yaml's
# requirements/run: python, numpy, the R stack + rpy2, mrtrix3, freeimage, libstdcxx-ng,
# and the Qt/GUI stack except PyQt6-WebEngine — see the comment at the top of recipe.yaml
# for why that split exists and how it was verified). Runs once, right after `conda`/`mamba`
# links nvitk into the target env.
#
# Version pins below are copied verbatim from pyproject.toml / [tool.pixi.*] — do not
# relax them here even where a conda package would technically satisfy the constraint.
set -euo pipefail

PIP="$PREFIX/bin/pip"
LOG="$PREFIX/.nvitk-post-link.log"

log() { echo "[nvitk post-link] $*" | tee -a "$LOG"; }

log "starting pip-only dependency install into $PREFIX"

log "installing core (data/plotting/logging)..."
"$PIP" install --no-cache-dir \
  "scipy>=1.15.3" "pandas>=2.3.2" "polars>=1.43.2" "pyarrow>=21.0.0" "click>=8.2.1" \
  "jupyter>=1.1.1" "ipywidgets>=8.1.5" "openpyxl>=3.1.5" "lazy-loader>=0.4" \
  >>"$LOG" 2>&1

log "installing visualization/plotting/logging..."
"$PIP" install --no-cache-dir \
  "matplotlib>=3.8.0" "plotly>=6.3.0" "kaleido>=1.0.0" "pyvista>=0.48.0" \
  "seaborn>=0.13.2" "rich>=14.1.0" "trame>=3.6.0" "trame-vtk>=2.5.8,<2.11.9" \
  >>"$LOG" 2>&1

log "installing image IO..."
"$PIP" install --no-cache-dir \
  "imageio>=2.37.0" "pillow>=11.0.0" "nibabel>=5.3.2" "tifffile>=2025.5.26" \
  "nd2>=0.10.3" "dicom2nifti>=2.6.1" "pylibjpeg>=2.0.1" "h5py>=3.16.0" \
  >>"$LOG" 2>&1

log "installing general image processing (simpleitk/antspyx: see recipe.yaml for why these can't be conda run-deps)..."
"$PIP" install --no-cache-dir \
  "scikit-image>=0.25.2" "opencv-python>=4.13.0.92" "vtk>=9.6.1" "SimpleITK==2.2.1" "antspyx>=0.6.3" \
  >>"$LOG" 2>&1

log "installing neuroimaging/segmentation tools..."
"$PIP" install --no-cache-dir \
  "nilearn>=0.11.0" "nipype>=1.11.0" "gudhi>=3.12.0" "antspynet>=0.3.2" \
  "totalsegmentator>=2.11.0" "nnunetv2>=2.6.2" "fireants>=1.5.0" \
  >>"$LOG" 2>&1

log "installing ML/statistics (pymer4: --no-deps, its pyarrow pin conflicts with ours above)..."
"$PIP" install --no-cache-dir \
  "pingouin>=0.6.1" "networkx>=3.3" "scikit-learn>=1.7.1" "scikit-posthocs>=0.14.0" \
  "semopy>=2.3.11" "great-tables" "formulae" \
  >>"$LOG" 2>&1
"$PIP" install --no-cache-dir --no-deps "pymer4>=0.9.2" \
  >>"$LOG" 2>&1

log "installing API/database..."
"$PIP" install --no-cache-dir \
  "xnat>=0.7.2" "pyyaml>=6.0" "paramiko>=3.4.0" "fastapi>=0.110.0" \
  "uvicorn>=0.27.0" "keyring>=24.0.0" \
  >>"$LOG" 2>&1

log "installing PyQt6-WebEngine (no conda-forge package yet; --no-deps keeps conda pyqt6)..."
"$PIP" install --no-cache-dir --no-deps "PyQt6-WebEngine>=6.11.0" \
  >>"$LOG" 2>&1

log "installing CUDA 13 PyTorch stack from the cu130 wheel index..."
"$PIP" install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "torch>=2.9.0" "torchvision>=0.24.0" \
  >>"$LOG" 2>&1
  
log "installing PyTorch/nnSSL extras..."
"$PIP" install --no-cache-dir \
  "lightning-bolts>=0.7.0" "pytorch-msssim>=1.0.0" "torchio>=1.2.1" \
  "lpips>=0.1.4" "loguru==0.7.3" "wandb==0.28.1" \
  >>"$LOG" 2>&1

log "installing remaining CUDA 13 GPU stack (cupy/cutensor/nccl)..."
"$PIP" install --no-cache-dir \
  "cupy-cuda13x>=13.6.0" "cutensor-cu13>=2.6.0" "nvidia-nccl-cu13>=2.27.7" \
  >>"$LOG" 2>&1

log "installing pyradiomics (no-build-isolation, matching pixi's handling)..."
"$PIP" install --no-cache-dir --no-build-isolation \
  "pyradiomics==3.0.1" \
  >>"$LOG" 2>&1

log "installing pydicom (no-build-isolation, matching pixi's handling)..."
"$PIP" install --no-cache-dir --no-build-isolation \
  "pydicom==3.0.1" \
  >>"$LOG" 2>&1

log "done."
