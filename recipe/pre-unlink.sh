#!/bin/bash
# Mirrors post-link.sh: removes every pip-only package it installed, so `conda remove
# nvitk` doesn't strand pip-managed packages in the environment. Runs before nvitk itself
# is unlinked. Best-effort (no `set -e`): a failed/absent uninstall shouldn't block removal.
set -uo pipefail

PIP="$PREFIX/bin/pip"
LOG="$PREFIX/.nvitk-post-link.log"

log() { echo "[nvitk pre-unlink] $*" | tee -a "$LOG"; }

log "removing pip-only dependencies installed by post-link.sh..."
"$PIP" uninstall -y \
  scipy pandas polars pyarrow click jupyter ipywidgets openpyxl lazy-loader \
  matplotlib plotly kaleido pyvista seaborn rich trame trame-vtk \
  imageio pillow nibabel tifffile nd2 dicom2nifti pylibjpeg h5py \
  scikit-image opencv-python vtk SimpleITK antspyx \
  nilearn nipype gudhi antspynet totalsegmentator nnunetv2 fireants \
  pingouin networkx scikit-learn scikit-posthocs semopy great-tables formulae pymer4 \
  xnat pyyaml paramiko fastapi uvicorn keyring \
  napari magicgui superqt PyQt6-WebEngine \
  pyradiomics pydicom \
  torch torchvision \
  cupy-cuda13x cutensor-cu13 nvidia-nccl-cu13 \
  >>"$LOG" 2>&1 || log "warning: some packages were already absent or failed to uninstall cleanly"

log "done."
