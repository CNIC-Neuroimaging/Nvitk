#!/bin/bash
# Installs every pip-only dependency of nvitk, i.e. everything in pyproject.toml except the
# conda-native core in recipe.yaml's requirements/run (python, numpy, the R stack + rpy2,
# freeimage, libstdcxx-ng, and the Qt/GUI stack except PyQt6-WebEngine — see the comment at
# the top of recipe.yaml for why that split exists). Runs once, right after `conda` links
# nvitk into the target env, and is also run as-is by ../install.sh for development/wheel
# installs, so the two install routes share one dependency list.
#
# NVITK_VARIANT selects the PyTorch build: `gpu` (default) installs CUDA 13 torch/torchvision
# plus cupy/cutensor/nccl; `cpu` installs CPU-only torch/torchvision and no CUDA packages.
# conda passes the caller's environment through to this script, so
# `NVITK_VARIANT=cpu conda create -n nvitk-cpu nvitk` selects the CPU build there too.
#
# Version pins below are copied verbatim from pyproject.toml — do not relax them here even
# where a conda package would technically satisfy the constraint.
set -euo pipefail

PIP="$PREFIX/bin/pip"
LOG="$PREFIX/.nvitk-post-link.log"
VARIANT="${NVITK_VARIANT:-gpu}"

log() { echo "[nvitk post-link] $*" | tee -a "$LOG"; }

trap 'tail -n 30 "$LOG" >&2; log "FAILED — full log: $LOG"' ERR

case "$VARIANT" in
  gpu) TORCH_INDEX=(--extra-index-url https://download.pytorch.org/whl/cu130) ;;
  # --index-url, not --extra-index-url: with PyPI also in play, pip could pick PyPI's default
  # torch wheel, which is the CUDA build (several GB of nvidia-* libraries).
  cpu) TORCH_INDEX=(--index-url https://download.pytorch.org/whl/cpu) ;;
  *) log "NVITK_VARIANT must be 'gpu' or 'cpu', got '$VARIANT'"; exit 1 ;;
esac

log "starting pip-only dependency install into $PREFIX (variant: $VARIANT)"

# torch goes first: totalsegmentator, nnunetv2, fireants, torchio, ... all require it, and if
# it is not already present pip resolves it from PyPI (the CUDA build) and the variant's own
# index is never consulted. The constraint file then holds torch/torchvision at exactly this
# build for every later step, so no dependency can swap it out for a PyPI one.
log "installing PyTorch ($VARIANT build)..."
"$PIP" install --no-cache-dir "${TORCH_INDEX[@]}" \
  "torch>=2.9.0" "torchvision>=0.24.0" \
  >>"$LOG" 2>&1
CONSTRAINTS="$(mktemp)"
trap 'rm -f "$CONSTRAINTS"' EXIT
"$PREFIX/bin/python" -c '
from importlib.metadata import version
for name in ("torch", "torchvision"):
    print(f"{name}=={version(name)}")
' >"$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"

log "installing core (data/plotting/logging)..."
"$PIP" install --no-cache-dir \
  "scipy>=1.15.3" "pandas>=2.3.2" "polars>=1.43.2" "pyarrow>=21.0.0" "click>=8.2.1" \
  "jupyter>=1.1.1" "ipywidgets>=8.1.5" "openpyxl>=3.1.5" "lazy-loader>=0.4" \
  >>"$LOG" 2>&1

log "installing visualization/plotting/logging..."
"$PIP" install --no-cache-dir \
  "matplotlib>=3.8.0" "plotly>=6.3.0" "kaleido>=1.0.0" "pyvista>=0.48.0" \
  "seaborn>=0.13.2" "rich>=14.1.0" "tensorboard>=2.16.0" "trame>=3.6.0" "trame-vtk>=2.5.8,<2.11.9" \
  >>"$LOG" 2>&1

log "installing image IO..."
"$PIP" install --no-cache-dir \
  "imageio>=2.37.0" "pillow>=11.0.0" "nibabel>=5.3.2" "tifffile>=2025.5.26" \
  "nd2>=0.10.3" "dicom2nifti>=2.6.1" "pylibjpeg>=2.0.1" "h5py>=3.16.0" \
  "blosc2>=3.0.0" \
  >>"$LOG" 2>&1

log "installing general image processing (simpleitk/antspyx: see recipe.yaml for why these can't be conda run-deps)..."
"$PIP" install --no-cache-dir \
  "scikit-image>=0.25.2" "opencv-python>=4.13.0.92" "vtk==9.6.2" "SimpleITK==2.2.1" "antspyx>=0.6.3" \
  >>"$LOG" 2>&1
# VMTK (installed last, below) is compiled against exactly this VTK wheel: hold it for every later
# step so no dependency can move it.
echo "vtk==9.6.2" >>"$CONSTRAINTS"

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

log "installing PyTorch/nnSSL extras..."
"$PIP" install --no-cache-dir \
  "lightning-bolts>=0.7.0" "pytorch-msssim>=1.0.0" "torchio>=1.2.1" \
  "lpips>=0.1.4" "loguru==0.7.3" "wandb==0.28.1" \
  >>"$LOG" 2>&1

if [[ "$VARIANT" == gpu ]]; then
  log "installing remaining CUDA 13 GPU stack (cupy/cutensor/nccl)..."
  "$PIP" install --no-cache-dir \
    "cupy-cuda13x>=13.6.0" "cutensor-cu13>=2.6.0" "nvidia-nccl-cu13>=2.27.7" \
    >>"$LOG" 2>&1
fi

log "installing pyradiomics (no-build-isolation)..."
"$PIP" install --no-cache-dir --no-build-isolation \
  "pyradiomics==3.0.1" \
  >>"$LOG" 2>&1

log "installing pydicom (no-build-isolation)..."
"$PIP" install --no-cache-dir --no-build-isolation \
  "pydicom==3.0.1" \
  >>"$LOG" 2>&1

log "installing pyvista deps (no-build-isolation)..."
"$PIP" install --no-cache-dir --no-build-isolation 'pyvista[notebook]' \
  >>"$LOG" 2>&1

log "installing mesh processing deps (no-build-isolation)..."
"$PIP" install --no-cache-dir --no-build-isolation 'pymeshlab>=2025.7.post1' \
  >>"$LOG" 2>&1

# VMTK 1.5.2 for CPython 3.11 (centerlines, surface tools, 4D-flow particle tracing). Upstream
# publishes cp312+ wheels only and conda-forge's py311 VMTK pins numpy<2, so nvitk ships its own
# cp311 build of the upstream sdist (recipe/vmtk/build_wheel.sh, manylinux_2_28) as a release
# asset. NVITK_VMTK_WHEEL_URL / NVITK_VMTK_WHEEL_SHA256 override the asset (a local path works
# too; set NVITK_VMTK_WHEEL_SHA256="" to skip the check for a self-built wheel). If it cannot be
# fetched, the wheel is built here from the sdist (~15-45 min: ITK compile).
VMTK_WHEEL_NAME="vmtk-1.5.2-cp311-cp311-manylinux_2_28_x86_64.whl"
VMTK_WHEEL_URL="${NVITK_VMTK_WHEEL_URL:-https://github.com/ignacio-ms/Nvitk/releases/download/vmtk-1.5.2-cp311/$VMTK_WHEEL_NAME}"
VMTK_WHEEL_SHA256="${NVITK_VMTK_WHEEL_SHA256:-91763226d87654bb9b58eae4279761fd64fad397c4ed92dc061420fed8467651}"
VMTK_BUILD_SCRIPT=""
for cand in "$PREFIX/share/nvitk/vmtk/build_wheel.sh" "${RECIPE_DIR:-}/vmtk/build_wheel.sh" \
            "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vmtk/build_wheel.sh"; do
  if [[ -n "$cand" && -f "$cand" ]]; then VMTK_BUILD_SCRIPT="$cand"; break; fi
done
log "installing VMTK 1.5.2 (cp311 wheel)..."
VMTK_TMP="$(mktemp -d)"
VMTK_WHEEL="$VMTK_TMP/$VMTK_WHEEL_NAME"
vmtk_fetched=0
if [[ -f "$VMTK_WHEEL_URL" ]]; then
  cp "$VMTK_WHEEL_URL" "$VMTK_WHEEL" && vmtk_fetched=1
elif curl -fsSL -o "$VMTK_WHEEL" "$VMTK_WHEEL_URL" >>"$LOG" 2>&1; then
  vmtk_fetched=1
fi
if [[ "$vmtk_fetched" == 1 && -n "$VMTK_WHEEL_SHA256" ]]; then
  if ! echo "$VMTK_WHEEL_SHA256  $VMTK_WHEEL" | sha256sum -c - >>"$LOG" 2>&1; then
    log "VMTK wheel checksum mismatch — discarding it"
    vmtk_fetched=0
  fi
fi
if [[ "$vmtk_fetched" != 1 ]]; then
  [[ -n "$VMTK_BUILD_SCRIPT" ]] || { log "VMTK wheel unavailable and build_wheel.sh not found"; exit 1; }
  log "VMTK wheel not downloadable; building it from the sdist (this takes a while)..."
  PYTHON="$PREFIX/bin/python" bash "$VMTK_BUILD_SCRIPT" native "$VMTK_TMP/built" >>"$LOG" 2>&1
  VMTK_WHEEL="$(ls "$VMTK_TMP"/built/vmtk-1.5.2-*.whl)"
fi
"$PIP" install --no-cache-dir --no-deps "$VMTK_WHEEL" >>"$LOG" 2>&1
rm -rf "$VMTK_TMP"
"$PREFIX/bin/python" -c 'import vtk, vmtk; from vmtk import vmtkscripts, vtkvmtk; print("vtk", vtk.vtkVersion.GetVTKVersion(), "| vmtk ok")' \
  | tee -a "$LOG"

log "done."
