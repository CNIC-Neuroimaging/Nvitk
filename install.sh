#!/usr/bin/env bash
# Development / wheel installer for nvitk: the from-source counterpart of the conda package.
#
# Builds the same environment `conda create -n ENV nvitk` does, from the same sources:
#   1. conda-native dependencies  <- recipe/recipe.yaml requirements/run (parsed below)
#   2. pip-only dependency stack  <- recipe/post-link.sh, run as-is with PREFIX=<env>
#      (FastSurfer's tree + checkpoints included: recipe/fastsurfer/install_fastsurfer.sh)
#   3. activation hooks           <- recipe/activate.d, recipe/deactivate.d
#   4. nvitk itself, --no-deps    <- this checkout (editable, default) or a wheel
#   5. nvitk's FSL core           <- `nvitk-fsl install` (needs conda, so not part of post-link)
# A dependency change made in recipe/ therefore reaches both install routes.
#
# Run ./install.sh --help for options.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE="$REPO/recipe"
# Same channel setup the conda package is built against (.github/workflows/publish-conda.yml).
CHANNELS=(--override-channels -c conda-forge --strict-channel-priority)

VARIANT=gpu
ENV_NAME=""
ENV_PREFIX=""
MODE=editable
WHEEL=""
DEV_TOOLS=""
FORCE=0
DRY_RUN=0
FASTSURFER=1
FSL=1
FSL_COMPONENTS="structural,registration,diffusion,functional,asl"

usage() {
  cat <<EOF
Usage: ./install.sh [options]

Creates a conda environment with nvitk and its full dependency stack (GUI, R stats,
PyTorch), following the same logic as the conda package in recipe/.

Build variant:
  --gpu               CUDA 13 torch/torchvision + cupy/cutensor/nccl (default)
  --cpu               CPU-only torch/torchvision, no CUDA packages

Target environment:
  -n, --name NAME     environment name (default: nvitk-dev, or nvitk with --wheel)
  -p, --prefix DIR    environment path, instead of a name
  --force             remove and recreate the environment if it already exists

How nvitk itself is installed:
  (default)           editable install of this checkout (pip install -e)
  --wheel [FILE]      non-editable: install FILE, or build a wheel from this checkout
                      into dist/ and install that
  --dev-tools         also install the [dev] extra: pytest, sphinx toolchain
                      (default for editable installs)
  --no-dev-tools      skip the [dev] extra (default for --wheel)

External suites:
  --no-fastsurfer     skip FastSurfer (tree + network weights, ~340 MB; later: nvitk-fastsurfer install)
  --no-fsl            skip nvitk's FSL core (later: nvitk-fsl install; an existing \$FSLDIR is used anyway)
  --fsl-components L  FSL components, comma-separated (default: $FSL_COMPONENTS)

Other:
  --dry-run           print what would be installed, change nothing
  -h, --help          show this help

Examples:
  ./install.sh                         # GPU, editable, env "nvitk-dev"
  ./install.sh --cpu -n nvitk-cpu      # CPU-only, editable
  ./install.sh --wheel dist/nvitk-0.1.0-py3-none-any.whl
EOF
}

die() { echo "install.sh: $*" >&2; exit 1; }
step() { echo; echo "==> $*"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) VARIANT=gpu ;;
    --cpu) VARIANT=cpu ;;
    -n|--name) [[ $# -ge 2 ]] || die "$1 needs a value"; ENV_NAME="$2"; shift ;;
    -p|--prefix) [[ $# -ge 2 ]] || die "$1 needs a value"; ENV_PREFIX="$2"; shift ;;
    --force) FORCE=1 ;;
    --wheel)
      MODE=wheel
      if [[ $# -ge 2 && "$2" == *.whl ]]; then WHEEL="$2"; shift; fi ;;
    --wheel=*) MODE=wheel; WHEEL="${1#--wheel=}" ;;
    --dev-tools) DEV_TOOLS=1 ;;
    --no-dev-tools) DEV_TOOLS=0 ;;
    --dry-run) DRY_RUN=1 ;;
    --no-fastsurfer) FASTSURFER=0 ;;
    --no-fsl) FSL=0 ;;
    --fsl-components) [[ $# -ge 2 ]] || die "$1 needs a value"; FSL_COMPONENTS="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
  shift
done

[[ -n "$ENV_NAME" && -n "$ENV_PREFIX" ]] && die "use either --name or --prefix, not both"
if [[ -z "$ENV_NAME" && -z "$ENV_PREFIX" ]]; then
  [[ "$MODE" == editable ]] && ENV_NAME=nvitk-dev || ENV_NAME=nvitk
fi
[[ -z "$DEV_TOOLS" ]] && { [[ "$MODE" == editable ]] && DEV_TOOLS=1 || DEV_TOOLS=0; }
if [[ -n "$WHEEL" ]]; then
  [[ -f "$WHEEL" ]] || die "wheel not found: $WHEEL"
  WHEEL="$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")"
fi

# `conda` is a shell function in interactive shells; scripts get the binary via CONDA_EXE.
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
[[ -n "$CONDA" ]] || die "conda not found — install Miniforge (https://conda-forge.org/download/) first"

# requirements/run from the recipe: every `- spec` line up to the next section, comments and
# blank lines skipped, "name ==version" squeezed to "name==version" for the conda CLI.
mapfile -t CONDA_DEPS < <(awk '
  /^  run:/ { inrun = 1; next }
  inrun && (/^[^ #]/ || /^  [^ #]/) { exit }
  inrun && /^ *- / { sub(/#.*/, ""); sub(/^ *- */, ""); gsub(/ /, ""); if ($0 != "") print }
' "$RECIPE/recipe.yaml")
[[ ${#CONDA_DEPS[@]} -gt 0 ]] || die "could not read requirements/run from $RECIPE/recipe.yaml"

if [[ -n "$ENV_NAME" ]]; then TARGET=(-n "$ENV_NAME"); TARGET_DESC="$ENV_NAME"
else TARGET=(-p "$ENV_PREFIX"); TARGET_DESC="$ENV_PREFIX"; fi

echo "nvitk install"
echo "  variant      : $VARIANT"
echo "  environment  : $TARGET_DESC"
echo "  nvitk        : $([[ "$MODE" == editable ]] && echo "editable ($REPO)" || echo "wheel (${WHEEL:-built from $REPO})")"
echo "  dev tools    : $([[ "$DEV_TOOLS" == 1 ]] && echo yes || echo no)"
echo "  FastSurfer   : $([[ "$FASTSURFER" == 1 ]] && echo yes || echo no)"
echo "  FSL core     : $([[ "$FSL" == 1 ]] && echo "yes ($FSL_COMPONENTS)" || echo no)"
echo "  conda deps   : ${CONDA_DEPS[*]}"

if [[ "$DRY_RUN" == 1 ]]; then
  echo; echo "(dry run: nothing installed)"
  exit 0
fi

env_prefix() {
  "$CONDA" run "${TARGET[@]}" python -c 'import sys; print(sys.prefix)' 2>/dev/null
}

if PREFIX="$(env_prefix)" && [[ -n "$PREFIX" ]]; then
  [[ "$FORCE" == 1 ]] || die "environment '$TARGET_DESC' already exists ($PREFIX) — pick another --name, or pass --force to recreate it"
  step "removing existing environment $PREFIX"
  "$CONDA" env remove -y "${TARGET[@]}"
fi

step "creating conda environment with the recipe's run requirements"
"$CONDA" create -y "${TARGET[@]}" "${CHANNELS[@]}" "${CONDA_DEPS[@]}"
PREFIX="$(env_prefix)"
[[ -n "$PREFIX" ]] || die "environment was created but its prefix could not be resolved"
PY="$PREFIX/bin/python"
PIP="$PREFIX/bin/pip"

step "installing pip-only dependencies (recipe/post-link.sh, log: $PREFIX/.nvitk-post-link.log)"
PREFIX="$PREFIX" NVITK_VARIANT="$VARIANT" NVITK_FASTSURFER="$FASTSURFER" bash "$RECIPE/post-link.sh"

step "installing activation hooks"
mkdir -p "$PREFIX/etc/conda/activate.d" "$PREFIX/etc/conda/deactivate.d"
install -m 755 "$RECIPE/activate.d/nvitk-gui.sh" "$PREFIX/etc/conda/activate.d/nvitk-gui.sh"
install -m 755 "$RECIPE/deactivate.d/nvitk-gui.sh" "$PREFIX/etc/conda/deactivate.d/nvitk-gui.sh"

# --no-deps, as in the recipe's build script: dependencies come from steps above only.
if [[ "$MODE" == editable ]]; then
  step "installing nvitk (editable) from $REPO"
  "$PIP" install --no-deps -e "$REPO"
else
  if [[ -z "$WHEEL" ]]; then
    step "building nvitk wheel into $REPO/dist"
    WHEEL_DIR="$(mktemp -d)"
    "$PIP" wheel --no-deps -w "$WHEEL_DIR" "$REPO"
    mkdir -p "$REPO/dist"
    WHEEL="$REPO/dist/$(basename "$WHEEL_DIR"/nvitk-*.whl)"
    mv "$WHEEL_DIR"/nvitk-*.whl "$WHEEL"
    rm -rf "$WHEEL_DIR"
  fi
  step "installing nvitk from $WHEEL"
  "$PIP" install --no-deps --force-reinstall "$WHEEL"
fi

if [[ "$DEV_TOOLS" == 1 ]]; then
  step "installing the [dev] extra"
  mapfile -t DEV_DEPS < <("$PY" -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f:
    print("\n".join(tomllib.load(f)["project"]["optional-dependencies"]["dev"]))
' "$REPO/pyproject.toml")
  "$PIP" install --no-cache-dir "${DEV_DEPS[@]}"
fi

if [[ "$FSL" == 1 ]]; then
  step "installing nvitk's FSL core ($FSL_COMPONENTS) into $PREFIX/share/nvitk/fsl"
  FSL_CUDA=(--no-cuda); [[ "$VARIANT" == gpu ]] && FSL_CUDA=(--cuda)
  CONDA_EXE="$CONDA" "$PY" -m nvitk.fsl.cli install -y --components "$FSL_COMPONENTS" "${FSL_CUDA[@]}" \
    || echo "warning: the FSL core could not be installed now; run 'nvitk-fsl install' later."
fi

step "verifying"
"$CONDA" run "${TARGET[@]}" python -c "import nvitk"
"$CONDA" run "${TARGET[@]}" pyhelp --no-interactive >/dev/null
CUDA_OK="$("$PY" -c 'import torch; print(torch.cuda.is_available())')"
echo "import nvitk: ok   pyhelp: ok   torch: $("$PY" -c 'import torch; print(torch.__version__)')   CUDA available: $CUDA_OK"
if [[ "$VARIANT" == gpu && "$CUDA_OK" != True ]]; then
  echo "note: GPU build installed but no CUDA device is visible here (expected on a login/CPU node)."
fi

echo
echo "Done. Activate with:  conda activate $TARGET_DESC"
