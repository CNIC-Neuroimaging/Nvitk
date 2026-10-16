#!/usr/bin/env bash
# Build a CPython 3.11 wheel of VMTK 1.5.2 that runs on the PyPI ``vtk==9.6.2`` wheels.
#
# Why this exists: conda-forge's only Python 3.11 VMTK (1.5.0) links libboost-python 1.82, which
# pins numpy<2, so it cannot live next to nvitk's numpy 2.3.3. Upstream VMTK 1.5.2 builds pip
# wheels against Kitware's VTK wheel SDK (numpy-2 compatible) but only publishes cp312-cp314; the
# 3.12 floor is a CI choice (vmtk commit 439791ff), not a code requirement. This script builds the
# same wheel for cp311 from the upstream sdist, following upstream's own recipe
# (distribution/pypi/README.md): VTK from the 9.6.2 wheel SDK, ITK 5.4.5 built once and bundled by
# ``auditwheel repair --exclude "libvtk*"`` (the VTK libraries come from the ``vtk`` wheel).
#
# Modes:
#   manylinux  build inside quay.io/pypa/manylinux_2_28_x86_64 (portable, glibc >= 2.28; what the
#              release asset is built with). Needs docker.
#   native     build with the host toolchain for the given $PYTHON (the post-link fallback when the
#              release asset cannot be downloaded). Needs gcc/g++, git and network access.
#   auto       manylinux when docker works, otherwise native (default).
#
# Usage:  build_wheel.sh [auto|manylinux|native] [output_dir]
# Env:    PYTHON (native mode, default python3), NVITK_VMTK_BUILD_DIR (scratch, default mktemp).
# Output: <output_dir>/vmtk-1.5.2-cp311-cp311-<platform>.whl and a matching .sha256 file.
set -euo pipefail

VMTK_VERSION="1.5.2"
VMTK_SDIST_URL="https://files.pythonhosted.org/packages/7e/bd/e9a14c5dc2ef73fbbec028d131c76b6717003b80f4e69bb5e71264069467/vmtk-${VMTK_VERSION}.tar.gz"
VMTK_SDIST_SHA256="8202880f68c97f14dad7df6816710b5aaafb26db7f09bf799358e61ee5ef6906"
VTK_VERSION="9.6.2"
ITK_VERSION="5.4.5"
MANYLINUX_IMAGE="quay.io/pypa/manylinux_2_28_x86_64"

MODE="${1:-auto}"
OUT_DIR="$(mkdir -p "${2:-$PWD/wheelhouse}" && cd "${2:-$PWD/wheelhouse}" && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

log() { echo "[vmtk-wheel] $*"; }

build_native() {
  local py="$1" plat="$2"
  local work="${NVITK_VMTK_BUILD_DIR:-$(mktemp -d -t nvitk-vmtk-XXXXXX)}"
  mkdir -p "$work"
  log "python: $("$py" -c 'import sys; print(sys.version.split()[0])')  work: $work"

  if [ ! -f "$work/vmtk-${VMTK_VERSION}.tar.gz" ]; then
    curl -fsSL -o "$work/vmtk-${VMTK_VERSION}.tar.gz" "$VMTK_SDIST_URL"
  fi
  echo "${VMTK_SDIST_SHA256}  $work/vmtk-${VMTK_VERSION}.tar.gz" | sha256sum -c -
  rm -rf "$work/vmtk-${VMTK_VERSION}"
  tar -xzf "$work/vmtk-${VMTK_VERSION}.tar.gz" -C "$work"
  local src="$work/vmtk-${VMTK_VERSION}"

  # The only change to upstream: allow CPython 3.11.
  sed -i 's/^requires-python = ">=3.12"/requires-python = ">=3.11"/' "$src/pyproject.toml"
  grep -q '^requires-python = ">=3.11"' "$src/pyproject.toml"

  "$py" -m venv "$work/venv"
  "$work/venv/bin/pip" install -q --upgrade pip auditwheel patchelf
  VMTK_VTK_VERSION="$VTK_VERSION" VMTK_ITK_VERSION="$ITK_VERSION" \
    "$work/venv/bin/pip" wheel --no-deps -w "$work/dist" "$src" -v
  local whl
  whl="$(ls "$work"/dist/vmtk-"${VMTK_VERSION}"-*.whl)"
  "$work/venv/bin/auditwheel" repair --exclude "libvtk*" --plat "$plat" -w "$OUT_DIR" "$whl"
  for w in "$OUT_DIR"/vmtk-"${VMTK_VERSION}"-*.whl; do
    (cd "$OUT_DIR" && sha256sum "$(basename "$w")" > "$(basename "$w").sha256")
    log "built $w"
  done
}

case "$MODE" in
  --inside-manylinux)
    # Running inside the manylinux image (see the manylinux branch below).
    dnf install -y -q mesa-libGL-devel libX11-devel libXt-devel libXcursor-devel git >/dev/null
    build_native /opt/python/cp311-cp311/bin/python manylinux_2_28_x86_64
    # Hand the results to whoever owns the mounted output directory (right for rootful and
    # rootless docker alike).
    chown -R --reference="$OUT_DIR" "$OUT_DIR"/vmtk-* || true
    ;;
  manylinux)
    command -v docker >/dev/null || { log "docker not found"; exit 1; }
    work="${NVITK_VMTK_BUILD_DIR:-$(mktemp -d -t nvitk-vmtk-XXXXXX)}"
    mkdir -p "$work"
    docker run --rm \
      -e NVITK_VMTK_BUILD_DIR=/work \
      -v "$SCRIPT_PATH:/build_wheel.sh:ro" -v "$OUT_DIR:/out" -v "$work:/work" \
      "$MANYLINUX_IMAGE" bash /build_wheel.sh --inside-manylinux /out
    ;;
  native)
    build_native "${PYTHON:-python3}" "linux_$(uname -m)"
    ;;
  auto)
    if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
      "$SCRIPT_PATH" manylinux "$OUT_DIR"
    else
      "$SCRIPT_PATH" native "$OUT_DIR"
    fi
    ;;
  *)
    echo "usage: $0 [auto|manylinux|native] [output_dir]" >&2
    exit 2
    ;;
esac
