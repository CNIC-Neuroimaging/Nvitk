#!/bin/bash
# Installs FastSurfer (https://github.com/Deep-MI/FastSurfer) into an nvitk environment.
#
# Used by recipe/post-link.sh (conda package), ../install.sh (development installs) and
# `nvitk-fastsurfer install`, so every route ends with the same tree:
#
#   $PREFIX/share/nvitk/fastsurfer/FastSurfer/     the FastSurfer source tree (VERSION marker inside)
#   $PREFIX/share/nvitk/fastsurfer/FastSurfer/checkpoints/   the network weights
#
# FastSurfer is not pip-installed: it runs from its own tree on PYTHONPATH with the
# environment's Python, which already holds its dependencies (torch, nibabel, scipy,
# SimpleITK, scikit-image, h5py, pandas, monai, lapy, yacs: see post-link.sh).
#
# The version is v2.5.4, the newest release that runs on nvitk's Python 3.11: v2.6.0 and
# later need Python >= 3.12 (PEP 695 generics throughout). FastSurfer's own torch pin
# (2.7.*) is advisory here: the tree runs on the environment's torch.
#
# Environment:
#   PREFIX                          target environment (required)
#   NVITK_FASTSURFER_VERSION        git tag to install (default v2.5.4)
#   NVITK_FASTSURFER_SHA256         sha256 of the tag tarball ("" skips the check)
#   NVITK_FASTSURFER_URL            tarball URL or local path (default: the GitHub tag archive)
#   NVITK_FASTSURFER_CHECKPOINTS    0 = do not download the network weights now (FastSurfer
#                                   fetches missing ones on first use)
set -euo pipefail

: "${PREFIX:?PREFIX must point at the target environment}"
VERSION="${NVITK_FASTSURFER_VERSION:-v2.5.4}"
DEFAULT_SHA256="0a290eb7409b7df42d47484b436fc9f0875b95927f3f8dd3a22608224521e1da"
if [[ "$VERSION" == "v2.5.4" ]]; then
  SHA256="${NVITK_FASTSURFER_SHA256-$DEFAULT_SHA256}"
else
  SHA256="${NVITK_FASTSURFER_SHA256-}"
fi
URL="${NVITK_FASTSURFER_URL:-https://github.com/Deep-MI/FastSurfer/archive/refs/tags/${VERSION}.tar.gz}"
ROOT="$PREFIX/share/nvitk/fastsurfer"
TREE="$ROOT/FastSurfer"
PY="$PREFIX/bin/python"

log() { echo "[nvitk fastsurfer] $*"; }

[[ -x "$PY" ]] || { log "no python at $PY"; exit 1; }
mkdir -p "$ROOT"

current=""
[[ -f "$TREE/VERSION" ]] && current="$(cat "$TREE/VERSION")"
if [[ "$current" == "$VERSION" ]]; then
  log "FastSurfer $VERSION already installed in $TREE"
else
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  log "fetching FastSurfer $VERSION..."
  if [[ -f "$URL" ]]; then
    cp "$URL" "$TMP/fastsurfer.tar.gz"
  else
    curl -fsSL --retry 3 -o "$TMP/fastsurfer.tar.gz" "$URL"
  fi
  if [[ -n "$SHA256" ]]; then
    echo "$SHA256  $TMP/fastsurfer.tar.gz" | sha256sum -c - >/dev/null \
      || { log "checksum mismatch for $URL (set NVITK_FASTSURFER_SHA256=\"\" to skip)"; exit 1; }
  fi
  mkdir -p "$TMP/src"
  tar -xzf "$TMP/fastsurfer.tar.gz" -C "$TMP/src"
  unpacked="$(find "$TMP/src" -mindepth 1 -maxdepth 1 -type d | head -1)"
  [[ -f "$unpacked/run_fastsurfer.sh" ]] || { log "unexpected archive layout (no run_fastsurfer.sh)"; exit 1; }
  # Keep downloaded weights across upgrades: they are versioned by file name.
  if [[ -d "$TREE/checkpoints" ]]; then
    mv "$TREE/checkpoints" "$TMP/checkpoints.keep"
  fi
  rm -rf "$TREE"
  mv "$unpacked" "$TREE"
  if [[ -d "$TMP/checkpoints.keep" ]]; then
    mkdir -p "$TREE/checkpoints"
    cp -an "$TMP/checkpoints.keep/." "$TREE/checkpoints/"
  fi
  echo "$VERSION" > "$TREE/VERSION"
  chmod +x "$TREE"/*.sh "$TREE"/recon_surf/*.sh 2>/dev/null || true
  log "installed FastSurfer $VERSION in $TREE"
fi

if [[ "${NVITK_FASTSURFER_CHECKPOINTS:-1}" != "0" ]]; then
  # Our downloader, not FastSurfer's: that one has no read timeout and hangs on a
  # stalled mirror (see download_checkpoints.py beside this script).
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  log "downloading network checkpoints (asegdkt, CerebNet, HypVINN, corpus callosum)..."
  "$PY" -s "$HERE/download_checkpoints.py" "$TREE"
fi

log "checking the modules import with $PY..."
(cd "$TREE" && PYTHONPATH="$TREE" "$PY" -s -c '
import FastSurferCNN.run_prediction, CerebNet.run_prediction, HypVINN.run_prediction
import CorpusCallosum.fastsurfer_cc
print("FastSurfer modules import ok")
')
log "done."
