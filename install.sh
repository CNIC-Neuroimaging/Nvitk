#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"

if [[ "$MODE" != "cpu" && "$MODE" != "gpu" ]]; then
  echo "Usage: ./install.sh [cpu|gpu]"
  exit 1
fi

if [[ "$MODE" == "gpu" ]]; then
  ENV_FILE="env/environment-gpu.yml"
  ENV_NAME="gpu-nvitk"
else
  ENV_FILE="env/environment.yml"
  ENV_NAME="nvitk"
fi

conda env create -f "$ENV_FILE"
conda run -n "$ENV_NAME" pip install --no-build-isolation pyradiomics==3.0.1

echo "Installation complete for '$MODE'."
echo "Activate with: conda activate $ENV_NAME"
