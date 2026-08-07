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
conda run -n "$ENV_NAME" pip install --no-build-isolation pydicom==3.0.1

conda install -n "$ENV_NAME" -c conda-forge r-base r-lme4 r-lmertest r-mmrm r-emmeans r-lmertest r-parameters r-performance r-report r-tibble r-broom r-broom.mixed r-insight rpy2
conda run -n "$ENV_NAME" pip install --no-deps pymer4
conda run -n "$ENV_NAME" pip install polars great-tables formulae
conda run -n "$ENV_NAME" R -e "install.packages(c('robustbase','emmeans'), repos='https://cloud.r-project.org')"

echo "Installation complete for '$MODE'."
echo "Activate with: conda activate $ENV_NAME"
