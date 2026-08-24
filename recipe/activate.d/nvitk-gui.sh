#!/bin/bash
# Prefer conda-native Qt/font libraries over HPC module paths when loading PyQt6.
# See recipe/recipe.yaml GUI run-deps + post-link PyQt6-WebEngine pip install.
export _NVITK_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
