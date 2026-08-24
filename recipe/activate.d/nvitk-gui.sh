#!/bin/bash
# Prefer conda-native Qt/font libraries over HPC module paths when loading PyQt6.
# See recipe/recipe.yaml GUI run-deps + post-link PyQt6-WebEngine pip install.
export _NVITK_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# conda-forge's pyqt6 build has no bundled Qt6 (unlike the PyPI wheel) — it loads
# plugins (platforms/xcb, imageformats, xcbglintegrations, ...) from the separate
# qt6-main package instead. Without QT_PLUGIN_PATH, Qt only falls back to an
# exe-relative "platforms" dir that doesn't exist, so it can never find libqxcb.so
# and the app aborts with "Could not find the Qt platform plugin xcb" (verified by
# reproducing the crash and confirming this fixes it — see conversation history).
export _NVITK_OLD_QT_PLUGIN_PATH="${QT_PLUGIN_PATH-}"
export QT_PLUGIN_PATH="${CONDA_PREFIX}/lib/qt6/plugins${QT_PLUGIN_PATH:+:${QT_PLUGIN_PATH}}"
