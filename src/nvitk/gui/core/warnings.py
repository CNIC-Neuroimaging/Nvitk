"""Suppress known Napari display warnings for oblique NIfTI affines."""

from __future__ import annotations

import warnings


def install_napari_display_warnings() -> None:
    """Filter non-orthogonal slice warnings (fires on load and while scrolling)."""
    warnings.filterwarnings(
        "ignore",
        message=r".*Non-orthogonal slicing.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*out-of-slice rotation or shear.*",
        category=UserWarning,
    )
