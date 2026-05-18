"""Morphological operators (dilation, erosion, opening, closing, fill-holes).

All functions are backend-aware (NumPy or CuPy), accept either an
:class:`~nvitk.types.Image` or a raw array, and return the same type as the
input (no in-place mutation of the caller's data).

This package re-exports the public API from :mod:`nvitk.morphology.binary` and
:mod:`nvitk.morphology._common`; implementation lives in those submodules.
"""

from __future__ import annotations

from ._common import make_ball_footprint
from .binary import close, dilate, erode, fill_holes, open
from .centerline import compute_centerlines, skeletonize_binary
from .components import (
    keep_component_closest_to_center,
    keep_components_touching_seeds,
    label_connected,
    remove_small_components,
    remove_small_components_by_fraction,
)

__all__ = [
    "close",
    "compute_centerlines",
    "dilate",
    "erode",
    "fill_holes",
    "keep_component_closest_to_center",
    "keep_components_touching_seeds",
    "label_connected",
    "make_ball_footprint",
    "open",
    "remove_small_components",
    "remove_small_components_by_fraction",
    "skeletonize_binary",
]
