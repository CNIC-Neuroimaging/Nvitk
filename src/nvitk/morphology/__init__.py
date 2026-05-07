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
from .centerline import compute_centerlines

__all__ = [
    "dilate",
    "erode",
    "open",
    "close",
    "fill_holes",
    "make_ball_footprint",
    "compute_centerlines",
]
