"""Volume measurements on a binary/label mask."""

from __future__ import annotations

from typing import Any

from nvitk.core.backend import setup
from nvitk.types import Image

from ._common import bool_mask, resolve_spacing

setup(globals())


def volume_mm3(
    mask: Image | Any,
    *,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """Return the number of non-zero voxels in *mask* × voxel volume (mm^3)."""
    m = bool_mask(mask)
    n = float(m.sum())
    sx, sy, sz = resolve_spacing(mask, spacing)[:3]
    return n * float(sx * sy * sz)


def volume_cc(
    mask: Image | Any,
    *,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """Return the mask's volume in cubic centimetres (mL)."""
    return float(volume_mm3(mask, spacing=spacing) / 1000.0)


__all__ = ["volume_mm3", "volume_cc"]
