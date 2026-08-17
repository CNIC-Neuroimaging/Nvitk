"""Binary morphology — NumPy/SciPy stand-in for ``nvitk.morphology.binary``.

Hand-written (not synced). Only :func:`close` is reachable from the
morphometrics pipeline, and only when ``BRIDGE_LABEL_CLOSE_RADIUS > 0``
(the shipped default is ``0``, so this normally never runs).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import ndimage as ndi


def _footprint(ndim: int, footprint: int | Any | None, connectivity: int) -> np.ndarray:
    """Resolve a footprint argument to a structuring element.

    An ``int`` means a ball of that radius (Chebyshev-free, matching upstream's
    ``generate_binary_structure`` + ``iterate_structure``); ``None`` means the
    plain connectivity element; an array is used as-is.
    """
    if footprint is None:
        return ndi.generate_binary_structure(ndim, max(1, min(int(connectivity), ndim)))
    if isinstance(footprint, (int, np.integer)):
        radius = int(footprint)
        base = ndi.generate_binary_structure(ndim, max(1, min(int(connectivity), ndim)))
        return base if radius <= 1 else ndi.iterate_structure(base, radius)
    return np.asarray(footprint, dtype=bool)


def dilate(
    img: Any,
    footprint: int | Any | None = None,
    *,
    iterations: int = 1,
    mode: str = "binary",
    isotropic: bool = False,
    spacing: Sequence[float] | None = None,
    connectivity: int = 1,
) -> np.ndarray:
    """Binary dilation."""
    arr = np.asarray(getattr(img, "data", img)).astype(bool, copy=False)
    return ndi.binary_dilation(
        arr, structure=_footprint(arr.ndim, footprint, connectivity), iterations=max(1, int(iterations))
    )


def erode(
    img: Any,
    footprint: int | Any | None = None,
    *,
    iterations: int = 1,
    mode: str = "binary",
    isotropic: bool = False,
    spacing: Sequence[float] | None = None,
    connectivity: int = 1,
) -> np.ndarray:
    """Binary erosion."""
    arr = np.asarray(getattr(img, "data", img)).astype(bool, copy=False)
    return ndi.binary_erosion(
        arr, structure=_footprint(arr.ndim, footprint, connectivity), iterations=max(1, int(iterations))
    )


def close(
    img: Any,
    footprint: int | Any | None = None,
    *,
    iterations: int = 1,
    mode: str = "binary",
    isotropic: bool = False,
    spacing: Sequence[float] | None = None,
    connectivity: int = 1,
) -> np.ndarray:
    """Morphological closing = dilate followed by erode."""
    dilated = dilate(
        img, footprint, iterations=iterations, mode=mode,
        isotropic=isotropic, spacing=spacing, connectivity=connectivity,
    )
    return erode(
        dilated, footprint, iterations=iterations, mode=mode,
        isotropic=isotropic, spacing=spacing, connectivity=connectivity,
    )


def open(  # noqa: A001 - mirrors the upstream public name
    img: Any,
    footprint: int | Any | None = None,
    *,
    iterations: int = 1,
    mode: str = "binary",
    isotropic: bool = False,
    spacing: Sequence[float] | None = None,
    connectivity: int = 1,
) -> np.ndarray:
    """Morphological opening = erode followed by dilate."""
    eroded = erode(
        img, footprint, iterations=iterations, mode=mode,
        isotropic=isotropic, spacing=spacing, connectivity=connectivity,
    )
    return dilate(
        eroded, footprint, iterations=iterations, mode=mode,
        isotropic=isotropic, spacing=spacing, connectivity=connectivity,
    )


__all__ = ["close", "dilate", "erode", "open"]
