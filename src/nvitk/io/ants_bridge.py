"""Convert nvitk :class:`~nvitk.types.Image` / arrays ↔ ANTsPy images."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import BackendUnavailableError


def require_ants():
    """Import ``ants`` or raise :class:`BackendUnavailableError`."""
    try:
        import ants
    except ImportError as exc:  # pragma: no cover
        raise BackendUnavailableError(
            "ANTsPy (antspyx) is required for this tool. "
            "Install with: pip install antspyx"
        ) from exc
    return ants


def require_antspynet():
    """Import ``antspynet`` or raise :class:`BackendUnavailableError`."""
    try:
        import antspynet
    except ImportError as exc:  # pragma: no cover
        raise BackendUnavailableError(
            "ANTsPyNet is required for this tool. "
            "Install with: pip install antspynet"
        ) from exc
    return antspynet


def _spacing_origin_direction(image: Any, ndim: int) -> tuple[
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    np.ndarray | None,
]:
    spacing = None
    origin = None
    direction = None
    meta = getattr(image, "metadata", None) or {}
    sp = getattr(image, "spacing", None)
    if sp is None and isinstance(meta, dict):
        sp = meta.get("spacing")
    if sp is not None:
        seq = tuple(float(v) for v in sp)
        if len(seq) >= ndim:
            spacing = seq[:ndim]
    if isinstance(meta, dict):
        if meta.get("origin") is not None:
            origin = tuple(float(v) for v in meta["origin"][:ndim])
        if meta.get("direction") is not None:
            direction = np.asarray(meta["direction"], dtype=float)
    return spacing, origin, direction


def to_ants_image(image: Any, *, dtype: Any = np.float32):
    """Build an ANTsPy image from an :class:`~nvitk.types.Image` or ndarray."""
    ants = require_ants()
    data = to_numpy(getattr(image, "data", image)).astype(dtype, copy=False)
    if data.ndim < 2 or data.ndim > 4:
        raise ValueError(f"ANTs bridge expects 2–4D data, got ndim={data.ndim}")
    spacing, origin, direction = _spacing_origin_direction(image, data.ndim)
    kw: dict[str, Any] = {}
    if spacing is not None:
        kw["spacing"] = spacing
    if origin is not None:
        kw["origin"] = origin
    if direction is not None:
        kw["direction"] = direction
    return ants.from_numpy(data, **kw)


def from_ants_image(ants_image: Any) -> np.ndarray:
    """Return a NumPy view/copy of an ANTsPy image array."""
    return np.asarray(ants_image.numpy())


__all__ = [
    "from_ants_image",
    "require_ants",
    "require_antspynet",
    "to_ants_image",
]
