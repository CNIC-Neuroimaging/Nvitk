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


def _affine_matrix(image: Any) -> np.ndarray | None:
    """Extract a 4x4 world affine from an Image-like object (attribute or metadata dict); ``None`` if unavailable."""
    aff = getattr(image, "affine", None)
    if aff is None:
        meta = getattr(image, "metadata", None) or {}
        if isinstance(meta, dict):
            aff = meta.get("affine")
    if aff is None:
        return None
    arr = np.asarray(aff, dtype=float)
    if arr.shape != (4, 4):
        return None
    return arr


def _spacing_origin_direction(image: Any, ndim: int) -> tuple[
    tuple[float, ...] | None,
    tuple[float, ...] | None,
    np.ndarray | None,
]:
    """Extract ``(spacing, origin, direction)`` for an ANTs image from an Image-like object's attributes/metadata."""
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

    # Derive missing / inconsistent geometry from the voxel→world affine
    # (critical for ANTsPyNet template registration / center-of-mass alignment).
    aff = _affine_matrix(image)
    if aff is not None:
        n = min(3, ndim)
        aff_spacing = tuple(float(np.linalg.norm(aff[:n, i])) for i in range(n))
        if spacing is None:
            spacing = aff_spacing
            if ndim > n:
                spacing = spacing + tuple(1.0 for _ in range(ndim - n))
        elif (
            len(spacing) >= n
            and all(abs(float(s) - 1.0) < 1e-3 for s in spacing[:n])
            and any(abs(a - 1.0) > 1e-3 for a in aff_spacing)
        ):
            # Napari often leaves scale at (1,1,1) while the affine still encodes
            # real mm spacing — prefer the affine.
            spacing = aff_spacing + (tuple(spacing[n:]) if len(spacing) > n else ())
            if ndim > len(spacing):
                spacing = spacing + tuple(1.0 for _ in range(ndim - len(spacing)))
        if origin is None:
            origin = tuple(float(aff[i, 3]) for i in range(n))
            if ndim > n:
                origin = origin + tuple(0.0 for _ in range(ndim - n))
        if direction is None and n >= 2:
            dirs = np.eye(n, dtype=float)
            for i in range(n):
                col = aff[:n, i]
                norm = float(np.linalg.norm(col))
                if norm > 0:
                    dirs[:, i] = col / norm
            direction = dirs
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


def ants_result_to_array(result: Any) -> np.ndarray:
    """Coerce an ANTsPyNet return value (image, list, or dict) to a NumPy array."""
    if isinstance(result, (list, tuple)) and result:
        # DKT and similar APIs return ``[segmentation, *probability_images]``.
        return from_ants_image(result[0])
    if isinstance(result, dict):
        for key in (
            "segmentation_image",
            "segmentation",
            "probability_image",
            "probability_brain_mask",
            "mask",
            "output_image",
            "super_resolution",
        ):
            if key in result and result[key] is not None:
                return from_ants_image(result[key])
        first = next(iter(result.values()))
        return from_ants_image(first)
    return from_ants_image(result)


__all__ = [
    "ants_result_to_array",
    "from_ants_image",
    "require_ants",
    "require_antspynet",
    "to_ants_image",
]
