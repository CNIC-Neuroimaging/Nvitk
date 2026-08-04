"""
Surface-distance metrics (Hausdorff, MSD, MDSD, STDSD).

Fixes the legacy BioImaging bug where spacing was used for the erosion kernel
but not for the KD-tree distance computation, so distances were in voxels
instead of mm. Here, contour point coordinates are multiplied by *spacing* so
that all returned values are in physical units.

Backend policy
--------------
Contour extraction uses the proxy ``ndi`` (so binary erosion runs on GPU when
the active backend is CuPy). The KD-tree step hands off to SciPy's
``cKDTree``, which is NumPy-only, so we materialize the contour coordinates
at that boundary.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as _host_np
from scipy.spatial import cKDTree

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup

from ._common import bool_mask, ensure_same_shape, resolve_spacing

setup(globals())


def _contour(mask_bool: Any, structure: Any | None = None) -> Any:
    """Surface voxels of a binary mask = mask minus its erosion (the boundary shell)."""
    eroded = ndi.binary_erosion(mask_bool, structure=structure)
    return mask_bool & ~eroded


def _isotropic_structure(ndim: int) -> Any:
    """6-connectivity structure on the active backend."""
    return ndi.generate_binary_structure(ndim, 1).astype(np.uint8)


def _distances(
    label_true: Any,
    label_pred: Any,
    spacing: tuple[float, ...],
) -> _host_np.ndarray:
    """Return a 1D array of symmetric contour-to-contour distances in mm."""
    a = bool_mask(label_true)
    b = bool_mask(label_pred)

    struct = _isotropic_structure(a.ndim)

    ca = _contour(a, structure=struct)
    cb = _contour(b, structure=struct)

    # argwhere stays on backend, but cKDTree is NumPy-only.
    coords_a = to_numpy(np.argwhere(ca)).astype(float)
    coords_b = to_numpy(np.argwhere(cb)).astype(float)

    if coords_a.size == 0 or coords_b.size == 0:
        return _host_np.array([0.0])

    sp = _host_np.asarray(spacing, dtype=float)
    coords_a_mm = coords_a * sp[None, :]
    coords_b_mm = coords_b * sp[None, :]

    tree_a = cKDTree(coords_a_mm)
    tree_b = cKDTree(coords_b_mm)

    d_b2a, _ = tree_a.query(coords_b_mm, k=1)
    d_a2b, _ = tree_b.query(coords_a_mm, k=1)
    return _host_np.concatenate([d_b2a, d_a2b])


def hausdorff(
    label_true: Any,
    label_pred: Any,
    *,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """Hausdorff distance (mm): the worst-case surface-to-surface distance between two masks."""
    ensure_same_shape(label_true, label_pred)
    sp = resolve_spacing(label_true, spacing)
    d = _distances(label_true, label_pred, sp)
    return float(d.max() if d.size else 0.0)


def hausdorff95(
    label_true: Any,
    label_pred: Any,
    *,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """95th-percentile Hausdorff distance (mm): robust to a few outlier surface voxels."""
    ensure_same_shape(label_true, label_pred)
    sp = resolve_spacing(label_true, spacing)
    d = _distances(label_true, label_pred, sp)
    return float(_host_np.percentile(d, 95) if d.size else 0.0)


def msd(
    label_true: Any,
    label_pred: Any,
    *,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """Mean surface distance (mm) between the two mask boundaries."""
    ensure_same_shape(label_true, label_pred)
    sp = resolve_spacing(label_true, spacing)
    d = _distances(label_true, label_pred, sp)
    return float(_host_np.mean(d) if d.size else 0.0)


def mdsd(
    label_true: Any,
    label_pred: Any,
    *,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """Median surface distance (mm) between the two mask boundaries."""
    ensure_same_shape(label_true, label_pred)
    sp = resolve_spacing(label_true, spacing)
    d = _distances(label_true, label_pred, sp)
    return float(_host_np.median(d) if d.size else 0.0)


def stdsd(
    label_true: Any,
    label_pred: Any,
    *,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """Standard deviation of surface distances (mm) — boundary agreement spread."""
    ensure_same_shape(label_true, label_pred)
    sp = resolve_spacing(label_true, spacing)
    d = _distances(label_true, label_pred, sp)
    return float(_host_np.std(d) if d.size else 0.0)


_METRIC_FUNCS = {
    "hd": hausdorff,
    "hd95": hausdorff95,
    "msd": msd,
    "mdsd": mdsd,
    "stdsd": stdsd,
}


def surface_metrics(
    label_true: Any,
    label_pred: Any,
    *,
    spacing: tuple[float, ...] | None = None,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """
    Compute multiple surface metrics in one pass (all sharing the same distance array).
    """
    ensure_same_shape(label_true, label_pred)
    sp = resolve_spacing(label_true, spacing)
    d = _distances(label_true, label_pred, sp)
    requested = tuple(_METRIC_FUNCS.keys()) if metrics is None else tuple(metrics)
    unknown = set(requested) - set(_METRIC_FUNCS.keys())
    if unknown:
        raise ValueError(
            f"Unknown surface metrics: {unknown}. Supported: {set(_METRIC_FUNCS.keys())}"
        )

    out: dict[str, float] = {}
    if d.size == 0:
        return {k: 0.0 for k in requested}
    if "hd" in requested:
        out["hd"] = float(d.max())
    if "hd95" in requested:
        out["hd95"] = float(_host_np.percentile(d, 95))
    if "msd" in requested:
        out["msd"] = float(_host_np.mean(d))
    if "mdsd" in requested:
        out["mdsd"] = float(_host_np.median(d))
    if "stdsd" in requested:
        out["stdsd"] = float(_host_np.std(d))
    return out


__all__ = [
    "hausdorff",
    "hausdorff95",
    "msd",
    "mdsd",
    "stdsd",
    "surface_metrics",
]
