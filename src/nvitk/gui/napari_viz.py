"""Napari-native visualization (hotspots, 4D flow vectors) without PyVista."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.gui.spatial import layer_affine, layer_spacing, layer_to_image
from nvitk.measure.hemodynamics import velocity_mm_s_from_phases
from nvitk.types import Image
from nvitk.viz.pet_hotspots import HotspotMode, _roi_mask, _select_hotspots

HOTSPOTS_LAYER = "SUV hotspots"
FLOW_VECTORS_LAYER = "Flow velocity"


def hotspot_points_from_volumes(
    suv: np.ndarray,
    mask: np.ndarray,
    *,
    label_ids: Sequence[int] | None = None,
    hotspot: HotspotMode = "top_percent",
    top_percent: float = 0.1,
    top_k: int | None = None,
    threshold: float | None = None,
    max_points: int = 20000,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Return Napari point coordinates (N,3), SUV values, and feature columns.

    Subsamples to *max_points* by highest SUV when needed.
    """
    suv_arr = to_numpy(suv).astype(np.float64)
    mask_arr = to_numpy(mask)
    roi = _roi_mask(mask_arr, label_ids)
    hot = _select_hotspots(
        suv_arr,
        roi,
        hotspot=hotspot,
        top_percent=top_percent,
        top_k=top_k,
        suv_threshold=threshold,
    )
    coords = np.argwhere(hot)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64), {}

    vals = suv_arr[hot].astype(np.float64)
    if coords.shape[0] > int(max_points):
        order = np.argsort(vals)[::-1][: int(max_points)]
        coords = coords[order]
        vals = vals[order]

    features = {
        "suv": vals,
        "i": coords[:, 0].astype(int),
        "j": coords[:, 1].astype(int),
        "k": coords[:, 2].astype(int),
    }
    return coords.astype(np.float64), vals, features


def add_hotspot_points_layer(
    viewer: Any,
    coords: np.ndarray,
    features: dict[str, np.ndarray],
    *,
    reference_layer: Any,
    name: str = HOTSPOTS_LAYER,
) -> Any:
    for lyr in list(viewer.layers):
        if lyr.name == name:
            viewer.layers.remove(lyr)
    kwargs: dict[str, Any] = {"size": 6, "symbol": "o"}
    if coords.shape[0] == 0:
        kwargs["face_color"] = "red"
    else:
        kwargs["face_color"] = "suv"
        kwargs["face_colormap"] = "viridis"
        vals = to_numpy(features["suv"]).astype(np.float64)
        lo = float(np.min(vals))
        hi = float(np.max(vals))
        kwargs["face_contrast_limits"] = (lo, hi if hi > lo else lo + 1.0)
    aff = layer_affine(reference_layer)
    if aff is not None:
        kwargs["affine"] = aff
    return viewer.add_points(coords, name=name, features=features, **kwargs)


def _subsample_mask_indices(mask: np.ndarray, max_points: int, label_ids: list[int] | None) -> np.ndarray:
    roi = _roi_mask(mask, label_ids)
    idx = np.argwhere(roi)
    if idx.shape[0] <= max_points:
        return idx
    step = max(1, idx.shape[0] // int(max_points))
    return idx[::step][: int(max_points)]


def flow_vectors_at_time(
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    mask: np.ndarray,
    time_index: int,
    *,
    label_ids: list[int] | None = None,
    max_points: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions Nx3, vectors Nx3) in mm/s for one cardiac phase."""
    ap_a = as_backend_array(ap).astype(np.float64)
    rl_a = as_backend_array(rl).astype(np.float64)
    fh_a = as_backend_array(fh).astype(np.float64)
    vx, vy, vz = velocity_mm_s_from_phases(ap_a, rl_a, fh_a)
    t = int(np.clip(time_index, 0, int(vx.shape[3]) - 1))
    idx = _subsample_mask_indices(mask, max_points, label_ids)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    pos = idx.astype(np.float64)
    vec = np.stack(
        [
            vx[idx[:, 0], idx[:, 1], idx[:, 2], t],
            vy[idx[:, 0], idx[:, 1], idx[:, 2], t],
            vz[idx[:, 0], idx[:, 1], idx[:, 2], t],
        ],
        axis=1,
    )
    vec = to_numpy(vec).astype(np.float64)
    mag = np.linalg.norm(vec, axis=1, keepdims=True)
    vec = np.where(mag > 1e-6, vec / np.maximum(mag, 1e-6), vec)
    return pos, vec * 2.0


def add_flow_vectors_layer(
    viewer: Any,
    positions: np.ndarray,
    vectors: np.ndarray,
    *,
    reference_layer: Any,
    name: str = FLOW_VECTORS_LAYER,
) -> Any:
    for lyr in list(viewer.layers):
        if lyr.name == name:
            viewer.layers.remove(lyr)
    kwargs: dict[str, Any] = {
        "name": name,
        "vector_style": "arrow",
        "edge_width": 0,
        "length": 1.0,
    }
    aff = layer_affine(reference_layer)
    if aff is not None:
        kwargs["affine"] = aff
    data = np.stack([positions, vectors], axis=1)
    return viewer.add_vectors(data, **kwargs)


def voxel_spacing_from_layer(layer: Any) -> tuple[float, float, float]:
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    return (1.0, 1.0, 1.0)
