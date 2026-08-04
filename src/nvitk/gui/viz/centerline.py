"""Napari helpers: skeleton junction detect / cut on centerline masks."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.backend import using
from nvitk.morphology.components import label_connected
from nvitk.morphology.polyline_graph import detect_junctions_from_centerline
from nvitk.pipes.qvtpy.util.centerline.venous_flexion import rasterize_polylines_to_volume, split_polyline_at_indices

JUNCTION_POINTS_LAYER = "Skeleton junctions"
JUNCTION_META_KEY = "nvitk_junction_source"


def _require_centerline_volume(layer: Any) -> np.ndarray:
    """Return *layer*'s data as a host 3D array with at least one foreground voxel; raises
    ``ValueError`` if the layer isn't 3D or is entirely empty."""
    arr = to_numpy(layer.data)
    if arr.ndim != 3:
        raise ValueError("Centerline tools expect a 3D centerline mask layer.")
    if not bool((arr > 0).any()):
        raise ValueError("Centerline layer has no foreground voxels.")
    return arr


def detect_junctions_from_layer(
    layer: Any,
    *,
    label_id: int | None,
    min_degree = 3,
    reskeletonize = False,
) -> np.ndarray:
    """Junction voxel coordinates from a 3D centerline mask."""
    arr = _require_centerline_volume(layer)
    return detect_junctions_from_centerline(
        arr,
        min_degree=min_degree,
        label_id=label_id,
        reskeletonize=reskeletonize,
    )


def add_junction_points_layer(
    viewer: Any,
    junctions: np.ndarray,
    *,
    reference_layer: Any,
    source_layer_name = "",
    min_degree = 3,
) -> Any:
    """Add or replace skeleton junction markers."""
    for lyr in list(viewer.layers):
        if lyr.name == JUNCTION_POINTS_LAYER:
            viewer.layers.remove(lyr)

    coords = to_numpy(junctions).astype(np.float64)
    if coords.ndim == 1:
        coords = coords.reshape(0, 3)
    features = {}
    if coords.shape[0] > 0:
        features["skeleton_degree"] = np.full(coords.shape[0], int(min_degree), dtype=int)

    kwargs = {
        "size": 12,
        "face_color": "magenta",
        "symbol": "star",
        "metadata": {
            JUNCTION_META_KEY: {
                "source_layer": source_layer_name,
                "min_degree": int(min_degree),
                "n_junctions": int(coords.shape[0]),
            }
        },
    }
    from nvitk.gui.core.spatial import layer_affine

    aff = layer_affine(reference_layer)
    if aff is not None:
        kwargs["affine"] = aff

    return viewer.add_points(
        coords,
        name=JUNCTION_POINTS_LAYER,
        features=features,
        **kwargs,
    )


def read_junction_coords(viewer: Any) -> np.ndarray:
    """Read junction voxel coordinates from the Skeleton junctions Points layer."""
    for lyr in viewer.layers:
        if lyr.name != JUNCTION_POINTS_LAYER:
            continue
        data = to_numpy(lyr.data)
        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError("Junction layer must be Nx3.")
        return data[:, :3].astype(np.int32)
    raise ValueError(
        f"No '{JUNCTION_POINTS_LAYER}' layer. Run Detect skeleton junctions on the centerline first."
    )


def cluster_junction_coords(
    junctions: np.ndarray,
    *,
    cluster_radius_vox = 1,
) -> np.ndarray:
    """Collapse nearby junction markers to one representative voxel per cluster."""
    pts = to_numpy(junctions).astype(np.int32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.int32)
    rad = max(0, int(cluster_radius_vox))
    visited = np.zeros(pts.shape[0], dtype=bool)
    reps = []
    for i in range(pts.shape[0]):
        if visited[i]:
            continue
        cluster_idx = [i]
        visited[i] = True
        q = deque([i])
        while q:
            cur = q.popleft()
            for j in range(pts.shape[0]):
                if visited[j]:
                    continue
                d = np.abs(pts[j] - pts[cur])
                if int(d.max()) <= rad:
                    visited[j] = True
                    cluster_idx.append(j)
                    q.append(j)
        block = pts[cluster_idx]
        reps.append(block[len(block) // 2])
    return to_numpy(reps).astype(np.int32)


def _nearest_polyline_indices(
    polyline: np.ndarray,
    junctions: np.ndarray,
    *,
    min_separation_points = 2,
) -> list[int]:
    """Map each junction cluster to one cut index on an ordered centerline polyline."""
    if junctions.size == 0 or polyline.shape[0] == 0:
        return []
    p = to_numpy(polyline).astype(np.float64)
    reps = cluster_junction_coords(junctions, cluster_radius_vox=1)
    cuts = []
    for j in reps.astype(np.float64):
        d2 = np.sum((p - j.reshape(1, 3)) ** 2, axis=1)
        cuts.append(int(np.argmin(d2)))
    sep = max(1, int(min_separation_points))
    ordered = sorted({int(c) for c in cuts if 0 < int(c) < int(p.shape[0])})
    merged = []
    for c in ordered:
        if merged and abs(c - merged[-1]) < sep:
            continue
        merged.append(c)
    return merged


def split_label_at_junctions(
    label_volume: np.ndarray,
    source_label_id: int,
    centerline_polyline: np.ndarray,
    junction_coords: np.ndarray,
    *,
    new_label_start = None,
) -> tuple[np.ndarray, list[int]]:
    """
    Split *source_label_id* into new labels at junctions along *centerline_polyline*.

    Junction voxels are removed from the source label; each branch between cuts is
    rasterized with a new label id (qvtpy-style branch split).
    """
    vol = to_numpy(label_volume).astype(np.int32, copy=True)
    cut_idx = _nearest_polyline_indices(centerline_polyline, junction_coords)

    if not cut_idx:
        # Fallback: remove one voxel per junction cluster
        mask = vol == int(source_label_id)
        work = mask.copy()
        for row in cluster_junction_coords(junction_coords, cluster_radius_vox=1):
            i, j, k = int(row[0]), int(row[1]), int(row[2])
            if 0 <= i < work.shape[0] and 0 <= j < work.shape[1] and 0 <= k < work.shape[2]:
                work[i, j, k] = False
        # 26-connectivity CC labeling (base tool); ``work`` is host-side here.
        with using("numpy"):
            labeled, n = label_connected(work, connectivity=3)
        labeled = to_numpy(labeled)
        if n < 1:
            return vol, [int(source_label_id)]
        existing = [int(x) for x in np.unique(vol) if int(x) != 0]
        start = (max(existing) + 1) if existing else int(source_label_id) + 1
        if new_label_start is not None and int(new_label_start) > 0:
            start = int(new_label_start)
        new_ids = [start + i for i in range(int(n))]
        vol[mask] = 0
        for comp in range(1, int(n) + 1):
            vol[labeled == comp] = new_ids[comp - 1]
        return vol, new_ids

    segments = split_polyline_at_indices(centerline_polyline, cut_idx)
    if not segments:
        return vol, [int(source_label_id)]

    existing = [int(x) for x in np.unique(vol) if int(x) != 0]
    start = (max(existing) + 1) if existing else int(source_label_id) + 1
    if new_label_start is not None and int(new_label_start) > 0:
        start = int(new_label_start)
    new_ids = [start + i for i in range(len(segments))]
    vol[vol == int(source_label_id)] = 0
    painted = rasterize_polylines_to_volume(vol.shape, segments, new_ids)
    vol[painted > 0] = painted[painted > 0]
    return vol, new_ids


def centerline_polyline_for_label(
    centerline_layer: Any,
    label_id: int,
    *,
    reskeletonize = False,
) -> np.ndarray:
    """Longest-path polyline through one label on a centerline mask (for junction cuts)."""
    from nvitk.morphology.centerline import compute_centerlines

    arr = _require_centerline_volume(centerline_layer)
    vol = np.zeros(arr.shape, dtype=np.int32)
    lid = int(label_id)
    vol[arr == lid] = lid
    lines = compute_centerlines(
        vol,
        centerline_mask=arr if not reskeletonize else None,
        labels=[lid],
        min_points=3,
    )
    if lid not in lines:
        coords = np.argwhere(arr == lid)
        if coords.shape[0] < 3:
            raise ValueError(f"Not enough centerline voxels for label {lid}.")
        from nvitk.morphology.centerline import _centerline_longest_path

        return _centerline_longest_path(coords.astype(np.float32))
    return lines[lid].astype(np.float32, copy=False)


POLYLINE_SHAPES_META = "nvitk_centerline_polylines"
DEFAULT_POLYLINE_LAYER = "Centerline polylines"


def centerline_mask_to_polylines(
    centerline: np.ndarray,
    *,
    labels: list[int] | None = None,
    min_branch_points: int | None = None,
    reskeletonize: bool = False,
    smooth: bool = True,
    smooth_window: int = 5,
) -> list[dict[str, Any]]:
    """Convert a labeled centerline mask into connected branch polylines.

    Uses :func:`~nvitk.morphology.centerline.compute_connected_centerline_tree`
    so each label keeps its main path plus every unique junction-split branch
    (shared only at bifurcation voxels — no dropped corridors).

    Parameters
    ----------
    min_branch_points
        Drop edges shorter than this many voxels. ``None`` / ``<= 0`` keeps
        every edge with at least 2 points.
    """
    from nvitk.morphology.centerline import compute_connected_centerline_tree
    from nvitk.pipes.qvtpy.util.centerline.centerline_io import smooth_centerline_polyline

    arr = to_numpy(centerline)
    if arr.ndim != 3:
        raise ValueError("Centerline mask must be 3D.")
    if not bool((arr > 0).any()):
        raise ValueError("Centerline mask has no foreground voxels.")

    labs = labels or sorted(int(v) for v in np.unique(arr) if int(v) != 0)
    if not labs:
        return []

    min_edge = 2
    if min_branch_points is not None and int(min_branch_points) > 0:
        min_edge = max(2, int(min_branch_points))

    # When the input is already a 1-voxel skeleton, pass it as centerline_mask
    # so we do not re-skeletonize (preserves every root voxel). Thick masks can
    # opt into reskeletonize.
    tree = compute_connected_centerline_tree(
        arr.astype(np.int32, copy=False),
        centerline_mask=None if reskeletonize else arr,
        labels=[int(x) for x in labs],
        min_points=2,
        min_edge_points=int(min_edge),
        keep_all_components=True,
        prune_short_spurs=False,
    )

    out: list[dict[str, Any]] = []
    for lid in labs:
        edges = tree.get(int(lid)) or []
        for bi, pts in enumerate(edges):
            poly = np.asarray(pts, dtype=np.float64)
            if poly.ndim != 2 or poly.shape[0] < 2 or poly.shape[1] < 3:
                continue
            if smooth and poly.shape[0] >= 3:
                poly = to_numpy(
                    smooth_centerline_polyline(poly, window=int(smooth_window))
                ).astype(np.float64, copy=False)
            role = "main" if bi == 0 else f"branch{bi}"
            out.append(
                {
                    "label": int(lid),
                    "branch_index": int(bi),
                    "role": role,
                    "points": np.asarray(poly[:, :3], dtype=np.float32),
                }
            )
    return out


def add_centerline_polylines_shapes(
    viewer: Any,
    polylines: list[dict[str, Any]],
    *,
    reference_layer: Any,
    layer_name: str = DEFAULT_POLYLINE_LAYER,
    edge_width: float = 0.35,
) -> Any | None:
    """Add Napari Shapes paths for centerline polylines (replace prior layer)."""
    for lyr in list(viewer.layers):
        meta = getattr(lyr, "metadata", None) or {}
        if lyr.name == layer_name or (
            isinstance(meta, dict) and meta.get(POLYLINE_SHAPES_META)
        ):
            try:
                viewer.layers.remove(lyr)
            except Exception:
                pass

    if not polylines:
        return None

    palette = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    )
    # Stable color per label id.
    label_color: dict[int, str] = {}
    paths: list[np.ndarray] = []
    edge_colors: list[str] = []
    for item in polylines:
        pts = np.asarray(item["points"], dtype=np.float64)
        if pts.shape[0] < 2:
            continue
        lid = int(item["label"])
        if lid not in label_color:
            label_color[lid] = palette[len(label_color) % len(palette)]
        paths.append(pts.astype(np.float32))
        edge_colors.append(label_color[lid])

    if not paths:
        return None

    from nvitk.gui.core.spatial import layer_affine

    kwargs: dict[str, Any] = {
        "name": str(layer_name),
        "shape_type": "path",
        "edge_color": edge_colors,
        "edge_width": float(edge_width),
        "opacity": 0.95,
        "metadata": {
            POLYLINE_SHAPES_META: True,
            "n_paths": int(len(paths)),
            "n_labels": int(len(label_color)),
            "source_layer": str(getattr(reference_layer, "name", "")),
        },
    }
    aff = layer_affine(reference_layer)
    if aff is not None:
        kwargs["affine"] = aff

    shapes = viewer.add_shapes(paths, **kwargs)
    try:
        shapes.editable = False
    except Exception:
        pass
    return shapes
