"""Centre centerline layers on the vessel lumen (Tools panel and Mesh panel).

Skeleton centerlines run off the lumen centre: they follow voxel corners, and near bends they
hug the inner wall. :func:`nvitk.morphology.centering.center_polyline` moves a path onto the
centre line (cross-section centroids, then the inscribed-radius ridge where VMTK's centerlines
run). This module feeds it any centerline layer the GUI makes:

- Shapes paths (*To polyline*, CPR, cross-sections, *Centerline network*);
- Vectors polylines (the Mesh panel's centerline lines; their polylines are in the layer metadata);
- Points ordered along the centerline, grouped by a ``branch`` feature when there is one;
- a skeleton mask (Labels / Image), split into branches between end points and junctions.

Everything is converted through world coordinates, so the centerline and the vessel need not
share a grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from nvitk.core.array import to_numpy

#: Point features that group an ordered points layer into separate centerlines.
_GROUP_FEATURES: tuple[str, ...] = ("branch", "branch_index", "path_id", "path")


def layer_world_matrix(layer: Any) -> np.ndarray:
    """4 × 4 data -> world transform of *layer*'s spatial (last three) dims."""
    try:
        mat = np.asarray(layer._data_to_world.affine_matrix, dtype=float)
    except Exception:  # noqa: BLE001 (not a Napari layer)
        return np.eye(4)
    out = np.eye(4)
    out[:3, :3] = mat[-4:-1, -4:-1]
    out[:3, 3] = mat[-4:-1, -1]
    return out


def data_to_world(layer: Any, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(len(points), -1)[:, -3:]
    mat = layer_world_matrix(layer)
    return pts @ mat[:3, :3].T + mat[:3, 3]


def world_to_data(layer: Any, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    inv = np.linalg.inv(layer_world_matrix(layer))
    return pts @ inv[:3, :3].T + inv[:3, 3]


def _split_at_jumps(points: np.ndarray) -> list[np.ndarray]:
    """An ordered point list cut where a step is far longer than the typical one."""
    if len(points) < 3:
        return [points]
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cut = np.flatnonzero(steps > 3.0 * float(np.median(steps)) + 1e-9) + 1
    return np.split(points, cut)


def _chain_segments(starts: np.ndarray, ends: np.ndarray) -> list[np.ndarray]:
    """Segments joined end to start into polylines (how polylines are stored in a Vectors layer)."""
    tol = 1e-6 * max(1.0, float(np.ptp(starts, axis=0).max()) if len(starts) else 1.0)
    lines: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for s, e in zip(starts, ends):
        if current and float(np.linalg.norm(current[-1] - s)) <= tol:
            current.append(e)
        else:
            if len(current) >= 2:
                lines.append(np.asarray(current))
            current = [s, e]
    if len(current) >= 2:
        lines.append(np.asarray(current))
    return lines


def layer_polylines_world(layer: Any) -> list[np.ndarray]:
    """The centerlines drawn by *layer*, as ordered world-coordinate polylines."""
    from nvitk.gui.mesh.layers import POLYLINES_META

    kind = type(layer).__name__
    if kind == "Shapes":
        types = list(getattr(layer, "shape_type", []) or [])
        return [
            data_to_world(layer, np.asarray(to_numpy(verts), dtype=float))
            for verts, st in zip(layer.data, types)
            if st in ("path", "line", "polygon") and len(verts) >= 2
        ]
    if kind == "Surface":  # a mesh carrying its centerline polylines (Mesh panel metadata)
        stored = (getattr(layer, "metadata", None) or {}).get(POLYLINES_META)
        if stored:
            return [data_to_world(layer, np.asarray(p, dtype=float)) for p in stored if len(p) >= 2]
    if kind == "Vectors":
        stored = (getattr(layer, "metadata", None) or {}).get(POLYLINES_META)
        if stored:
            return [data_to_world(layer, np.asarray(p, dtype=float)) for p in stored if len(p) >= 2]
        data = np.asarray(to_numpy(layer.data), dtype=float)
        if data.ndim != 3 or data.shape[1] != 2:
            raise ValueError("Unexpected Vectors layer data.")
        return _chain_segments(data_to_world(layer, data[:, 0]), data_to_world(layer, data[:, 0] + data[:, 1]))
    if kind == "Points":
        world = data_to_world(layer, np.asarray(to_numpy(layer.data), dtype=float))
        features = getattr(layer, "features", None)
        groups = None
        if features is not None and len(features) == len(world):
            for key in _GROUP_FEATURES:
                if key in features.columns:
                    groups = np.asarray(features[key])
                    break
        if groups is None:
            return [p for p in _split_at_jumps(world) if len(p) >= 2]
        order = list(dict.fromkeys(groups.tolist()))
        lines = []
        for g in order:
            lines.extend(p for p in _split_at_jumps(world[groups == g]) if len(p) >= 2)
        return lines
    if kind in ("Labels", "Image"):
        from nvitk.core.backend import using
        from nvitk.gui.viz.centerline import mask_is_thin
        from nvitk.morphology.centerline import skeletonize_binary
        from nvitk.morphology.polyline_graph import branch_polylines_from_skeleton

        arr = np.asarray(to_numpy(layer.data))
        if arr.ndim != 3:
            raise ValueError("A centerline mask must be 3D.")
        mask = arr > 0
        with using("cpu"):
            skel = mask if mask_is_thin(mask) else np.asarray(to_numpy(skeletonize_binary(mask))).astype(bool)
            chains = branch_polylines_from_skeleton(np.argwhere(skel).astype(np.float32), min_points=2)
        return [data_to_world(layer, np.asarray(to_numpy(c), dtype=float)) for c in chains]
    raise ValueError(
        f"Layer '{getattr(layer, 'name', '?')}' ({kind}) is not a centerline: use a Shapes path, "
        "Vectors polyline, ordered Points or skeleton mask layer."
    )


def layer_polylines_world_labelled(layer: Any) -> list[tuple[int | None, np.ndarray]]:
    """:func:`layer_polylines_world` with each polyline's label id when the layer records one (a
    ``label`` feature per Shapes path or per Points point, as To polyline and the centerline network
    write them), else ``None``."""
    kind = type(layer).__name__
    lines = layer_polylines_world(layer)
    features = getattr(layer, "features", None)
    has = features is not None and "label" in getattr(features, "columns", ())
    if kind == "Shapes" and has and len(features) == len(layer.data):
        types = list(getattr(layer, "shape_type", []) or [])
        labs = [int(v) for v, st, verts in zip(features["label"], types, layer.data)
                if st in ("path", "line", "polygon") and len(verts) >= 2]
        if len(labs) == len(lines):
            return list(zip(labs, lines))
    if kind == "Points" and has and len(features) == len(layer.data):
        world = data_to_world(layer, np.asarray(to_numpy(layer.data), dtype=float))
        labs = np.asarray(features["label"])
        out: list[tuple[int | None, np.ndarray]] = []
        for line in lines:  # each line's label: that of the point nearest its first vertex
            j = int(np.argmin(np.linalg.norm(world - line[0], axis=1)))
            out.append((int(labs[j]), line))
        return out
    return [(None, line) for line in lines]


def is_polyline_layer(layer: Any) -> bool:
    """Whether *layer* holds centerline polylines rather than a centerline image: Shapes paths,
    Vectors (segments chained into polylines, or the Mesh panel's stored ones), a Surface carrying the
    Mesh panel's polylines, or ordered Points."""
    from nvitk.gui.mesh.layers import POLYLINES_META

    kind = type(layer).__name__
    if kind in ("Shapes", "Vectors", "Points"):
        return True
    return kind == "Surface" and bool((getattr(layer, "metadata", None) or {}).get(POLYLINES_META))


def polylines_to_centerline_image(grid_layer: Any, polyline_layer: Any, *, label_id: int = 0,
                                  label_volume: Any = None) -> tuple[np.ndarray, dict[int, int]]:
    """The centerline image (int32, on *grid_layer*'s grid) of the polylines of *polyline_layer*
    (:func:`is_polyline_layer`): each drawn as a 26-connected voxel chain of its label. A path
    without a recorded label takes *label_id*, or with 0 the label under most of its voxels in
    *label_volume* (default: *grid_layer*'s data when it is a label map), else 1. Returns the image
    and ``{label: paths}``."""
    from nvitk.morphology.skeleton_methods import rasterize_labelled_polylines

    grid = np.asarray(to_numpy(grid_layer.data))
    if grid.ndim != 3:
        raise ValueError("The grid (active) layer must be 3D.")
    lines = [(lab, world_to_data(grid_layer, pts)) for lab, pts in layer_polylines_world_labelled(polyline_layer)]
    lines = [(lab, pts) for lab, pts in lines if len(pts) >= 2]
    if not lines:
        raise ValueError(f"No polyline in '{polyline_layer.name}'.")
    lookup = None
    if label_volume is not None:
        lookup = np.asarray(to_numpy(label_volume))
    elif type(grid_layer).__name__ == "Labels":
        lookup = grid
    labels: list[int] = []
    for lab, pts in lines:
        if lab is None or int(lab) <= 0:
            lab = int(label_id)
            if lab <= 0 and lookup is not None and lookup.shape == grid.shape:  # the mask's label under the path
                one = rasterize_labelled_polylines(grid.shape, [pts], [1]) > 0
                under = lookup[one]
                under = under[under > 0]
                lab = int(np.bincount(under.astype(np.int64)).argmax()) if under.size else 1
            lab = lab if lab > 0 else 1
        labels.append(int(lab))
    image = rasterize_labelled_polylines(grid.shape, [p for _l, p in lines], labels)
    counts: dict[int, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    return image, counts


@dataclass
class CentredLine:
    """One centerline moved onto the lumen centre."""

    points: np.ndarray
    #: Inscribed radius (distance to the wall) along the centred line.
    radius: np.ndarray
    #: Distance from each centred point to the original line.
    shift: np.ndarray
    #: Median inscribed radius of the original line (how far off-centre it was).
    radius_before: float
    length_before: float
    length_after: float


def _length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _densify(points: np.ndarray, step: float) -> np.ndarray:
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return points
    ss = np.arange(0.0, s[-1] + 0.5 * step, step)
    return np.column_stack([np.interp(ss, s, points[:, k]) for k in range(3)])


def center_lines(
    polylines: Sequence[np.ndarray],
    surface: Any,
    *,
    iterations: int = 5,
    smooth_mm: float = 0.3,
    step_mm: float = 0.25,
) -> list[CentredLine]:
    """Centre each polyline on *surface* (same frame, mm)."""
    from scipy.spatial import cKDTree

    from nvitk.morphology.centering import center_polyline, inscribed_radius

    out = []
    for line in polylines:
        pts = np.asarray(line, dtype=float).reshape(-1, 3)
        if len(pts) < 2:
            continue
        before = inscribed_radius(surface, pts)
        if len(pts) < 3 or _length(pts) < 2.0 * step_mm:
            centred = pts.copy()
        else:
            centred = center_polyline(pts, surface, iterations=int(iterations), smooth_mm=float(smooth_mm),
                                      sample_step_mm=float(step_mm))
        shift, _ = cKDTree(_densify(pts, 0.25 * step_mm)).query(centred)
        out.append(CentredLine(
            points=centred, radius=inscribed_radius(surface, centred), shift=np.asarray(shift, dtype=float),
            radius_before=float(np.median(before)), length_before=_length(pts), length_after=_length(centred),
        ))
    return out


def summary(lines: Sequence[CentredLine]) -> str:
    """One-line report: shifts and how much the inscribed radius grew (it peaks on the centre line)."""
    if not lines:
        return "nothing centred"
    shift = np.concatenate([c.shift for c in lines])
    r_before = float(np.median([c.radius_before for c in lines]))
    r_after = float(np.median([float(np.median(c.radius)) for c in lines]))
    return (f"{len(lines)} centerline(s): shift mean {shift.mean():.2f} mm, max {shift.max():.2f} mm; "
            f"median inscribed radius {r_before:.2f} → {r_after:.2f} mm")


__all__ = [
    "CentredLine",
    "center_lines",
    "data_to_world",
    "layer_polylines_world",
    "layer_world_matrix",
    "summary",
    "world_to_data",
]
