"""Napari-native mouse picking helpers for Labels (and generic layers).

Prefer these over hand-rolled ray marches so hover/status and click agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy


def _event_view_direction(event: Any, viewer: Any | None) -> Any | None:
    """Resolve the current 3D view direction: from *event*, else the viewer's dims, else its camera."""
    view_dir = getattr(event, "view_direction", None)
    if view_dir is not None:
        return view_dir
    if viewer is None:
        return None
    view_dir = getattr(getattr(viewer, "dims", None), "view_direction", None)
    if view_dir is not None:
        return view_dir
    return getattr(getattr(viewer, "camera", None), "view_direction", None)


def _dims_displayed(layer: Any, viewer: Any | None) -> list[int] | None:
    """Currently displayed dimension indices, preferring *layer*'s own slice input over the viewer's
    dims; ``None`` if neither is available."""
    try:
        displayed = getattr(getattr(layer, "_slice_input", None), "displayed", None)
        if displayed is not None:
            return list(displayed)
    except Exception:
        pass
    if viewer is not None:
        try:
            return list(viewer.dims.displayed)
        except Exception:
            pass
    return None


def _as_positive_label(value: Any) -> int | None:
    """Coerce a picked label *value* (scalar, array, or single-item sequence) to a positive int label
    id, or ``None`` if it's missing, unparseable, or non-positive (background)."""
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and value:
        value = value[0]
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        lid = int(arr.ravel()[0])
    except (TypeError, ValueError):
        return None
    return lid if lid > 0 else None


def label_id_under_mouse(
    layer: Any,
    event: Any,
    *,
    viewer: Any | None = None,
) -> int | None:
    """Return positive label id under the cursor using Napari's Labels API.

    Matches the status-bar / hover value: ``layer.get_value`` with
    ``view_direction`` and ``dims_displayed`` in 3D, plain sample in 2D.
    Falls back to :func:`napari.layers.labels._labels_utils.mouse_event_to_labels_coordinate`
    when available.
    """
    if layer is None:
        return None
    pos = getattr(event, "position", None)
    if pos is None and viewer is not None:
        pos = getattr(getattr(viewer, "cursor", None), "position", None)
    if pos is None:
        return None

    view_dir = _event_view_direction(event, viewer)
    dims = _dims_displayed(layer, viewer)

    # Primary path: same API Napari uses for status / tooltip.
    try:
        kwargs: dict[str, Any] = {"world": True}
        if view_dir is not None and dims is not None and len(dims) >= 3:
            kwargs["view_direction"] = view_dir
            kwargs["dims_displayed"] = dims
        val = layer.get_value(pos, **kwargs)
        lid = _as_positive_label(val)
        if lid is not None:
            return lid
    except TypeError:
        try:
            val = layer.get_value(pos, world=True)
            lid = _as_positive_label(val)
            if lid is not None:
                return lid
        except Exception:
            pass
    except Exception:
        pass

    # Secondary: Napari Labels paint helper (ray → first nonzero voxel).
    try:
        from napari.layers.labels._labels_utils import mouse_event_to_labels_coordinate

        coords = mouse_event_to_labels_coordinate(layer, event)
        if coords is None:
            return None
        idx = tuple(int(round(float(c))) for c in to_numpy(coords).ravel())
        data = getattr(layer, "data", None)
        if data is None:
            return None
        shape = getattr(data, "shape", ())
        if len(idx) != len(shape):
            idx = idx[-len(shape) :]
        if any(i < 0 or i >= s for i, s in zip(idx, shape, strict=False)):
            return None
        return _as_positive_label(data[idx])
    except Exception:
        return None


def view_ray_via_layer(
    layer: Any,
    event: Any,
    *,
    viewer: Any | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (origin, into-scene unit direction) in *layer* data coords via Napari.

    Uses ``layer.get_ray_intersections`` when possible so 3D picks share Napari's
    transform / dims handling (e.g. vessel cross-section centerline rays).
    """
    from nvitk.gui.core.spatial import view_direction_into_scene, world_to_data_coords

    pos = getattr(event, "position", None)
    if pos is None and viewer is not None:
        pos = getattr(getattr(viewer, "cursor", None), "position", None)
    if pos is None:
        return None, None

    view_dir = _event_view_direction(event, viewer)
    dims = _dims_displayed(layer, viewer)

    if view_dir is not None and dims is not None and hasattr(layer, "get_ray_intersections"):
        try:
            start, end = layer.get_ray_intersections(
                position=pos,
                view_direction=view_dir,
                dims_displayed=dims,
                world=True,
            )
            if start is not None and end is not None:
                start_a = to_numpy(start).astype(np.float64).ravel()
                end_a = to_numpy(end).astype(np.float64).ravel()
                n = min(3, start_a.size, end_a.size)
                origin = start_a[-n:]
                direction = end_a[-n:] - origin
                norm = float(np.linalg.norm(direction))
                if norm > 1e-9:
                    return origin.astype(np.float64), (direction / norm).astype(np.float64)
        except Exception:
            pass

    origin = world_to_data_coords(layer, pos)
    if origin is None or view_dir is None:
        return origin, None
    direction = view_direction_into_scene(layer, view_dir, event)
    return origin, direction


__all__ = [
    "label_id_under_mouse",
    "view_ray_via_layer",
]
