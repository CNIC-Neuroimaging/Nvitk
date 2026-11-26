"""Camera rays on the 3D canvas, in a layer's data coordinates.

A click on Napari's 3D canvas carries a world point (on the plane through the camera centre) and
the view direction. Grabbing something drawn in 3D (an edge of the ROI box, the crosshair point)
and dragging it needs the ray in the layer's voxel space, the closest points between that ray and
a segment, and how far apart two points look on screen. Every helper works on full data vectors
(all of the layer's axes, time included), so a 3D+t layer's transforms apply unchanged.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def ray_in_data(layer: Any, position: Sequence[float], direction: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """``(point, direction)`` of the camera ray through world *position*, in *layer*'s data axes."""
    position = np.asarray(position, dtype=float)
    direction = np.asarray(direction, dtype=float)
    n = min(len(position), len(direction))
    position, direction = position[-n:], direction[-n:]
    a = np.asarray(layer.world_to_data(position), dtype=float)
    b = np.asarray(layer.world_to_data(position + direction), dtype=float)
    return a, b - a


def to_world(layer: Any, point: Sequence[float]) -> np.ndarray:
    return np.asarray(layer.data_to_world(np.asarray(point, dtype=float)), dtype=float)


def screen_px(viewer: Any, layer: Any, a: Sequence[float], b: Sequence[float]) -> float:
    """How far apart data points *a* and *b* look on the canvas, in screen pixels (orthographic
    camera: world distance times the zoom)."""
    distance = float(np.linalg.norm(to_world(layer, a) - to_world(layer, b)))
    return distance * float(getattr(viewer.camera, "zoom", 1.0) or 1.0)


def closest_on_ray(point: np.ndarray, direction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """The point of the line ``point + t·direction`` nearest *target*, and its ``t``."""
    dd = float(direction @ direction)
    t = float((target - point) @ direction) / dd if dd > 1e-12 else 0.0
    return point + t * direction, t


def segment_ray_closest(point: np.ndarray, direction: np.ndarray, a: np.ndarray,
                        b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Closest points between the line ``point + t·direction`` and segment *a*-*b*:
    ``(on the line, on the segment, t)``."""
    u, v, w = direction, b - a, point - a
    uu, uv, vv, uw, vw = float(u @ u), float(u @ v), float(v @ v), float(u @ w), float(v @ w)
    denom = uu * vv - uv * uv
    s = (uu * vw - uv * uw) / denom if abs(denom) > 1e-12 else 0.0
    s = min(max(s, 0.0), 1.0)
    on_segment = a + s * v
    on_line, t = closest_on_ray(point, direction, on_segment)
    return on_line, on_segment, t


def drag_point(layer: Any, position: Sequence[float], direction: Sequence[float], anchor: np.ndarray,
               normal_axis: int | None = None) -> np.ndarray | None:
    """Where a point grabbed at *anchor* (data coordinates) goes while the cursor is at *position*.

    With *normal_axis*, the point slides in the plane across that data axis through *anchor* (an
    edge of a box keeps its own axis); when that plane is seen edge-on, or with no axis, in the
    plane facing the camera through *anchor*.
    """
    point, ray = ray_in_data(layer, position, direction)
    norm = float(np.linalg.norm(ray))
    if norm <= 1e-12:
        return None
    if normal_axis is not None and abs(float(ray[normal_axis])) / norm > 0.2:
        t = (float(anchor[normal_axis]) - float(point[normal_axis])) / float(ray[normal_axis])
        return point + t * ray
    hit, _t = closest_on_ray(point, ray, np.asarray(anchor, dtype=float))
    return hit


__all__ = ["closest_on_ray", "drag_point", "ray_in_data", "screen_px", "segment_ray_closest", "to_world"]
