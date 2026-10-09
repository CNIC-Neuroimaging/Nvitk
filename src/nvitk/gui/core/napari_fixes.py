"""Fixes for Napari behaviour nvitk runs into, installed once at start-up.

* **Rays through a layer with fewer dimensions than the viewer** (Napari 0.7).
  ``Layer._get_ray_intersections`` builds its result from the *world* position,
  so a 3D layer shown next to a 3D+t one gets 4-entry ray points. Painting on
  the 3D canvas then fails inside ``first_nonzero_coordinate`` (operands could
  not be broadcast), and 3D value picks read the wrong voxel. The points are
  brought back to the layer's own dimensions.
* **Painting on the 3D canvas into empty space.** Napari paints where the ray
  meets the first existing label and does nothing when there is none — so an
  empty label layer cannot be painted in 3D at all. A registered fallback (the
  Labeling tab's: the voxel the reference image shows there) gives the brush a
  place to land. Only the brush uses it: a bucket fill landing on background
  would flood the whole background.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

#: Called as ``fallback(layer, event)`` when a 3D brush ray meets no label;
#: returns a data coordinate for the layer, or ``None``.
_LABELS_3D_FALLBACKS: list[Callable[[Any, Any], Any]] = []


def register_labels_3d_fallback(fn: Callable[[Any, Any], Any]) -> None:
    """Add *fn* to the places a 3D brush stroke may land when it meets no label."""
    if fn not in _LABELS_3D_FALLBACKS:
        _LABELS_3D_FALLBACKS.append(fn)


def unregister_labels_3d_fallback(fn: Callable[[Any, Any], Any]) -> None:
    """Remove *fn* (see :func:`register_labels_3d_fallback`)."""
    try:
        _LABELS_3D_FALLBACKS.remove(fn)
    except ValueError:
        pass


def _fix_ray_intersections() -> None:
    from napari.layers.base.base import Layer

    if getattr(Layer._get_ray_intersections, "_nvitk_fixed", False):
        return
    original = Layer._get_ray_intersections

    def _get_ray_intersections(self: Any, position: Any, view_direction: Any, dims_displayed: Any,
                               bounding_box: Any, world: bool = True) -> Any:
        start, end = original(self, position=position, view_direction=view_direction,
                              dims_displayed=dims_displayed, bounding_box=bounding_box, world=world)
        if start is None or end is None or not world:
            return start, end
        nd = int(self.ndim)
        if len(start) == nd:
            return start, end
        # The displayed entries are right (data coordinates, at the layer's own
        # displayed dims); the rest came from the world position — take them
        # from the position mapped into the layer instead.
        base = np.asarray(self.world_to_data(np.asarray(position, dtype=float)), dtype=float)
        dims = [int(d) for d in dims_displayed]
        fixed = []
        for point in (start, end):
            out = base.copy()
            out[dims] = np.asarray(point, dtype=float)[dims]
            fixed.append(out)
        return fixed[0], fixed[1]

    _get_ray_intersections._nvitk_fixed = True  # type: ignore[attr-defined]
    Layer._get_ray_intersections = _get_ray_intersections


def _fix_labels_3d_brush() -> None:
    from napari.layers.labels import _labels_mouse_bindings as bindings

    if getattr(bindings.mouse_event_to_labels_coordinate, "_nvitk_fixed", False):
        return
    original = bindings.mouse_event_to_labels_coordinate

    def mouse_event_to_labels_coordinate(layer: Any, event: Any) -> Any:
        coordinates = original(layer, event)
        layer._nvitk_paint_anywhere = False
        if coordinates is not None or len(layer._slice_input.displayed) != 3:
            return coordinates
        if str(getattr(layer, "mode", "")) != "paint":
            return coordinates
        for fallback in list(_LABELS_3D_FALLBACKS):
            try:
                hit = fallback(layer, event)
            except Exception:  # noqa: BLE001 — a fallback must never break the brush
                hit = None
            if hit is not None:
                # Napari's _draw skips background voxels in 3D (it paints on labels
                # only); this dab is meant for background.
                layer._nvitk_paint_anywhere = True
                return np.asarray(hit, dtype=int)
        return None

    mouse_event_to_labels_coordinate._nvitk_fixed = True  # type: ignore[attr-defined]
    bindings.mouse_event_to_labels_coordinate = mouse_event_to_labels_coordinate

    from napari.layers import Labels
    from napari.layers.labels._labels_constants import Mode
    from napari.layers.labels._labels_utils import interpolate_coordinates

    original_draw = Labels._draw

    def _draw(self: Any, new_label: Any, last_cursor_coord: Any, coordinates: Any) -> None:
        if (coordinates is None or self._slice_input.ndisplay != 3
                or not getattr(self, "_nvitk_paint_anywhere", False) or self._mode != Mode.PAINT):
            return original_draw(self, new_label, last_cursor_coord, coordinates)
        for c in interpolate_coordinates(last_cursor_coord, coordinates, self.brush_size):
            self.paint(c, new_label, refresh=False)
        self._partial_labels_refresh()
        return None

    Labels._draw = _draw


def install_napari_fixes() -> None:
    """Install every fix above (idempotent; a missing Napari internal skips its fix)."""
    for fix in (_fix_ray_intersections, _fix_labels_3d_brush):
        try:
            fix()
        except Exception:  # noqa: BLE001 — another Napari version: leave it as it is
            pass


__all__ = ["install_napari_fixes", "register_labels_3d_fallback", "unregister_labels_3d_fallback"]
