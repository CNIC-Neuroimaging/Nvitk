"""Points and meshes in the slice views.

The views composite voxel layers (images, labels); points and surfaces are stored in world
space and have no voxels to slice. They are drawn on top instead, per view, through its
:class:`~nvitk.gui.viz.ortho.geometry.PlaneMap`:

* **Points**: each point the plane passes through, as a disc of its cross-section (the point's
  size is its diameter), in its face colour with its border; points within half a slice of the
  plane at least as a small dot, so a thin slice does not lose them.
* **Surfaces**: as the mesh is displayed (faces / wireframe / points, see
  :meth:`SurfaceCutter.cut`): the cut where the plane slices it, the edges of the triangles it
  passes through, their vertices; in the mesh's colours (vertex colours, or its colormap over
  the vertex values).

Both follow the layer's opacity, visibility and transform (a mesh moved in Transform mode is
re-cut). The heavy part of a surface, its vertices in the views' frame, is cached per layer
and placement, so scrolling only re-cuts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Layer types drawn as overlays.
OVERLAY_TYPES: tuple[str, ...] = ("Points", "Surface")


@dataclass
class Overlay:
    """What one layer draws in one view (slice pixels, voxel-centred)."""

    kind: str  # "points" or "lines"
    #: points: ``(N, 2)`` row/col; lines: ``(N, 4)`` row0, col0, row1, col1.
    coords: np.ndarray
    #: ``(N, 4)`` RGBA floats 0..1 (fill for points, stroke for lines).
    colors: np.ndarray
    #: points: ``(N, 2)`` radius down / across in slice pixels.
    radii: np.ndarray | None = None
    #: points: ``(N, 4)`` border RGBA, and the border width in screen pixels.
    edge_colors: np.ndarray | None = None
    edge_width: float = 1.0
    #: lines: pen width in screen pixels.
    width: float = 1.5
    #: points: a fixed on-screen diameter (pixels) instead of :attr:`radii` (mesh vertices).
    screen_size: float | None = None
    opacity: float = 1.0
    name: str = ""
    extra: dict = field(default_factory=dict)


def overlay_layers(viewer: Any) -> list[Any]:
    """The visible points and surface layers, bottom to top."""
    out = []
    for layer in list(getattr(viewer, "layers", []) or []):
        if type(layer).__name__ not in OVERLAY_TYPES or not bool(getattr(layer, "visible", True)):
            continue
        if float(getattr(layer, "opacity", 1.0) or 0.0) <= 0.0:
            continue
        if bool(getattr(layer, "_nvitk_slices_hidden", False)):
            continue  # its "Slices" box in the 3D options' layers table is unticked
        out.append(layer)
    return out


def _world_matrix(layer: Any) -> np.ndarray | None:
    from nvitk.gui.viz.main_canvas import real_data_to_world

    try:
        return np.asarray(real_data_to_world(layer), dtype=float)
    except Exception:  # noqa: BLE001
        return None


def _to_world(layer: Any, coords: np.ndarray) -> np.ndarray | None:
    """Data coordinates ``(N, D)`` of *layer* as world points ``(N, 3)`` (last three axes)."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 3:
        return None
    matrix = _world_matrix(layer)
    nd = coords.shape[1]
    if matrix is None or matrix.shape != (nd + 1, nd + 1):
        return None
    world = coords @ matrix[:nd, :nd].T + matrix[:nd, -1]
    return world[:, -3:]


def _linear_scale(layer: Any) -> float:
    """World units per data unit (geometric mean over the spatial axes)."""
    matrix = _world_matrix(layer)
    if matrix is None:
        return 1.0
    lin = matrix[-4:-1, -4:-1]
    det = abs(float(np.linalg.det(lin)))
    return det ** (1.0 / 3.0) if det > 1e-12 else 1.0


def points_overlay(layer: Any, plane: Any) -> Overlay | None:
    """*layer*'s points on *plane*'s view."""
    data = np.asarray(getattr(layer, "data", np.empty((0, 3))), dtype=float)
    if data.ndim != 2 or len(data) == 0:
        return None
    world = _to_world(layer, data)
    if world is None:
        return None
    rows, cols, depth = plane.world_to_pixels(world)
    n = len(data)
    size = np.asarray(getattr(layer, "size", 10.0), dtype=float).reshape(-1)
    size = np.broadcast_to(size if size.size in (1, n) else size[:1], (n,)) * _linear_scale(layer)
    radius = size / 2.0
    half = plane.thickness_mm() / 2.0
    shown = np.asarray(getattr(layer, "shown", np.ones(n, bool)), dtype=bool).reshape(-1)
    if shown.size != n:
        shown = np.ones(n, bool)
    keep = shown & (np.abs(depth) <= np.maximum(radius, half))
    if not keep.any():
        return None
    # The disc the plane cuts from each point's sphere; a point merely within half a slice of
    # the plane keeps a small dot.
    cut = np.sqrt(np.clip(radius[keep] ** 2 - depth[keep] ** 2, 0.0, None))
    steps = np.asarray(plane.steps_mm, dtype=float)
    radii = np.stack([cut / max(steps[0], 1e-9), cut / max(steps[1], 1e-9)], axis=1)
    face = np.asarray(getattr(layer, "face_color", np.ones((n, 4))), dtype=float).reshape(-1, 4)
    edge = getattr(layer, "border_color", None)
    if edge is None:
        edge = getattr(layer, "edge_color", None)
    edge = np.asarray(edge if edge is not None else face, dtype=float).reshape(-1, 4)
    pick = np.flatnonzero(keep)
    width = np.asarray(getattr(layer, "border_width", 0.05), dtype=float).reshape(-1)
    rel = bool(getattr(layer, "border_width_is_relative", True))
    w = float(np.median(width)) if width.size else 0.0
    return Overlay(
        kind="points",
        coords=np.stack([rows[keep], cols[keep]], axis=1),
        colors=face[pick % len(face)],
        radii=radii,
        edge_colors=edge[pick % len(edge)],
        edge_width=(1.0 if rel and w > 0 else max(w, 0.0)) if w > 0 else 0.0,
        opacity=float(getattr(layer, "opacity", 1.0) or 1.0),
        name=str(getattr(layer, "name", "")),
    )


class SurfaceCutter:
    """Cuts surface layers with view planes, caching each mesh in the views' frame."""

    def __init__(self) -> None:
        self._cache: dict[int, tuple[tuple, np.ndarray, np.ndarray, np.ndarray]] = {}

    def clear(self) -> None:
        self._cache.clear()

    def _prepared(self, layer: Any, plane: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """``(vertices in plane millimetres (relative to the crosshair), faces, vertex RGBA)``."""
        try:
            vertices = np.asarray(layer.vertices, dtype=float)
            faces = np.asarray(layer.faces, dtype=np.int64)
        except Exception:  # noqa: BLE001
            return None
        if vertices.ndim != 2 or vertices.shape[1] < 3 or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            return None
        matrix = _world_matrix(layer)
        colors_key = self._colour_key(layer)
        key = (id(layer.vertices), id(layer.faces), None if matrix is None else matrix.tobytes(),
               np.asarray(plane.matrix).tobytes(), np.asarray(plane.spacing).tobytes(), colors_key)
        hit = self._cache.get(id(layer))
        if hit is None or hit[0] != key:
            world = _to_world(layer, vertices)
            if world is None:
                return None
            # In the bound volume's spacing-scaled voxel axes, not yet relative to the crosshair:
            # moving the crosshair then only shifts them (no re-projection of the whole mesh).
            inv = np.linalg.inv(np.asarray(plane.matrix, dtype=float))
            vox_mm = (world @ inv[:3, :3].T + inv[:3, 3]) * np.asarray(plane.spacing, dtype=float)
            hit = (key, vox_mm, faces, self._vertex_colours(layer, len(vertices)))
            self._cache[id(layer)] = hit
        centre_mm = np.asarray(plane.centre_vox, dtype=float) * np.asarray(plane.spacing, dtype=float)
        return hit[1] - centre_mm, hit[2], hit[3]

    @staticmethod
    def _colour_key(layer: Any) -> tuple:
        vc = getattr(layer, "vertex_colors", None)
        cmap = getattr(getattr(layer, "colormap", None), "name", "")
        return (id(vc) if vc is not None else None, str(cmap),
                tuple(np.asarray(getattr(layer, "contrast_limits", (0, 1)), float).tolist()),
                id(getattr(layer, "vertex_values", None)), float(getattr(layer, "gamma", 1.0) or 1.0))

    @staticmethod
    def _vertex_colours(layer: Any, n: int) -> np.ndarray:
        vc = getattr(layer, "vertex_colors", None)
        if vc is not None:
            vc = np.asarray(vc, dtype=float)
            if vc.ndim == 2 and len(vc) == n and vc.shape[1] in (3, 4):
                return vc if vc.shape[1] == 4 else np.hstack([vc, np.ones((n, 1))])
        values = np.asarray(getattr(layer, "vertex_values", np.ones(n)), dtype=float).reshape(-1)
        if values.size != n:
            values = np.ones(n)
        lo, hi = (np.asarray(getattr(layer, "contrast_limits", (0.0, 1.0)), dtype=float).tolist() + [0.0, 1.0])[:2]
        norm = np.clip((values - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.ones(n)
        gamma = float(getattr(layer, "gamma", 1.0) or 1.0)
        if gamma != 1.0:
            norm = norm ** gamma
        try:
            return np.asarray(layer.colormap.map(norm), dtype=float).reshape(n, 4)
        except Exception:  # noqa: BLE001
            return np.tile([1.0, 1.0, 0.0, 1.0], (n, 1))

    def cut(self, layer: Any, plane: Any) -> list[Overlay]:
        """What *plane* shows of *layer*'s mesh, by its display parts (the Meshlab panel's
        Faces / Wireframe / Points, as in 3D):

        * **faces**: the cut, where the plane slices the surface (solid lines);
        * **wireframe**: the edges of the triangles the plane passes through or that lie within
          half a slice of it, flattened into the view (fainter under the cut when faces are on);
        * **points**: the vertices of those triangles, as dots of the point size.
        """
        prepared = self._prepared(layer, plane)
        if prepared is None:
            return []
        faces_on, wire_on, points_on, point_size = surface_parts(layer)
        if not (faces_on or wire_on or points_on):
            return []
        mm, faces, colours = prepared
        normal = np.asarray(plane.frame.normal, dtype=float)
        depth = mm @ normal
        above = depth > 0.0
        fa = above[faces]
        crossing = fa.any(axis=1) & ~fa.all(axis=1)
        if not crossing.any():
            return []
        tri = faces[crossing]
        # The wireframe and the dots also show the triangles lying within half a slice of the
        # plane, so a surface running along the slice shows its mesh, not just a line.
        half = plane.thickness_mm() / 2.0
        near_tri = faces[crossing | (np.abs(depth)[faces].min(axis=1) <= half)]
        opacity = float(getattr(layer, "opacity", 1.0) or 1.0)
        name = str(getattr(layer, "name", ""))
        out: list[Overlay] = []
        if wire_on:
            edges = np.sort(np.concatenate([near_tri[:, [0, 1]], near_tri[:, [1, 2]], near_tri[:, [2, 0]]]), axis=1)
            edges = np.unique(edges, axis=0)
            ra, ca, _ = plane.mm_to_pixels(mm[edges[:, 0]])
            rb, cb, _ = plane.mm_to_pixels(mm[edges[:, 1]])
            out.append(Overlay(kind="lines", coords=np.stack([ra, ca, rb, cb], axis=1),
                               colors=colours[edges[:, 0]], width=1.0,
                               opacity=opacity * (0.45 if faces_on else 1.0), name=name))
        if faces_on:
            d = depth[tri]                      # (K, 3)
            p = mm[tri]                         # (K, 3, 3)
            ends = []
            hits = []
            for i, j in ((0, 1), (1, 2), (2, 0)):
                di, dj = d[:, i], d[:, j]
                hit = (di > 0.0) != (dj > 0.0)
                denom = np.where(hit, di - dj, 1.0)
                t = np.where(hit, di / denom, 0.0)[:, None]
                ends.append(p[:, i] + t * (p[:, j] - p[:, i]))
                hits.append(hit)
            ends_arr = np.stack(ends, axis=1)    # (K, 3 edges, 3)
            hits_arr = np.stack(hits, axis=1)    # (K, 3)
            order = np.argsort(~hits_arr, axis=1, kind="stable")[:, :2]
            k = np.arange(len(tri))
            a = ends_arr[k, order[:, 0]]
            b = ends_arr[k, order[:, 1]]
            ra, ca, _ = plane.mm_to_pixels(a)
            rb, cb, _ = plane.mm_to_pixels(b)
            out.append(Overlay(kind="lines", coords=np.stack([ra, ca, rb, cb], axis=1),
                               colors=colours[tri[:, 0]], width=2.0 if wire_on else 1.5,
                               opacity=opacity, name=name))
        if points_on:
            verts = np.unique(near_tri.reshape(-1))
            rows, cols, _ = plane.mm_to_pixels(mm[verts])
            out.append(Overlay(kind="points", coords=np.stack([rows, cols], axis=1), colors=colours[verts],
                               screen_size=float(point_size), edge_width=0.0, opacity=opacity, name=name))
        return out


def surface_parts(layer: Any) -> tuple[bool, bool, bool, float]:
    """``(faces, wireframe, points, point size)`` a surface is drawn with (the Meshlab panel's
    Display; a surface it never touched shows its faces)."""
    try:
        from nvitk.gui.mesh.layers import display_of

        disp = display_of(layer)
        return bool(disp.faces), bool(disp.wireframe), bool(disp.points), float(disp.point_size)
    except Exception:  # noqa: BLE001
        return True, False, False, 4.0


def view_overlays(viewer: Any, plane: Any, cutter: SurfaceCutter) -> list[Overlay]:
    """Everything the overlay layers draw on *plane*'s view, bottom to top."""
    out: list[Overlay] = []
    for layer in overlay_layers(viewer):
        try:
            if type(layer).__name__ == "Points":
                item = points_overlay(layer, plane)
                items = [item] if item is not None else []
            else:
                items = cutter.cut(layer, plane)
        except Exception:  # noqa: BLE001 (one odd layer must not blank the others)
            items = []
        out.extend(items)
    return out


def overlay_signature(layers: Any) -> tuple:
    """What, besides events, changes the overlays: each surface's display parts."""
    return tuple((id(layer), surface_parts(layer) if type(layer).__name__ == "Surface" else None)
                 for layer in layers)


__all__ = ["OVERLAY_TYPES", "Overlay", "SurfaceCutter", "overlay_layers", "overlay_signature", "points_overlay",
           "surface_parts", "view_overlays"]
