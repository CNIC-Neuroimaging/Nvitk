"""A box on a volume — drawn on the canvas or in the orthogonal views, or typed —
to crop to, to read intensity ranges in, and to keep edits inside.

One box per viewer (:func:`roi_box`), on one voxel grid: it applies to every
layer with that spatial shape (a scan and its labels alike). Its corners are
voxel indices along the layer's spatial axes (the time axis of a 3D+t layer is
never cut). The Labeling tab's Box card edits it; the canvas shows it as an
outline (a rectangle on the slice in 2D, the twelve edges in 3D), the
orthogonal views as a rectangle in each plane.

Cropping (:func:`crop_layers`) adds a copy of each layer cut to the box, placed
where it came from: its affine — and the file affine export writes — shift by
the box's corner (:func:`~nvitk.gui.core.spatial.crop_placement`).
"""

from __future__ import annotations

import weakref
from typing import Any, Sequence

from qtpy.QtCore import QObject, Signal

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.gui.labels import editing as E

setup(globals())

#: The box's outline on the canvas (an internal Shapes layer).
BOX_LAYER_NAME = "▭ box (Labeling)"
_EDGE = "#4dd2ff"


def spatial_dims(layer: Any) -> list[int]:
    """*layer*'s spatial axes (every axis but the time axis of a 3D+t layer)."""
    t_ax = E.time_axis(layer)
    return [d for d in range(int(layer.data.ndim)) if d != t_ax]


def spatial_shape(layer: Any) -> tuple[int, ...]:
    return tuple(int(layer.data.shape[d]) for d in spatial_dims(layer))


class RoiBox(QObject):
    """The viewer's box (see the module docstring); ``changed`` fires on every change."""

    changed = Signal()

    def __init__(self, viewer: Any) -> None:
        super().__init__()
        self._viewer = viewer
        self._layer: Any = lambda: None
        self.shape: tuple[int, ...] | None = None
        self.lo: list[int] = []
        self.hi: list[int] = []
        self._outline: Any = lambda: None
        try:
            viewer.dims.events.current_step.connect(self._on_dims)
            viewer.dims.events.ndisplay.connect(self._on_dims)
            viewer.layers.events.removed.connect(self._on_removed)
        except Exception:  # noqa: BLE001
            pass

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self.shape is not None and bool(self.lo)

    @property
    def layer(self) -> Any | None:
        """The layer the box was drawn on (for its placement), while it is loaded."""
        layer = self._layer()
        return layer if layer is not None and layer in self._viewer.layers else None

    def applies(self, layer: Any) -> bool:
        """True when the box is on *layer*'s voxel grid."""
        try:
            return self.active and spatial_shape(layer) == self.shape
        except Exception:  # noqa: BLE001 — a layer without a voxel array
            return False

    def set(self, layer: Any, lo: Sequence[int], hi: Sequence[int], *, emit: bool = True) -> None:
        """The box from voxel *lo* (included) to *hi* (excluded) along *layer*'s
        spatial axes, clipped to the volume."""
        shape = spatial_shape(layer)
        if len(lo) != len(shape) or len(hi) != len(shape):
            raise ValueError(f"A box on this volume needs {len(shape)} corners' coordinates.")
        a = [min(max(int(v), 0), n - 1) for v, n in zip(lo, shape)]
        b = [min(max(int(v), a_ + 1), n) for v, a_, n in zip(hi, a, shape)]
        self._layer = weakref.ref(layer)
        self.shape, self.lo, self.hi = shape, a, b
        self.sync_outline()
        if emit:
            self.changed.emit()

    def set_axes(self, layer: Any, ranges: dict[int, tuple[int, int]], *, emit: bool = True) -> None:
        """Change the box along some of *layer*'s axes (``{axis: (lo, hi)}``, data
        axes); the others keep their range — the whole volume when the box was
        not on this grid yet."""
        dims = spatial_dims(layer)
        shape = spatial_shape(layer)
        if self.applies(layer):
            lo, hi = list(self.lo), list(self.hi)
        else:
            lo, hi = [0] * len(shape), list(shape)
        for axis, (a, b) in ranges.items():
            if int(axis) in dims:
                k = dims.index(int(axis))
                lo[k], hi[k] = sorted((int(a), int(b)))
                hi[k] = max(hi[k], lo[k] + 1)
        self.set(layer, lo, hi, emit=emit)

    def clear(self) -> None:
        self.shape, self.lo, self.hi = None, [], []
        self._layer = lambda: None
        self.sync_outline()
        self.changed.emit()

    def index(self, layer: Any) -> tuple[Any, ...]:
        """The box as an index into ``layer.data`` (the time axis whole)."""
        dims = spatial_dims(layer)
        out: list[Any] = [slice(None)] * int(layer.data.ndim)
        for k, d in enumerate(dims):
            out[d] = slice(self.lo[k], self.hi[k])
        return tuple(out)

    def limit(self, layer: Any, index: Sequence[Any]) -> tuple[Any, ...]:
        """*index* (an edit's region) cut down to the box, on its whole axes."""
        if not self.applies(layer):
            return tuple(index)
        dims = spatial_dims(layer)
        out = list(index)
        for k, d in enumerate(dims):
            ix = out[d]
            if isinstance(ix, slice):
                start, stop, _s = ix.indices(int(layer.data.shape[d]))
                start, stop = max(start, self.lo[k]), min(stop, self.hi[k])
                out[d] = slice(start, max(stop, start))
        return tuple(out)

    def contains(self, layer: Any, voxel: Sequence[int]) -> bool:
        if not self.applies(layer):
            return True
        return all(self.lo[k] <= int(voxel[d]) < self.hi[k] for k, d in enumerate(spatial_dims(layer)))

    def intensity_range(self, image: Any, low: float = 1.0, high: float = 99.0) -> tuple[float, float] | None:
        """The *low*–*high* percentiles of *image*'s voxels in the box (``None``
        when the box is not on its grid or holds nothing finite)."""
        if not self.applies(image):
            return None
        values = as_backend_array(image.data[self.index(image)]).astype(np.float32, copy=False).ravel()
        values = values[np.isfinite(values)]
        if int(values.size) == 0:
            return None
        if int(values.size) > 2_000_000:
            values = values[:: int(values.size) // 2_000_000 + 1]
        lo, hi = (float(v) for v in np.percentile(values, [float(low), float(high)]))
        if hi <= lo:
            lo, hi = float(values.min()), float(values.max())
        return (lo, hi) if hi > lo else (lo, lo + 1.0)

    def describe(self, layer: Any | None = None) -> str:
        """``40 × 50 × 60 voxels (32 × 40 × 72 mm)``."""
        if not self.active:
            return "No box."
        from nvitk.gui.core.spatial import layer_spacing

        n = [b - a for a, b in zip(self.lo, self.hi)]
        text = " × ".join(str(v) for v in n) + " voxels"
        layer = layer if layer is not None else self.layer
        spacing = layer_spacing(layer) if layer is not None else None
        if spacing is not None:
            dims = spatial_dims(layer)
            if len(spacing) >= max(dims) + 1:
                text += " (" + " × ".join(f"{v * abs(float(spacing[d])):.1f}" for v, d in zip(n, dims)) + " mm)"
        return text

    # ── the outline on the canvas ────────────────────────────────────────────

    def _on_dims(self, _event: Any = None) -> None:
        if self.active:
            self.sync_outline()

    def _on_removed(self, event: Any) -> None:
        if getattr(event, "value", None) is self._outline():
            self._outline = lambda: None

    def _corner_world(self, layer: Any, voxel: Sequence[float]) -> list[float]:
        return [float(v) for v in layer.data_to_world(list(voxel))]

    def _outline_shapes(self, layer: Any) -> tuple[list[Any], str, list[str]]:
        """The outline's shapes (world coordinates), their type and colours."""
        viewer = self._viewer
        dims = spatial_dims(layer)
        here = list(E.current_point(viewer, layer))
        # Voxel i spans i ± 0.5: the box's faces lie on those half-voxel edges.
        lo = {d: self.lo[k] - 0.5 for k, d in enumerate(dims)}
        hi = {d: self.hi[k] - 0.5 for k, d in enumerate(dims)}
        if E.is_3d_view(viewer):
            corners = {}
            for bits in range(1 << len(dims)):
                point = [float(v) for v in here]
                for k, d in enumerate(dims):
                    point[d] = hi[d] if bits >> k & 1 else lo[d]
                corners[bits] = self._corner_world(layer, point)
            edges = []
            for bits in corners:
                for k in range(len(dims)):
                    other = bits | (1 << k)
                    if other != bits:
                        edges.append([corners[bits], corners[other]])
            return edges, "line", [_EDGE] * len(edges)
        a, b = [int(d) for d in E.displayed_axes(viewer, layer)][-2:]
        normal = [d for d in dims if d not in (a, b)]
        inside = all(self.lo[dims.index(n)] <= here[n] < self.hi[dims.index(n)] for n in normal)
        ring = []
        for ra, rb in ((lo[a], lo[b]), (lo[a], hi[b]), (hi[a], hi[b]), (hi[a], lo[b])):
            point = [float(v) for v in here]
            point[a], point[b] = ra, rb
            ring.append(self._corner_world(layer, point))
        # Outside the box's depth: drawn faint, so it still shows where it is.
        return [ring], "polygon", [_EDGE if inside else "#4dd2ff59"]

    def sync_outline(self) -> None:
        """Draw the box on the canvas (or remove the outline when there is none)."""
        outline = self._outline()
        if outline is not None and outline not in self._viewer.layers:
            outline = None
        layer = self.layer
        if not self.active or layer is None:
            if outline is not None:
                try:
                    self._viewer.layers.remove(outline)
                except Exception:  # noqa: BLE001
                    pass
            self._outline = lambda: None
            return
        try:
            shapes, kind, colours = self._outline_shapes(layer)
        except Exception:  # noqa: BLE001 — a layer mid-change: next time
            return
        from nvitk.gui.core.spatial import layer_spacing

        spacing = layer_spacing(layer) or (1.0,)
        width = 0.6 * float(sum(abs(float(v)) for v in spacing[:3]) / max(len(spacing[:3]), 1))
        try:
            if outline is None or int(outline.ndim) != len(shapes[0][0]):
                if outline is not None:
                    self._viewer.layers.remove(outline)
                active = self._viewer.layers.selection.active
                outline = self._viewer.add_shapes(shapes, shape_type=kind, edge_color=colours, face_color="transparent",
                                                  edge_width=width, name=BOX_LAYER_NAME, opacity=0.9)
                outline._nvitk_internal = True
                self._outline = weakref.ref(outline)
                if active is not None and active in self._viewer.layers:
                    self._viewer.layers.selection.active = active
            else:
                outline.selected_data = set()
                outline.data = []
                outline.add(shapes, shape_type=kind, edge_color=colours, face_color="transparent", edge_width=width)
        except Exception:  # noqa: BLE001 — the outline is a convenience
            pass


def roi_box(viewer: Any) -> RoiBox:
    """The viewer's box (made on first use)."""
    box = getattr(viewer, "_nvitk_roi_box", None)
    if box is None:
        box = RoiBox(viewer)
        viewer._nvitk_roi_box = box
    return box


def crop_layer(viewer: Any, layer: Any, box: RoiBox) -> Any:
    """A copy of *layer* cut to *box*, placed where it came from (a new layer)."""
    from nvitk.gui.core.spatial import crop_placement
    from nvitk.gui.labels.visibility import copy_layer_metadata_for_output

    if not box.applies(layer):
        raise ValueError(f"“{layer.name}” is not on the box's grid.")
    index = box.index(layer)
    data = to_numpy(layer.data[index]).copy()
    lo = [ix.indices(int(n))[0] for ix, n in zip(index, layer.data.shape)]
    placement, metadata = crop_placement(layer, lo)
    kwargs = {"name": f"{layer.name}_crop", "metadata": copy_layer_metadata_for_output(metadata), **placement}
    kind = type(layer).__name__
    if kind == "Labels":
        out = viewer.add_labels(data, colormap=layer.colormap, opacity=float(layer.opacity), **kwargs)
    else:
        extra = {"colormap": getattr(layer, "colormap", None), "blending": getattr(layer, "blending", None),
                 "opacity": float(getattr(layer, "opacity", 1.0))}
        if getattr(layer, "contrast_limits", None) is not None and not getattr(layer, "_nvitk_label_like", False):
            extra["contrast_limits"] = tuple(float(v) for v in layer.contrast_limits)
        out = viewer.add_image(data, **kwargs, **{k: v for k, v in extra.items() if v is not None})
        if getattr(layer, "_nvitk_label_like", None) is True:
            out._nvitk_label_like = True
    return out


def crop_layers(viewer: Any, layers: Sequence[Any], box: RoiBox) -> list[Any]:
    """:func:`crop_layer` for each of *layers*, in order; the new layers."""
    return [crop_layer(viewer, layer, box) for layer in layers]


__all__ = [
    "BOX_LAYER_NAME",
    "RoiBox",
    "crop_layer",
    "crop_layers",
    "roi_box",
    "spatial_dims",
    "spatial_shape",
]
