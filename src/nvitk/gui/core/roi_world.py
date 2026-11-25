"""A region-of-interest box in world space, for cropping or masking any layers.

The Labeling box (:class:`nvitk.gui.labels.roi_box.RoiBox`) is a voxel range on one grid, which
is what painting needs. To cut several layers at once (a CT and its PET, a TOF and its masks,
volumes on different grids) the box has to live in the space they share: :class:`WorldBox` is an
axis-aligned box in world (scanner) millimetres, and each layer is cropped to the voxels it
covers through that layer's own transform. It converts to and from the Labeling box, fits a
label map's extent, and saves to / loads from JSON.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nvitk.core.array import to_numpy

#: Name of the outline layer drawn for the box.
OUTLINE_NAME = "▭ ROI box"


def _spatial_dims(layer: Any) -> list[int]:
    from nvitk.gui.labels.roi_box import spatial_dims

    return list(spatial_dims(layer))


def _corners(lo: Sequence[float], hi: Sequence[float]) -> np.ndarray:
    return np.array(list(itertools.product(*[(float(a), float(b)) for a, b in zip(lo, hi)])), dtype=float)


@dataclass
class WorldBox:
    """Axis-aligned box in world coordinates (the three spatial world axes, in world order)."""

    lo: np.ndarray
    hi: np.ndarray

    def __post_init__(self) -> None:
        lo, hi = np.asarray(self.lo, dtype=float), np.asarray(self.hi, dtype=float)
        self.lo, self.hi = np.minimum(lo, hi), np.maximum(lo, hi)

    # construction
    @classmethod
    def from_voxels(cls, layer: Any, lo: Sequence[float], hi: Sequence[float]) -> "WorldBox":
        """The world box enclosing voxel range ``[lo, hi)`` (spatial axes) of *layer*."""
        dims = _spatial_dims(layer)
        nd = int(layer.ndim)
        points = []
        for corner in _corners([float(v) - 0.5 for v in lo], [float(v) - 0.5 for v in hi]):
            full = np.zeros(nd)
            full[dims] = corner
            points.append(np.asarray(layer.data_to_world(full), dtype=float)[dims])
        points = np.asarray(points)
        return cls(points.min(axis=0), points.max(axis=0))

    @classmethod
    def from_layer(cls, layer: Any) -> "WorldBox":
        """The box around all of *layer*."""
        shape = [int(layer.data.shape[d]) for d in _spatial_dims(layer)]
        return cls.from_voxels(layer, [0, 0, 0], shape)

    @classmethod
    def from_labels(cls, layer: Any, label: int | None = None, *, margin_mm: float = 0.0) -> "WorldBox":
        """The box around *layer*'s non-zero voxels (or one *label*), widened by *margin_mm*."""
        data = to_numpy(layer.data)
        dims = _spatial_dims(layer)
        if data.ndim > len(dims):
            # 3D+t: the union over time.
            other = tuple(i for i in range(data.ndim) if i not in dims)
            data = np.any(data != 0, axis=other) if label is None else np.any(data == int(label), axis=other)
        else:
            data = data != 0 if label is None else data == int(label)
        hits = np.argwhere(data)
        if hits.size == 0:
            raise ValueError(f"“{getattr(layer, 'name', 'layer')}” has no {'labelled' if label is None else f'label {label}'} voxels.")
        box = cls.from_voxels(layer, hits.min(axis=0), hits.max(axis=0) + 1)
        return cls(box.lo - margin_mm, box.hi + margin_mm)

    @classmethod
    def from_roi_box(cls, viewer: Any) -> "WorldBox | None":
        """The Labeling box, as a world box (``None`` when it is not set)."""
        from nvitk.gui.labels.roi_box import roi_box

        box = roi_box(viewer)
        layer = box.layer
        if not box.active or layer is None:
            return None
        return cls.from_voxels(layer, box.lo, box.hi)

    # use
    def voxel_slices(self, layer: Any) -> tuple[slice, ...] | None:
        """*layer*'s index covering the box (full slices on non-spatial axes); ``None`` if they
        do not overlap."""
        dims = _spatial_dims(layer)
        nd = int(layer.ndim)
        shape = layer.data.shape
        points = []
        for corner in _corners(self.lo, self.hi):
            world = np.zeros(nd)
            world[dims] = corner
            points.append(np.asarray(layer.world_to_data(world), dtype=float)[dims])
        points = np.asarray(points)
        lo = np.floor(points.min(axis=0) + 0.5).astype(int)
        hi = np.ceil(points.max(axis=0) + 0.5).astype(int)
        index: list[slice] = [slice(None)] * nd
        for d, a, b in zip(dims, lo, hi):
            a, b = max(int(a), 0), min(int(b), int(shape[d]))
            if b <= a:
                return None
            index[d] = slice(a, b)
        return tuple(index)

    def size_mm(self) -> np.ndarray:
        """Edge lengths in world units."""
        return self.hi - self.lo

    def to_dict(self) -> dict[str, Any]:
        return {"lo": [float(v) for v in self.lo], "hi": [float(v) for v in self.hi], "space": "world"}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldBox":
        return cls(np.asarray(data["lo"], dtype=float), np.asarray(data["hi"], dtype=float))

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> "WorldBox":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_roi_box(self, viewer: Any, layer: Any) -> bool:
        """Set the Labeling box to this box on *layer*'s grid; False if they do not overlap."""
        from nvitk.gui.labels.roi_box import roi_box

        index = self.voxel_slices(layer)
        if index is None:
            return False
        dims = _spatial_dims(layer)
        roi_box(viewer).set(layer, [index[d].start for d in dims], [index[d].stop for d in dims])
        return True


# viewer-level state and operations
def current_box(viewer: Any) -> WorldBox | None:
    """The ROI box set on *viewer* (``None`` when there is none)."""
    return getattr(viewer, "_nvitk_world_box", None)


def set_box(viewer: Any, box: WorldBox | None, *, outline: bool = True) -> None:
    """Make *box* the viewer's ROI box (and draw / remove its outline)."""
    try:
        viewer._nvitk_world_box = box
    except Exception:  # noqa: BLE001
        pass
    if outline and box is not None:
        show_outline(viewer, box)
    elif box is None:
        hide_outline(viewer)


def show_outline(viewer: Any, box: WorldBox) -> Any:
    """Draw *box*'s twelve edges as a non-editable shapes layer."""
    corners = _corners(box.lo, box.hi)
    edges = []
    for i, a in enumerate(corners):
        for b in corners[i + 1:]:
            if int(np.sum(~np.isclose(a, b))) == 1:
                edges.append(np.array([a, b]))
    existing = next((l for l in viewer.layers if l.name == OUTLINE_NAME), None)
    if existing is None:
        active = viewer.layers.selection.active
        existing = viewer.add_shapes(edges, shape_type="line", name=OUTLINE_NAME, edge_color="#4dd2ff", edge_width=0.8)
        existing.editable = False
        existing._nvitk_internal = True
        if active is not None:
            viewer.layers.selection.active = active
    else:
        existing.data = edges
    return existing


def hide_outline(viewer: Any) -> None:
    for layer in list(viewer.layers):
        if layer.name == OUTLINE_NAME:
            viewer.layers.remove(layer)


def _add_like(viewer: Any, layer: Any, data: Any, lo: Sequence[int], suffix: str) -> Any:
    from nvitk.gui.core.spatial import crop_placement
    from nvitk.gui.labels.visibility import copy_layer_metadata_for_output

    placement, metadata = crop_placement(layer, lo)
    kwargs = {"name": f"{layer.name}{suffix}", "metadata": copy_layer_metadata_for_output(metadata), **placement}
    if type(layer).__name__ == "Labels":
        return viewer.add_labels(data, colormap=layer.colormap, opacity=float(layer.opacity), **kwargs)
    extra = {"colormap": getattr(layer, "colormap", None), "blending": getattr(layer, "blending", None),
             "opacity": float(getattr(layer, "opacity", 1.0))}
    if getattr(layer, "contrast_limits", None) is not None:
        extra["contrast_limits"] = tuple(float(v) for v in layer.contrast_limits)
    return viewer.add_image(data, **kwargs, **{k: v for k, v in extra.items() if v is not None})


def crop_layers(viewer: Any, layers: Sequence[Any], box: WorldBox) -> tuple[list[Any], list[str]]:
    """A cropped copy of each layer, placed where it came from; ``(new layers, skipped names)``."""
    made, skipped = [], []
    for layer in layers:
        if type(layer).__name__ not in ("Image", "Labels"):
            skipped.append(str(layer.name))
            continue
        index = box.voxel_slices(layer)
        if index is None:
            skipped.append(str(layer.name))
            continue
        data = to_numpy(layer.data[index]).copy()
        lo = [ix.indices(int(n))[0] for ix, n in zip(index, layer.data.shape)]
        made.append(_add_like(viewer, layer, data, lo, "_crop"))
    from nvitk.gui.core.journal import record

    record(viewer, "roi_crop", "crop", params={"box": box.to_dict()}, inputs=[str(l.name) for l in layers if str(l.name) not in skipped],
           outputs=[str(l.name) for l in made], label="Crop to ROI box")
    return made, skipped


def mask_outside(viewer: Any, layers: Sequence[Any], box: WorldBox, *, fill: float = 0.0) -> tuple[list[Any], list[str]]:
    """A copy of each layer with everything outside the box set to *fill* (same grid)."""
    made, skipped = [], []
    for layer in layers:
        if type(layer).__name__ not in ("Image", "Labels"):
            skipped.append(str(layer.name))
            continue
        index = box.voxel_slices(layer)
        if index is None:
            skipped.append(str(layer.name))
            continue
        src = to_numpy(layer.data)
        out = np.full_like(src, fill)
        out[index] = src[index]
        made.append(_add_like(viewer, layer, out, [0] * src.ndim, "_inbox"))
    from nvitk.gui.core.journal import record

    record(viewer, "roi_mask", "mask", params={"box": box.to_dict(), "fill": fill},
           inputs=[str(l.name) for l in layers if str(l.name) not in skipped], outputs=[str(l.name) for l in made],
           label="Mask outside ROI box")
    return made, skipped


__all__ = [
    "OUTLINE_NAME",
    "WorldBox",
    "crop_layers",
    "current_box",
    "hide_outline",
    "mask_outside",
    "set_box",
    "show_outline",
]
