"""Spatial metadata and orientation helpers for Napari layers."""

from __future__ import annotations

from typing import Any

import numpy as np
from nvitk.core.array import to_numpy

from nvitk.types import Image


def nvitk_metadata_from_layer(layer: Any) -> dict[str, Any]:
    """Merged nvitk metadata (nested ``nvitk_metadata`` + top-level keys)."""
    meta = dict(getattr(layer, "metadata", None) or {})
    nested = meta.get("nvitk_metadata")
    if isinstance(nested, dict):
        out = dict(nested)
        for key, val in meta.items():
            if key != "nvitk_metadata":
                out[key] = val
        return out
    return meta


def layer_affine(layer: Any) -> np.ndarray | None:
    """4x4 transform used by the Napari layer (voxel index → world)."""
    aff = getattr(layer, "affine", None)
    if aff is not None:
        aff = to_numpy(aff).astype(float)
        if aff.shape == (4, 4):
            return aff
    meta = nvitk_metadata_from_layer(layer)
    src = meta.get("affine_source")
    if src is None:
        src = meta.get("affine")
    if src is not None:
        aff = to_numpy(src).astype(float)
        if aff.shape == (4, 4):
            return aff
    return None


def layer_spacing(layer: Any) -> tuple[float, ...] | None:
    """Voxel spacing in layer axis order."""
    scale = getattr(layer, "scale", None)
    if scale is not None:
        return tuple(float(x) for x in scale)
    meta = nvitk_metadata_from_layer(layer)
    sp = meta.get("spacing")
    if sp is not None:
        return tuple(float(x) for x in sp[:3])
    if all(k in meta for k in ("x_res", "y_res", "z_res")):
        return (float(meta["x_res"]), float(meta["y_res"]), float(meta["z_res"]))
    aff = layer_affine(layer)
    if aff is not None:
        r = aff[:3, :3]
        return tuple(float(np.linalg.norm(r[:, i])) for i in range(3))
    return None


def layer_spatial_kwargs(layer: Any) -> dict[str, Any]:
    """Keyword args for ``add_image`` / ``add_labels`` / ``add_surface``."""
    kwargs: dict[str, Any] = {}
    aff = getattr(layer, "affine", None)
    if aff is not None:
        kwargs["affine"] = to_numpy(aff).astype(float)
    elif getattr(layer, "scale", None) is not None:
        kwargs["scale"] = tuple(float(x) for x in layer.scale)
    return kwargs


def find_spatial_reference_layer(viewer: Any, layer: Any) -> Any:
    """Pick a layer with affine/scale on the same voxel grid as *layer*."""
    if layer_spatial_kwargs(layer):
        return layer
    target_shape = tuple(layer.data.shape)
    for lyr in viewer.layers:
        if lyr is layer:
            continue
        if tuple(lyr.data.shape) != target_shape:
            continue
        if layer_spatial_kwargs(lyr):
            return lyr
    return layer


def layer_to_image(layer: Any, data: np.ndarray | None = None) -> Image:
    """Build :class:`~nvitk.types.Image` using file metadata and layer geometry."""
    arr = to_numpy(data if data is not None else layer.data)
    meta = nvitk_metadata_from_layer(layer)
    aff = layer_affine(layer)
    if aff is not None:
        meta["affine"] = aff
    sp = layer_spacing(layer)
    if sp is not None and len(sp) >= min(3, arr.ndim):
        meta["spacing"] = sp
        meta["x_res"], meta["y_res"], meta["z_res"] = sp[0], sp[1], sp[2]
    axes = meta.get("axes")
    if axes is None:
        axes = "XYZ" if arr.ndim == 3 else "YX"
    return Image(
        data=arr,
        metadata=meta,
        axes=axes,
        name=getattr(layer, "name", "layer"),
    )


def _axis_direction_label(code: str) -> str:
    c = str(code).upper()
    pairs = {
        "R": "R+ = patient Right",
        "L": "L+ = patient Left",
        "A": "A+ = Anterior",
        "P": "P+ = Posterior",
        "S": "S+ = Superior",
        "I": "I+ = Inferior",
    }
    return pairs.get(c, f"{c}+")


def orientation_text(layer: Any, viewer: Any | None = None) -> str:
    """
    Describe voxel-axis world directions (standard NIfTI affine interpretation).

    Shown in the viewer to clarify R/L, A/P, S/I along array axes 0, 1, 2.
    """
    aff = layer_affine(layer)
    if aff is None:
        sp = layer_spacing(layer)
        if sp is not None:
            return (
                f"No affine — voxel spacing (axis 0,1,2): "
                f"{sp[0]:.4g}, {sp[1]:.4g}, {sp[2]:.4g} mm"
            )
        return "No affine — display uses voxel indices"

    try:
        import nibabel as nib
    except Exception:
        return "Affine set (install nibabel for axis labels)"

    try:
        codes = nib.orientations.aff2axcodes(aff[:3, :3])
    except Exception:
        return "Affine set — axis codes unavailable"

    parts = [
        f"Axis {i}: {_axis_direction_label(c)}"
        for i, c in enumerate(codes[:3])
    ]
    text = "  |  ".join(parts)
    if viewer is not None and getattr(layer, "data", None) is not None:
        try:
            from nvitk.gui.orientation import axial_dim_order, superior_voxel_axis

            ndim = int(layer.data.ndim)
            sup = superior_voxel_axis(aff, ndim)
            order = axial_dim_order(aff, ndim)
            in_plane = [order[1], order[2]] if len(order) >= 3 else []
            plane = ", ".join(
                f"dim {d} ({codes[d] if d < len(codes) else '?'})" for d in in_plane
            )
            text += f"  —  View: axial (scroll axis {order[0]} = {codes[sup]})  |  In-plane: {plane}"
        except Exception:
            pass
    return text


def attach_orientation_status(viewer: Any, label_widget: Any) -> None:
    """Keep *label_widget* (Qt QLabel) in sync with the active layer / dims."""

    def _refresh() -> None:
        if not viewer.layers:
            label_widget.setText("Orientation: —")
            return
        layer = viewer.layers.selection.active or viewer.layers[-1]
        label_widget.setText(orientation_text(layer, viewer))

    try:
        viewer.layers.selection.events.active.connect(_refresh)
        viewer.layers.events.inserted.connect(_refresh)
        viewer.layers.events.removed.connect(_refresh)
        viewer.dims.events.current_step.connect(_refresh)
        viewer.dims.events.order.connect(_refresh)
    except Exception:
        pass
    _refresh()
