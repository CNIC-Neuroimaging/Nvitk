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
    """Voxel spacing in layer axis order.

    Prefer file metadata / affine column norms over Napari ``scale``. When a
    layer is displayed with an affine, Napari typically leaves ``scale`` at
    ``(1,1,1)``, which must not override the real mm spacing.
    """
    meta = nvitk_metadata_from_layer(layer)
    sp = meta.get("spacing")
    if sp is not None:
        try:
            seq = tuple(float(v) for v in sp)
            if len(seq) >= 3:
                return seq[:3]
            if seq:
                return seq
        except Exception:
            pass
    if all(k in meta for k in ("x_res", "y_res", "z_res")):
        return (float(meta["x_res"]), float(meta["y_res"]), float(meta["z_res"]))

    aff = layer_affine(layer)
    aff_norms: tuple[float, ...] | None = None
    if aff is not None:
        r = aff[:3, :3]
        aff_norms = tuple(float(np.linalg.norm(r[:, i])) for i in range(3))

    scale = getattr(layer, "scale", None)
    scale_t: tuple[float, ...] | None = None
    if scale is not None and len(scale) >= 3:
        scale_t = tuple(float(x) for x in scale[:3])

    # Identity affine + meaningful scale → 4D scale-only display path.
    if (
        aff_norms is not None
        and scale_t is not None
        and all(abs(a - 1.0) < 1e-6 for a in aff_norms)
        and any(abs(s - 1.0) > 1e-6 for s in scale_t)
    ):
        return scale_t

    if aff_norms is not None and max(aff_norms) > 1e-8:
        return aff_norms
    if scale_t is not None:
        return scale_t
    return None


def layer_spatial_kwargs(layer: Any) -> dict[str, Any]:
    """Keyword args for ``add_image`` / ``add_labels`` / ``add_surface``."""
    kwargs = {}
    aff = getattr(layer, "affine", None)
    if aff is not None:
        kwargs["affine"] = to_numpy(aff).astype(float)
    elif getattr(layer, "scale", None) is not None:
        kwargs["scale"] = tuple(float(x) for x in layer.scale)
    return kwargs


def data_indices_to_world(points: np.ndarray, layer: Any) -> np.ndarray:
    """Map voxel indices in layer data axis order to world coordinates."""
    pts = to_numpy(points).astype(np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[1] < 3:
        return pts
    aff = layer_affine(layer)
    if aff is None:
        return pts[:, :3]
    a = to_numpy(aff).astype(np.float64)
    if a.shape != (4, 4):
        return pts[:, :3]
    homog = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1))], axis=1)
    return (homog @ a.T)[:, :3]


def spatial_vector_3d(vector: Any, event: Any | None = None) -> np.ndarray | None:
    """Map Napari ndim-D *vector* (e.g. view_direction) to 3 displayed spatial components."""
    if vector is None:
        return None
    arr = to_numpy(vector).astype(np.float64).ravel()
    if arr.size == 3:
        return arr.reshape(3)
    displayed = getattr(event, "dims_displayed", None) if event is not None else None
    if displayed is not None:
        disp = [int(d) for d in displayed if 0 <= int(d) < arr.size]
        if len(disp) >= 3:
            return arr[disp[:3]].reshape(3)
    if arr.size >= 3:
        return arr[:3].reshape(3)
    return None


def world_vector_to_data(
    layer: Any,
    vector: Any,
    event: Any | None = None,
) -> np.ndarray | None:
    """Map a Napari world direction to a unit vector in layer data (voxel) space."""
    v = spatial_vector_3d(vector, event)
    if v is None:
        return None
    aff = layer_affine(layer)
    if aff is None:
        n = float(np.linalg.norm(v))
        return v.reshape(3) / n if n > 1e-9 else None
    r = to_numpy(aff).astype(np.float64)[:3, :3]
    try:
        v_d = np.linalg.solve(r, v.reshape(3))
    except np.linalg.LinAlgError:
        v_d = np.linalg.pinv(r) @ v.reshape(3)
    n = float(np.linalg.norm(v_d))
    return (v_d / n).astype(np.float64) if n > 1e-9 else None


def view_direction_into_scene(
    layer: Any,
    view_direction: Any,
    event: Any | None = None,
) -> np.ndarray | None:
    """
    Unit view ray direction in data space, from the camera through the scene.

    Napari ``dims.view_direction`` points from the scene toward the camera; the pick
    ray uses the opposite (into the volume).
    """
    d = world_vector_to_data(layer, view_direction, event)
    if d is None:
        return None
    return -d


def world_to_data_coords(layer: Any, position: Any) -> np.ndarray | None:
    """Map a Napari world click to sub-voxel data coordinates on *layer*'s 3D grid."""
    if position is None:
        return None
    shape = getattr(getattr(layer, "data", None), "shape", None)
    if shape is None or len(shape) < 3:
        return None
    try:
        data_pos = layer.world_to_data(position)
        pos = to_numpy(data_pos).astype(np.float64).ravel()
    except Exception:
        pos = to_numpy(position).astype(np.float64).ravel()
        aff = layer_affine(layer)
        if aff is not None and pos.size >= 3:
            inv = np.linalg.inv(to_numpy(aff).astype(np.float64))
            homog = np.array([pos[0], pos[1], pos[2], 1.0], dtype=np.float64)
            pos = (inv @ homog)[:3]
    if pos.size < 3:
        return None
    return np.array(
        [
            np.clip(float(pos[0]), 0.0, float(shape[0] - 1)),
            np.clip(float(pos[1]), 0.0, float(shape[1] - 1)),
            np.clip(float(pos[2]), 0.0, float(shape[2] - 1)),
        ],
        dtype=np.float64,
    )


def world_to_data_indices(layer: Any, position: Any) -> np.ndarray | None:
    """Map a Napari world click to rounded voxel indices on *layer*'s 3D grid."""
    coords = world_to_data_coords(layer, position)
    if coords is None:
        return None
    return np.round(coords).astype(np.float32)


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
        from nvitk.io._common import default_nifti_axes

        axes = default_nifti_axes(arr.ndim)
    return Image(
        data=arr,
        metadata=meta,
        axes=axes,
        name=getattr(layer, "name", "layer"),
    )


def _time_axis_index(layer: Any) -> int:
    """Array axis index for cardiac phase (``T`` / ``C``) on a 4D+ layer."""
    from nvitk.gui.core.orientation import _axes_string_from_layer

    axes = (_axes_string_from_layer(layer) or "").upper()
    nd = int(getattr(getattr(layer, "data", None), "ndim", 0) or 0)
    if len(axes) == nd:
        if "T" in axes:
            return int(axes.index("T"))
        if "C" in axes:
            return int(axes.index("C"))
    return max(0, nd - 1)


def spatial_volume_image(layer: Any, data: np.ndarray | None = None) -> Image:
    """3D spatial volume from a 3D or 4D+ layer (cardiac phase index 0)."""
    img = layer_to_image(layer, data)
    arr = to_numpy(img.data)
    if arr.ndim <= 3:
        return img

    t_ax = _time_axis_index(layer)
    sl = [slice(None)] * arr.ndim
    sl[t_ax] = 0
    data3 = np.ascontiguousarray(arr[tuple(sl)])
    meta = dict(img.metadata or {})
    meta["shape"] = tuple(int(x) for x in data3.shape)
    spatial_axes = [i for i in range(arr.ndim) if i != t_ax]
    axes_str = str(meta.get("axes") or "").upper()
    if len(axes_str) == arr.ndim:
        meta["axes"] = "".join(axes_str[i] for i in spatial_axes)
    else:
        from nvitk.io._common import default_nifti_axes

        meta["axes"] = default_nifti_axes(data3.ndim)

    scale = getattr(layer, "scale", None)
    if scale is not None and len(scale) >= arr.ndim:
        sc = tuple(float(x) for x in scale)
        sp3 = tuple(sc[i] for i in spatial_axes[:3])
        meta["spacing"] = sp3
        meta["x_res"], meta["y_res"], meta["z_res"] = sp3[0], sp3[1], sp3[2]

    aff = meta.get("affine")
    if aff is not None:
        aff_np = to_numpy(aff).astype(float)
        if aff_np.shape == (4, 4) and arr.ndim > 3:
            meta["affine"] = np.eye(4, dtype=float)
        else:
            meta["affine"] = aff_np

    return Image(
        data=data3,
        metadata=meta,
        axes=meta.get("axes"),
        name=getattr(layer, "name", "layer"),
    )


def layers_need_resample(
    mask_layer: Any,
    reference_layer: Any,
    mask_img: Image,
    reference_img: Image,
) -> bool:
    """True when mask and reference are not on the same voxel grid."""
    mask_sp = spatial_volume_image(mask_layer, to_numpy(mask_img.data)) if mask_img.ndim > 3 else mask_img
    ref_sp = (
        spatial_volume_image(reference_layer)
        if reference_img.ndim > 3
        else reference_img
    )
    if tuple(mask_sp.data.shape) != tuple(ref_sp.data.shape):
        return True
    aff_m = layer_affine(mask_layer)
    aff_r = layer_affine(reference_layer)
    if aff_m is None or aff_r is None:
        return tuple(mask_sp.data.shape) != tuple(ref_sp.data.shape)
    return not np.allclose(aff_m, aff_r, rtol=0, atol=1e-3)


def align_mask_to_reference_layer(
    mask_layer: Any,
    reference_layer: Any,
    mask_data = None,
    *,
    order = 0,
) -> tuple[Image, Image, bool]:
    """
    Return ``(reference_image, mask_on_reference_grid, was_resampled)``.

    When shapes or affines differ, resamples the mask onto the reference grid
    (same convention as :meth:`~nvitk.measure.Measurer.align` ``mask_to_raw``).
    """
    ref_img = layer_to_image(reference_layer)
    mask_img = layer_to_image(mask_layer, mask_data)
    ref_sp = spatial_volume_image(reference_layer) if ref_img.ndim > 3 else ref_img
    mask_sp = (
        spatial_volume_image(mask_layer, mask_data)
        if mask_img.ndim > 3
        else mask_img
    )
    if not layers_need_resample(mask_layer, reference_layer, mask_img, ref_img):
        return ref_sp, mask_sp, False

    if layer_affine(mask_layer) is None or layer_affine(reference_layer) is None:
        if tuple(mask_sp.data.shape) == tuple(ref_sp.data.shape):
            return ref_sp, mask_sp, False
        raise ValueError(
            f"Shape mismatch ({mask_sp.data.shape} vs {ref_sp.data.shape}) but affine "
            f"metadata is missing on "
            f"'{getattr(mask_layer, 'name', 'mask')}' or "
            f"'{getattr(reference_layer, 'name', 'reference')}'; "
            "cannot resample for measurement."
        )

    from nvitk.measure.measurer import Measurer

    aligned = Measurer(ref_sp, mask_sp).align("mask_to_raw", order=order)
    return ref_sp, aligned.mask, True


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
    layer_type = type(layer).__name__
    if layer_type in ("Shapes", "Points", "Vectors", "Tracks", "Surface"):
        return f"{getattr(layer, 'name', layer_type)} ({layer_type})"

    aff = layer_affine(layer)
    if aff is None:
        sp = layer_spacing(layer) or ()
        if len(sp) >= 3:
            return (
                f"No affine — voxel spacing (axis 0,1,2): "
                f"{sp[0]:.4g}, {sp[1]:.4g}, {sp[2]:.4g} mm"
            )
        if len(sp) == 2:
            return (
                f"No affine — voxel spacing (axis 0,1): "
                f"{sp[0]:.4g}, {sp[1]:.4g} mm"
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
            from nvitk.gui.core.orientation import (
                axial_dim_order,
                napari_dim_order_3d,
                superior_voxel_axis,
            )

            ndim = int(layer.data.ndim)
            sup = superior_voxel_axis(aff, ndim)
            order = (
                napari_dim_order_3d(aff, 3)
                if ndim == 3
                else axial_dim_order(aff, ndim)
            )
            in_plane = [order[1], order[2]] if len(order) >= 3 else []
            plane = ", ".join(
                f"dim {d} ({codes[d] if d < len(codes) else '?'})" for d in in_plane
            )
            text += f"  —  View: axial (scroll axis {order[0]} = {codes[sup]})  |  In-plane: {plane}"
        except Exception:
            pass
    return text


def format_layer_spatial_info(layer: Any) -> str:
    """Human-readable spatial metadata for the active Napari layer."""
    name = getattr(layer, "name", "?")
    data = getattr(layer, "data", None)
    shape = tuple(int(s) for s in data.shape) if data is not None else ()
    dtype = getattr(data, "dtype", None)
    meta = nvitk_metadata_from_layer(layer)
    aff = layer_affine(layer)
    sp = layer_spacing(layer)
    scale = getattr(layer, "scale", None)
    scale_t = tuple(float(x) for x in scale) if scale is not None else None
    axes = meta.get("axes") or getattr(layer, "axis_labels", None)
    orient = meta.get("orientation")
    if orient is None and aff is not None:
        try:
            import nibabel as nib

            orient = "".join(nib.orientations.aff2axcodes(aff[:3, :3]))
        except Exception:
            orient = None
    origin = None
    direction = None
    if aff is not None:
        origin = tuple(float(aff[i, 3]) for i in range(min(3, aff.shape[0])))
        direction = aff[:3, :3].copy()
        for i in range(3):
            nrm = float(np.linalg.norm(direction[:, i]))
            if nrm > 0:
                direction[:, i] /= nrm
    fov = None
    if sp is not None and len(shape) >= len(sp):
        fov = tuple(float(shape[i]) * float(sp[i]) for i in range(len(sp)))

    lines = [
        f"Layer: {name}",
        f"shape: {shape}",
        f"dtype: {dtype}",
        f"axes: {axes!r}",
        f"orientation: {orient!r}",
        f"spacing (mm): {sp}",
        f"fov (mm): {fov}",
        f"origin: {origin}",
        f"Napari scale: {scale_t}",
        f"source: {meta.get('source')!r}",
    ]
    if direction is not None:
        lines.append("direction:")
        lines.append(np.array2string(np.asarray(direction, dtype=float), precision=4))
    if aff is not None:
        lines.append("affine:")
        lines.append(np.array2string(np.asarray(aff, dtype=float), precision=6))
    src = meta.get("affine_source")
    if src is not None:
        try:
            src_a = to_numpy(src).astype(float)
            if aff is None or not np.allclose(src_a, aff):
                lines.append("affine_source (file):")
                lines.append(np.array2string(src_a, precision=6))
        except Exception:
            pass
    return "\n".join(lines)


def _layer_for_orientation_status(viewer: Any) -> Any | None:
    """Prefer a volume layer over Shapes/Points overlays for the status bar."""
    active = viewer.layers.selection.active
    if active is not None and type(active).__name__ not in (
        "Shapes",
        "Points",
        "Vectors",
        "Tracks",
        "Surface",
    ):
        return active
    for lyr in reversed(list(viewer.layers)):
        if type(lyr).__name__ in ("Image", "Labels"):
            return lyr
    return viewer.layers[-1] if viewer.layers else None


def attach_orientation_status(viewer: Any, label_widget: Any) -> None:
    """Keep *label_widget* (Qt QLabel) in sync with the active layer / dims."""

    def _refresh() -> None:
        if not viewer.layers:
            label_widget.setText("Orientation: —")
            return
        layer = _layer_for_orientation_status(viewer)
        if layer is None:
            label_widget.setText("Orientation: —")
            return
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
