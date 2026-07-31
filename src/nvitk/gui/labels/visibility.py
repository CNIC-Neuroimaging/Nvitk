"""Live napari display filter: show only label ids selected in the label picker."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy

_NVITK_LABEL_SOURCE_KEY = "nvitk_label_source"
_NVITK_VISIBLE_IDS_KEY = "nvitk_visible_ids"
_NVITK_COLOR_BACKUP_KEY = "nvitk_label_color_backup"
NVITK_LAYER_METADATA_KEYS = frozenset(
    {_NVITK_LABEL_SOURCE_KEY, _NVITK_VISIBLE_IDS_KEY, _NVITK_COLOR_BACKUP_KEY}
)
_MAX_LABEL_LIKE_IDS = 64


def copy_layer_metadata_for_output(metadata: dict[str, Any] | None) -> dict[str, Any]:
    """Copy user metadata for a new layer; drop live label-visibility cache."""
    if not metadata:
        return {}
    return {k: v for k, v in metadata.items() if k not in NVITK_LAYER_METADATA_KEYS}


def unique_layer_labels(data: np.ndarray, *, max_labels: int = 500) -> list[int]:
    flat = to_numpy(data).ravel()
    if flat.size == 0:
        return []
    if np.issubdtype(flat.dtype, np.floating):
        vals = np.unique(flat[np.isfinite(flat)])
        labels = [int(round(v)) for v in vals if v != 0]
    else:
        vals = np.unique(flat)
        labels = [int(v) for v in vals if int(v) != 0]
    labels.sort()
    if len(labels) > max_labels:
        return labels[:max_labels]
    return labels


def _layer_metadata(layer: Any) -> dict[str, Any]:
    meta = getattr(layer, "metadata", None)
    return dict(meta) if isinstance(meta, dict) else {}


def label_source_data(layer: Any) -> np.ndarray:
    """Full label array (before visibility filter), for tools that need the source."""
    current = to_numpy(layer.data)
    meta = _layer_metadata(layer)
    cached = meta.get(_NVITK_LABEL_SOURCE_KEY)
    if cached is not None:
        src = np.asarray(cached)
        if src.shape == current.shape:
            return src
    return current


def _compute_is_label_like(layer: Any) -> bool:
    if type(layer).__name__ in ("Shapes", "Points", "Vectors", "Tracks", "Surface"):
        return False
    if type(layer).__name__ == "Labels":
        return True
    if getattr(layer, "data", None) is None:
        return False
    try:
        arr = to_numpy(label_source_data(layer))
    except (ValueError, TypeError):
        return False
    if arr.ndim not in (2, 3):
        return False

    flat = arr.ravel()
    if flat.size == 0:
        return False

    if np.issubdtype(arr.dtype, np.floating):
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return False
        if np.unique(finite).size > _MAX_LABEL_LIKE_IDS:
            return False
        if not np.allclose(finite, np.round(finite), rtol=0, atol=1e-3):
            return False

    labels = unique_layer_labels(arr, max_labels=_MAX_LABEL_LIKE_IDS + 1)
    return 0 < len(labels) <= _MAX_LABEL_LIKE_IDS


def is_label_like_layer(layer: Any | None) -> bool:
    """True for napari Labels layers and discrete multi-value mask Image layers."""
    if layer is None:
        return False
    cached = getattr(layer, "_nvitk_label_like", None)
    if cached is not None:
        return bool(cached)
    result = _compute_is_label_like(layer)
    try:
        layer._nvitk_label_like = result
    except Exception:
        pass
    return result


def infer_target_mode(
    layer: Any | None,
    *,
    label_ids = None,
) -> str:
    """Infer process target from the active layer type and label selection."""
    if layer is None or not is_label_like_layer(layer):
        return "raw"

    ids = [int(x) for x in (label_ids or [])]
    all_ids = unique_layer_labels(label_source_data(layer))
    if not all_ids:
        return "raw"

    if len(all_ids) == 1:
        if ids and ids != all_ids:
            return "label"
        return "binary_mask"

    if ids:
        return "label"
    return "all_labels"


def ensure_label_source(layer: Any) -> np.ndarray:
    """Return unfiltered label data, caching a copy in layer metadata once."""
    current = to_numpy(layer.data)
    meta = _layer_metadata(layer)
    src = meta.get(_NVITK_LABEL_SOURCE_KEY)
    if src is not None:
        src_arr = np.asarray(src)
        if src_arr.shape == current.shape:
            return src_arr
        meta.pop(_NVITK_LABEL_SOURCE_KEY, None)
        meta.pop(_NVITK_VISIBLE_IDS_KEY, None)
    src = np.array(current, copy=True)
    meta[_NVITK_LABEL_SOURCE_KEY] = src
    layer.metadata = meta
    return src


def _visibility_key(selected_ids: list[int]) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in selected_ids))


def _apply_image_data_visibility(layer: Any, selected_ids: list[int]) -> None:
    src = ensure_label_source(layer)
    if not selected_ids:
        filtered = np.zeros_like(src)
    else:
        ids = np.asarray(selected_ids, dtype=src.dtype)
        filtered = np.where(np.isin(src, ids), src, 0)

    current = to_numpy(layer.data)
    if current.shape == filtered.shape and np.array_equal(current, filtered):
        return
    layer.data = np.asarray(filtered, dtype=current.dtype, copy=False)


def _apply_labels_color_visibility(layer: Any, selected_ids: list[int]) -> None:
    """Hide unselected ids via transparent colors (no volume copy / re-texture)."""
    try:
        from napari.utils.colormaps import label_colormap
    except Exception:
        _apply_image_data_visibility(layer, selected_ids)
        return

    src = label_source_data(layer)
    all_ids = unique_layer_labels(src)
    selected = set(int(x) for x in selected_ids)
    meta = _layer_metadata(layer)
    backup = meta.get(_NVITK_COLOR_BACKUP_KEY)
    if backup is None:
        backup = {int(k): np.asarray(v) for k, v in dict(getattr(layer, "color", {})).items()}
        meta[_NVITK_COLOR_BACKUP_KEY] = backup
        layer.metadata = meta

    palette = label_colormap(max(len(all_ids), 1))
    colors = np.asarray(getattr(palette, "colors", None), dtype=np.float32)
    if colors.ndim != 2 or colors.shape[0] == 0:
        colors = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    color_dict = {}
    for i, lid in enumerate(sorted(all_ids)):
        if lid in selected:
            color_dict[lid] = backup.get(lid, colors[i % len(colors)])
        else:
            color_dict[lid] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    layer.color = color_dict


def get_label_color(layer: Any, label_id: int) -> np.ndarray:
    """Return RGBA (float 0–1) for *label_id*, preferring the visibility backup."""
    lid = int(label_id)
    meta = _layer_metadata(layer)
    backup = meta.get(_NVITK_COLOR_BACKUP_KEY)
    if isinstance(backup, dict) and lid in backup:
        return np.asarray(backup[lid], dtype=np.float32)

    colors_dict = dict(getattr(layer, "color", {}) or {})
    if lid in colors_dict:
        rgba = np.asarray(colors_dict[lid], dtype=np.float32)
        if rgba.shape[-1] >= 3 and float(rgba[-1] if rgba.shape[-1] > 3 else 1.0) > 0:
            return rgba

    # Fall back to napari's default label palette index.
    try:
        from napari.utils.colormaps import label_colormap

        all_ids = unique_layer_labels(label_source_data(layer))
        palette = label_colormap(max(len(all_ids), 1))
        colors = np.asarray(getattr(palette, "colors", None), dtype=np.float32)
        if colors.ndim != 2 or colors.shape[0] == 0:
            return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        if lid in all_ids:
            i = sorted(all_ids).index(lid)
            return np.asarray(colors[i % len(colors)], dtype=np.float32)
        return np.asarray(colors[0], dtype=np.float32)
    except Exception:
        return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def set_label_color(
    layer: Any,
    label_id: int,
    rgba: Any,
    *,
    selected_ids: list[int] | None = None,
) -> None:
    """Set the display color for one label id on a Napari Labels layer.

    Updates the color backup used by :func:`apply_label_visibility` so custom
    colors survive show/hide toggles in the label picker.
    """
    if type(layer).__name__ != "Labels" or not hasattr(layer, "color"):
        raise TypeError("Per-label colors require a Napari Labels layer.")

    lid = int(label_id)
    color = np.asarray(rgba, dtype=np.float32).reshape(-1)
    if color.size == 3:
        color = np.concatenate([color, np.array([1.0], dtype=np.float32)])
    if color.size < 4:
        raise ValueError("rgba must have 3 or 4 channels")
    color = color[:4].astype(np.float32, copy=False)
    # Napari expects 0–1 floats.
    if float(np.nanmax(color)) > 1.0:
        color = color / 255.0

    meta = _layer_metadata(layer)
    backup = meta.get(_NVITK_COLOR_BACKUP_KEY)
    if backup is None:
        backup = {int(k): np.asarray(v) for k, v in dict(getattr(layer, "color", {})).items()}
    backup = {int(k): np.asarray(v, dtype=np.float32) for k, v in dict(backup).items()}
    backup[lid] = color
    meta[_NVITK_COLOR_BACKUP_KEY] = backup
    # Force visibility re-apply so the live color dict picks up the backup.
    meta.pop(_NVITK_VISIBLE_IDS_KEY, None)
    layer.metadata = meta

    if selected_ids is not None:
        apply_label_visibility(layer, list(selected_ids))
        return

    live = dict(getattr(layer, "color", {}) or {})
    live[lid] = color
    layer.color = live


def ensure_labels_layer(viewer: Any, layer: Any) -> Any:
    """Return a Napari Labels layer for *layer*, converting Image masks in place.

    Discrete segmentation Image layers (e.g. QC ``seg_4dflow``) cannot use
    per-label ``layer.color``; this replaces them with an equivalent Labels
    layer at the same stack index so color editing works.
    """
    if layer is None:
        raise ValueError("No layer to convert.")
    if type(layer).__name__ == "Labels" and hasattr(layer, "color"):
        return layer
    if viewer is None:
        raise TypeError("A Napari viewer is required to convert Image → Labels.")

    from nvitk.gui.core.spatial import layer_spatial_kwargs

    name = str(getattr(layer, "name", "labels") or "labels")
    src = np.asarray(label_source_data(layer))
    labels = np.rint(src).astype(np.int32, copy=False) if np.issubdtype(src.dtype, np.floating) else src.astype(np.int32, copy=False)

    meta = dict(getattr(layer, "metadata", None) or {})
    for key in (_NVITK_LABEL_SOURCE_KEY, _NVITK_VISIBLE_IDS_KEY, _NVITK_COLOR_BACKUP_KEY):
        meta.pop(key, None)

    spatial = layer_spatial_kwargs(layer)
    opacity = float(getattr(layer, "opacity", 0.7) or 0.7)
    visible = bool(getattr(layer, "visible", True))
    blending = getattr(layer, "blending", "translucent")

    try:
        idx = list(viewer.layers).index(layer)
    except ValueError:
        idx = len(viewer.layers)

    # Drop Image first so the new Labels can reuse the same name.
    viewer.layers.remove(layer)
    new_layer = viewer.add_labels(
        labels,
        name=name,
        opacity=opacity,
        affine=spatial.get("affine"),
        scale=spatial.get("scale"),
        metadata=meta,
    )
    try:
        new_layer.visible = visible
        if blending is not None:
            new_layer.blending = blending
    except Exception:
        pass

    # Restore stack order.
    try:
        current = list(viewer.layers).index(new_layer)
        if current != idx:
            viewer.layers.move(current, idx)
    except Exception:
        pass

    try:
        viewer.layers.selection.active = new_layer
    except Exception:
        pass

    try:
        new_layer._nvitk_label_like = True
    except Exception:
        pass
    return new_layer


def apply_label_visibility(layer: Any, selected_ids: list[int]) -> None:
    """Show only *selected_ids* in the napari canvas."""
    key = _visibility_key(selected_ids)
    meta = _layer_metadata(layer)
    if meta.get(_NVITK_VISIBLE_IDS_KEY) == key:
        return
    meta[_NVITK_VISIBLE_IDS_KEY] = key
    layer.metadata = meta

    if type(layer).__name__ == "Labels" and hasattr(layer, "color"):
        _apply_labels_color_visibility(layer, selected_ids)
    else:
        _apply_image_data_visibility(layer, selected_ids)


def layer_in_viewer(layer: Any | None, viewer: Any | None) -> bool:
    """True if *layer* is still attached to *viewer*."""
    if layer is None or viewer is None:
        return False
    try:
        return layer in viewer.layers
    except Exception:
        return False


def restore_label_visibility(
    layer: Any | None,
    *,
    viewer = None,
) -> None:
    """Restore full label display after live filtering."""
    if layer is None:
        return
    if viewer is not None and not layer_in_viewer(layer, viewer):
        return
    meta = _layer_metadata(layer)
    meta.pop(_NVITK_VISIBLE_IDS_KEY, None)

    backup = meta.pop(_NVITK_COLOR_BACKUP_KEY, None)
    if backup is not None and hasattr(layer, "color"):
        layer.color = backup
        layer.metadata = meta
        return

    src = meta.pop(_NVITK_LABEL_SOURCE_KEY, None)
    if src is not None:
        current = to_numpy(layer.data)
        src_arr = np.asarray(src)
        if src_arr.shape == current.shape:
            layer.data = np.asarray(src_arr, dtype=current.dtype)
    layer.metadata = meta


__all__ = [
    "NVITK_LAYER_METADATA_KEYS",
    "apply_label_visibility",
    "copy_layer_metadata_for_output",
    "ensure_label_source",
    "ensure_labels_layer",
    "get_label_color",
    "infer_target_mode",
    "is_label_like_layer",
    "label_source_data",
    "layer_in_viewer",
    "restore_label_visibility",
    "set_label_color",
    "unique_layer_labels",
]
