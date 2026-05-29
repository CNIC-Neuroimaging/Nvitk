"""Live napari display filter: show only label ids selected in the label picker."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy

_NVITK_LABEL_SOURCE_KEY = "nvitk_label_source"
_NVITK_VISIBLE_IDS_KEY = "nvitk_visible_ids"
_NVITK_COLOR_BACKUP_KEY = "nvitk_label_color_backup"
_MAX_LABEL_LIKE_IDS = 64


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
    meta = _layer_metadata(layer)
    if _NVITK_LABEL_SOURCE_KEY in meta:
        return np.asarray(meta[_NVITK_LABEL_SOURCE_KEY])
    return to_numpy(layer.data)


def _compute_is_label_like(layer: Any) -> bool:
    if type(layer).__name__ == "Labels":
        return True
    if getattr(layer, "data", None) is None:
        return False
    arr = to_numpy(label_source_data(layer))
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
    meta = _layer_metadata(layer)
    src = meta.get(_NVITK_LABEL_SOURCE_KEY)
    if src is None:
        src = np.array(to_numpy(layer.data), copy=True)
        meta[_NVITK_LABEL_SOURCE_KEY] = src
        layer.metadata = meta
    return np.asarray(src)


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
    color_dict = {}
    for i, lid in enumerate(sorted(all_ids)):
        if lid in selected:
            color_dict[lid] = backup.get(lid, palette[i % len(palette)])
        else:
            color_dict[lid] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    layer.color = color_dict


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
        layer.data = np.asarray(src, dtype=to_numpy(layer.data).dtype, copy=False)
    layer.metadata = meta


__all__ = [
    "apply_label_visibility",
    "ensure_label_source",
    "infer_target_mode",
    "is_label_like_layer",
    "label_source_data",
    "layer_in_viewer",
    "restore_label_visibility",
    "unique_layer_labels",
]
