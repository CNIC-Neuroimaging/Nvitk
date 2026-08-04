"""Pipeline-default parameter presets for interactive GUI tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.backend import using
from nvitk.morphology import keep_largest_components


@dataclass(frozen=True)
class ToolPreset:
    """A named set of default parameter values for a GUI tool (key, display title, params, hint text)."""

    key: str
    title: str
    params: dict[str, Any]
    hint: str = ""


_PRESETS: dict[str, dict[str, ToolPreset]] = {
    "seg_region_grow": {
        "custom": ToolPreset("custom", "Custom", {}, "Manual seed and intensity fraction."),
        "qvtpy_default": ToolPreset(
            "qvtpy_default",
            "QVTpy default (frac=0.45)",
            {"threshold": 0.45},
            "Matches stage-4 default rg_intensity_frac.",
        ),
        "qvtpy_explore": ToolPreset(
            "qvtpy_explore",
            "QVTpy explore (frac=0.25)",
            {
                "threshold": 0.25,
                "mask_barrier_dilation_vox": 1,
                "centerline_barrier_dilation_vox": 3,
            },
            "ACA / MCA / PCA explore RG in QVTpy (mask barrier @1, CL @3).",
        ),
        "qvtpy_ica": ToolPreset(
            "qvtpy_ica",
            "QVTpy ICA test (frac=0.45)",
            {"threshold": 0.45, "seed_from_label": True},
            "Use label id 1 (LICA) under qvtpy-4dflow mapping as seed.",
        ),
    },
}


def presets_for_tool(tool_id: str) -> tuple[ToolPreset, ...]:
    """All registered presets for *tool_id*, empty if the tool has none."""
    return tuple(_PRESETS.get(tool_id, {}).values())


def preset_choices_for_tool(tool_id: str) -> tuple[str, ...]:
    """Display titles of *tool_id*'s presets, for populating a combo box."""
    return tuple(p.title for p in presets_for_tool(tool_id))


def preset_key_from_title(tool_id: str, title: str) -> str:
    """Resolve a preset's display *title* back to its key for *tool_id*; ``"custom"`` if unmatched."""
    for p in presets_for_tool(tool_id):
        if p.title == title:
            return p.key
    return "custom"


def get_preset(tool_id: str, key: str) -> ToolPreset | None:
    """Look up the preset registered under *key* for *tool_id*, or ``None``."""
    return _PRESETS.get(tool_id, {}).get(key)


def apply_preset_to_panel(panel: Any, tool_id: str, preset_key: str) -> None:
    """Copy preset values onto magicgui tool_panel widgets when they exist."""
    preset = get_preset(tool_id, preset_key)
    if preset is None:
        return
    for name, val in preset.params.items():
        w = getattr(panel, name, None)
        if w is not None and hasattr(w, "value"):
            w.value = val


def label_centroid_voxel(mask: np.ndarray, label_id: int) -> tuple[int, int, int]:
    """Voxel index of the largest connected component for *label_id*."""
    arr = to_numpy(mask)
    comp = arr == int(label_id)
    if not np.any(comp):
        raise ValueError(f"Label {label_id} not found in the active mask.")
    # Reduce to the largest connected component (base tool) so the seed centroid
    # is not dragged toward a stray island. ``comp`` is already host-side.
    try:
        with using("numpy"):
            comp = to_numpy(keep_largest_components(comp, n=1)).astype(bool)
    except Exception:
        pass
    coords = np.argwhere(comp)
    if coords.size == 0:
        raise ValueError(f"Empty component for label {label_id}.")
    center = np.mean(coords, axis=0)
    z, y, x = (int(round(center[0])), int(round(center[1])), int(round(center[2])))
    shape = arr.shape
    z = max(0, min(z, shape[0] - 1))
    y = max(0, min(y, shape[1] - 1))
    x = max(0, min(x, shape[2] - 1))
    return z, y, x


def cursor_voxel_indices(viewer: Any, layer: Any) -> tuple[int, int, int]:
    """Map Napari cursor position to voxel indices for *layer*."""
    pos = getattr(getattr(viewer, "cursor", None), "position", None)
    if pos is None:
        raise ValueError("Cursor position is not available.")
    data = to_numpy(layer.data)
    if data.ndim != 3:
        raise ValueError("Cursor seed requires a 3D layer.")
    # Napari cursor.position is typically in layer data coordinates (z, y, x).
    z = int(round(float(pos[0])))
    y = int(round(float(pos[1])))
    x = int(round(float(pos[2])))
    z = max(0, min(z, data.shape[0] - 1))
    y = max(0, min(y, data.shape[1] - 1))
    x = max(0, min(x, data.shape[2] - 1))
    return z, y, x
