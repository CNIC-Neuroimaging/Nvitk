"""Viewport layout presets: which panes are shown and how they are arranged.

Pane ids: ``"ax"``, ``"cor"``, ``"sag"`` (the three slice views, in the orthogonal panel's
order) and ``"3d"`` (Napari's canvas). A spec is a pane id, or ``(orientation, children)`` /
``(orientation, children, weights)`` with ``"h"`` side by side and ``"v"`` stacked (see
:class:`nvitk.gui.viz.ortho.layout.LayoutGrid`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

SLICE_PANES: tuple[str, ...] = ("ax", "cor", "sag")
PANE_3D = "3d"
ALL_PANES: tuple[str, ...] = (*SLICE_PANES, PANE_3D)


@dataclass(frozen=True)
class LayoutPreset:
    """A named arrangement of the viewport's panes."""

    key: str
    label: str
    icon: str
    spec: Any
    #: Offered as a one-click button on the top bar (the rest are in its menu).
    quick: bool = False


PRESETS: tuple[LayoutPreset, ...] = (
    LayoutPreset("2x2", "Four panes (2 x 2)", "layout_2x2",
                 ("v", [("h", ["ax", "cor"]), ("h", ["sag", "3d"])]), quick=True),
    LayoutPreset("3+1", "Three slices beside a large 3D view", "layout_3plus1",
                 ("h", [("v", ["ax", "cor", "sag"]), "3d"], [1, 3]), quick=True),
    LayoutPreset("1+3", "Large 3D view beside three slices", "layout_1plus3",
                 ("h", ["3d", ("v", ["ax", "cor", "sag"])], [3, 1])),
    LayoutPreset("3d_over", "Large 3D view over three slices", "layout_3d_over_axes",
                 ("v", ["3d", ("h", ["ax", "cor", "sag"])], [3, 1]), quick=True),
    LayoutPreset("3d", "3D view only", "layout_3d", "3d", quick=True),
    LayoutPreset("axes", "Three slices, side by side", "layout_axes_row",
                 ("h", ["ax", "cor", "sag"]), quick=True),
    LayoutPreset("axes_2x2", "Three slices in a grid", "layout_2x2",
                 ("v", [("h", ["ax", "cor"]), ("h", ["sag"])])),
    LayoutPreset("3d+ax", "3D view beside the first slice view", "layout_3d_axis",
                 ("h", ["ax", "3d"])),
    LayoutPreset("ax", "First slice view only (axial)", "layout_single", "ax"),
    LayoutPreset("cor", "Second slice view only (coronal)", "layout_single", "cor"),
    LayoutPreset("sag", "Third slice view only (sagittal)", "layout_single", "sag"),
)

PRESETS_BY_KEY: dict[str, LayoutPreset] = {p.key: p for p in PRESETS}
DEFAULT_PRESET = "2x2"


def spec_has_slices(spec: Any) -> bool:
    """Whether a layout shows at least one slice view."""
    from nvitk.gui.viz.ortho.layout import spec_panes

    return any(pane in SLICE_PANES for pane in spec_panes(spec))


def preset(key: str | None) -> LayoutPreset:
    """The preset named *key*, or the default."""
    return PRESETS_BY_KEY.get(str(key or ""), PRESETS_BY_KEY[DEFAULT_PRESET])


__all__ = [
    "ALL_PANES",
    "DEFAULT_PRESET",
    "LayoutPreset",
    "PANE_3D",
    "PRESETS",
    "PRESETS_BY_KEY",
    "SLICE_PANES",
    "preset",
    "spec_has_slices",
]
