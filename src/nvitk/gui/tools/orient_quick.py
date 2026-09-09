"""Quick-access orientation switcher for the active Napari layer.

One button in the tools dock that reports the active layer's axis codes and
applies a new target in a single click — the same display reorientation as
Transform → *View / reorient orientation* with ``Action = reorient``, without
walking through the tool form.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QMenu, QToolButton

from nvitk.gui.core.orientation import (
    ORIENTATION_CODES,
    apply_target_orientation,
    layer_orientation_codes,
)

_NO_LAYER_TEXT = "Orientation: —"


def _active_layer(viewer: Any) -> Any | None:
    """The viewer's active (or last) layer, or ``None`` if there are no layers."""
    if not viewer.layers:
        return None
    return viewer.layers.selection.active or viewer.layers[-1]


def _is_reorientable(layer: Any | None) -> bool:
    """True when *layer* is a 3D array layer whose affine gives usable axis codes."""
    return layer is not None and layer_orientation_codes(layer) is not None


def build_orientation_quick_button(viewer: Any) -> QToolButton:
    """Menu button that reorients the active image/mask layer to a chosen orientation.

    The button label tracks the active layer's current codes; picking an entry
    from its menu applies that target immediately. Disabled while the active
    layer has no usable 3D affine (points, shapes, 2D or affine-less layers).
    """
    btn = QToolButton()
    btn.setPopupMode(QToolButton.InstantPopup)
    btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
    btn.setText(_NO_LAYER_TEXT)

    menu = QMenu(btn)
    btn.setMenu(menu)

    def _apply(target: str) -> None:
        """Reorient the active layer to *target*, logging the outcome."""
        from nvitk.gui.tools.runner import log_tool_failure, notify

        layer = _active_layer(viewer)
        if not _is_reorientable(layer):
            notify("Select a 3D image or mask layer with an affine first.", error=True)
            return
        try:
            previous, applied = apply_target_orientation(viewer, layer, target)
        except Exception as exc:  # noqa: BLE001
            log_tool_failure(exc)
            notify(f"Could not reorient to {target}: {exc}", error=True)
            return
        if previous == applied:
            notify(f"“{layer.name}” is already {applied}.")
        else:
            notify(f"Reoriented “{layer.name}” {previous} → {applied}.")
        sync()

    for code in ORIENTATION_CODES:
        action = menu.addAction(code)
        action.setData(code)
        action.triggered.connect(lambda _checked=False, c=code: _apply(c))

    def sync() -> None:
        """Refresh the button label / enabled state from the active layer."""
        layer = _active_layer(viewer)
        codes = layer_orientation_codes(layer) if layer is not None else None
        enabled = codes is not None
        btn.setEnabled(enabled)
        if not enabled:
            btn.setText(_NO_LAYER_TEXT)
            btn.setToolTip(
                "Reorient the active layer's display (RAS/LAS/…). "
                "Needs a 3D layer with an affine."
            )
            return
        btn.setText(f"Orientation: {codes}")
        btn.setToolTip(
            f"“{layer.name}” is {codes}. Pick a target to mirror / permute the "
            "Napari display (same as Transform → View / reorient orientation)."
        )
        for action in menu.actions():
            action.setEnabled(str(action.data()) != codes)

    # Napari emits the active-layer event before the layer is fully swapped in,
    # so refresh on the next event-loop pass.
    resync = QTimer(btn)
    resync.setSingleShot(True)
    resync.setInterval(0)
    resync.timeout.connect(sync)

    def _schedule_sync(_event: Any = None) -> None:
        """Debounce a label refresh onto the next event-loop pass."""
        resync.start()

    viewer.layers.selection.events.active.connect(_schedule_sync)
    viewer.layers.events.inserted.connect(_schedule_sync)
    viewer.layers.events.removed.connect(_schedule_sync)

    sync()
    return btn


__all__ = ["build_orientation_quick_button"]
