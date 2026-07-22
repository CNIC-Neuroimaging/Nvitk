"""Shared Napari left-dock placement for cross-section and hemodynamics panels."""

from __future__ import annotations

from typing import Any


def _napari_left_layer_docks(viewer: Any) -> tuple[Any | None, Any | None]:
    """Return (layer controls dock, layer list dock) on Napari's left edge."""
    try:
        qt_viewer = viewer.window._qt_viewer
    except Exception:
        return None, None
    controls = getattr(qt_viewer, "dockLayerControls", None)
    layer_list = getattr(qt_viewer, "dockLayerList", None)
    return controls, layer_list


def attach_left_inspection_dock(
    viewer: Any,
    panel: Any,
    *,
    object_name: str,
    title: str,
    tabify_with: str | list[str] | None = None,
    minimum_width: int = 280,
) -> Any:
    """Attach *panel* on Napari's left edge, optionally tabified with another dock.

    ``tabify_with`` may be a single dock object name or an ordered list of
    candidates; the first existing dock is used as the tab target.
    """
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QDockWidget, QSizePolicy
    except Exception:
        return None

    try:
        win = viewer.window._qt_window
    except Exception:
        try:
            win = viewer.window.qt_viewer.parent()
        except Exception:
            return None

    existing: QDockWidget | None = None
    for child in win.findChildren(QDockWidget):
        if child.objectName() == object_name:
            existing = child
            break

    if existing is not None:
        existing.setWindowTitle(title)
        existing.setWidget(panel)
        existing.show()
        existing.raise_()
        return existing

    dock = QDockWidget(title, win)
    dock.setObjectName(object_name)
    dock.setWidget(panel)
    dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
    panel.setMinimumSize(minimum_width, 220)
    dock.setMinimumWidth(minimum_width)

    controls, layer_list = _napari_left_layer_docks(viewer)
    win.addDockWidget(Qt.LeftDockWidgetArea, dock)

    tab_target = None
    if tabify_with:
        candidates = [tabify_with] if isinstance(tabify_with, str) else list(tabify_with)
        existing_docks = {
            child.objectName(): child
            for child in win.findChildren(QDockWidget)
            if child is not dock
        }
        for candidate in candidates:
            if candidate in existing_docks:
                tab_target = existing_docks[candidate]
                break

    if tab_target is not None:
        win.tabifyDockWidget(tab_target, dock)
        dock.show()
        dock.raise_()
    elif controls is not None and layer_list is not None:
        win.splitDockWidget(controls, dock, Qt.Vertical)
        win.splitDockWidget(dock, layer_list, Qt.Vertical)
    elif layer_list is not None:
        win.splitDockWidget(dock, layer_list, Qt.Vertical)
    elif controls is not None:
        win.splitDockWidget(controls, dock, Qt.Vertical)

    dock.show()
    dock.raise_()
    return dock


__all__ = ["attach_left_inspection_dock"]
