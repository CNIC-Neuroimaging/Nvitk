"""The 3D pane: Napari's own viewer widget under a slim header with the 3D options pop-over.

Napari's ``QtViewer`` (canvas + dims sliders) is moved in whole: its canvas calls into its
parent on mouse enter/leave, so the canvas cannot be taken out alone. It is reparented once,
before the window is first shown, and never leaves the main window afterwards (a GL widget
moved to another top-level window loses its context), which is why the 3D pane can be
maximised or hidden but not popped out.
"""

from __future__ import annotations

from typing import Any, Callable

from qtpy.QtWidgets import QFrame, QMenu, QScrollArea, QToolButton, QVBoxLayout, QWidget

from nvitk.gui.core import design
from nvitk.gui.core.widgets import IconButton, PaneHeader, Popover


class Pane3D(QWidget):
    """Header (title, options, maximise, hide) over Napari's viewer widget."""

    def __init__(
        self,
        qt_viewer: QWidget,
        controls: QWidget,
        *,
        on_maximize: Callable[[], Any],
        on_hide: Callable[[], Any],
        layers_getter: Callable[[], list[Any]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("nvitkPane3D")
        self._qt_viewer = qt_viewer
        self._controls = controls
        self._popover: Popover | None = None

        self.header = PaneHeader("3D view")
        self.header.setToolTip("Drag: rotate · scroll: zoom · Shift-drag: pan · Ctrl+click: move the 2D crosshair to the voxel clicked")
        design.apply_theme(self.header)
        self._layers_getter = layers_getter
        if layers_getter is not None:
            self._render_btn = IconButton("droplet", "Render preset for the selected layers (CT, CTA, MRA, PET, labels...)", icon_size=13)
            self._render_btn.setMinimumSize(20, 20)
            self._render_btn.setPopupMode(QToolButton.InstantPopup)
            self._render_menu = QMenu(self._render_btn)
            self._render_menu.aboutToShow.connect(self._fill_render_menu)
            self._render_btn.setMenu(self._render_menu)
            self.header.add_widget(self._render_btn)
        self._options = self.header.add_action(
            "sliders", "3D view options: resampling, slices in 3D, see-inside cut, per-layer table", self._toggle_options
        )
        self.header.add_action("maximize", "This pane alone (again: back to the layout)", on_maximize)
        self.header.add_action("eye_off", "Hide the 3D view (Layout menu or View ▸ Show main canvas brings it back)", on_hide)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self.header)
        root.addWidget(qt_viewer, 1)
        self.setMinimumSize(80, 80)

    def qt_viewer(self) -> QWidget:
        """Napari's viewer widget hosted here."""
        return self._qt_viewer

    def options_popover(self) -> Popover:
        """The 3D options pop-over, built on first use."""
        if self._popover is None:
            scroll = QScrollArea()
            scroll.setWidget(self._controls)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setMinimumSize(380, 460)
            self._popover = Popover(scroll, title="3D view options", parent=self.window())
        return self._popover

    def _fill_render_menu(self) -> None:
        from nvitk.gui.viz.render_presets import build_preset_menu

        self._render_menu.clear()
        build_preset_menu(self._render_menu, self._layers_getter)

    def _toggle_options(self) -> None:
        self.options_popover().toggle_at(self._options, side="below_end")


__all__ = ["Pane3D"]
