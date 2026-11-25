"""The central viewport: the three slice views and Napari's 3D canvas, in one switchable layout.

What 3D Slicer opens with: axial / coronal / sagittal views and a 3D view side by side, in a
layout picked from a menu (four equal panes, three slices beside a large 3D view, 3D only,
slices only, one slice...). Every divider can be dragged, double-clicked to even the panes out,
and a pane can be shown alone (double-click a view's title, or its maximise button) or hidden.

The slice views and their logic are the orthogonal-views panel's
(:class:`~nvitk.gui.viz.ortho.panel.OrthoViewerPanel`, built ``embedded``: it keeps binding the
active layer, slicing, drawing and pushing cuts to the 3D canvas, and only its widgets move
here). Its 3D options open from the 3D pane's header as a pop-over instead of taking a pane.

The layout, the pane sizes of every layout and the hidden panes are remembered between
sessions (``gui.json``, ``ortho_view.viewport``).
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import QTimer, Qt
from qtpy.QtWidgets import QHBoxLayout, QMenu, QSizePolicy, QToolButton, QVBoxLayout, QWidget

from nvitk.gui.core import design
from nvitk.gui.core.icons import icon as make_icon
from nvitk.gui.core.widgets import IconButton, SegmentedButtons, separator
from nvitk.gui.viewport.layouts import (
    ALL_PANES,
    DEFAULT_PRESET,
    PANE_3D,
    PRESETS,
    SLICE_PANES,
    preset,
    spec_has_slices,
)
from nvitk.gui.viewport.pane3d import Pane3D
from nvitk.gui.viz.ortho.layout import LayoutGrid, _ortho_prefs, _save_ortho_pref

#: Key inside the ``ortho_view`` preferences.
PREFS_KEY = "viewport"

_PANE_TITLES = {"ax": "First slice view (axial)", "cor": "Second slice view (coronal)",
                "sag": "Third slice view (sagittal)", "3d": "3D view"}


class Viewport(QWidget):
    """Top bar (layout, projection, status) over the pane grid; the time row when 4D."""

    def __init__(self, viewer: Any, panel: Any, qt_viewer: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("nvitkViewport")
        self._viewer = viewer
        self._panel = panel
        stored = dict(_ortho_prefs().get(PREFS_KEY) or {})
        self._fractions: dict[str, Any] = dict(stored.get("fractions") or {})
        self._hidden: set[str] = {p for p in stored.get("hidden") or () if p in ALL_PANES}
        self._preset = preset(stored.get("preset")).key

        # The panel is the controller: kept alive (and hidden) inside the viewport.
        panel.setParent(self)
        panel.hide()
        design.apply_theme(panel)

        views = panel.slice_views()
        for index, (pane_id, view) in enumerate(zip(SLICE_PANES, views)):
            design.apply_theme(view)
            maximize = IconButton("maximize", "This view alone (again: back to the layout)", icon_size=12)
            maximize.clicked.connect(lambda _=False, p=pane_id: self.toggle_maximized(p))
            hide = IconButton("eye_off", "Hide this view (the Layout menu brings it back)", icon_size=12)
            hide.clicked.connect(lambda _=False, p=pane_id: self.set_pane_visible(p, False))
            view.add_header_widget(maximize)
            view.add_header_widget(hide)
        panel.maximize_hook = lambda index: self.toggle_maximized(SLICE_PANES[int(index)])

        self._pane3d = Pane3D(
            qt_viewer,
            panel.controls3d(),
            on_maximize=lambda: self.toggle_maximized(PANE_3D),
            on_hide=lambda: self.set_pane_visible(PANE_3D, False),
            layers_getter=self.selected_layers,
        )
        panes = {pane_id: view for pane_id, view in zip(SLICE_PANES, views)}
        panes[PANE_3D] = self._pane3d

        self._bar = self._build_bar()
        design.apply_theme(self._bar)
        time_bar = panel.time_bar()
        design.apply_theme(time_bar)
        hint = panel.draw_hint_label()

        # The 3D pane floats: it is laid over its slot, never moved between splitters (GL).
        self._grid = LayoutGrid(panes, preset(self._preset).spec, floating=(PANE_3D,))
        self._grid.set_fractions(self._fractions.get(self._preset))
        for pane_id in self._hidden:
            self._grid.set_pane_visible(pane_id, False)

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 2, 4, 2)
        root.setSpacing(design.SPACE_TIGHT)
        root.addWidget(self._bar)
        root.addWidget(hint)
        root.addWidget(time_bar)
        root.addWidget(self._grid, 1)
        panel.install_shortcuts(self)

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(400)
        self._save_timer.timeout.connect(self._save)
        self._grid.changed.connect(self._save_timer.start)
        self._sync_bar()

    # top bar
    def _build_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("nvitkViewportBar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(design.SPACE_TIGHT)

        quick = [(p.key, p.label, p.icon, p.label) for p in PRESETS if p.quick]
        self._quick = SegmentedButtons(quick, current=self._preset, icon_only=True, icon_size=15)
        self._quick.changed.connect(self.set_preset)
        row.addWidget(self._quick)

        self._layout_btn = QToolButton()
        self._layout_btn.setText("Layout")
        self._layout_btn.setToolTip("Every layout, which panes are shown, and evening the panes out.")
        self._layout_btn.setPopupMode(QToolButton.InstantPopup)
        self._layout_btn.setFocusPolicy(Qt.NoFocus)
        self._layout_menu = QMenu(self._layout_btn)
        self._layout_menu.aboutToShow.connect(self._fill_layout_menu)
        self._layout_btn.setMenu(self._layout_menu)
        row.addWidget(self._layout_btn)

        self._cross_btn = IconButton("crosshair", "Show the crosshair lines in the slice views", checkable=True, icon_size=15)
        self._cross_btn.setChecked(self._panel.crosshair_visible())
        self._cross_btn.toggled.connect(self._panel.set_crosshair_visible)
        row.addWidget(self._cross_btn)
        # Beside it, the same for the 3D view: the crosshair drawn as three lines through the
        # volume (kept in step with the "Crosshair lines in 3D" box of the 3D options).
        self._cross3d_btn = IconButton("crosshair_3d", "Show the crosshair lines in the 3D view", checkable=True, icon_size=15)
        box = getattr(self._panel, "_cross3d_box", None)
        if box is not None:
            self._cross3d_btn.setChecked(box.isChecked())
            self._cross3d_btn.toggled.connect(lambda on: box.setChecked(on) if box.isChecked() != on else None)
            box.toggled.connect(lambda on: self._cross3d_btn.setChecked(on) if self._cross3d_btn.isChecked() != on else None)
        else:
            self._cross3d_btn.setEnabled(False)
        row.addWidget(self._cross3d_btn)
        from nvitk.gui.viz.roi_box_tool import roi_box_button

        row.addWidget(roi_box_button(self._viewer))

        row.addWidget(separator(vertical=True))
        projection = self._panel.projection_bar()
        projection.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        row.addWidget(projection)
        row.addWidget(separator(vertical=True))

        status = self._panel.status_label()
        status.setWordWrap(False)
        status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        status.setStyleSheet(f"color: {design.COLOR_MUTED}; font-size: {design.FONT_S}px;")
        row.addWidget(status, 1)
        readout = self._panel.readout_label()
        readout.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(readout)
        return bar

    def _fill_layout_menu(self) -> None:
        menu = self._layout_menu
        menu.clear()
        for p in PRESETS:
            action = menu.addAction(make_icon(p.icon), p.label)
            action.setCheckable(True)
            action.setChecked(p.key == self._preset)
            action.triggered.connect(lambda _=False, k=p.key: self.set_preset(k))
        menu.addSeparator()
        shown = set(self._grid.pane_ids())
        for pane_id in ALL_PANES:
            action = menu.addAction(f"Show {_PANE_TITLES[pane_id]}")
            action.setCheckable(True)
            action.setEnabled(pane_id in shown)
            action.setChecked(pane_id in shown and pane_id not in self._hidden)
            action.toggled.connect(lambda on, p=pane_id: self.set_pane_visible(p, on))
        menu.addSeparator()
        even = menu.addAction(make_icon("layout_2x2"), "Even out the panes")
        even.triggered.connect(self._grid.even)

    def _sync_bar(self) -> None:
        current = self._preset if any(p.key == self._preset and p.quick for p in PRESETS) else None
        if current is not None:
            self._quick.set_current(current)
        else:
            for p in PRESETS:
                button = self._quick.button(p.key)
                if button is not None and button.isChecked():
                    group = button.group()
                    if group is not None:
                        group.setExclusive(False)
                        button.setChecked(False)
                        group.setExclusive(True)

    def selected_layers(self) -> list[Any]:
        """The layers selected in the layer list (the active one first), else the active one."""
        layers = self._viewer.layers
        active = layers.selection.active
        selected = [l for l in layers if l in layers.selection]
        if active is not None and active in selected:
            selected.remove(active)
            selected.insert(0, active)
        return selected or ([active] if active is not None else [])

    # layout
    def preset_key(self) -> str:
        """Key of the layout in force."""
        return self._preset

    def grid(self) -> LayoutGrid:
        """The pane grid."""
        return self._grid

    def pane3d(self) -> Pane3D:
        """The 3D pane."""
        return self._pane3d

    def set_preset(self, key: str) -> None:
        """Switch to layout *key*, with the pane sizes it had last time."""
        chosen = preset(key)
        if self._grid.maximized() is None:
            self._fractions[self._preset] = self._grid.fractions()
        self._preset = chosen.key
        self._grid.set_spec(chosen.spec, self._fractions.get(chosen.key))
        for pane_id in ALL_PANES:
            if pane_id in self._hidden:
                self._grid.set_pane_visible(pane_id, False)
        self._sync_bar()
        self._save()

    def toggle_maximized(self, pane_id: str) -> None:
        """Show *pane_id* alone, or put the layout back."""
        if pane_id not in self._grid.pane_ids():
            return
        if pane_id in self._hidden:
            self.set_pane_visible(pane_id, True)
        self._grid.toggle_maximized(pane_id)

    def set_pane_visible(self, pane_id: str, visible: bool) -> None:
        """Hide a pane (its neighbours take the room) or bring it back.

        The 3D pane goes through the main-canvas switch, so the 3D options' "Show the main
        canvas" and *View ▸ Show main canvas* stay in step (and a hidden canvas stops rendering).
        """
        if pane_id == PANE_3D:
            canvas = getattr(self._viewer, "_nvitk_main_canvas", None)
            if canvas is not None and canvas.canvas_shown() != bool(visible):
                canvas.set_canvas_shown(bool(visible))
                return
        self.apply_pane_visible(pane_id, visible)

    def apply_pane_visible(self, pane_id: str, visible: bool) -> None:
        """Show or hide a pane in the grid (no side effects on the main-canvas switch)."""
        if visible:
            self._hidden.discard(pane_id)
        else:
            self._hidden.add(pane_id)
        self._grid.set_pane_visible(pane_id, bool(visible))
        self._save()

    def ensure_slices_visible(self) -> None:
        """Make sure the slice views are on screen (the *Orthogonal views* button)."""
        for pane_id in SLICE_PANES:
            if pane_id in self._hidden:
                self.apply_pane_visible(pane_id, True)
        if not spec_has_slices(self._grid.spec()):
            self.set_preset(DEFAULT_PRESET)
        elif self._grid.maximized() == PANE_3D:
            self._grid.toggle_maximized(PANE_3D)

    def _save(self) -> None:
        if self._grid.maximized() is None:
            self._fractions[self._preset] = self._grid.fractions()
        _save_ortho_pref(
            PREFS_KEY,
            {"preset": self._preset, "fractions": self._fractions, "hidden": sorted(self._hidden)},
        )


def install_viewport(viewer: Any) -> Viewport:
    """Make the viewport the main window's central widget, with Napari's viewer in its 3D pane.

    Call before the window is first shown, so the 3D canvas's GL context is created in its final
    place. Builds the orthogonal-views panel (embedded), keeps it on the active layer, and stores
    both on the viewer (``_nvitk_viewport``, ``_nvitk_ortho_panel``).
    """
    from nvitk.gui.viz.ortho.panel import OrthoViewerPanel, connect_ortho_panel

    existing = getattr(viewer, "_nvitk_viewport", None)
    if existing is not None:
        return existing
    panel = OrthoViewerPanel(viewer, embedded=True)
    viewer._nvitk_ortho_panel = panel
    qt_main = viewer.window._qt_window
    qt_viewer = viewer.window._qt_viewer
    viewport = Viewport(viewer, panel, qt_viewer)
    # The old central widget (Napari's margin box) is deleted here; its viewer widget already
    # moved into the 3D pane.
    qt_main.setCentralWidget(viewport)
    viewer._nvitk_viewport = viewport
    connect_ortho_panel(viewer, panel)
    QTimer.singleShot(0, panel.restore_marker_preference)
    QTimer.singleShot(0, panel.restore_canvas_preference)
    return viewport


__all__ = ["PREFS_KEY", "Viewport", "install_viewport"]
