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

    # Nothing named matched, but another panel may already be occupying the left
    # edge — one added by ``add_dock_widget`` rather than by this helper, so it
    # cannot be named in advance. Tabify with it. Re-splitting Napari's own docks
    # around an existing arrangement tears that arrangement apart: the panel that
    # got there first ends up the only one with usable geometry.
    if tab_target is None:
        napari_own = {id(controls), id(layer_list)}
        for child in win.findChildren(QDockWidget):
            if child is dock or id(child) in napari_own:
                continue
            if win.dockWidgetArea(child) == Qt.LeftDockWidgetArea:
                tab_target = child
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

    install_expand_button(dock, title)
    dock.show()
    dock.raise_()
    return dock


#: Size a panel is given when the screen cannot be measured. Wide enough for the
#: horizontal panels to be usable.
_FLOAT_FALLBACK = (1100, 700)


def install_expand_button(dock: Any, title: str) -> Any:
    """Put an expand/restore button in *dock*'s title bar.

    Napari's own docks carry only a float and a close button, and floating one
    leaves it at whatever size the sidebar had — so every panel that wants room
    starts with a drag to resize and a drag to place. This makes it one click:
    float and fill the screen, click again to put it back where it was.
    """
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

        from nvitk.gui.core.design import (
            COLOR_ACCENT,
            COLOR_CONTROL_HOVER,
            COLOR_DISABLED,
            COLOR_SURFACE,
            COLOR_TEXT,
        )
    except Exception:
        return None
    if dock is None or getattr(dock, "_nvitk_expand_button", None) is not None:
        return getattr(dock, "_nvitk_expand_button", None)

    def _button(glyph: str) -> QPushButton:
        """One title-bar button, styled to read on the dark chrome."""
        widget = QPushButton(glyph)
        widget.setFlat(True)
        widget.setFixedSize(22, 22)
        widget.setCursor(Qt.PointingHandCursor)
        # Explicit colours: a flat button inherits the dock's own stylesheet, and
        # on the dark theme that renders the glyph at almost the background's
        # value — the control is there but invisible.
        widget.setStyleSheet(
            f"QPushButton {{ color: {COLOR_TEXT}; background: transparent;"
            f" border: none; font-size: 13px; }}"
            f"QPushButton:hover {{ color: {COLOR_ACCENT};"
            f" background: {COLOR_CONTROL_HOVER}; border-radius: 3px; }}"
            f"QPushButton:disabled {{ color: {COLOR_DISABLED}; }}"
        )
        return widget

    pop = _button("⧉")
    button = _button("⛶")

    # One title bar for every nvitk panel, ours. Napari's own cannot host these:
    # its stylesheet clamps every direct QPushButton child to 12x12
    # (``#QtCustomTitleBar > QPushButton { max-width: 12px; max-height: 12px }``),
    # which crops a text glyph to nothing — the button is laid out and clickable
    # but draws blank. Its three controls have equivalents here: pop-out floats,
    # and the close button closes.
    bar = QWidget(dock)
    layout = QHBoxLayout(bar)
    layout.setContentsMargins(6, 2, 4, 2)
    layout.setSpacing(2)
    caption = QLabel(title)
    caption.setStyleSheet(f"color: {COLOR_TEXT}; font-weight: bold;")
    layout.addWidget(caption)
    layout.addStretch(1)
    layout.addWidget(pop)
    layout.addWidget(button)
    close = _button("✕")
    close.setToolTip("Close this panel.")
    close.clicked.connect(dock.close)
    layout.addWidget(close)
    bar.setStyleSheet(f"background: {COLOR_SURFACE};")

    def _screen_rect() -> Any:
        """Available geometry of the screen the dock is on."""
        try:
            handle = dock.screen() or dock.window().screen()
            return handle.availableGeometry()
        except Exception:
            return None

    def _sync() -> None:
        """Keep both buttons showing what they will do next."""
        floating = bool(dock.isFloating())
        full = getattr(dock, "_nvitk_fullscreen_from", None) is not None
        pop.setText("⧈" if floating else "⧉")
        pop.setToolTip(
            "Dock this panel back into the window." if floating
            else "Pop this panel out into its own window."
        )
        button.setText("⤡" if full else "⛶")
        button.setToolTip(
            "Restore this window's previous size." if full
            else "Fill the screen with this panel."
        )
        # Fullscreen only means anything once the panel is its own window.
        button.setEnabled(floating or not full)

    def _toggle_float() -> None:
        """Pop the panel out into its own window, or put it back."""
        if dock.isFloating():
            _restore_size()
            dock.setFloating(False)
        else:
            dock._nvitk_docked_geometry = dock.geometry()
            dock.setFloating(True)
            rect = _screen_rect()
            if rect is not None:
                # Half the screen, centred: a panel that opens already filling the
                # display hides the viewer it is meant to be read alongside.
                dock.setGeometry(
                    rect.x() + rect.width() // 4, rect.y() + rect.height() // 6,
                    max(rect.width() // 2, 480), max(int(rect.height() * 0.66), 360),
                )
            else:
                dock.resize(*_FLOAT_FALLBACK)
        dock.show()
        dock.raise_()
        _sync()

    def _restore_size() -> None:
        """Undo a fullscreen, leaving the window floating at its previous size."""
        previous = getattr(dock, "_nvitk_fullscreen_from", None)
        dock._nvitk_fullscreen_from = None
        if previous is not None:
            dock.setGeometry(previous)

    def _toggle_fullscreen() -> None:
        """Fill the screen with the panel, or give it its previous size back."""
        if getattr(dock, "_nvitk_fullscreen_from", None) is not None:
            _restore_size()
            dock.show()
            dock.raise_()
            _sync()
            return
        if not dock.isFloating():
            dock._nvitk_docked_geometry = dock.geometry()
            dock.setFloating(True)
        dock._nvitk_fullscreen_from = dock.geometry()
        rect = _screen_rect()
        if rect is not None:
            dock.setGeometry(rect)
        else:
            dock.resize(*_FLOAT_FALLBACK)
        dock.show()
        dock.raise_()
        _sync()

    pop.clicked.connect(_toggle_float)
    button.clicked.connect(_toggle_fullscreen)
    # Double-clicking a title bar normally floats it; keep that gesture, since it
    # is the one people already reach for.
    bar.mouseDoubleClickEvent = lambda _event: _toggle_float()
    dock.setTitleBarWidget(bar)
    dock.topLevelChanged.connect(lambda _floating: _sync())
    dock._nvitk_expand_button = button
    dock._nvitk_float_button = pop
    dock._nvitk_expand_toggle = _toggle_fullscreen
    dock._nvitk_float_toggle = _toggle_float
    dock._nvitk_fullscreen_from = None
    _sync()
    return button


__all__ = ["attach_left_inspection_dock", "install_expand_button"]
