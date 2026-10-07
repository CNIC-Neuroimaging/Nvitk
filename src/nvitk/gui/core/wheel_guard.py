"""The mouse wheel scrolls the panels; it does not change their settings.

Qt gives every dropdown, spin box and slider the wheel as soon as the pointer is
over it, so scrolling down a panel with a long form flips whatever option happens
to pass under the mouse — a tool's method, a colormap, a parameter — without
anyone noticing. With the guard installed:

* a dropdown (``QComboBox``) never takes the wheel — choose with a click or the keys;
* a spin box or slider takes it only once it has been clicked into (keyboard focus),
  so it can still be nudged on purpose;
* otherwise the wheel goes on to the panel, which scrolls as if the control were not
  there. Scroll bars, lists, the canvas and open dropdown lists are not affected.

Qt focuses a control on the wheel before any filter sees the event when its focus
policy is ``WheelFocus`` (the default for spin boxes and dropdowns), so the guard
also turns those into click-focus (``StrongFocus``) as they appear.
"""

from __future__ import annotations

import atexit
import sys
from typing import Any

from qtpy.QtCore import QEvent, QObject, Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollBar,
)


def _is_value_control(obj: Any) -> bool:
    return isinstance(obj, (QComboBox, QAbstractSpinBox)) or (
        isinstance(obj, QAbstractSlider) and not isinstance(obj, QScrollBar))


def _click_focus(widget: Any) -> None:
    """No focus from the wheel: ``WheelFocus`` → ``StrongFocus`` (tab and click still focus)."""
    try:
        if widget.focusPolicy() == Qt.FocusPolicy.WheelFocus:
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    except Exception:  # noqa: BLE001
        pass


def _scroll_area_around(widget: Any) -> Any | None:
    """The nearest scroll area *widget* sits in (its viewport is an ancestor)."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea) and not isinstance(parent, QComboBox):
            return parent
        parent = parent.parentWidget()
    return None


class WheelGuard(QObject):
    """Application-wide event filter implementing the rules above."""

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 — Qt naming
        if sys.is_finalizing():
            return False
        kind = event.type()
        if kind == QEvent.Type.Polish:
            if _is_value_control(obj):
                _click_focus(obj)
            return False
        if kind != QEvent.Type.Wheel:
            return False
        if isinstance(obj, QComboBox):
            blocked = True
        elif isinstance(obj, QAbstractSpinBox) or (isinstance(obj, QAbstractSlider) and not isinstance(obj, QScrollBar)):
            blocked = not obj.hasFocus()
        else:
            return False
        if not blocked:
            return False
        area = _scroll_area_around(obj)
        if area is None:
            # Ignored and filtered: Qt offers a real wheel event to the parents next.
            event.ignore()
            return True
        # Handed straight to the scroll area holding the control, which scrolls as if
        # the control were not under the pointer.
        viewport = area.viewport()
        pos = viewport.mapFromGlobal(event.globalPosition().toPoint())
        forwarded = QWheelEvent(
            pos.toPointF() if hasattr(pos, "toPointF") else pos, event.globalPosition(), event.pixelDelta(),
            event.angleDelta(), event.buttons(), event.modifiers(), event.phase(), event.inverted(),
        )
        QApplication.sendEvent(viewport, forwarded)
        event.accept()
        return True


def install_wheel_guard(app: QApplication | None = None) -> WheelGuard | None:
    """Install the guard on the application (once); returns it."""
    app = app or QApplication.instance()
    if app is None:
        return None
    guard = getattr(app, "_nvitk_wheel_guard", None)
    if guard is None:
        guard = WheelGuard(app)
        app.installEventFilter(guard)
        app._nvitk_wheel_guard = guard
        # Off again before Qt tears the widgets down: an application-wide Python
        # filter called during interpreter shutdown crashes the exit.
        app.aboutToQuit.connect(lambda: uninstall_wheel_guard(app))
        atexit.register(uninstall_wheel_guard, app)
        # Controls created before the guard: already polished.
        for widget in app.allWidgets():
            if _is_value_control(widget):
                _click_focus(widget)
    return guard


def uninstall_wheel_guard(app: QApplication | None = None) -> None:
    """Remove the guard (on quit; safe to call twice)."""
    try:
        app = app or QApplication.instance()
        guard = getattr(app, "_nvitk_wheel_guard", None) if app is not None else None
        if guard is not None:
            app.removeEventFilter(guard)
            app._nvitk_wheel_guard = None
    except (RuntimeError, AttributeError):
        pass


__all__ = ["WheelGuard", "install_wheel_guard", "uninstall_wheel_guard"]
