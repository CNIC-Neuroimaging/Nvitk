"""Napari notification toasts that close themselves after a fixed time."""

from __future__ import annotations

from typing import Any

NOTIFICATION_TIMEOUT_MS = 3000


def install_notification_timeout(timeout_ms: int = NOTIFICATION_TIMEOUT_MS) -> None:
    """
    Close every Napari notification toast *timeout_ms* after it appears.

    Napari stops every open toast's dismiss timer whenever a new one shows, and never
    arms one while the window is inactive, so a burst of messages piles up on the
    canvas until each is closed by hand. Each toast here gets its own timer, parented
    to the toast so it dies with it.
    """
    try:
        from napari._qt.dialogs.qt_notification import NapariQtNotification
        from qtpy.QtCore import QTimer
    except Exception:  # noqa: BLE001 — headless or a Napari without Qt toasts
        return
    if getattr(NapariQtNotification, "_nvitk_timeout_ms", None) is not None:
        NapariQtNotification._nvitk_timeout_ms = int(timeout_ms)
        return

    original_show = NapariQtNotification.show

    def show(self: Any) -> None:
        """Napari's slide-in, then a dismiss timer no other toast can stop."""
        original_show(self)
        try:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self.close_with_fade)
            timer.start(int(NapariQtNotification._nvitk_timeout_ms))
        except Exception:  # noqa: BLE001 — a toast must never break the caller
            pass

    NapariQtNotification.DISMISS_AFTER = int(timeout_ms)
    NapariQtNotification._nvitk_timeout_ms = int(timeout_ms)
    NapariQtNotification.show = show
