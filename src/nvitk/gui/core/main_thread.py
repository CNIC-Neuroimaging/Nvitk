"""Run a callable on the GUI thread from any thread.

Qt widgets and Napari layers belong to the GUI thread. Work done on a worker thread (a tool
running in the background) hands anything that touches them, a log line, a notification, the
layers it made, to :func:`call_on_main`, which runs it at once on the GUI thread and queues it
there from any other thread. The relay is a plain ``threading`` + ``QObject`` signal (no
``QThread``: those crashed nvitk at exit).
"""

from __future__ import annotations

import functools
import threading
from typing import Any, Callable

from qtpy.QtCore import QCoreApplication, QObject, Qt, Signal, Slot

_RELAY: "_Relay | None" = None
_LOCK = threading.Lock()


class _Relay(QObject):
    call = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.call.connect(self._run, Qt.QueuedConnection)

    @Slot(object)
    def _run(self, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 (a relayed call must not take the event loop down)
            print(f"nvitk: a call relayed to the GUI thread failed: {exc!r}", flush=True)


def on_main_thread() -> bool:
    """True on the GUI (main) thread."""
    return threading.current_thread() is threading.main_thread()


def _relay() -> _Relay | None:
    global _RELAY
    if _RELAY is not None:
        return _RELAY
    app = QCoreApplication.instance()
    if app is None:
        return None
    with _LOCK:
        if _RELAY is None:
            relay = _Relay()
            if relay.thread() is not app.thread():
                relay.moveToThread(app.thread())
            _RELAY = relay
    return _RELAY


def install_main_thread_relay() -> None:
    """Create the relay on the GUI thread (at start-up)."""
    _relay()


def call_on_main(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """``fn(*args, **kwargs)`` on the GUI thread: now when already there, else queued.
    Without a Qt application it just runs here."""
    call = functools.partial(fn, *args, **kwargs) if (args or kwargs) else fn
    if on_main_thread():
        call()
        return
    relay = _relay()
    if relay is None:
        call()
        return
    relay.call.emit(call)


__all__ = ["call_on_main", "install_main_thread_relay", "on_main_thread"]
