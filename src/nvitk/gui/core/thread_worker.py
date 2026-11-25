"""A ``QThread``-shaped worker that runs in a plain Python thread.

``QThread`` workers segfault at interpreter exit here (a Qt-owned thread object outliving the
application), so background work runs in a :class:`threading.Thread` instead and reports
through the signals of a ``QObject`` that lives on the GUI thread: Qt queues a signal emitted
from another thread to the receivers' thread, so slots still run on the GUI thread.

:class:`ThreadWorker` keeps the ``QThread`` API the panels were written against (``run``,
``start``, ``isRunning``, ``wait``, a ``finished`` signal), so a ``QThread`` subclass moves over
by changing its base class only.
"""

from __future__ import annotations

import threading
from typing import Any

from qtpy.QtCore import QObject, Signal


class ThreadWorker(QObject):
    """Subclass and override :meth:`run`; :meth:`start` runs it off the GUI thread."""

    #: Emitted after :meth:`run` returns (or raises), as ``QThread.finished`` is.
    finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: threading.Thread | None = None
        self._interrupt = False

    def run(self) -> None:  # pragma: no cover - overridden
        """The work; runs on the worker thread. Report through signals only."""

    def _main(self) -> None:
        try:
            self.run()
        finally:
            try:
                self.finished.emit()
            except RuntimeError:  # the QObject went away with the application
                pass

    def start(self) -> None:
        """Start :meth:`run` on a daemon thread (a no-op while it is running)."""
        if self.isRunning():
            return
        self._interrupt = False
        self._thread = threading.Thread(target=self._main, name=type(self).__name__, daemon=True)
        self._thread.start()

    def isRunning(self) -> bool:  # noqa: N802 (QThread API)
        return self._thread is not None and self._thread.is_alive()

    def isFinished(self) -> bool:  # noqa: N802 (QThread API)
        return self._thread is not None and not self._thread.is_alive()

    def wait(self, msecs: int | None = None) -> bool:
        """Join the thread (at most *msecs*); True when it has finished."""
        if self._thread is None:
            return True
        self._thread.join(None if msecs is None else max(msecs, 0) / 1000.0)
        return not self._thread.is_alive()

    def requestInterruption(self) -> None:  # noqa: N802 (QThread API)
        self._interrupt = True

    def isInterruptionRequested(self) -> bool:  # noqa: N802 (QThread API)
        return self._interrupt

    def __repr__(self) -> str:  # pragma: no cover
        state = "running" if self.isRunning() else "idle"
        return f"<{type(self).__name__} {state}>"


def is_worker(obj: Any) -> bool:
    """Whether *obj* is a running-capable worker (``ThreadWorker`` or ``QThread``)."""
    return hasattr(obj, "isRunning") and hasattr(obj, "start")


__all__ = ["ThreadWorker", "is_worker"]
