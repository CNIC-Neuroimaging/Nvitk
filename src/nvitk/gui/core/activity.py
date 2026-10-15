"""What the GUI is busy with: an activity indicator in the main window's status bar.

Every load and every tool run registers a *task* (:func:`activity`, or
:func:`start_task` for work that finishes later, a worker thread's): the status
bar shows its message, a bar — filled when the task reports how far it got,
sweeping when it does not — and the time it has been running; with several at
once, the latest and how many more. A finished task leaves "Done …" (or
"Failed …") for a moment when it took long enough to be worth saying.

Two kinds of work, two behaviours:

* **Work handed to a thread** (:func:`run_in_background`: reading image files,
  DICOM series, writing a scene…): the GUI keeps repainting while it runs — the
  bar moves, the time counts — though clicks and keys wait until it is done.
* **Work on the GUI thread** (a tool from the Imaging tab, most edits): the
  indicator is painted before the work starts and says what is running; the
  window cannot repaint until it ends, so the bar holds still meanwhile.
* **Work elsewhere** — a job sent to the SGE cluster (``kind="cluster"``): the task
  stays while the job is queued and running (its state in the message), for as long
  as that takes, and the GUI is free meanwhile. Elapsed times read as minutes and
  hours, and the tooltip lists every task when several are running.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Any, Callable, Iterator

from qtpy.QtCore import QEventLoop, QObject, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget

from nvitk.gui.core.design import COLOR_ACCENT, COLOR_BORDER, COLOR_ERROR, COLOR_MUTED, COLOR_OK

#: A finished task is reported ("Done in 3.2 s") only when it took at least this long.
_REPORT_AFTER_S = 0.6
#: How long the "Done" / "Failed" line stays.
_LINGER_MS = 2500
#: Animation and clock tick.
_TICK_MS = 80


class Task:
    """One piece of work the indicator shows. :meth:`update` is safe from any thread."""

    def __init__(self, hub: "ActivityHub", message: str, total: int | None = None, *, kind: str = "") -> None:
        self._hub = hub
        #: ``"cluster"`` for a job running elsewhere (the GUI is not busy with it).
        self.kind = str(kind or "")
        self.message = str(message)
        self.total = int(total) if total else None
        self.done = 0
        self.started = time.monotonic()
        self.finished: float | None = None
        self.ok = True

    def update(self, done: int | None = None, total: int | None = None, message: str | None = None) -> None:
        """How far it got (*done* of *total*), and/or a new message."""
        if total is not None:
            self.total = int(total) if total else None
        if done is not None:
            self.done = int(done)
        if message is not None:
            self.message = str(message)

    @property
    def fraction(self) -> float | None:
        if not self.total:
            return None
        return max(0.0, min(1.0, self.done / float(self.total)))

    @property
    def elapsed(self) -> float:
        return (self.finished or time.monotonic()) - self.started

    def finish(self, ok: bool = True) -> None:
        """The work is over (GUI thread)."""
        if self.finished is not None:
            return
        self.finished = time.monotonic()
        self.ok = bool(ok)
        self._hub._finished(self)


class ActivityHub(QObject):
    """The running tasks (one per application)."""

    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.tasks: list[Task] = []
        self.last: Task | None = None

    def start(self, message: str, total: int | None = None, *, kind: str = "") -> Task:
        task = Task(self, message, total, kind=kind)
        self.tasks.append(task)
        self.changed.emit()
        return task

    def _finished(self, task: Task) -> None:
        if task in self.tasks:
            self.tasks.remove(task)
        self.last = task
        self.changed.emit()

    def busy(self) -> bool:
        return bool(self.tasks)


_HUB: ActivityHub | None = None


def elapsed_text(seconds: float) -> str:
    """``"8 s"``, ``"12 min"``, ``"1 h 05 min"``: how long something has been running."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min" + (f" {s % 60:02d} s" if s < 600 else "")
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


def activity_hub() -> ActivityHub:
    """The application's task list (made on first use)."""
    global _HUB
    if _HUB is None:
        _HUB = ActivityHub()
    return _HUB


class _Bar(QWidget):
    """A thin bar: filled to the task's fraction, or a segment sweeping across."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(110, 8)
        self.fraction: float | None = None
        self.colour = QColor(COLOR_ACCENT)

    def paintEvent(self, _event: Any) -> None:  # noqa: N802 - Qt naming
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        painter.setPen(QColor(COLOR_BORDER))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.colour)
        inner = rect.adjusted(1.5, 1.5, -1.5, -1.5)
        if self.fraction is not None:
            painter.drawRoundedRect(QRectF(inner.left(), inner.top(), inner.width() * self.fraction,
                                           inner.height()), 2, 2)
        else:
            width = inner.width() * 0.3
            phase = (time.monotonic() * 0.9) % 1.0
            left = inner.left() - width + phase * (inner.width() + width)
            seg = QRectF(max(left, inner.left()), inner.top(),
                         min(left + width, inner.right()) - max(left, inner.left()), inner.height())
            if seg.width() > 0:
                painter.drawRoundedRect(seg, 2, 2)
        painter.end()


class ActivityIndicator(QWidget):
    """The status-bar widget showing the :class:`ActivityHub`'s tasks."""

    def __init__(self, hub: ActivityHub, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._hub = hub
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(6)
        self._label = QLabel("")
        self._label.setStyleSheet(f"color: {COLOR_MUTED};")
        self._bar = _Bar()
        layout.addWidget(self._label)
        layout.addWidget(self._bar)
        self._tick = QTimer(self)
        self._tick.setInterval(_TICK_MS)
        self._tick.timeout.connect(self.refresh)
        self._linger = QTimer(self)
        self._linger.setSingleShot(True)
        self._linger.setInterval(_LINGER_MS)
        self._linger.timeout.connect(self._after_linger)
        hub.changed.connect(self._on_changed)
        self.hide()

    def _on_changed(self) -> None:
        hub = self._hub
        if hub.busy():
            self._linger.stop()
            if not self._tick.isActive():
                self._tick.start()
            self.refresh()
            return
        self._tick.stop()
        last = hub.last
        if last is not None and last.elapsed >= _REPORT_AFTER_S:
            mark = "✓ Done" if last.ok else "✗ Failed"
            took = f"{last.elapsed:.1f} s" if last.elapsed < 60 else elapsed_text(last.elapsed)
            self._label.setText(f"{mark}: {last.message} ({took})")
            self._label.setToolTip("")
            self._label.setStyleSheet(f"color: {COLOR_OK if last.ok else COLOR_ERROR};")
            self._bar.hide()
            self.show()
            self._linger.start()
        else:
            self.hide()

    def _after_linger(self) -> None:
        if not self._hub.busy():
            self.hide()

    def refresh(self) -> None:
        """Repaint from the tasks' state (also called mid-work, to paint at once)."""
        tasks = self._hub.tasks
        if not tasks:
            return
        task = tasks[-1]
        text = f"⏳ {task.message}"
        if task.total:
            text += f" ({task.done}/{task.total})"
        if len(tasks) > 1:
            text += f" · +{len(tasks) - 1} more"
        if task.elapsed >= 1.0:
            text += f" · {elapsed_text(task.elapsed)}"
        self._label.setText(text)
        self._label.setToolTip("\n".join(
            f"{t.message}" + (f" ({t.done}/{t.total})" if t.total else "") + f" · {elapsed_text(t.elapsed)}"
            for t in reversed(tasks)))
        self._label.setStyleSheet(f"color: {COLOR_MUTED};")
        self._bar.fraction = task.fraction
        self._bar.show()
        self._bar.update()
        if not self.isVisible():
            self.show()


def install_activity_indicator(viewer: Any) -> ActivityIndicator | None:
    """Put the indicator in *viewer*'s main-window status bar (once)."""
    existing = getattr(viewer, "_nvitk_activity_indicator", None)
    if existing is not None:
        return existing
    try:
        bar = viewer.window._qt_window.statusBar()
    except Exception:  # noqa: BLE001 — headless
        return None
    indicator = ActivityIndicator(activity_hub())
    bar.addPermanentWidget(indicator)
    viewer._nvitk_activity_indicator = indicator
    return indicator


def _paint_now() -> None:
    """Paint the indicators right away (the work that follows may hold the GUI thread)."""
    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        for indicator in widget.findChildren(ActivityIndicator):
            try:
                indicator.refresh()
                indicator.repaint()
            except RuntimeError:
                continue


def start_task(message: str, total: int | None = None, *, kind: str = "") -> Task:
    """A task that finishes later (call :meth:`Task.finish`), e.g. a worker thread's, or a
    job on the cluster (*kind* ``"cluster"``)."""
    task = activity_hub().start(message, total, kind=kind)
    _paint_now()
    return task


@contextlib.contextmanager
def activity(message: str, *, total: int | None = None, cursor: bool = True) -> Iterator[Task]:
    """Show *message* while the block runs (with a wait cursor); the task is yielded
    for :meth:`Task.update`. A failure marks it failed and is re-raised."""
    app = QApplication.instance()
    if app is None:
        yield Task(activity_hub(), message, total)
        return
    task = start_task(message, total)
    if cursor:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    ok = True
    try:
        yield task
    except BaseException:
        ok = False
        raise
    finally:
        if cursor:
            QApplication.restoreOverrideCursor()
        task.finish(ok)


def run_in_background(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """``fn(*args, **kwargs)`` on a worker thread, the GUI repainting meanwhile
    (indicators move, windows redraw; clicks and keys wait). Returns its result or
    raises its exception. Off the GUI thread, or without a Qt application, it just
    calls *fn*.

    *fn* must not touch Qt or Napari objects: it is for reading files, computing
    arrays, writing archives — the layers are added afterwards, back here.
    """
    app = QApplication.instance()
    if app is None or threading.current_thread() is not threading.main_thread():
        return fn(*args, **kwargs)
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — handed back to the caller
            box["error"] = exc

    thread = threading.Thread(target=work, name="nvitk-background", daemon=True)
    thread.start()
    loop = QEventLoop()
    poll = QTimer()
    poll.setInterval(30)
    poll.timeout.connect(lambda: None if thread.is_alive() else loop.quit())
    poll.start()
    if thread.is_alive():
        loop.exec(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
    poll.stop()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("value")


__all__ = [
    "ActivityHub",
    "ActivityIndicator",
    "Task",
    "activity",
    "activity_hub",
    "elapsed_text",
    "install_activity_indicator",
    "run_in_background",
    "start_task",
]
