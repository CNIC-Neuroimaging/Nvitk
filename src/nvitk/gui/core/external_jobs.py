"""Long external commands that run while the GUI stays usable.

:func:`~nvitk.gui.core.log_panel.run_subprocess_logged` keeps the window painting
but holds clicks and keys until the command exits — right for a tool that takes
seconds, wrong for FastSurfer's surface stage or an FSL ``bedpostx`` that take an
hour. An :class:`ExternalJob` runs the command in its own process group and returns
at once:

* its output streams into the log panel, and an activity task counts the time;
* the *Jobs* window lists every running job with its latest stage and a **Cancel**
  button (the whole process group is stopped: the shell scripts start children);
* ``on_finished(code)`` runs on the GUI thread when the command exits — that is
  where results are loaded into the viewer;
* closing the main window with jobs still running asks first, and stops them.

The output is read on a plain :class:`threading.Thread` and handed to the GUI
thread through a QObject's signals (a ``QThread`` worker segfaults at exit here).
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from qtpy.QtCore import QEvent, QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.activity import elapsed_text, start_task
from nvitk.gui.core.design import COLOR_MUTED, COLOR_TEXT
from nvitk.gui.core.log_panel import gui_log

#: Seconds a cancelled job gets to stop on SIGTERM before it is killed.
_TERM_GRACE_S = 5.0


class _Relay(QObject):
    """Carries the reader thread's output to the GUI thread (queued signals)."""

    line = Signal(str)
    finished = Signal(int)


class ExternalJob(QObject):
    """One external command running in the background.

    *stage_of(line)* may return a short description of where the command is (shown
    in the Jobs window and the activity bar) or ``None`` to leave it unchanged.
    """

    #: Emitted on the GUI thread with the exit code, after ``on_finished`` ran.
    done = Signal(int)

    def __init__(
        self,
        argv: Sequence[str],
        *,
        title: str,
        env: dict[str, str] | None = None,
        cwd: str | Path | None = None,
        on_finished: Callable[[int], None] | None = None,
        stage_of: Callable[[str], str | None] | None = None,
        log_file: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.argv = [str(a) for a in argv]
        self.title = str(title)
        self.env = env
        self.cwd = str(cwd) if cwd else None
        self.on_finished = on_finished
        self.stage_of = stage_of
        self.log_file = Path(log_file) if log_file else None
        self.stage = "starting"
        self.returncode: int | None = None
        self.cancelled = False
        self.started = time.monotonic()
        self._proc: subprocess.Popen | None = None
        self._task: Any | None = None
        self._relay = _Relay()
        self._relay.line.connect(self._on_line)
        self._relay.finished.connect(self._on_finished)

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._proc is not None and self.returncode is None

    def start(self) -> "ExternalJob":
        """Launch the command and return immediately."""
        gui_log(f"$ {' '.join(self.argv)}" + (f"  (cwd={self.cwd})" if self.cwd else ""))
        log_handle = None
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = self.log_file.open("a", encoding="utf-8")
        try:
            self._proc = subprocess.Popen(
                self.argv,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # Its own process group, so Cancel reaches the children a shell
                # script starts, not just the script.
                start_new_session=True,
            )
        except Exception as exc:  # noqa: BLE001
            if log_handle is not None:
                log_handle.close()
            gui_log(f"Failed to start {self.title}: {exc}", error=True)
            raise
        self._task = start_task(f"{self.title}", kind="job")
        _registry().add(self)
        proc = self._proc
        relay = self._relay

        def read() -> None:
            try:
                assert proc.stdout is not None
                for raw in proc.stdout:
                    if log_handle is not None:
                        log_handle.write(raw)
                    relay.line.emit(raw.rstrip("\n"))
            finally:
                code = int(proc.wait())
                if log_handle is not None:
                    log_handle.close()
                relay.finished.emit(code)

        threading.Thread(target=read, name=f"nvitk-job-{proc.pid}", daemon=True).start()
        return self

    def cancel(self) -> None:
        """Stop the whole process group (SIGTERM, then SIGKILL after a grace period)."""
        if not self.running or self._proc is None:
            return
        self.cancelled = True
        self.stage = "cancelling"
        _registry().refresh(self)
        pid = self._proc.pid
        _signal_group(pid, signal.SIGTERM)
        QTimer.singleShot(int(_TERM_GRACE_S * 1000), lambda: self.running and _signal_group(pid, signal.SIGKILL))

    def kill_now(self) -> None:
        """Stop the job before the application exits (blocking, a few seconds at most)."""
        if not self.running or self._proc is None:
            return
        self.cancelled = True
        _signal_group(self._proc.pid, signal.SIGTERM)
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _signal_group(self._proc.pid, signal.SIGKILL)

    # ── GUI-thread callbacks ─────────────────────────────────────────────────

    def _on_line(self, text: str) -> None:
        if not text.strip():
            return
        gui_log(text)
        if self.stage_of is not None:
            try:
                stage = self.stage_of(text)
            except Exception:  # noqa: BLE001
                stage = None
            if stage:
                self.stage = stage
                if self._task is not None:
                    self._task.update(message=f"{self.title} — {stage}")
                _registry().refresh(self)

    def _on_finished(self, code: int) -> None:
        self.returncode = int(code)
        ok = code == 0 and not self.cancelled
        took = elapsed_text(time.monotonic() - self.started)
        if self.cancelled:
            gui_log(f"{self.title}: cancelled after {took}.", error=True)
        elif ok:
            gui_log(f"{self.title}: finished in {took}.")
        else:
            gui_log(f"{self.title}: failed (exit code {code}) after {took}.", error=True)
        if self._task is not None:
            self._task.finish(ok)
        _registry().remove(self)
        if self.on_finished is not None and not self.cancelled:
            try:
                self.on_finished(self.returncode)
            except Exception as exc:  # noqa: BLE001 — a failed load must not kill the GUI
                gui_log(f"{self.title}: could not load the results: {exc}", error=True)
                try:
                    from nvitk.gui.tools.runner import notify

                    notify(f"{self.title}: could not load the results: {exc}", error=True)
                except Exception:  # noqa: BLE001
                    pass
        self.done.emit(self.returncode)


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


# ── the Jobs window ──────────────────────────────────────────────────────────


class _JobRow(QWidget):
    def __init__(self, job: ExternalJob, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.job = job
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        text = QVBoxLayout()
        self._title = QLabel(job.title)
        self._title.setStyleSheet(f"color: {COLOR_TEXT}; font-weight: bold;")
        self._stage = QLabel("")
        self._stage.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 10px;")
        self._stage.setWordWrap(True)
        text.addWidget(self._title)
        text.addWidget(self._stage)
        layout.addLayout(text, stretch=1)
        self._cancel = QPushButton("Cancel")
        self._cancel.setToolTip("Stop this job and everything it started.")
        self._cancel.clicked.connect(self._confirm_cancel)
        layout.addWidget(self._cancel)
        self.refresh()

    def _confirm_cancel(self) -> None:
        answer = QMessageBox.question(
            self, "Cancel job", f"Stop “{self.job.title}”? Its partial output stays on disk."
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.job.cancel()

    def refresh(self) -> None:
        self._stage.setText(f"{self.job.stage} · {elapsed_text(time.monotonic() - self.job.started)}")
        self._cancel.setEnabled(not self.job.cancelled)


class JobsWindow(QWidget):
    """Non-modal list of the running external jobs (shown while there are any)."""

    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.Tool)
        self.setWindowTitle("nvitk — running jobs")
        self.setMinimumWidth(380)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._rows: dict[int, _JobRow] = {}
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._refresh_all)
        try:
            from nvitk.gui.core.design import apply_theme

            apply_theme(self)
        except Exception:  # noqa: BLE001
            pass

    def add(self, job: ExternalJob) -> None:
        row = _JobRow(job, self)
        self._rows[id(job)] = row
        self._layout.addWidget(row)
        self._tick.start()
        self.adjustSize()
        self.show()
        self.raise_()

    def remove(self, job: ExternalJob) -> None:
        row = self._rows.pop(id(job), None)
        if row is not None:
            row.hide()
            row.setParent(None)
            row.deleteLater()
        if not self._rows:
            self._tick.stop()
            self.hide()
        else:
            self.adjustSize()

    def refresh(self, job: ExternalJob) -> None:
        row = self._rows.get(id(job))
        if row is not None:
            row.refresh()

    def _refresh_all(self) -> None:
        for row in self._rows.values():
            row.refresh()


class _Registry(QObject):
    """Every running job, the Jobs window, and the exit guard."""

    def __init__(self) -> None:
        super().__init__()
        self.jobs: list[ExternalJob] = []
        self._window: JobsWindow | None = None
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.kill_all)
        self._guarded: set[int] = set()

    def window(self) -> JobsWindow:
        if self._window is None:
            self._window = JobsWindow()
        return self._window

    def add(self, job: ExternalJob) -> None:
        self.jobs.append(job)
        self._guard_main_windows()
        self.window().add(job)

    def remove(self, job: ExternalJob) -> None:
        if job in self.jobs:
            self.jobs.remove(job)
        if self._window is not None:
            self._window.remove(job)

    def refresh(self, job: ExternalJob) -> None:
        if self._window is not None:
            self._window.refresh(job)

    def running(self) -> list[ExternalJob]:
        return [j for j in self.jobs if j.running]

    def kill_all(self) -> None:
        for job in list(self.running()):
            gui_log(f"Stopping “{job.title}” (the application is closing).", error=True)
            job.kill_now()

    def _guard_main_windows(self) -> None:
        """Ask before the main window closes with jobs still running."""
        app = QApplication.instance()
        if app is None:
            return
        for widget in app.topLevelWidgets():
            if type(widget).__name__ != "_QtMainWindow" or id(widget) in self._guarded:
                continue
            widget.installEventFilter(self)
            self._guarded.add(id(widget))

    def eventFilter(self, obj: Any, event: Any) -> bool:  # noqa: N802 — Qt naming
        if event.type() == QEvent.Type.Close and self.running():
            names = "\n".join(f"  • {j.title}" for j in self.running())
            answer = QMessageBox.question(
                obj,
                "Jobs still running",
                f"These jobs are still running:\n{names}\n\nQuit and stop them?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return True
            self.kill_all()
        return False


_REGISTRY: _Registry | None = None


def _registry() -> _Registry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _Registry()
    return _REGISTRY


def running_jobs() -> list[ExternalJob]:
    """The external jobs still running."""
    return _registry().running() if _REGISTRY is not None else []


def start_external_job(argv: Sequence[str], *, title: str, **kwargs: Any) -> ExternalJob:
    """Create an :class:`ExternalJob` and start it (see the class for *kwargs*)."""
    return ExternalJob(argv, title=title, **kwargs).start()


__all__ = ["ExternalJob", "JobsWindow", "running_jobs", "start_external_job"]
