"""Scrollable log area for tool and pipeline subprocess output."""

from __future__ import annotations

import os

import subprocess
from datetime import datetime
from typing import Any, Callable

from qtpy.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


_log_widget: QPlainTextEdit | None = None


def set_log_widget(widget: QPlainTextEdit | None) -> None:
    """Register *widget* as the log panel that :func:`gui_log` appends to (module-level singleton)."""
    global _log_widget
    _log_widget = widget


def gui_log(message: str, *, error: bool = False) -> None:
    """Append a timestamped log line to the registered log widget (if any) and stdout."""
    stamp = datetime.now().strftime("%H:%M:%S")
    level = "ERROR" if error else "INFO"
    line = f"[{stamp}] {level}: {message}"
    if _log_widget is not None:
        _log_widget.appendPlainText(line)
        sb = _log_widget.verticalScrollBar()
        sb.setValue(sb.maximum())
    print(line, flush=True)


def build_log_dock_widget() -> QWidget:
    """Bottom dock: plain-text log of tools and pipelines."""
    w = QPlainTextEdit()
    w.setReadOnly(True)
    w.setPlaceholderText("Tool and pipeline logs appear here…")
    w.setMaximumBlockCount(5000)
    set_log_widget(w)
    container = QWidget()
    lay = QVBoxLayout()
    lay.setContentsMargins(4, 4, 4, 4)
    lay.addWidget(w)
    container.setLayout(lay)
    return container


def run_subprocess_logged(
    argv: list[str],
    *,
    cwd = None,
    on_line = None,
    env = None,
) -> int:
    """Run a command and stream stdout/stderr into :func:`gui_log`.

    *env* replaces the child's whole environment when given: pass ``{**os.environ, ...}`` to add
    to it rather than to start from nothing. It exists so a caller can hand a subprocess
    credentials it must not print: the command line is echoed to the log, the environment is not.
    """
    cmd = " ".join(argv)
    gui_log(f"$ {cmd}" + (f"  (cwd={cwd})" if cwd else ""))
    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        gui_log(f"Failed to start: {exc}", error=True)
        return -1

    assert proc.stdout is not None
    import queue
    import threading

    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QApplication

    def show(line: str) -> None:
        text = line.rstrip()
        if text:
            gui_log(text)
            if on_line is not None:
                on_line(text)

    if QApplication.instance() is None or threading.current_thread() is not threading.main_thread():
        for line in proc.stdout:
            show(line)
        return int(proc.wait())

    # The output is read on a worker thread and shown here, on the GUI thread, as it
    # comes: the window keeps repainting (and the activity indicator moving) while
    # the command runs, instead of freezing until it exits.
    from nvitk.gui.core.activity import activity, run_in_background

    lines: "queue.Queue[str]" = queue.Queue()

    def read() -> int:
        for line in proc.stdout:
            lines.put(line)
        return int(proc.wait())

    def drain() -> None:
        while True:
            try:
                show(lines.get_nowait())
            except queue.Empty:
                return

    timer = QTimer()
    timer.setInterval(100)
    timer.timeout.connect(drain)
    timer.start()
    try:
        with activity(f"Running {os.path.basename(str(argv[0]))}", cursor=False):
            code = run_in_background(read)
    finally:
        timer.stop()
        drain()
    return int(code)
