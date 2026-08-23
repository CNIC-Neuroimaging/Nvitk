"""Scrollable log area for tool and pipeline subprocess output."""

from __future__ import annotations

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

    *env* replaces the child's whole environment when given — pass ``{**os.environ, ...}`` to add
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
    for line in proc.stdout:
        text = line.rstrip()
        if text:
            gui_log(text)
            if on_line is not None:
                on_line(text)
    return int(proc.wait())
