"""CPU performance preferences for the GUI: worker threads and Napari async slicing.

The heavy host work behind the interactive views — slicing and compositing the
orthogonal views, resampling a layer that sits on another grid, oblique and CPR
reformats, FFTs — runs on nvitk's shared worker pool
(:mod:`nvitk.core.parallel`). This module is the GUI side of it: the stored
preference, applying it at start-up, and the small dialog that changes it.

The default is :data:`~nvitk.core.parallel.DEFAULT_WORKER_FRACTION` of the
machine's cores (three quarters), so a workstation is put to use without the
desktop going unresponsive while a whole-body CT resamples. ``$NVITK_WORKERS``
still overrides the stored value for one session.

Stored in ``gui.json`` under ``"performance"``::

    {"performance": {"workers": "75%", "async_slicing": false}}
"""

from __future__ import annotations

import os
from typing import Any

from nvitk.core.logger import Logger

log = Logger()

#: ``gui.json`` key holding the performance preferences.
PERF_KEY = "performance"

#: What a fresh install uses.
DEFAULT_PERFORMANCE: dict[str, Any] = {"workers": "75%", "async_slicing": False}


# ──────────────────────────────────────────────────────────────────────────────
# Stored preference
# ──────────────────────────────────────────────────────────────────────────────


def stored_performance() -> dict[str, Any]:
    """The saved performance preferences merged over the defaults."""
    from nvitk.gui.core.prefs import load_prefs

    raw = load_prefs().get(PERF_KEY)
    out = dict(DEFAULT_PERFORMANCE)
    if isinstance(raw, dict):
        out.update({k: v for k, v in raw.items() if k in DEFAULT_PERFORMANCE})
    return out


def save_performance(values: dict[str, Any]) -> bool:
    """Persist *values* (merged with what is stored); ``True`` when written."""
    from nvitk.gui.core.prefs import save_prefs

    merged = stored_performance()
    merged.update({k: v for k, v in values.items() if k in DEFAULT_PERFORMANCE})
    return save_prefs({PERF_KEY: merged})


def apply_performance(values: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Put the preferences into effect for this process and return what applied.

    ``$NVITK_WORKERS`` wins over the stored worker count, so a session launched
    with an explicit budget (a shared node, a benchmark) keeps it.
    """
    from nvitk.core.parallel import WORKERS_ENV_VAR, cpu_count, set_worker_count

    prefs = dict(values) if values is not None else stored_performance()
    env = os.environ.get(WORKERS_ENV_VAR, "").strip()
    spec = env or prefs.get("workers") or DEFAULT_PERFORMANCE["workers"]
    try:
        workers = set_worker_count(spec)
    except ValueError:
        log.warning("Unreadable worker setting %r; using the default.", spec)
        workers = set_worker_count(DEFAULT_PERFORMANCE["workers"])
    async_on = bool(prefs.get("async_slicing", False))
    set_async_slicing(async_on)
    applied = {
        "workers": workers,
        "cpu_count": cpu_count(),
        "async_slicing": async_on,
        "source": "env" if env else "prefs",
    }
    log.info(
        "CPU workers: %d of %d cores (%s)%s",
        workers, applied["cpu_count"], applied["source"],
        "; Napari async slicing on" if async_on else "",
    )
    return applied


def set_async_slicing(enabled: bool) -> bool:
    """Turn Napari's experimental asynchronous slicing on or off.

    Slicing then happens off the GUI thread, so dragging a slider through a
    whole-body CT no longer blocks the window while each slice is read. It is
    Napari's *experimental* setting, and Napari stores it in its own settings
    file — only written when it actually changes.
    """
    try:
        from napari.settings import get_settings

        experimental = get_settings().experimental
        if bool(getattr(experimental, "async_", False)) != bool(enabled):
            experimental.async_ = bool(enabled)
        return True
    except Exception:  # noqa: BLE001 — an older Napari without the setting
        return False


def performance_label() -> str:
    """Short button text: ``CPU: 24 / 32 threads``."""
    from nvitk.core.parallel import cpu_count, get_worker_count

    return f"CPU: {get_worker_count()} / {cpu_count()} threads"


# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────


def open_performance_dialog(parent: Any = None) -> bool:
    """Show the performance dialog; ``True`` when the user applied a change."""
    from qtpy.QtWidgets import (
        QCheckBox,
        QDialog,
        QDialogButtonBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
    )

    from nvitk.core.parallel import cpu_count, get_worker_count, resolve_workers
    from nvitk.gui.core.design import COLOR_MUTED, SPACE, SPACE_TIGHT, apply_theme

    cores = cpu_count()
    dialog = QDialog(parent)
    dialog.setWindowTitle("nvitk performance")
    root = QVBoxLayout(dialog)
    root.setContentsMargins(SPACE, SPACE, SPACE, SPACE)
    root.setSpacing(SPACE_TIGHT)

    intro = QLabel(
        f"Threads nvitk may use for slicing, compositing and resampling the "
        f"views, oblique/CPR reformats and FFTs. This machine has {cores} cores."
    )
    intro.setWordWrap(True)
    root.addWidget(intro)

    row = QHBoxLayout()
    row.addWidget(QLabel("Worker threads"))
    spin = QSpinBox()
    spin.setRange(1, cores)
    spin.setValue(get_worker_count())
    row.addWidget(spin, 1)
    root.addLayout(row)

    presets = QHBoxLayout()
    for label, spec in (("25%", "25%"), ("50%", "50%"), ("75% (default)", "75%"), ("All", "all")):
        btn = QPushButton(label)
        btn.clicked.connect(lambda _c=False, s=spec: spin.setValue(resolve_workers(s)))
        presets.addWidget(btn)
    root.addLayout(presets)

    env = os.environ.get("NVITK_WORKERS", "").strip()
    if env:
        note = QLabel(f"$NVITK_WORKERS={env} is set and overrides this for the current session.")
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 10px;")
        root.addWidget(note)

    async_box = QCheckBox("Asynchronous slicing (Napari, experimental)")
    async_box.setChecked(bool(stored_performance().get("async_slicing")))
    async_box.setToolTip(
        "Read each canvas slice off the GUI thread, so scrolling a very large "
        "volume does not freeze the window. Napari marks it experimental."
    )
    root.addWidget(async_box)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    root.addWidget(buttons)
    apply_theme(dialog)
    dialog.setMinimumWidth(380)

    if dialog.exec() != QDialog.Accepted:
        return False
    # Stored as a share of the machine when it matches one of the presets, so the
    # same gui.json does the sensible thing on a laptop and on a workstation.
    n = int(spin.value())
    spec: Any = n
    for share in ("25%", "50%", "75%", "all"):
        if resolve_workers(share) == n:
            spec = share
            break
    values = {"workers": spec, "async_slicing": bool(async_box.isChecked())}
    save_performance(values)
    apply_performance(values)
    return True


def build_performance_button(parent: Any = None) -> Any:
    """``CPU: N / M threads`` button opening :func:`open_performance_dialog`."""
    from qtpy.QtWidgets import QPushButton

    btn = QPushButton(performance_label())
    btn.setToolTip(
        "CPU threads for the interactive views, resampling, reformats and FFTs "
        "(default: 75% of the cores). Click to change; remembered in gui.json."
    )

    def _open() -> None:
        if open_performance_dialog(btn.window()):
            btn.setText(performance_label())

    btn.clicked.connect(_open)
    return btn


__all__ = [
    "DEFAULT_PERFORMANCE",
    "PERF_KEY",
    "apply_performance",
    "build_performance_button",
    "open_performance_dialog",
    "performance_label",
    "save_performance",
    "set_async_slicing",
    "stored_performance",
]
