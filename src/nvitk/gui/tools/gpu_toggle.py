"""Global GPU/CPU backend toggle for the nvitk GUI.

Uses the same backend selection as CLI entry points
(:func:`~nvitk.core.click_backend.apply_cli_backend` / ``--backend cpu|gpu``).
"""

from __future__ import annotations

from typing import Callable

from nvitk.core.backend import BackendName, get_global_backend, is_gpu_available
from nvitk.core.click_backend import apply_cli_backend
from nvitk.gui.core.log_panel import gui_log


def backend_label() -> str:
    """Human-readable label for the active global backend."""
    return "GPU (CuPy)" if get_global_backend() == "cupy" else "CPU (NumPy)"


def set_gui_backend(backend: str, *, log: bool = True) -> BackendName:
    """Set process-wide backend from ``cpu`` / ``gpu`` (same as ``nvitk-gui --backend``)."""
    requested = str(backend).strip().lower()
    if requested == "gpu" and not is_gpu_available():
        resolved = apply_cli_backend("cpu")
        if log:
            gui_log(
                "GPU backend unavailable (no CUDA/CuPy). Using CPU (NumPy).",
                error=True,
            )
        return resolved
    resolved = apply_cli_backend(requested)
    if log:
        gui_log(f"Backend: {backend_label()} ({resolved})")
    return resolved


def set_gpu_backend(enabled: bool, *, log: bool = True) -> BackendName:
    """Toggle GPU on/off via :func:`set_gui_backend` (``gpu`` / ``cpu``)."""
    return set_gui_backend("gpu" if enabled else "cpu", log=log)


def build_gpu_toggle_button(*, on_changed: Callable[[], None] | None = None):
    """Checkable button that toggles the global nvitk backend."""
    from qtpy.QtWidgets import QPushButton

    btn = QPushButton()
    btn.setCheckable(True)
    btn.setToolTip(
        "Toggle process-wide compute backend (--backend cpu|gpu). "
        "Applies to all nvitk tools until changed again."
    )

    def _sync_checked() -> None:
        btn.blockSignals(True)
        btn.setChecked(get_global_backend() == "cupy")
        btn.setText(f"GPU computing: {'ON' if btn.isChecked() else 'OFF'}")
        btn.blockSignals(False)

    def _on_toggled(checked: bool) -> None:
        set_gpu_backend(checked)
        _sync_checked()
        if on_changed is not None:
            on_changed()

    btn.toggled.connect(_on_toggled)
    _sync_checked()
    return btn
