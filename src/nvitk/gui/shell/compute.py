"""The compute button: CPU / GPU backend and CPU threads, from one place at the foot of the
activity bar (it used to be two wide buttons, "GPU computing: ON/OFF" and "CPU: N / M threads",
over the Tools panel).

The icon is tinted in the accent while the GPU backend is on; its tooltip says what is in force.
Clicking opens a pop-over with the backend switch (process-wide, as ``--backend cpu|gpu``) and
the CPU threads (the performance dialog: workers for the views, resampling, reformats, FFTs).
"""

from __future__ import annotations

from qtpy.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from nvitk.gui.core import design
from nvitk.gui.core.widgets import IconButton, Popover, SegmentedButtons, hint_label


def _gpu_on() -> bool:
    from nvitk.core.backend import get_global_backend

    return get_global_backend() == "cupy"


def compute_summary() -> str:
    """``GPU (CuPy) · CPU: 24 / 32 threads``."""
    from nvitk.gui.core.performance import performance_label
    from nvitk.gui.tools.gpu_toggle import backend_label

    return f"{backend_label()} · {performance_label()}"


class ComputeButton(IconButton):
    """Icon button with the compute pop-over (see the module docstring)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("gpu", "", icon_size=18, parent=parent)
        self._popover: Popover | None = None
        self.clicked.connect(self._toggle)
        self.sync()

    def sync(self) -> None:
        """Tint and tooltip from the backend and threads in force."""
        self._hue = "green" if _gpu_on() else None
        self._restyle()
        self.setToolTip(f"Compute: {compute_summary()}\nClick to switch CPU / GPU or change the CPU threads.")

    def _content(self) -> QWidget:
        from nvitk.core.backend import is_gpu_available
        from nvitk.gui.core.performance import open_performance_dialog, performance_label
        from nvitk.gui.tools.gpu_toggle import set_gui_backend

        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(design.SPACE_TIGHT)
        lay.addWidget(hint_label("Backend for every nvitk tool (same as --backend cpu|gpu)"))
        backend = SegmentedButtons([("cpu", "CPU · NumPy", "tools"), ("gpu", "GPU · CuPy", "gpu")],
                                   current="gpu" if _gpu_on() else "cpu")
        gpu_ok = is_gpu_available()
        if not gpu_ok and backend.button("gpu") is not None:
            backend.button("gpu").setEnabled(False)
            backend.button("gpu").setToolTip("No CUDA / CuPy in this environment")
        lay.addWidget(backend)
        threads = hint_label(performance_label())
        row = QHBoxLayout()
        row.addWidget(threads, 1)
        change = QPushButton("Change…")
        change.setToolTip("CPU threads for the views, resampling, reformats and FFTs (remembered)")
        row.addWidget(change)
        lay.addLayout(row)

        def _backend(key: str) -> None:
            set_gui_backend(key)
            backend.set_current("gpu" if _gpu_on() else "cpu")
            self.sync()

        def _threads() -> None:
            if open_performance_dialog(self.window()):
                threads.setText(performance_label())
                self.sync()

        backend.changed.connect(_backend)
        change.clicked.connect(_threads)
        design.apply_theme(panel)
        return panel

    def _toggle(self) -> None:
        if self._popover is None:
            self._popover = Popover(title="Compute", parent=self.window())
        if not self._popover.isVisible():
            self._popover.set_content(self._content())  # fresh: the backend may have changed
        self._popover.toggle_at(self, side="right")


__all__ = ["ComputeButton", "compute_summary"]
