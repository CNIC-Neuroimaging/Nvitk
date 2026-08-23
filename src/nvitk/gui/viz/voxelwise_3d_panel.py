"""
Configuration window for the 3-D voxelwise scene.

The scene has a dozen knobs — which map, which contrasts, the threshold window, surfaces or
points, the shell and its opacity — and they interact: the sensible threshold for a 1−p map is
nothing like the one for a t-statistic, and the range a slider should span comes from the data.
That is more than a couple of buttons on the analysis dialog can carry, so it gets its own window.

One button does everything. Choosing a folder loads it, fills the map and contrast pickers from
what is actually in it, and picks a defensible threshold from the data; **Show** draws the scene.
There is no separate load step to get out of step with the display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.gui.core.log_panel import gui_log
from nvitk.gui.viz.voxelwise_3d import (
    SCENE_COLORMAPS,
    SCENE_MODES,
    SceneSpec,
    build_scene,
    clear_voxelwise_layers,
    map_data,
    suggest_band,
    value_range,
)

log = Logger()

DIALOG_OBJECT_NAME = "nvitk_voxelwise_3d_dialog"


class Voxelwise3DPanel(QDialog):
    """Pick a result, a map and a threshold, and draw it in 3-D."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        """Build the form and wire it up; nothing is loaded until a folder is chosen."""
        super().__init__(parent)
        self.setObjectName(DIALOG_OBJECT_NAME)
        self.setWindowTitle("Voxelwise 3-D scene")
        self.setMinimumWidth(560)
        self._viewer = viewer
        self._data: np.ndarray | None = None

        outer = QVBoxLayout(self)
        outer.addWidget(self._build_source_group())
        outer.addWidget(self._build_threshold_group())
        outer.addWidget(self._build_style_group())

        self._status = QLabel("Choose a results folder to begin.")
        self._status.setWordWrap(True)
        self._status.setTextFormat(Qt.PlainText)
        outer.addWidget(self._status)
        outer.addLayout(self._build_buttons())

    # -- construction ---------------------------------------------------------
    def _build_source_group(self) -> QGroupBox:
        """Results folder, map kind and contrast selection."""
        box = QGroupBox("Result")
        form = QFormLayout(box)

        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        self._dir = QLineEdit()
        self._dir.setPlaceholderText("randomise results folder…")
        self._dir.editingFinished.connect(self._reload)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_dir)
        row_lay.addWidget(self._dir, 1)
        row_lay.addWidget(browse)
        form.addRow("Folder", row)

        self._kind = QComboBox()
        self._kind.setToolTip(
            "Which map to draw.\n\n"
            "The corrected 1 − p maps say where the evidence is; the t-statistic says how large "
            "and in which direction the effect is. They are thresholded differently, so changing "
            "this resets the window below to something sensible for the new map."
        )
        self._kind.currentIndexChanged.connect(self._on_kind_changed)
        form.addRow("Map", self._kind)

        self._contrasts = QListWidget()
        self._contrasts.setSelectionMode(QAbstractItemView.MultiSelection)
        self._contrasts.setMaximumHeight(70)
        self._contrasts.setToolTip("Contrasts to draw. Several at once are drawn as separate layers.")
        form.addRow("Contrasts", self._contrasts)
        return box

    def _build_threshold_group(self) -> QGroupBox:
        """The value window, whose meaning follows the map kind."""
        box = QGroupBox("Threshold")
        form = QFormLayout(box)

        row = QWidget()
        row_lay = QHBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 0, 0)
        self._lo = QDoubleSpinBox()
        self._hi = QDoubleSpinBox()
        for spin in (self._lo, self._hi):
            spin.setDecimals(3)
            spin.setRange(0.0, 1.0)
            spin.setSingleStep(0.005)
            spin.setMaximumWidth(110)
        self._lo.setValue(0.950)
        self._hi.setValue(1.000)
        row_lay.addWidget(QLabel("from"))
        row_lay.addWidget(self._lo)
        row_lay.addWidget(QLabel("to"))
        row_lay.addWidget(self._hi)
        row_lay.addStretch(1)
        form.addRow("Window", row)

        self._threshold_hint = QLabel()
        self._threshold_hint.setWordWrap(True)
        form.addRow("", self._threshold_hint)

        reset = QPushButton("Suggest from data")
        reset.setToolTip("Pick a window from this map's own distribution.")
        reset.clicked.connect(self._suggest)
        form.addRow("", reset)
        return box

    def _build_style_group(self) -> QGroupBox:
        """How the clusters and the shell are drawn."""
        box = QGroupBox("Appearance")
        form = QFormLayout(box)

        self._mode = QComboBox()
        for label, key in SCENE_MODES:
            self._mode.addItem(label, key)
        self._mode.setToolTip(
            "Iso-surfaces give each cluster a solid shape — best for extent and location.\n"
            "Points colour every voxel by its own value, showing the gradient inside a cluster."
        )
        self._mode.currentIndexChanged.connect(self._sync_style_enabled)
        form.addRow("Clusters as", self._mode)

        self._colormap = QComboBox()
        for name in SCENE_COLORMAPS:
            self._colormap.addItem(name, name)
        form.addRow("Colormap", self._colormap)

        self._point_size = QDoubleSpinBox()
        self._point_size.setRange(0.5, 20.0)
        self._point_size.setSingleStep(0.5)
        self._point_size.setValue(2.5)
        self._point_size.setMaximumWidth(90)
        self._point_size_label = QLabel("Point size")
        form.addRow(self._point_size_label, self._point_size)

        self._cluster_opacity = QDoubleSpinBox()
        self._cluster_opacity.setRange(0.05, 1.0)
        self._cluster_opacity.setSingleStep(0.05)
        self._cluster_opacity.setValue(1.0)
        self._cluster_opacity.setMaximumWidth(90)
        form.addRow("Cluster opacity", self._cluster_opacity)

        self._shell = QCheckBox("Show brain shell")
        self._shell.setChecked(True)
        self._shell.setToolTip(
            "A translucent surface of the brain, so the clusters have something to sit inside.\n\n"
            "Built from the analysis mask when the folder has one, otherwise from the MNI152 brain "
            "mask resampled onto this result's grid — an analysis run with --mask does not leave "
            "its mask behind, so the fallback is the usual case rather than the exception."
        )
        self._shell.stateChanged.connect(self._sync_style_enabled)
        form.addRow("", self._shell)

        self._shell_opacity = QDoubleSpinBox()
        self._shell_opacity.setRange(0.02, 1.0)
        self._shell_opacity.setSingleStep(0.02)
        self._shell_opacity.setValue(0.22)
        self._shell_opacity.setMaximumWidth(90)
        self._shell_opacity_label = QLabel("Shell opacity")
        form.addRow(self._shell_opacity_label, self._shell_opacity)
        return box

    def _build_buttons(self) -> QHBoxLayout:
        """Show, clear, close."""
        row = QHBoxLayout()
        self._show_btn = QPushButton("Show in 3D")
        self._show_btn.setDefault(True)
        self._show_btn.clicked.connect(self._show)
        clear = QPushButton("Clear layers")
        clear.clicked.connect(self._clear)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(self._show_btn)
        row.addWidget(clear)
        row.addStretch(1)
        row.addWidget(close)
        return row

    # -- state ----------------------------------------------------------------
    def _pick_dir(self) -> None:
        """Choose a folder and load it immediately — one action, not two."""
        chosen = QFileDialog.getExistingDirectory(
            self, "Select a voxelwise results folder", self._dir.text()
        )
        if chosen:
            self._dir.setText(chosen)
            self._reload()

    def set_directory(self, out_dir: str | Path) -> None:
        """Point the window at *out_dir* and load it (used when opened from the analysis dialog)."""
        self._dir.setText(str(out_dir))
        self._reload()

    def _reload(self) -> None:
        """Read the folder and repopulate the pickers from what is actually in it."""
        directory = self._dir.text().strip()
        if not directory:
            return
        try:
            from nvitk.measure.voxelwise import STAT_KINDS, load_voxelwise_result

            result = load_voxelwise_result(directory)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not read that folder: {exc}")
            return

        previous = str(self._kind.currentData() or "")
        self._kind.blockSignals(True)
        self._kind.clear()
        for kind in sorted(result.maps):
            self._kind.addItem(STAT_KINDS.get(kind, kind), kind)
        index = self._kind.findData(previous or result.primary_kind())
        self._kind.setCurrentIndex(index if index >= 0 else 0)
        self._kind.blockSignals(False)

        chosen = {i.text() for i in self._contrasts.selectedItems()}
        self._contrasts.clear()
        for name in result.contrast_names:
            self._contrasts.addItem(name)
        for i in range(self._contrasts.count()):
            item = self._contrasts.item(i)
            item.setSelected(item.text() in chosen or not chosen)

        self._on_kind_changed()

    def _on_kind_changed(self) -> None:
        """Re-read the map and reset the window to something meaningful for this kind."""
        directory = self._dir.text().strip()
        kind = str(self._kind.currentData() or "")
        if not directory or not kind:
            return
        try:
            self._data, _kind, _names = map_data(directory, kind=kind)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not read the {kind} map: {exc}")
            self._data = None
            return

        low, high = value_range(self._data, kind)
        for spin in (self._lo, self._hi):
            spin.blockSignals(True)
            spin.setRange(float(low), float(high))
            spin.setSingleStep(max(0.001, (high - low) / 200.0))
            spin.setDecimals(3 if high <= 1.0 else 2)
            spin.blockSignals(False)
        self._suggest()
        self._sync_style_enabled()

    def _suggest(self) -> None:
        """Set the window from the current map's own distribution."""
        if self._data is None:
            return
        kind = str(self._kind.currentData() or "")
        lo, hi = suggest_band(self._data, kind)
        self._lo.setValue(lo)
        self._hi.setValue(hi)

        from nvitk.stats.voxelwise_map import is_corrp_kind

        if is_corrp_kind(kind):
            self._threshold_hint.setText(
                f"1 − p, so {lo:g} means p < {1.0 - lo:g}. Voxels outside the window are hidden — "
                "an upper edge below 1 also hides the most significant ones."
            )
        else:
            self._threshold_hint.setText(
                f"Signed statistic: the window is on |value|, so both tails are kept. "
                f"This map runs to {self._hi.maximum():g}; there is no conventional cut, and "
                f"{lo:g} is just its 99th percentile."
            )

    def _sync_style_enabled(self) -> None:
        """Show only the controls the current mode and shell setting use."""
        points = str(self._mode.currentData() or "surface") == "points"
        self._point_size.setVisible(points)
        self._point_size_label.setVisible(points)
        shell = self._shell.isChecked()
        self._shell_opacity.setVisible(shell)
        self._shell_opacity_label.setVisible(shell)

    def spec(self) -> SceneSpec:
        """The current form state as a :class:`SceneSpec`."""
        directory = self._dir.text().strip()
        if not directory:
            raise ValueError("Choose a results folder.")
        if self._hi.value() <= self._lo.value():
            raise ValueError(
                f"The window is empty: {self._lo.value():g} is not below {self._hi.value():g}."
            )
        selected = tuple(i.text() for i in self._contrasts.selectedItems())
        return SceneSpec(
            out_dir=Path(directory),
            kind=str(self._kind.currentData() or ""),
            contrasts=selected,
            lo=float(self._lo.value()),
            hi=float(self._hi.value()),
            mode=str(self._mode.currentData() or "surface"),
            colormap=str(self._colormap.currentData() or "hot"),
            point_size=float(self._point_size.value()),
            show_shell=self._shell.isChecked(),
            shell_opacity=float(self._shell_opacity.value()),
            cluster_opacity=float(self._cluster_opacity.value()),
        )

    # -- actions --------------------------------------------------------------
    def _show(self) -> None:
        """Build the scene."""
        try:
            spec = self.spec()
        except ValueError as exc:
            QMessageBox.warning(self, "Voxelwise 3-D", str(exc))
            return
        self._show_btn.setEnabled(False)
        try:
            layers, caption = build_scene(self._viewer, spec)
        except Exception as exc:  # noqa: BLE001
            log.debug("Voxelwise 3-D scene failed: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Voxelwise 3-D", f"Could not build the scene:\n{exc}")
            return
        finally:
            self._show_btn.setEnabled(True)
        gui_log(f"Voxelwise 3-D: {len(layers)} layer(s) from {spec.out_dir.name}")
        self._status.setText(caption)

    def _clear(self) -> None:
        """Remove this module's layers, leaving anything else the user opened."""
        clear_voxelwise_layers(self._viewer)
        self._status.setText("Cleared the voxelwise layers.")


def start_voxelwise_3d(viewer: Any, out_dir: str | Path | None = None) -> Voxelwise3DPanel:
    """Open the 3-D window for *viewer* (reusing an open one), optionally pointed at a folder."""
    for widget in viewer.window._qt_window.findChildren(QDialog):
        if widget.objectName() == DIALOG_OBJECT_NAME:
            if out_dir:
                widget.set_directory(out_dir)  # type: ignore[attr-defined]
            widget.show()
            widget.raise_()
            return widget  # type: ignore[return-value]
    panel = Voxelwise3DPanel(viewer, parent=viewer.window._qt_window)
    if out_dir:
        panel.set_directory(out_dir)
    panel.show()
    return panel


__all__ = ["DIALOG_OBJECT_NAME", "Voxelwise3DPanel", "start_voxelwise_3d"]
