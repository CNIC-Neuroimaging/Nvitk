"""ROI box panel: set a world-space box and crop / mask any selected layers with it.

Opened from the viewport's top bar (the dashed-box icon) and from the layer menu. The box comes
from the Labeling box (drawn in the views), from the active label map's extent (optionally with
a margin), from the whole active layer, or typed in millimetres; it can be saved and loaded as
JSON and sent back to the Labeling box on any grid.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtWidgets import QDoubleSpinBox, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from nvitk.gui.core import design
from nvitk.gui.core.roi_world import WorldBox, crop_layers, current_box, mask_outside, set_box
from nvitk.gui.core.widgets import Section, hint_label

_AXES = ("axis 0", "axis 1", "axis 2")


def _notify(text: str, *, error: bool = False) -> None:
    try:
        from nvitk.gui.tools.runner import notify

        notify(text, error=error)
    except Exception:  # noqa: BLE001
        pass


class RoiBoxPanel(QWidget):
    """Build, edit and apply the viewer's world ROI box."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self.setMinimumWidth(360)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)

        source = Section("Set the box", icon_name="roi_box", hue="cyan", pref_key="roi.source")
        row = QHBoxLayout()
        for text, tip, slot in (
            ("From Labeling box", "The box drawn in the views with the Labeling panel's box tool.", self._from_labeling),
            ("Fit labels", "Around the active label map's non-zero voxels (plus the margin).", self._from_labels),
            ("Whole layer", "Around all of the active layer.", self._from_layer),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        source.add_layout(row)
        self._draw = QPushButton("Draw in a view")
        self._draw.setCheckable(True)
        self._draw.setToolTip("Drag a rectangle in an orthogonal view (or the 2D canvas); the box takes "
                              "its two in-plane sides, then drag in another view for the third.")
        self._draw.toggled.connect(self._arm_draw)
        source.add(self._draw)
        margin_row = QHBoxLayout()
        margin_row.addWidget(QLabel("Margin"))
        self._margin = QDoubleSpinBox()
        self._margin.setRange(0.0, 200.0)
        self._margin.setSuffix(" mm")
        self._margin.setValue(5.0)
        margin_row.addWidget(self._margin)
        margin_row.addStretch(1)
        source.add_layout(margin_row)
        root.addWidget(source)

        bounds = Section("Bounds (world, mm)", icon_name="ruler", hue="blue", pref_key="roi.bounds")
        grid = QGridLayout()
        grid.addWidget(QLabel("min"), 0, 1)
        grid.addWidget(QLabel("max"), 0, 2)
        self._spins: list[tuple[QDoubleSpinBox, QDoubleSpinBox]] = []
        for i, name in enumerate(_AXES):
            grid.addWidget(QLabel(name), i + 1, 0)
            pair = []
            for j in range(2):
                spin = QDoubleSpinBox()
                spin.setRange(-1e5, 1e5)
                spin.setDecimals(1)
                spin.setSuffix(" mm")
                spin.editingFinished.connect(self._from_spins)
                grid.addWidget(spin, i + 1, j + 1)
                pair.append(spin)
            self._spins.append((pair[0], pair[1]))
        bounds.add_layout(grid)
        self._summary = hint_label("No box yet.")
        bounds.add(self._summary)
        root.addWidget(bounds)

        apply = Section("Apply to the selected layers", icon_name="crop", hue="orange", pref_key="roi.apply")
        row = QHBoxLayout()
        for text, tip, slot in (
            ("Crop", "A cropped copy of each selected image / label layer, placed where it was.", self._crop),
            ("Mask outside", "A copy of each with everything outside the box set to 0 (same grid).", self._mask),
            ("To Labeling box", "Make this the Labeling box on the active layer's grid.", self._to_labeling),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        apply.add_layout(row)
        row = QHBoxLayout()
        for text, slot in (("Save…", self._save), ("Load…", self._load), ("Clear", self._clear)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        apply.add_layout(row)
        root.addWidget(apply)
        self._sync_spins()

    # drawing (the Labeling panel's box tool draws; the box follows it)
    def _labeling(self) -> Any:
        return getattr(self._viewer, "_nvitk_labeling_panel", None)

    def _arm_draw(self, on: bool) -> None:
        from nvitk.gui.labels.roi_box import roi_box

        labeling = self._labeling()
        if labeling is None:
            if on:
                _notify("Drawing needs the Labeling panel.", error=True)
                self._draw.setChecked(False)
            return
        box = roi_box(self._viewer)
        if on:
            if not getattr(self, "_follow_connected", False):
                box.changed.connect(self._on_drawn)
                self._follow_connected = True
            labeling._box_draw.setChecked(True)
            if not labeling.box_armed():
                self._draw.setChecked(False)
                _notify("Select an image or a label layer to draw the box on.", error=True)
                return
            _notify("Drag a rectangle in a view.")
        elif labeling.box_armed():
            labeling._box_draw.setChecked(False)

    def _on_drawn(self) -> None:
        if not self._draw.isChecked():
            return
        box = WorldBox.from_roi_box(self._viewer)
        if box is not None:
            self._set(box)
        self._draw.blockSignals(True)
        self._draw.setChecked(False)
        self._draw.blockSignals(False)

    # sources
    def _active(self) -> Any:
        return self._viewer.layers.selection.active

    def _set(self, box: WorldBox | None) -> None:
        set_box(self._viewer, box)
        self._sync_spins()

    def _from_labeling(self) -> None:
        box = WorldBox.from_roi_box(self._viewer)
        if box is None:
            _notify("Draw a box first (Labeling ▸ Box, then drag in a view).", error=True)
            return
        self._set(box)

    def _from_labels(self) -> None:
        layer = self._active()
        if layer is None:
            _notify("Select a label map (or mask) first.", error=True)
            return
        try:
            self._set(WorldBox.from_labels(layer, margin_mm=float(self._margin.value())))
        except Exception as exc:  # noqa: BLE001
            _notify(str(exc), error=True)

    def _from_layer(self) -> None:
        layer = self._active()
        if layer is None or getattr(layer, "data", None) is None:
            _notify("Select an image or label layer first.", error=True)
            return
        try:
            self._set(WorldBox.from_layer(layer))
        except Exception as exc:  # noqa: BLE001
            _notify(str(exc), error=True)

    def _from_spins(self) -> None:
        lo = [pair[0].value() for pair in self._spins]
        hi = [pair[1].value() for pair in self._spins]
        if any(b <= a for a, b in zip(lo, hi)):
            return
        set_box(self._viewer, WorldBox(lo, hi))
        self._sync_spins()

    def _sync_spins(self) -> None:
        box = current_box(self._viewer)
        for i, (a, b) in enumerate(self._spins):
            for spin, value in ((a, box.lo[i] if box is not None else 0.0), (b, box.hi[i] if box is not None else 0.0)):
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)
        if box is None:
            self._summary.setText("No box yet.")
        else:
            size = " × ".join(f"{v:.1f}" for v in box.size_mm())
            self._summary.setText(f"{size} mm")

    # actions
    def _targets(self) -> list[Any]:
        layers = self._viewer.layers
        chosen = [l for l in layers if l in layers.selection]
        return chosen or ([layers.selection.active] if layers.selection.active is not None else [])

    def _require(self) -> WorldBox | None:
        box = current_box(self._viewer)
        if box is None:
            _notify("Set the ROI box first.", error=True)
        return box

    def _crop(self) -> None:
        box = self._require()
        if box is None:
            return
        made, skipped = crop_layers(self._viewer, self._targets(), box)
        _notify(f"Cropped {len(made)} layer(s)" + (f"; skipped {', '.join(skipped)}" if skipped else "") + ".")

    def _mask(self) -> None:
        box = self._require()
        if box is None:
            return
        made, skipped = mask_outside(self._viewer, self._targets(), box)
        _notify(f"Masked {len(made)} layer(s) outside the box" + (f"; skipped {', '.join(skipped)}" if skipped else "") + ".")

    def _to_labeling(self) -> None:
        box = self._require()
        layer = self._active()
        if box is None or layer is None:
            return
        if not box.to_roi_box(self._viewer, layer):
            _notify(f"The box does not overlap “{layer.name}”.", error=True)

    def _save(self) -> None:
        box = self._require()
        if box is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save ROI box", "roi_box.json", "JSON (*.json)")
        if path:
            box.save(path)

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load ROI box", "", "JSON (*.json)")
        if path:
            self._set(WorldBox.load(path))

    def _clear(self) -> None:
        self._set(None)


def roi_box_button(viewer: Any) -> Any:
    """Top-bar button opening the ROI box panel in a pop-over."""
    from nvitk.gui.core.widgets import popover_button

    return popover_button("roi_box", "ROI box: crop or mask the selected layers with a world-space box",
                          lambda: _themed(RoiBoxPanel(viewer)), title="ROI box", icon_size=15)


def _themed(widget: QWidget) -> QWidget:
    design.apply_theme(widget)
    return widget


__all__ = ["RoiBoxPanel", "roi_box_button"]
