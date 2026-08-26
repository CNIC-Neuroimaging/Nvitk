"""CT display-window picker for the selected image layer(s).

Applies a named Hounsfield-unit window (see :mod:`nvitk.viz.ct_windows`) to a napari Image
layer's contrast limits. Display only — voxels are never modified.

Why fixed windows rather than auto-contrast
-------------------------------------------
Napari's default is to stretch contrast to each layer's own min/max. For CT that is actively
unhelpful: Hounsfield units are physically calibrated, so a *fixed* window shows the same tissue
contrast on every scan, while auto-contrast makes two scans of the same anatomy look different
because one happened to include more bone in the field of view.

Only CT is offered a window. MR intensities are arbitrary units with no fixed zero, so an HU
range means nothing there; the panel detects that and says so rather than applying a window that
would merely look wrong.
"""

from __future__ import annotations

from typing import Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.viz.ct_windows import (
    DEFAULT_WINDOW_KEY,
    get_window,
    suggest_window,
    window_from_limits,
    window_keys,
)

log = Logger()

#: Combo entry for limits that match no registered window.
_CUSTOM = "__custom__"

#: HU bounds the spin boxes accept. Wide enough for metal artefact and padding values, which can
#: sit far outside the nominal 12-bit reconstruction range.
_HU_MIN, _HU_MAX = -10000.0, 10000.0


def is_image_layer(layer: Any) -> bool:
    """Whether *layer* is a napari Image layer that has contrast limits to set."""
    return (
        layer is not None
        and hasattr(layer, "contrast_limits")
        and layer.__class__.__name__ == "Image"
    )


class CTWindowPanel(QGroupBox):
    """Pick a CT display window and apply it to the selected image layer(s)."""

    #: Emitted after limits are applied, with ``(low, high)``.
    window_applied = Signal(float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the picker. Call :meth:`set_viewer` to bind it to a napari viewer."""
        super().__init__("CT display window", parent)
        self._viewer: Any | None = None
        self._layer: Any | None = None
        self._updating = False

        root = QVBoxLayout()
        root.setSpacing(4)

        # ---- preset ----------------------------------------------------------
        self._combo = QComboBox()
        for key in window_keys():
            window = get_window(key)
            self._combo.addItem(window.label, key)
            self._combo.setItemData(
                self._combo.count() - 1, window.description, Qt.ToolTipRole
            )
        self._combo.addItem("Custom", _CUSTOM)
        self._combo.currentIndexChanged.connect(self._on_preset_changed)
        root.addWidget(self._combo)

        # ---- level / width ---------------------------------------------------
        self._level = self._spin(" HU")
        self._width = self._spin(" HU")
        self._width.setMinimum(1.0)
        for spin in (self._level, self._width):
            spin.valueChanged.connect(self._on_level_width_changed)

        for label, spin in (("Level", self._level), ("Width", self._width)):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            row.addWidget(spin, stretch=1)
            root.addLayout(row)

        # ---- range readout ---------------------------------------------------
        self._range = QLabel("—")
        self._range.setStyleSheet("color: palette(mid);")
        root.addWidget(self._range)

        # ---- options ---------------------------------------------------------
        self._all_layers = QCheckBox("Apply to all image layers")
        self._all_layers.setToolTip(
            "Apply to every Image layer in the viewer, not only the selected one.\n"
            "Layers whose data is not in Hounsfield units are skipped."
        )
        root.addWidget(self._all_layers)

        buttons = QHBoxLayout()
        self._btn_apply = QPushButton("Apply")
        self._btn_apply.clicked.connect(self.apply_to_layers)
        self._btn_auto = QPushButton("Auto")
        self._btn_auto.setToolTip("Reset to napari's per-layer min/max contrast.")
        self._btn_auto.clicked.connect(self.reset_to_auto)
        buttons.addWidget(self._btn_apply)
        buttons.addWidget(self._btn_auto)
        root.addLayout(buttons)

        self._status = QLabel("No image layer selected.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: palette(mid);")
        root.addWidget(self._status)

        self.setLayout(root)
        self._select_key(DEFAULT_WINDOW_KEY)
        self._set_enabled(False)

    # ---- construction helpers ------------------------------------------------

    @staticmethod
    def _spin(suffix: str) -> QDoubleSpinBox:
        """A HU spin box with sensible bounds and step."""
        spin = QDoubleSpinBox()
        spin.setRange(_HU_MIN, _HU_MAX)
        spin.setDecimals(0)
        spin.setSingleStep(10.0)
        spin.setSuffix(suffix)
        return spin

    # ---- viewer binding ------------------------------------------------------

    def set_viewer(self, viewer: Any | None) -> None:
        """Bind to a napari viewer and follow its layer selection."""
        self._viewer = viewer
        if viewer is None:
            return
        try:
            viewer.layers.selection.events.active.connect(self._on_selection_changed)
        except Exception as exc:  # napari version differences; the panel still works manually
            log.debug("Could not subscribe to layer selection events: %s", exc)
        self._on_selection_changed()

    def _on_selection_changed(self, _event: Any = None) -> None:
        """Re-bind to whichever layer is active now."""
        layer = None
        if self._viewer is not None:
            layer = getattr(self._viewer.layers.selection, "active", None)
        self.set_layer(layer)

    def set_layer(self, layer: Any | None) -> None:
        """Bind to *layer*, enabling the controls only when a window can apply to it."""
        self._layer = layer if is_image_layer(layer) else None

        if self._layer is None:
            self._set_enabled(False)
            self._status.setText("Select an Image layer to set its display window.")
            return

        name = getattr(self._layer, "name", "layer")
        modality, minimum, maximum = self._layer_intensity_info(self._layer)
        suggestion = suggest_window(modality, minimum=minimum, maximum=maximum)

        if suggestion is None:
            # Not CT: a Hounsfield window would be meaningless. Say why rather than silently
            # offering a control that produces a nonsense result.
            self._set_enabled(False)
            reason = (
                f"modality {modality!r}" if modality
                else f"intensity range [{minimum:.0f}, {maximum:.0f}]" if minimum is not None
                else "unknown intensities"
            )
            self._status.setText(
                f"'{name}' does not look like CT ({reason}). "
                f"Hounsfield windows apply to CT only."
            )
            return

        self._set_enabled(True)
        # Show the window already in effect, if it is one we know.
        current = self._current_limits(self._layer)
        matched = window_from_limits(*current) if current else None
        self._select_key(matched.key if matched else suggestion)
        if matched is None and current is not None:
            self._set_level_width(
                (current[0] + current[1]) / 2.0, abs(current[1] - current[0])
            )
            self._select_key(_CUSTOM)
        self._status.setText(f"'{name}' — suggested: {get_window(suggestion).title}")

    # ---- state ---------------------------------------------------------------

    def _set_enabled(self, enabled: bool) -> None:
        """Enable or disable every control at once."""
        for widget in (self._combo, self._level, self._width, self._all_layers,
                       self._btn_apply, self._btn_auto):
            widget.setEnabled(enabled)

    def _select_key(self, key: str) -> None:
        """Select a preset without triggering an apply."""
        index = self._combo.findData(key)
        if index < 0:
            return
        self._updating = True
        try:
            self._combo.setCurrentIndex(index)
            if key != _CUSTOM:
                window = get_window(key)
                self._level.setValue(window.level)
                self._width.setValue(window.width)
        finally:
            self._updating = False
        self._refresh_range()

    def _set_level_width(self, level: float, width: float) -> None:
        """Set the spin boxes without triggering an apply."""
        self._updating = True
        try:
            self._level.setValue(float(level))
            self._width.setValue(max(1.0, float(width)))
        finally:
            self._updating = False
        self._refresh_range()

    def _refresh_range(self) -> None:
        """Update the ``[low, high] HU`` readout from the current level/width."""
        low, high = self.limits()
        self._range.setText(f"Range: [{low:.0f}, {high:.0f}] HU")

    def limits(self) -> tuple[float, float]:
        """The ``(low, high)`` HU bounds currently configured."""
        half = self._width.value() / 2.0
        return (self._level.value() - half, self._level.value() + half)

    # ---- signals -------------------------------------------------------------

    def _on_preset_changed(self, _index: int) -> None:
        """Preset chosen: load its level/width and apply."""
        if self._updating:
            return
        key = self._combo.currentData()
        if key and key != _CUSTOM:
            window = get_window(key)
            self._set_level_width(window.level, window.width)
        self.apply_to_layers()

    def _on_level_width_changed(self, _value: float) -> None:
        """Level or width edited by hand: switch to Custom unless it matches a preset."""
        if self._updating:
            return
        self._refresh_range()
        matched = window_from_limits(*self.limits())
        self._updating = True
        try:
            index = self._combo.findData(matched.key if matched else _CUSTOM)
            if index >= 0:
                self._combo.setCurrentIndex(index)
        finally:
            self._updating = False
        self.apply_to_layers()

    # ---- application ---------------------------------------------------------

    @staticmethod
    def _current_limits(layer: Any) -> tuple[float, float] | None:
        """A layer's contrast limits as a plain tuple, or ``None`` if unreadable."""
        try:
            low, high = layer.contrast_limits
            return (float(low), float(high))
        except Exception:
            return None

    @staticmethod
    def _layer_intensity_info(layer: Any) -> tuple[str | None, float | None, float | None]:
        """``(modality, min, max)`` for *layer*, from metadata and the data range.

        The range is read from ``contrast_limits_range``, which napari computes once on load —
        far cheaper than scanning a 70 M-voxel volume on every selection change.
        """
        from nvitk.gui.core.spatial import nvitk_metadata_from_layer

        modality = None
        try:
            meta = nvitk_metadata_from_layer(layer)
            raw = meta.get("modality") or meta.get("Modality")
            modality = str(raw).strip().lower() if raw else None
        except Exception:
            pass

        minimum = maximum = None
        try:
            low, high = layer.contrast_limits_range
            minimum, maximum = float(low), float(high)
        except Exception:
            pass
        return modality, minimum, maximum

    def _target_layers(self) -> list[Any]:
        """Layers the current settings should be applied to."""
        if self._all_layers.isChecked() and self._viewer is not None:
            return [layer for layer in self._viewer.layers if is_image_layer(layer)]
        return [self._layer] if self._layer is not None else []

    def apply_to_layers(self) -> None:
        """Apply the configured window to the target layers."""
        targets = self._target_layers()
        if not targets:
            return

        low, high = self.limits()
        applied, skipped = 0, 0
        for layer in targets:
            modality, minimum, maximum = self._layer_intensity_info(layer)
            if suggest_window(modality, minimum=minimum, maximum=maximum) is None:
                # Applying an HU range to arbitrary-unit MR would blank the layer.
                skipped += 1
                continue
            try:
                layer.contrast_limits = (low, high)
                applied += 1
            except Exception as exc:
                log.warning("Could not set contrast limits on %r: %s",
                            getattr(layer, "name", "?"), exc)
                skipped += 1

        message = f"Applied [{low:.0f}, {high:.0f}] HU to {applied} layer(s)"
        if skipped:
            message += f"; skipped {skipped} non-CT layer(s)"
        self._status.setText(message + ".")
        if applied:
            self.window_applied.emit(low, high)

    def reset_to_auto(self) -> None:
        """Restore napari's per-layer min/max contrast on the target layers."""
        targets = self._target_layers()
        reset = 0
        for layer in targets:
            try:
                layer.contrast_limits = list(layer.contrast_limits_range)
                reset += 1
            except Exception as exc:
                log.warning("Could not reset contrast on %r: %s",
                            getattr(layer, "name", "?"), exc)
        self._select_key(_CUSTOM)
        self._status.setText(f"Reset {reset} layer(s) to automatic contrast.")


__all__ = ["CTWindowPanel", "is_image_layer"]
