"""Small options popup for the quick image operations.

An operation with something to choose gets a compact dialog built from its
:class:`~nvitk.gui.tools.quick_ops.OpParam` list, rather than running blind on a
default. Where a live preview is meaningful — thresholding above all — the dialog
maintains a temporary layer that follows the control, so the cut is chosen by
looking at it. Cancelling removes the preview and changes nothing.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import COLOR_MUTED, SPACE, SPACE_TIGHT, apply_theme
from nvitk.gui.tools import quick_ops

#: Steps a float slider resolves its range into.
_SLIDER_STEPS = 1000

#: Delay before a preview is recomputed, so dragging a slider does not rethreshold
#: the whole volume on every intermediate value.
_PREVIEW_MS = 60

#: Name of the temporary layer a previewing operation maintains.
PREVIEW_LAYER = "nvitk_quick_preview"

#: Which quick operations take options, and how to build them.
PARAM_BUILDERS: dict[str, Callable[[Any], tuple[quick_ops.OpParam, ...]]] = {
    "threshold_at_display": quick_ops.threshold_params,
    "gaussian": quick_ops.gaussian_params,
    "median": quick_ops.median_params,
    "project": quick_ops.projection_params,
    "rotate90": quick_ops.rotate_params,
    "convert_dtype": quick_ops.dtype_params,
    "crop_to_content": quick_ops.crop_params,
    "auto_contrast": quick_ops.contrast_params,
}

#: Operations whose result can be shown while it is being chosen.
PREVIEWABLE: frozenset[str] = frozenset(
    {"threshold_at_display", "auto_contrast", "gaussian", "median", "crop_to_content"}
)

#: Filtering a whole volume per slider step is only interactive up to a point.
#: Past this many voxels the filter previews are skipped and the dialog says so,
#: rather than making every drag stutter.
_FILTER_PREVIEW_VOXELS = 8_000_000


class _FloatSlider(QWidget):
    """A slider over a float range, with the value shown beside it."""

    def __init__(self, param: quick_ops.OpParam, parent: QWidget | None = None) -> None:
        """Build a slider spanning *param*'s range, starting at its default."""
        super().__init__(parent)
        self._lo, self._hi = float(param.minimum), float(param.maximum)
        if self._hi <= self._lo:
            self._hi = self._lo + 1.0
        self._decimals = int(param.decimals)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, _SLIDER_STEPS)
        self._readout = QLabel("")
        self._readout.setMinimumWidth(84)
        self._readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACE_TIGHT)
        row.addWidget(self._slider, stretch=1)
        row.addWidget(self._readout)

        self._slider.valueChanged.connect(lambda _v: self._sync_readout())
        self.set_value(float(param.default))

    def _sync_readout(self) -> None:
        """Show the slider's current value."""
        self._readout.setText(f"{self.value():.{self._decimals}g}")

    def value(self) -> float:
        """Current value in the parameter's own units."""
        frac = self._slider.value() / float(_SLIDER_STEPS)
        return self._lo + frac * (self._hi - self._lo)

    def set_value(self, value: float) -> None:
        """Move the slider to *value*, clamped into range."""
        frac = (float(value) - self._lo) / (self._hi - self._lo)
        self._slider.setValue(int(round(float(np.clip(frac, 0.0, 1.0)) * _SLIDER_STEPS)))
        self._sync_readout()

    def on_change(self, callback: Callable[[], None]) -> None:
        """Call *callback* whenever the slider moves."""
        self._slider.valueChanged.connect(lambda _v: callback())


class QuickOpDialog(QDialog):
    """Choose an operation's options, previewing the result where that makes sense."""

    def __init__(
        self,
        viewer: Any,
        op_name: str,
        params: tuple[quick_ops.OpParam, ...],
        title: str,
        parent: QWidget | None = None,
    ) -> None:
        """Build one control per parameter, plus a live preview when supported."""
        super().__init__(parent)
        self._viewer = viewer
        self._op_name = op_name
        self._widgets: dict[str, Any] = {}
        # Pin the layer now: the preview layer becomes the active one as soon as
        # it is added, and resolving the source later would preview the preview.
        self._source = quick_ops._active(viewer)

        self.setWindowTitle(title)
        self.setModal(True)

        form = QFormLayout()
        form.setHorizontalSpacing(SPACE)
        form.setVerticalSpacing(SPACE_TIGHT)
        hints: list[str] = []
        for param in params:
            widget = self._build_widget(param)
            self._widgets[param.name] = widget
            form.addRow(param.label, widget)
            if param.hint:
                hints.append(param.hint)

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE, SPACE, SPACE, SPACE)
        root.setSpacing(SPACE_TIGHT)
        root.addLayout(form)
        for hint in hints:
            label = QLabel(hint)
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 10px;")
            root.addWidget(label)

        if op_name in ("gaussian", "median") and self._source is not None:
            try:
                too_big = quick_ops.layer_data(self._source).size > _FILTER_PREVIEW_VOXELS
            except Exception:
                too_big = False
            if too_big:
                note = QLabel("Volume too large to preview live — OK applies it.")
                note.setWordWrap(True)
                note.setStyleSheet(f"color: {COLOR_MUTED}; font-size: 10px;")
                root.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        # Contrast is previewed on the source itself; remember the window so a
        # cancelled dialog leaves the layer exactly as it was found.
        self._original_limits = None
        if op_name == "auto_contrast" and self._source is not None:
            limits = getattr(self._source, "contrast_limits", None)
            self._original_limits = tuple(limits) if limits else None

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(_PREVIEW_MS)
        self._preview_timer.timeout.connect(self._update_preview)

        if op_name in PREVIEWABLE:
            for widget in self._widgets.values():
                _connect_change(widget, self._preview_timer.start)
            self._update_preview()

        apply_theme(self)
        self.setMinimumWidth(360)

    def _build_widget(self, param: quick_ops.OpParam) -> Any:
        """The control for one parameter."""
        if param.kind == "choice":
            combo = QComboBox()
            for label, value in param.choices:
                combo.addItem(str(label), value)
            index = combo.findData(param.default)
            combo.setCurrentIndex(max(index, 0))
            return combo
        if param.kind == "int":
            spin = QSpinBox()
            spin.setRange(int(param.minimum), int(param.maximum))
            spin.setValue(int(param.default))
            return spin
        # A float over a known range reads far better as a slider than a spin box.
        return _FloatSlider(param)

    def values(self) -> dict[str, Any]:
        """The chosen option values, keyed by parameter name."""
        out: dict[str, Any] = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, QComboBox):
                out[name] = widget.currentData()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                out[name] = widget.value()
            else:
                out[name] = widget.value()
        return out

    # ── preview ──────────────────────────────────────────────────────────────

    def _preview_result(self) -> tuple[np.ndarray, dict[str, Any], bool] | None:
        """``(data, spatial kwargs, is_labels)`` to preview, or ``None`` for no layer.

        Contrast is the odd one out: it changes how the source is *displayed*
        rather than producing anything, so it previews by adjusting the source and
        is handled separately.
        """
        from nvitk.gui.core.spatial import layer_spatial_kwargs

        values = self.values()
        source = self._source
        spatial = layer_spatial_kwargs(source) if source is not None else {}

        if self._op_name == "threshold_at_display":
            return quick_ops.threshold_mask(source, values["value"]), spatial, True

        data = quick_ops.layer_data(source)
        if self._op_name == "crop_to_content":
            lo, hi = quick_ops.crop_bounds(data, int(values["pad"]))
            out = data[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))]
            return out, quick_ops.cropped_spatial(source, lo), False

        if data.size > _FILTER_PREVIEW_VOXELS:
            return None
        if self._op_name == "gaussian":
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(
                data.astype(np.float32, copy=False), sigma=float(values["sigma"])
            ), spatial, False
        if self._op_name == "median":
            from scipy.ndimage import median_filter

            return median_filter(data, size=int(values["size"])), spatial, False
        return None

    def _preview_contrast(self) -> None:
        """Apply the chosen window to the source layer itself, live."""
        try:
            lo, hi = quick_ops.contrast_window(
                self._source, float(self.values()["low"]), float(self.values()["high"])
            )
        except Exception:
            return
        try:
            self._source.contrast_limits = (lo, hi)
        except Exception:
            pass

    def _update_preview(self) -> None:
        """Refresh the preview for the current values."""
        if self._op_name not in PREVIEWABLE or self._source is None:
            return
        if self._op_name == "auto_contrast":
            self._preview_contrast()
            return
        try:
            result = self._preview_result()
        except Exception:
            return
        if result is None:
            return
        data, spatial, labels = result

        existing = self._preview_layer()
        # Reuse the layer while the shape holds: replacing it on every step makes
        # the canvas rebuild a whole volume texture per slider move.
        if existing is not None and tuple(existing.data.shape) == tuple(data.shape):
            existing.data = data
            return
        self._clear_preview()
        try:
            # The preview must sit on the source's grid. Without its affine it is
            # drawn in raw voxel space and floats away from the image it came
            # from, only snapping into place once the real result is added.
            if labels:
                layer = self._viewer.add_labels(
                    data, name=PREVIEW_LAYER, opacity=0.5, **spatial
                )
                layer._nvitk_label_like = True
            else:
                self._viewer.add_image(data, name=PREVIEW_LAYER, opacity=0.9, **spatial)
        except Exception:
            pass

    def source_layer(self) -> Any:
        """The layer this dialog was opened for."""
        return self._source

    def _preview_layer(self) -> Any:
        """The temporary preview layer, if it is present."""
        for layer in getattr(self._viewer, "layers", []) or []:
            if str(getattr(layer, "name", "")) == PREVIEW_LAYER:
                return layer
        return None

    def _clear_preview(self) -> None:
        """Remove the preview layer, if any."""
        layer = self._preview_layer()
        if layer is not None:
            try:
                self._viewer.layers.remove(layer)
            except Exception:
                pass

    def done(self, result: int) -> None:
        """Drop the preview however the dialog closes — accepted or cancelled."""
        self._preview_timer.stop()
        self._clear_preview()
        # Restore a contrast preview on cancel; on accept the operation re-applies
        # it properly and reports what it did.
        if self._original_limits is not None and self._source is not None:
            try:
                self._source.contrast_limits = self._original_limits
            except Exception:
                pass
        super().done(result)


def _connect_change(widget: Any, callback: Callable[[], None]) -> None:
    """Call *callback* whenever *widget*'s value changes, whatever kind it is."""
    if isinstance(widget, QComboBox):
        widget.currentIndexChanged.connect(lambda _i: callback())
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.valueChanged.connect(lambda _v: callback())
    elif isinstance(widget, _FloatSlider):
        widget.on_change(callback)


def run_quick_op(
    viewer: Any,
    op_name: str,
    kwargs: dict[str, Any],
    *,
    title: str = "",
    parent: QWidget | None = None,
) -> str | None:
    """Run quick operation *op_name*, asking for its options first when it has any.

    Returns the operation's message, or ``None`` when the user cancelled. Options
    the caller already fixed (a palette entry for "Gaussian blur (sigma 2)") are
    passed through and no dialog is shown.
    """
    builder = PARAM_BUILDERS.get(op_name) if not kwargs else None
    if builder is None:
        return str(getattr(quick_ops, op_name)(viewer, **kwargs))

    params = builder(viewer)
    dialog = QuickOpDialog(viewer, op_name, params, title or op_name, parent=parent)
    if dialog.exec() != dialog.Accepted:
        return None
    # Put the selection back on the layer the dialog was opened for, in case the
    # preview moved it while the user was choosing.
    source = dialog.source_layer()
    if source is not None:
        try:
            viewer.layers.selection.active = source
        except Exception:
            pass
    return str(getattr(quick_ops, op_name)(viewer, **dialog.values()))


__all__ = [
    "PARAM_BUILDERS",
    "PREVIEWABLE",
    "PREVIEW_LAYER",
    "QuickOpDialog",
    "run_quick_op",
]
