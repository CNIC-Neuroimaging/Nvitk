"""Checkbox list of labels with optional pipeline / model name mapping."""

from __future__ import annotations

from typing import Any

import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import COLOR_BORDER_STRONG, COLOR_FAINT, clear_layout
from nvitk.gui.labels.catalog import (
    all_schemas,
    get_schema,
    guess_schema_from_layer,
    remember_layer_schema,
    schema_keys,
)
from nvitk.gui.labels.visibility import (
    LABEL_COLORMAPS,
    apply_label_colormap,
    ensure_labels_layer,
    get_label_color,
    is_label_like_layer,
    label_source_data,
    layer_in_viewer,
    restore_label_visibility,
    apply_label_visibility,
    set_label_color,
    stored_label_colormap,
    stored_visible_ids,
    supports_per_label_color,
    layer_label_ids,
)

LABEL_SELECTOR_SCROLL_MIN = 80


class _LabelFilterHub(QObject):
    """Announces that a layer's live label filter changed, so every picker bound to
    that layer — the Labels panel, the Tools picker, a layer-list popup — shows it."""

    changed = Signal(object)


_HUB: _LabelFilterHub | None = None


def label_filter_hub() -> _LabelFilterHub:
    """The process-wide :class:`_LabelFilterHub`."""
    global _HUB
    if _HUB is None:
        _HUB = _LabelFilterHub()
    return _HUB


def apply_selection_to_layer(selector: "LabelSelectorWidget", viewer: Any) -> None:
    """Filter *selector*'s bound layer to its checked ids and tell the other pickers.

    The filter is recorded on the layer itself, so a layer keeps the selection it
    was given while other layers are active; checking every id present is what
    clears it again.
    """
    layer = selector.current_layer()
    if layer is None or not is_label_like_layer(layer):
        return
    if not layer_in_viewer(layer, viewer):
        return
    set_layer_visible_ids(layer, selector.selected_ids(), viewer, present=selector.available_ids())


def set_layer_visible_ids(
    layer: Any, ids: list[int], viewer: Any, *, present: list[int] | None = None
) -> None:
    """Show only *ids* of *layer* — every id present clears the filter — and tell
    every picker bound to it."""
    if present is None:
        present = layer_label_ids(layer)
    ids = sorted(int(i) for i in ids)
    if ids and present and set(ids) >= set(present):
        restore_label_visibility(layer, viewer=viewer)
    else:
        apply_label_visibility(layer, ids)
    label_filter_hub().changed.emit(layer)


def _rgba_to_qcolor(rgba: np.ndarray) -> QColor:
    """Convert a 3- or 4-channel float RGBA array in ``[0, 1]`` to a Qt ``QColor``."""
    arr = np.asarray(rgba, dtype=float).reshape(-1)
    r = int(np.clip(arr[0], 0.0, 1.0) * 255)
    g = int(np.clip(arr[1], 0.0, 1.0) * 255)
    b = int(np.clip(arr[2], 0.0, 1.0) * 255)
    a = int(np.clip(arr[3] if arr.size > 3 else 1.0, 0.0, 1.0) * 255)
    return QColor(r, g, b, a)


def _qcolor_to_rgba(color: QColor) -> np.ndarray:
    """Convert a Qt ``QColor`` to a 4-channel float32 RGBA array in ``[0, 1]``."""
    return np.array(
        [
            color.red() / 255.0,
            color.green() / 255.0,
            color.blue() / 255.0,
            color.alpha() / 255.0,
        ],
        dtype=np.float32,
    )


def _swatch_stylesheet(rgba: np.ndarray) -> str:
    """Qt stylesheet giving a ``QToolButton`` a solid background swatch of color *rgba*."""
    c = _rgba_to_qcolor(rgba)
    # Circular, so a colour swatch is never mistaken for the square checkbox
    # indicator sitting right beside it.
    return (
        f"QToolButton {{ background-color: rgba({c.red()},{c.green()},{c.blue()},{c.alpha()}); "
        f"border: 1px solid {COLOR_BORDER_STRONG}; border-radius: 9px; }}"
    )


class LabelSelectorWidget(QGroupBox):
    """Select label ids using an optional named vocabulary (eICAB, QVTpy, TS, …)."""

    selection_changed = Signal()

    def __init__(self, parent: QWidget | None = None, *, compact: bool = False) -> None:
        """Build the schema picker, filter, All/None/Refresh buttons, and scrollable checkbox list.

        *compact* drops the colormap, full-schema and Refresh controls, for the
        small per-layer popup opened from the layer list.
        """
        super().__init__("" if compact else "Label selection", parent)
        self._compact = compact
        self._checks: list[QCheckBox] = []
        self._color_buttons: dict[int, QToolButton] = {}
        self._layer_ids: list[int] = []
        self._base_hint = ""
        self._schema_key = "generic"
        self._hint = QLabel("Choose a label mapping, then select labels below.")
        self._hint.setWordWrap(True)

        schema_row = QHBoxLayout()
        schema_row.addWidget(QLabel("Mapping:"))
        self._schema_combo = QComboBox()
        self._schema_combo.setMinimumWidth(180)
        for key in schema_keys():
            sch = all_schemas()[key]
            self._schema_combo.addItem(sch.title, key)
        schema_row.addWidget(self._schema_combo, stretch=1)
        self._btn_guess = QPushButton("Guess")
        self._btn_guess.setToolTip("Guess mapping from layer filename / metadata")
        schema_row.addWidget(self._btn_guess)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colours:"))
        self._cmap_combo = QComboBox()
        self._cmap_combo.setToolTip(
            "Recolour every label at once. Qualitative maps (tab10, Set1) keep "
            "neighbouring ids distinct; sequential maps (viridis, turbo) ramp across "
            "them, which suits labels that have an order."
        )
        for label, key in LABEL_COLORMAPS:
            self._cmap_combo.addItem(label, key)
        cmap_row.addWidget(self._cmap_combo, stretch=1)

        self._show_full = QCheckBox("Show full schema")
        self._show_full.setToolTip(
            "List every id in the mapping, not only ids present in the active layer"
        )

        # Long vocabularies (TotalSegmentator lists over a hundred structures) are
        # unusable as a bare checkbox column: type part of a name or an id instead.
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter by name or id…")
        self._filter.setClearButtonEnabled(True)
        self._filter.setToolTip(
            "Show only matching labels. All / None then act on the matches only."
        )
        self._filter.textChanged.connect(lambda _text: self._apply_filter())

        btn_row = QHBoxLayout()
        self._btn_all = QPushButton("All")
        self._btn_all.setToolTip("Show every label listed (only the filtered ones while filtering).")
        self._btn_none = QPushButton("None")
        self._btn_none.setToolTip("Hide every label listed (only the filtered ones while filtering).")
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setToolTip("Re-read the labels present in the layer.")
        btn_row.addWidget(self._btn_all)
        btn_row.addWidget(self._btn_none)
        btn_row.addWidget(self._btn_refresh)
        btn_row.addStretch(1)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setMinimumHeight(80)
        self._inner = QWidget()
        self._inner_layout = QVBoxLayout()
        self._inner_layout.setAlignment(Qt.AlignTop)
        self._inner.setLayout(self._inner_layout)
        self._scroll.setWidget(self._inner)

        root = QVBoxLayout()
        root.addWidget(self._hint)
        root.addLayout(schema_row)
        root.addLayout(cmap_row)
        root.addWidget(self._show_full)
        root.addWidget(self._filter)
        root.addLayout(btn_row)
        root.addWidget(self._scroll, stretch=1)
        self.setLayout(root)
        if compact:
            root.setContentsMargins(4, 4, 4, 4)
            self._btn_guess.setVisible(False)
            self._show_full.setVisible(False)
            self._btn_refresh.setVisible(False)
            for i in range(cmap_row.count()):
                item = cmap_row.itemAt(i).widget()
                if item is not None:
                    item.setVisible(False)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        self._btn_all.clicked.connect(self.select_all)
        self._btn_none.clicked.connect(self.select_none)
        self._schema_combo.currentIndexChanged.connect(self._on_schema_changed)
        # Only a choice made by hand is remembered for the layer; a guess is
        # re-derived, and a programmatic switch must not stick to the wrong layer.
        self._schema_combo.activated.connect(
            lambda _i: remember_layer_schema(self._layer_ref, self._schema_combo.currentData())
        )
        self._show_full.toggled.connect(lambda _: self._refresh_current_layer())
        self._btn_guess.clicked.connect(self._guess_schema)
        self._cmap_combo.currentIndexChanged.connect(self._on_colormap_changed)

        self._layer_ref: Any | None = None
        self._viewer: Any | None = None
        label_filter_hub().changed.connect(self._on_layer_filter_changed)

    def _on_layer_filter_changed(self, layer: Any) -> None:
        """Another picker filtered *layer*: show its selection here too."""
        try:
            if layer is not None and layer is self._layer_ref:
                self.sync_checks_from_layer()
        except RuntimeError:  # this widget's C++ side is already gone
            pass

    def sync_checks_from_layer(self) -> None:
        """Tick exactly the ids the bound layer's live filter keeps, without rebuilding."""
        layer = self._layer_ref
        if layer is None:
            return
        remembered = stored_visible_ids(layer)
        present = set(self._layer_ids)
        for cb in self._checks:
            lid = int(cb.property("label_id"))
            want = (lid in present) if remembered is None else (lid in remembered)
            if cb.isChecked() != want:
                cb.blockSignals(True)
                cb.setChecked(want)
                cb.blockSignals(False)

    def set_viewer(self, viewer: Any | None) -> None:
        """Napari viewer used to promote Image masks to Labels for color editing."""
        self._viewer = viewer

    def _emit_selection_changed(self) -> None:
        """Emit the ``selection_changed`` Qt signal."""
        self.selection_changed.emit()

    def _wire_checkbox(self, cb: QCheckBox) -> None:
        """Connect *cb*'s toggle event to emit ``selection_changed``."""
        cb.toggled.connect(lambda _checked: self._emit_selection_changed())

    def _on_schema_changed(self, _index: int) -> None:
        """Update the active schema key and re-render the checkbox list for it."""
        key = self._schema_combo.currentData()
        if key:
            self._schema_key = str(key)
        self._refresh_current_layer()

    def _guess_schema(self) -> None:
        """Attempt to auto-detect and select the label schema from the current layer's name/path."""
        if self._layer_ref is None:
            self._hint.setText("No layer to guess from.")
            return
        guessed = guess_schema_from_layer(self._layer_ref)
        if not guessed:
            self._hint.setText("Could not guess mapping from layer name/path.")
            return
        remember_layer_schema(self._layer_ref, guessed)
        idx = self._schema_combo.findData(guessed)
        if idx >= 0:
            self._schema_combo.setCurrentIndex(idx)

    def _refresh_current_layer(self) -> None:
        """Re-render the checkbox list for whichever layer is currently bound."""
        self.refresh_from_layer(self._layer_ref)

    def set_schema_key(self, key: str, *, refresh: bool = True) -> None:
        """Select a catalog schema by key (e.g. ``ts:total``, ``eicab``).

        With ``refresh=False`` only the choice changes — for a caller about to
        bind a new layer, which would otherwise rebuild the list for the old one
        first.
        """
        idx = self._schema_combo.findData(key)
        if not refresh:
            self._schema_combo.blockSignals(True)
            if idx >= 0:
                self._schema_combo.setCurrentIndex(idx)
            self._schema_combo.blockSignals(False)
            self._schema_key = key
            return
        if idx >= 0:
            self._schema_combo.setCurrentIndex(idx)
        else:
            self._schema_key = key
            self._refresh_current_layer()

    def schema_key(self) -> str:
        """Currently selected label schema key."""
        return self._schema_key

    def current_layer(self) -> Any | None:
        """Layer currently bound to the selector (may be Labels after Image promote)."""
        return self._layer_ref

    def set_expanded(self, expanded: bool) -> None:
        """Fill remaining dock height when visible; collapse when hidden."""
        if expanded:
            expanding = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
            self.setSizePolicy(expanding)
            self._scroll.setSizePolicy(expanding)
            self._scroll.setMinimumHeight(LABEL_SELECTOR_SCROLL_MIN)
            self._scroll.setMaximumHeight(16777215)
        else:
            compact = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self.setSizePolicy(compact)
            self._scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
            self._scroll.setMinimumHeight(0)
            self._scroll.setMaximumHeight(200)

    def _supports_color_edit(self, layer: Any) -> bool:
        """True if per-label colors on *layer* can be edited (a Labels layer, or a label-like Image
        that can be promoted to one given a bound viewer)."""
        if layer is None:
            return False
        if supports_per_label_color(layer):
            return True
        # Discrete Image masks (e.g. QC segs) can be promoted to Labels.
        return bool(self._viewer is not None and is_label_like_layer(layer))

    def _ensure_colorable_layer(self, layer: Any) -> Any:
        """Return *layer* if it already supports per-label colors, else promote it to a Labels layer
        (updating the bound layer reference); raises ``TypeError`` if no viewer is bound."""
        if supports_per_label_color(layer):
            return layer
        if self._viewer is None:
            raise TypeError("No viewer available to convert the mask to a Labels layer.")
        new_layer = ensure_labels_layer(self._viewer, layer)
        self._layer_ref = new_layer
        return new_layer

    def _make_color_button(self, layer: Any, lid: int) -> QToolButton:
        """Build a small clickable color swatch for label *lid*, opening the color editor on click."""
        btn = QToolButton()
        btn.setFixedSize(18, 18)
        btn.setToolTip(f"Change the display colour of label {lid}")
        btn.setProperty("label_id", lid)
        rgba = get_label_color(layer, lid)
        btn.setStyleSheet(_swatch_stylesheet(rgba))
        btn.clicked.connect(lambda _checked=False, label_id=lid: self._edit_label_color(label_id))
        self._color_buttons[lid] = btn
        return btn

    def _edit_label_color(self, label_id: int) -> None:
        """Open a Qt color dialog for *label_id* and apply the chosen color to the layer and swatch."""
        layer = self._layer_ref
        if layer is None or not self._supports_color_edit(layer):
            return
        try:
            layer = self._ensure_colorable_layer(layer)
        except Exception:
            return
        current = _rgba_to_qcolor(get_label_color(layer, label_id))
        try:
            options = QColorDialog.ColorDialogOption.ShowAlphaChannel
        except AttributeError:
            options = QColorDialog.ShowAlphaChannel  # type: ignore[attr-defined]
        chosen = QColorDialog.getColor(current, self, f"Label {label_id} color", options)
        if not chosen.isValid():
            return
        rgba = _qcolor_to_rgba(chosen)
        set_label_color(layer, label_id, rgba, selected_ids=self.selected_ids())
        btn = self._color_buttons.get(int(label_id))
        if btn is not None:
            btn.setStyleSheet(_swatch_stylesheet(rgba))

    def _on_colormap_changed(self, _index: int) -> None:
        """Recolour the bound layer's whole label set from the chosen colormap."""
        layer = self._layer_ref
        if layer is None or not self._supports_color_edit(layer):
            return
        try:
            layer = self._ensure_colorable_layer(layer)
        except Exception as exc:  # noqa: BLE001
            self._hint.setText(f"Could not recolour: {exc}")
            return
        colormap = str(self._cmap_combo.currentData() or "")
        try:
            count = apply_label_colormap(layer, colormap, selected_ids=self.selected_ids())
        except Exception as exc:  # noqa: BLE001
            self._hint.setText(f"Could not apply that colormap: {exc}")
            return
        self._sync_color_buttons(layer)
        name = self._cmap_combo.currentText()
        self._hint.setText(
            f"Recoloured {count} label(s) from Napari's default palette."
            if not colormap
            else f"Recoloured {count} label(s) — {name}."
        )

    def _sync_color_buttons(self, layer: Any) -> None:
        """Repaint each swatch from the layer's current colours."""
        for lid, btn in self._color_buttons.items():
            btn.setStyleSheet(_swatch_stylesheet(get_label_color(layer, int(lid))))

    def colormap_key(self) -> str:
        """Currently selected label colormap key (``""`` for Napari's own colours)."""
        return str(self._cmap_combo.currentData() or "")

    def set_colormap_key(self, key: str) -> None:
        """Select a label colormap by key without re-applying it."""
        idx = self._cmap_combo.findData(str(key))
        if idx < 0:
            return
        self._cmap_combo.blockSignals(True)
        self._cmap_combo.setCurrentIndex(idx)
        self._cmap_combo.blockSignals(False)

    def refresh_from_layer(self, layer: Any | None) -> None:
        """Rebuild the checkbox list (and color swatches, if supported) for *layer*'s label ids under
        the current schema, promoting a discrete Image mask to Labels first when a viewer is bound."""
        # Promote discrete Image masks once so swatches / color visibility work.
        if (
            layer is not None
            and self._viewer is not None
            and type(layer).__name__ != "Labels"
            and is_label_like_layer(layer)
        ):
            try:
                layer = ensure_labels_layer(self._viewer, layer)
            except Exception:
                pass

        self._layer_ref = layer
        clear_layout(self._inner_layout)
        self._checks.clear()
        self._color_buttons.clear()
        self._layer_ids = []

        if layer is None:
            self._hint.setText("No layer selected.")
            return

        schema = get_schema(self._schema_key)
        layer_ids = layer_label_ids(layer)
        self._layer_ids = list(layer_ids)
        if self._show_full.isChecked() and schema and schema.id_to_name:
            ids = sorted(set(schema.id_to_name.keys()) | set(layer_ids))
        else:
            ids = layer_ids

        if not ids:
            self._hint.setText(f"No labels to show in “{layer.name}”.")
            return

        mapped = sum(1 for lid in ids if schema and schema.name_for(lid))
        schema_title = schema.title if schema else "Generic"
        color_hint = (
            " — click a colour dot to change it"
            if self._supports_color_edit(layer)
            else ""
        )
        self._base_hint = (
            f"{len(ids)} label(s) in “{layer.name}” — {schema_title}"
            + (f" ({mapped} named)" if schema and schema.id_to_name else "")
            + color_hint
        )
        self._hint.setText(self._base_hint)

        # The combo describes *this* layer, so a layer using Napari's own colours
        # must not read as though a colormap were applied to it.
        self.set_colormap_key(stored_label_colormap(layer))

        can_color = self._supports_color_edit(layer)
        # Each layer keeps its own selection: reuse whatever the live filter is
        # already showing for it, so switching layers does not reset the picker.
        remembered = stored_visible_ids(layer)
        for lid in ids:
            text = schema.display(lid) if schema else f"Label {lid}"
            in_layer = lid in layer_ids
            checked = in_layer if remembered is None else lid in remembered
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            if can_color and in_layer:
                row_layout.addWidget(self._make_color_button(layer, lid))
            elif can_color:
                spacer = QWidget()
                spacer.setFixedSize(18, 18)
                row_layout.addWidget(spacer)
            cb = QCheckBox(text)
            cb.setProperty("label_id", lid)
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
            if self._show_full.isChecked() and not in_layer:
                cb.setEnabled(False)
                cb.setStyleSheet(f"color: {COLOR_FAINT};")
            self._wire_checkbox(cb)
            row_layout.addWidget(cb, stretch=1)
            self._inner_layout.addWidget(row)
            self._checks.append(cb)
        self._apply_filter()
        # No selection_changed here: the boxes now show the layer's own filter, so
        # there is nothing to apply. Emitting made every picker re-apply its state
        # a beat after a rebind — over the top of an edit made meanwhile in another.

    def filter_text(self) -> str:
        """The current filter text."""
        return self._filter.text()

    def set_filter_text(self, text: str) -> None:
        """Filter the list to labels whose name or id contains *text*."""
        self._filter.setText(text)

    def _apply_filter(self) -> None:
        """Hide the rows whose label text and id do not contain the filter text."""
        needle = self._filter.text().strip().lower()
        shown = 0
        for cb in self._checks:
            row = cb.parentWidget()
            hit = (
                not needle
                or needle in cb.text().lower()
                or needle == str(cb.property("label_id"))
            )
            if row is not None:
                row.setVisible(hit)
            shown += int(hit)
        if needle and self._layer_ref is not None:
            self._hint.setText(f"{shown} of {len(self._checks)} label(s) match “{needle}”.")
        elif self._layer_ref is not None and self._checks:
            self._hint.setText(self._base_hint)

    def _row_shown(self, cb: QCheckBox) -> bool:
        """True when *cb*'s row passes the filter."""
        row = cb.parentWidget()
        return row is None or not row.isHidden()

    def select_all(self) -> None:
        """Check every enabled label checkbox the filter shows."""
        changed = False
        for cb in self._checks:
            if cb.isEnabled() and not cb.isChecked() and self._row_shown(cb):
                cb.blockSignals(True)
                cb.setChecked(True)
                cb.blockSignals(False)
                changed = True
        if changed:
            self._emit_selection_changed()

    def select_none(self) -> None:
        """Uncheck every enabled label checkbox the filter shows."""
        changed = False
        for cb in self._checks:
            if cb.isEnabled() and cb.isChecked() and self._row_shown(cb):
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)
                changed = True
        if changed:
            self._emit_selection_changed()

    def selected_ids(self) -> list[int]:
        """Sorted label ids whose checkbox is both checked and enabled."""
        out = []
        for cb in self._checks:
            if cb.isChecked() and cb.isEnabled():
                out.append(int(cb.property("label_id")))
        return sorted(out)

    def available_ids(self) -> list[int]:
        """Label ids actually present in the bound layer, as of the last refresh."""
        return list(self._layer_ids)

    def selected_names(self) -> list[str]:
        """Human names for checked ids (falls back to ``Label_<id>``)."""
        schema = get_schema(self._schema_key)
        names = []
        for lid in self.selected_ids():
            if schema and schema.name_for(lid):
                names.append(schema.name_for(lid))
            else:
                names.append(f"Label_{lid}")
        return names
