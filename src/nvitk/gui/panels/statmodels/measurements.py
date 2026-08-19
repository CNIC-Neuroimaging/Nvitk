"""
Measurement selection — one or several image measurements per analysis frame.

Description
-----------
A model like ``att_mean ~ pi + age_c`` needs 4D-flow and ASL measurements side by side in one frame.
Each measurement carries its *own* pipeline kind, pipeline, feature, atlas and grouping, and the
frames are joined on ``(subject_uid, territory)`` — so the groupings must be chosen to produce
commensurate keys (qvtpy ``hemisphere`` → ``MCA``/``ACA``/``PCA`` lines up with ASL ``territory``;
qvtpy ``vessel`` → ``LMCA``/``RMCA`` does not).

``MeasurementForm`` is the reusable selector: the window shows one inline for the primary
measurement, and :class:`MeasurementDialog` wraps another for adding/editing the rest.
Loading runs on :class:`FrameLoadWorker` because it issues one ``repo.image()`` query per
measurement.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Sequence

import pandas as pd
from qtpy.QtCore import QThread, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.stats import (
    KIND_MODALITY,
    MeasurementSpec,
    build_multi_feature_analysis_frame,
    features_for_kind,
    grouping_choices_for,
    resolve_feature_id,
)
from nvitk.stats._statmodels_frames import (
    COMPOSITE_DEFINITIONS,
    composites_for,
    is_timeseries_feature,
)
from nvitk.stats.frame_ops import IDENTIFIER_RE

from .constants import (
    ASL_ATLASES,
    JOIN_GRAINS,
    JOIN_MODES,
    PIPELINE_KIND_ASL,
    PIPELINE_KIND_FLAIR,
    PIPELINE_KIND_ITEMS,
    PIPELINE_KIND_QVTPY,
    PIPELINE_KIND_T1,
    PIPELINE_KIND_TOF,
)
from .theme import COLOR_WARN, muted_label_style

log = Logger()


# ──────────────────────────────────────────────────────────────────────────────
# Single-measurement selector
# ──────────────────────────────────────────────────────────────────────────────
class MeasurementForm(QWidget):
    """
    Pipeline-kind / pipeline / feature / atlas / grouping selector for one measurement.

    Every combo is repopulated from the one above it, so the offered features and groupings are
    always valid for the selected pipeline kind. The ASL atlas row only appears for ASL.
    """

    changed = Signal()

    def __init__(
        self,
        repo: Any,
        parent: QWidget | None = None,
        *,
        show_alias: bool = False,
    ) -> None:
        """Build the form against *repo*'s catalog; *show_alias* adds the output-column override."""
        super().__init__(parent)
        self._repo = repo

        lay = QFormLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self._pipeline_kind = QComboBox()
        for label, key in PIPELINE_KIND_ITEMS:
            self._pipeline_kind.addItem(label, key)

        self._pipeline = QComboBox()
        self._feature = QComboBox()
        self._feature.setEditable(True)
        self._grouping = QComboBox()
        self._atlas = QComboBox()
        for label, key in ASL_ATLASES:
            self._atlas.addItem(label, key)
        atlas_idx = self._atlas.findData("vascular-8")
        if atlas_idx >= 0:
            self._atlas.setCurrentIndex(atlas_idx)

        self._atlas_label = QLabel("ASL atlas / smoothing")
        self._alias = QLineEdit()
        self._alias.setPlaceholderText("(defaults to the feature name)")
        self._alias.setToolTip(
            "Column name this measurement takes in the analysis frame. Set it when two "
            "measurements would otherwise collide. Must be a valid identifier."
        )
        self._alias_label = QLabel("Column name")

        # Composite territory labels (e.g. TCBF) — populated per pipeline kind / feature.
        self._composites_host = QWidget()
        self._composites_layout = QVBoxLayout(self._composites_host)
        self._composites_layout.setContentsMargins(0, 0, 0, 0)
        self._composites_layout.setSpacing(2)
        self._composite_boxes: dict[str, QCheckBox] = {}
        self._composites_label = QLabel("Extra labels")

        # Optional region pre-filter: load only some of the measurement's published regions.
        self._regions_host = QWidget()
        regions_row = QHBoxLayout(self._regions_host)
        regions_row.setContentsMargins(0, 0, 0, 0)
        self._regions = QLineEdit()
        self._regions.setPlaceholderText("(all regions)")
        self._regions.setToolTip(
            "Comma-separated region ids to keep, before grouping. Blank loads every region.\n\n"
            "Use it to pull a single scalar out of a multi-region variable — 't1_volume_mm3' "
            "publishes 61 regions of which 'etiv' is one — or to restrict a flow measurement to "
            "the vessels you actually model. A composite label such as TCBF can be named here too: "
            "the filter runs after composites are built, so it keeps the composite and drops the "
            "vessels it was summed from."
        )
        self._regions.textChanged.connect(lambda *_: self.changed.emit())
        regions_row.addWidget(self._regions, stretch=1)
        self._btn_pick_regions = QPushButton("Pick…")
        self._btn_pick_regions.setToolTip("Choose from the regions this measurement publishes.")
        self._btn_pick_regions.clicked.connect(self._on_pick_regions)
        regions_row.addWidget(self._btn_pick_regions)
        self._regions_label = QLabel("Regions")

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet(muted_label_style())

        lay.addRow("Pipeline kind", self._pipeline_kind)
        lay.addRow("Measurement pipeline", self._pipeline)
        lay.addRow("Image feature", self._feature)
        lay.addRow(self._atlas_label, self._atlas)
        lay.addRow("Grouping", self._grouping)
        lay.addRow(self._composites_label, self._composites_host)
        lay.addRow(self._regions_label, self._regions_host)
        lay.addRow(self._alias_label, self._alias)
        lay.addRow("", self._hint)

        self._alias.setVisible(show_alias)
        self._alias_label.setVisible(show_alias)

        self._pipeline_kind.currentIndexChanged.connect(self._on_kind_changed)
        self._feature.currentIndexChanged.connect(self._on_feature_changed)
        self._feature.editTextChanged.connect(self._on_feature_changed)
        for widget in (self._pipeline, self._grouping, self._atlas):
            widget.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self._alias.textChanged.connect(lambda *_: self.changed.emit())

        self._on_kind_changed()

    # ---- combo population -----------------------------------------------------
    def _current_kind(self) -> str:
        """The selected pipeline kind."""
        return str(self._pipeline_kind.currentData() or PIPELINE_KIND_QVTPY)

    def _current_feature(self) -> str:
        """Entered/selected feature, falling back to the first available for the kind."""
        text = self._feature.currentText().strip()
        if text:
            return text
        feats = features_for_kind(self._current_kind())
        return feats[0] if feats else "flow_mean"

    def _populate_pipeline_combo(self) -> None:
        """Repopulate the pipeline combo with entries registered for the current modality."""
        modality = KIND_MODALITY.get(self._current_kind(), "4dflow")
        current = str(self._pipeline.currentData() or "latest")
        self._pipeline.blockSignals(True)
        self._pipeline.clear()
        self._pipeline.addItem("latest (catalog default)", "latest")
        try:
            entries = self._repo.catalog.list_pipelines(modality=modality)
        except Exception as exc:
            log.debug("Could not list %s pipelines: %s", modality, exc)
            entries = []
        for entry in entries:
            pid = str(entry.get("pipeline_id", "")).strip()
            if not pid:
                continue
            name = str(entry.get("pipeline_name") or pid)
            self._pipeline.addItem(f"{name} ({pid})", pid)
        idx = self._pipeline.findData(current)
        self._pipeline.setCurrentIndex(idx if idx >= 0 else 0)
        self._pipeline.blockSignals(False)

    def _populate_feature_combo(self) -> None:
        """Repopulate the feature combo for the current kind, preserving a still-valid choice."""
        feats = features_for_kind(self._current_kind())
        current = self._feature.currentText().strip()
        self._feature.blockSignals(True)
        self._feature.clear()
        for feat in feats:
            self._feature.addItem(feat)
        if current and current in feats:
            self._feature.setCurrentText(current)
        elif feats:
            self._feature.setCurrentIndex(0)
        self._feature.blockSignals(False)

    def _populate_grouping_combo(self) -> None:
        """Repopulate the grouping combo with choices valid for the current kind/feature."""
        choices = grouping_choices_for(self._current_kind(), self._current_feature())
        current = str(self._grouping.currentData() or "")
        self._grouping.blockSignals(True)
        self._grouping.clear()
        for label, key in choices:
            self._grouping.addItem(label, key)
        idx = self._grouping.findData(current)
        self._grouping.setCurrentIndex(idx if idx >= 0 else 0)
        self._grouping.blockSignals(False)

    def _populate_composites(self) -> None:
        """
        Rebuild the composite-label checkboxes for the current kind/feature, preserving ticks.

        Composites are aggregates over raw vessels (TCBF = RICA + LICA + BASILAR), so they only
        exist for the pipelines and features that publish those vessels — the row hides entirely
        when nothing applies.
        """
        checked = {name for name, box in self._composite_boxes.items() if box.isChecked()}
        while self._composites_layout.count():
            item = self._composites_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._composite_boxes = {}

        available = composites_for(self._current_kind(), self._current_feature())
        for name, label in available:
            box = QCheckBox(label)
            box.setToolTip(str(COMPOSITE_DEFINITIONS[name].get("description") or label))
            box.setChecked(name in checked)
            box.stateChanged.connect(lambda *_: self.changed.emit())
            self._composites_layout.addWidget(box)
            self._composite_boxes[name] = box

        self._composites_host.setVisible(bool(available))
        self._composites_label.setVisible(bool(available))

    def _sync_atlas_visibility(self) -> None:
        """Show the ASL-atlas picker only when the current pipeline kind is ASL."""
        is_asl = self._current_kind() == PIPELINE_KIND_ASL
        self._atlas.setEnabled(is_asl)
        self._atlas_label.setEnabled(is_asl)
        self._atlas.setVisible(is_asl)
        self._atlas_label.setVisible(is_asl)

    def _sync_hint(self) -> None:
        """Update the explanatory hint for the current pipeline kind/feature."""
        kind = self._current_kind()
        vid = resolve_feature_id(self._current_feature())
        if kind == PIPELINE_KIND_QVTPY:
            if is_timeseries_feature(vid):
                self._hint.setText(
                    "Time-resolved flow: loads one column per cardiac frame — flow_tseries, "
                    "flow_tseries_f1 … — rather than a single number, since averaging the "
                    "waveform back just reproduces flow_mean.\n"
                    "Right-click a subject_uid cell to play the cycle on the vascular map."
                )
            elif vid in {"pwv", "pwv_fielding_xcor", "pitc_slope", "pitc_intercept"}:
                self._hint.setText(
                    "Tree metrics: one value per arterial root (L_ICA / R_ICA / Basilar). "
                    "Hemisphere grouping averages L/R ICA and keeps Basilar."
                )
            else:
                self._hint.setText(
                    "Vessel-wise LOC metrics (flow_mean / pi / ri). "
                    "Hemisphere grouping averages left/right pairs (e.g. LMCA+RMCA → MCA)."
                )
        elif kind == PIPELINE_KIND_ASL:
            self._hint.setText(
                "Pick Desikan or one vascular-atlas smoothing (0 / 8 / 12). "
                "Only that atlas’s regions are loaded."
            )
        elif kind == PIPELINE_KIND_T1:
            self._hint.setText(
                "T1 cortical vs subcortical volume — regions come from the matching atlas."
            )
        elif kind == PIPELINE_KIND_FLAIR:
            self._hint.setText("FLAIR WMH metrics by published region_id.")
        elif kind == PIPELINE_KIND_TOF:
            self._hint.setText(
                "TOF morphometrics by eICAB vessel. "
                "Hemisphere grouping averages L/R pairs (e.g. LICA+RICA → ICA)."
            )
        else:
            self._hint.setText("")

    def _on_kind_changed(self) -> None:
        """Repopulate every dependent combo and sync visibility/hints for the new kind."""
        self._populate_pipeline_combo()
        self._populate_feature_combo()
        self._populate_grouping_combo()
        self._populate_composites()
        self._sync_atlas_visibility()
        self._sync_hint()
        self.changed.emit()

    def _on_feature_changed(self, *_args: Any) -> None:
        """Repopulate the grouping/composite options and refresh the hint for the new feature."""
        self._populate_grouping_combo()
        self._populate_composites()
        self._sync_hint()
        self.changed.emit()

    # ---- spec round-trip ------------------------------------------------------
    def spec(self) -> MeasurementSpec:
        """The measurement currently described by the form."""
        kind = self._current_kind()
        choices = grouping_choices_for(kind, self._current_feature())
        default_grouping = choices[0][1] if choices else "vessel"
        alias = self._alias.text().strip() or None
        return MeasurementSpec(
            pipeline_kind=kind,
            pipeline=str(self._pipeline.currentData() or "latest"),
            feature=self._current_feature(),
            grouping=str(self._grouping.currentData() or default_grouping),
            atlas=(str(self._atlas.currentData() or "vascular-8") if kind == PIPELINE_KIND_ASL else None),
            alias=alias,
            composites=tuple(
                name for name, box in self._composite_boxes.items() if box.isChecked()
            ),
            regions=tuple(
                token.strip() for token in self._regions.text().split(",") if token.strip()
            ),
        )

    def _on_pick_regions(self) -> None:
        """Offer the measurement's published regions as a checkable list."""
        from nvitk.stats._statmodels_frames import available_region_ids

        available = available_region_ids(
            self._repo,
            pipeline_kind=self._current_kind(),
            feature=self._current_feature(),
            pipeline=str(self._pipeline.currentData() or "latest"),
        )
        # Composites are not published rows — they are computed at load time — so they are offered
        # alongside whatever the table holds rather than looked up in it.
        composites = [name for name, box in self._composite_boxes.items() if box.isChecked()]
        available = [*composites, *available]
        if not available:
            self._hint.setText(
                f"No published regions found for {self._current_feature()!r} — type region ids "
                f"directly, or check the measurement is imported."
            )
            return

        current = {
            token.strip().lower() for token in self._regions.text().split(",") if token.strip()
        }
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Regions — {self._current_feature()}")
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"{len(available)} region(s). Nothing checked loads them all."))

        listing = QListWidget()
        listing.setSelectionMode(QListWidget.NoSelection)
        for region in available:
            item = QListWidgetItem(str(region))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if str(region).lower() in current else Qt.Unchecked
            )
            listing.addItem(item)
        layout.addWidget(listing, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.resize(360, 480)

        if dialog.exec() != QDialog.Accepted:
            return
        chosen = [
            listing.item(i).text()
            for i in range(listing.count())
            if listing.item(i).checkState() == Qt.Checked
        ]
        self._regions.setText(", ".join(chosen))

    def apply_spec(self, spec: MeasurementSpec) -> None:
        """Load *spec* into the form without emitting intermediate change signals."""
        self.blockSignals(True)
        try:
            idx = self._pipeline_kind.findData(spec.pipeline_kind)
            if idx >= 0:
                self._pipeline_kind.setCurrentIndex(idx)
            self._populate_pipeline_combo()
            self._populate_feature_combo()

            pidx = self._pipeline.findData(spec.pipeline)
            if pidx >= 0:
                self._pipeline.setCurrentIndex(pidx)
            self._feature.setCurrentText(spec.feature)
            self._populate_grouping_combo()
            gidx = self._grouping.findData(spec.grouping)
            if gidx >= 0:
                self._grouping.setCurrentIndex(gidx)
            self._populate_composites()
            wanted = {str(c) for c in spec.composites}
            for name, box in self._composite_boxes.items():
                box.setChecked(name in wanted)
            if spec.atlas:
                aidx = self._atlas.findData(spec.atlas)
                if aidx >= 0:
                    self._atlas.setCurrentIndex(aidx)
            self._alias.setText(spec.alias or "")
            self._regions.setText(", ".join(spec.regions))
            self._sync_atlas_visibility()
            self._sync_hint()
        finally:
            self.blockSignals(False)
        self.changed.emit()

    def validation_error(self) -> str:
        """Why the current form cannot be used, or ``""``."""
        alias = self._alias.text().strip()
        if alias and not IDENTIFIER_RE.match(alias):
            return (
                f"{alias!r} is not a valid column name — use letters, digits and underscores, "
                "starting with a letter or underscore."
            )
        return ""


class MeasurementDialog(QDialog):
    """Modal wrapper around a :class:`MeasurementForm`, used to add or edit a list entry."""

    def __init__(self, parent: QWidget | None, repo: Any, *, spec: MeasurementSpec | None = None) -> None:
        """Show the form, seeded with *spec* when editing."""
        super().__init__(parent)
        self.setWindowTitle("Edit measurement" if spec else "Add measurement")
        self.resize(480, 300)

        lay = QVBoxLayout(self)
        self._form = MeasurementForm(repo, self, show_alias=True)
        lay.addWidget(self._form)

        self._error = QLabel("")
        self._error.setWordWrap(True)
        self._error.setStyleSheet(f"color: {COLOR_WARN}; font-weight: normal;")
        lay.addWidget(self._error)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        if spec is not None:
            self._form.apply_spec(spec)

    def _on_accept(self) -> None:
        """Accept only when the form validates."""
        problem = self._form.validation_error()
        if problem:
            self._error.setText(problem)
            return
        self.accept()

    def spec(self) -> MeasurementSpec:
        """The configured measurement."""
        return self._form.spec()


# ──────────────────────────────────────────────────────────────────────────────
# Measurement list
# ──────────────────────────────────────────────────────────────────────────────
class MeasurementsWidget(QGroupBox):
    """
    The measurements that make up the analysis frame, plus the join mode and load diagnostics.

    Entry 0 is the *primary* measurement — it is the one the window's inline Data-selection form
    edits, the one whose column the default formulas and plots fall back to, and the one every other
    measurement's ``(subject_uid, territory)`` keys are compared against.
    """

    changed = Signal()

    def __init__(self, repo: Any, parent: QWidget | None = None) -> None:
        """Build the list, its buttons and the join-mode selector."""
        super().__init__("Measurements", parent)
        self._repo = repo
        self._specs: list[MeasurementSpec] = []

        lay = QVBoxLayout(self)

        self._list = QListWidget()
        self._list.setToolTip(
            "Measurements are joined on (subject_uid, territory). Pick groupings that produce the "
            "same keys — e.g. qvtpy 'hemisphere' against ASL 'territory'."
        )
        self._list.setMinimumHeight(60)
        self._list.itemDoubleClicked.connect(lambda *_: self._on_edit())
        lay.addWidget(self._list, stretch=1)

        row = QHBoxLayout()
        for label, slot in (
            ("Add…", self._on_add),
            ("Edit…", self._on_edit),
            ("Duplicate", self._on_duplicate),
            ("Remove", self._on_remove),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

        join_row = QHBoxLayout()
        join_row.addWidget(QLabel("Join"))
        self._join = QComboBox()
        for label, key in JOIN_MODES:
            self._join.addItem(label, key)
        self._join.currentIndexChanged.connect(lambda *_: self.changed.emit())
        join_row.addWidget(self._join, stretch=1)
        lay.addLayout(join_row)

        grain_row = QHBoxLayout()
        grain_row.addWidget(QLabel("Grain"))
        self._grain = QComboBox()
        for label, key in JOIN_GRAINS:
            self._grain.addItem(label, key)
        self._grain.setToolTip(
            "What a row is.\n\n"
            "territory — one row per subject × region, joined on a shared region key. The default, "
            "and the only grain in which a model can carry a territory term.\n\n"
            "subject — one row per subject, with each measurement's regions spread into "
            "'value@region' columns. Use this to relate measurements from different parcellations: "
            "an ASL whole-brain CBF against a 4D-flow vessel, or a FreeSurfer parcel against "
            "either. They name different kinds of region, so no shared key exists."
        )
        self._grain.currentIndexChanged.connect(lambda *_: self.changed.emit())
        grain_row.addWidget(self._grain, stretch=1)
        lay.addLayout(grain_row)

        self._attach_qc = QCheckBox("Load automatic QC columns")
        self._attach_qc.setChecked(False)
        self._attach_qc.setToolTip(
            "Merge the published per-vessel autoQC metrics (qc_flow_plausible, qc_conservation, "
            "qc_score, …) onto the frame when it loads.\n\n"
            "They are what the QC filter presets read, so the presets stay unavailable while this "
            "is off. Left off by default: they add five columns to every frame, including ones "
            "built from modalities the QC stage never covered."
        )
        self._attach_qc.stateChanged.connect(lambda *_: self.changed.emit())
        lay.addWidget(self._attach_qc)

        self._diagnostics = QLabel("")
        self._diagnostics.setWordWrap(True)
        self._diagnostics.setStyleSheet(muted_label_style())
        lay.addWidget(self._diagnostics)

    # ---- state ----------------------------------------------------------------
    def specs(self) -> list[MeasurementSpec]:
        """The configured measurements, primary first."""
        return list(self._specs)

    def set_specs(self, specs: Sequence[MeasurementSpec]) -> None:
        """Replace the measurement list (always keeping at least one entry)."""
        self._specs = list(specs) or [MeasurementSpec()]
        self._refresh()
        self.changed.emit()

    def set_primary(self, spec: MeasurementSpec) -> None:
        """Update entry 0 in place, from the window's inline Data-selection form."""
        if self._specs and self._specs[0] == spec:
            return
        if self._specs:
            self._specs[0] = spec
        else:
            self._specs = [spec]
        self._refresh()
        self.changed.emit()

    def join(self) -> str:
        """The selected join mode."""
        return str(self._join.currentData() or "inner")

    def set_join(self, mode: str) -> None:
        """Select the join mode named *mode*, if known."""
        idx = self._join.findData(mode)
        if idx >= 0:
            self._join.setCurrentIndex(idx)

    def attach_qc(self) -> bool:
        """Whether the automatic QC columns should be merged onto the frame."""
        return bool(self._attach_qc.isChecked())

    def set_attach_qc(self, enabled: bool) -> None:
        """Set the automatic-QC toggle without emitting an intermediate change signal."""
        self._attach_qc.blockSignals(True)
        self._attach_qc.setChecked(bool(enabled))
        self._attach_qc.blockSignals(False)

    def grain(self) -> str:
        """The selected join grain (``"territory"`` or ``"subject"``)."""
        return str(self._grain.currentData() or "territory")

    def set_grain(self, grain: str) -> None:
        """Select the join grain named *grain*, if known."""
        idx = self._grain.findData(grain)
        if idx >= 0:
            self._grain.setCurrentIndex(idx)

    def columns(self) -> list[str]:
        """Output column name of every measurement."""
        return [spec.column() for spec in self._specs]

    def _refresh(self) -> None:
        """Redraw the list, marking the primary entry."""
        current = self._list.currentRow()
        self._list.clear()
        for i, spec in enumerate(self._specs):
            prefix = "★ " if i == 0 else "   "
            item = QListWidgetItem(f"{prefix}{spec.label()}")
            item.setToolTip(
                "Primary measurement: edited by the Data selection box above, and used as the "
                "reference for key overlap." if i == 0 else spec.label()
            )
            self._list.addItem(item)
        if 0 <= current < self._list.count():
            self._list.setCurrentRow(current)

    def set_diagnostics(self, meta: dict[str, Any] | None) -> None:
        """Report per-measurement counts, key overlap, and any warnings from the last load."""
        if not meta:
            self._diagnostics.setText("")
            return
        parts = [
            f"{entry['column']}: {entry['n_rows']} rows / {entry['n_subjects']} subjects"
            for entry in meta.get("measurements", [])
        ]
        overlap = meta.get("key_overlap")
        if len(meta.get("measurements", [])) > 1 and overlap is not None:
            parts.append(f"key overlap {overlap:.0%}")
        text = "  ·  ".join(parts)
        warnings = list(meta.get("warnings") or [])
        if warnings:
            self._diagnostics.setStyleSheet(f"color: {COLOR_WARN}; font-weight: normal;")
            text = f"{text}\n⚠ " + "\n⚠ ".join(warnings)
        else:
            self._diagnostics.setStyleSheet(muted_label_style())
        self._diagnostics.setText(text)

    # ---- list actions ---------------------------------------------------------
    def _on_add(self) -> None:
        """Append a measurement configured in a modal dialog."""
        dialog = MeasurementDialog(self, self._repo)
        if dialog.exec():
            self._specs.append(dialog.spec())
            self._refresh()
            self._list.setCurrentRow(len(self._specs) - 1)
            self.changed.emit()

    def _on_edit(self) -> None:
        """Edit the selected measurement."""
        row = self._list.currentRow()
        if not (0 <= row < len(self._specs)):
            return
        dialog = MeasurementDialog(self, self._repo, spec=self._specs[row])
        if dialog.exec():
            self._specs[row] = dialog.spec()
            self._refresh()
            self.changed.emit()

    def _on_duplicate(self) -> None:
        """Copy the selected measurement, giving the copy a distinct alias."""
        row = self._list.currentRow()
        if not (0 <= row < len(self._specs)):
            return
        source = self._specs[row]
        taken = set(self.columns())
        base = source.column()
        alias = next(f"{base}_{i}" for i in range(2, 100) if f"{base}_{i}" not in taken)
        self._specs.insert(row + 1, MeasurementSpec.from_dict({**source.to_dict(), "alias": alias}))
        self._refresh()
        self._list.setCurrentRow(row + 1)
        self.changed.emit()

    def _on_remove(self) -> None:
        """Drop the selected measurement (never the last remaining one)."""
        row = self._list.currentRow()
        if len(self._specs) <= 1:
            self._diagnostics.setText("At least one measurement is required.")
            return
        if 0 <= row < len(self._specs):
            del self._specs[row]
            self._refresh()
            self.changed.emit()


# ──────────────────────────────────────────────────────────────────────────────
# Background loading
# ──────────────────────────────────────────────────────────────────────────────
class FrameLoadWorker(QThread):
    """
    Background thread that builds the analysis frame, so the GUI stays responsive.

    One ``repo.image()`` query runs per measurement plus the covariate queries, which is seconds on
    a large dataset. Emits ``finished_ok((frame, meta))`` or ``failed(message)``.
    """

    finished_ok = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        repo: Any,
        *,
        measurements: Sequence[MeasurementSpec],
        clinical_vars: list[str],
        cognitive_vars: list[str],
        join: str,
        grain: str = "territory",
        attach_qc: bool = True,
    ) -> None:
        """Store the query parameters; the repo is only touched from :meth:`run`."""
        super().__init__()
        self._repo = repo
        self._measurements = list(measurements)
        self._clinical_vars = list(clinical_vars)
        self._cognitive_vars = list(cognitive_vars)
        self._join = join
        self._grain = grain
        self._attach_qc = attach_qc

    def run(self) -> None:
        """Build the frame and emit it, or report the failure."""
        try:
            self.progress.emit(f"Loading {len(self._measurements)} measurement(s)…")
            frame, meta = build_multi_feature_analysis_frame(
                self._repo,
                measurements=self._measurements,
                clinical_vars=self._clinical_vars,
                cognitive_vars=self._cognitive_vars,
                join=self._join,
                grain=self._grain,
                attach_qc=self._attach_qc,
            )
        except Exception as exc:
            log.exception("Analysis frame load failed.")
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit((frame, meta))


__all__ = [
    "FrameLoadWorker",
    "MeasurementDialog",
    "MeasurementForm",
    "MeasurementsWidget",
]
