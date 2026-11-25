"""PESA-Fat QC in the GUI: load a subject's CT-PET or Dixon results and review them.

The same review the web portal records (``pesa_fat_qc_reviews``: one status per structure and
aspect, SEGMENTATION and MEASUREMENT), done beside the images:

* pick the pipeline (CT-PET v5 / Dixon v5), the cohort (batch) and a subject;
* *Load* opens the subject's input volumes and post-processing masks into a layer folder;
* the review table lists every (structure, aspect) the pipeline requires, with the subject's
  measurements for that structure, tinted by the expected ranges (amber: outside the usual
  range, red: implausible), and the status already recorded;
* *Save reviews* writes the statuses and comments through
  :func:`nvitk.pipes.pesa_fat.common.db_publish.publish_qc_review`.
"""

from __future__ import annotations

import getpass
from pathlib import Path
from typing import Any


from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QListWidget,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core import design
from nvitk.gui.core.widgets import Section, hint_label

#: pipeline key (as the portal names it) -> (label, data preset, stage-2 dir, stage-3 module)
PIPELINES: dict[str, tuple[str, str]] = {
    "ct-pet-v5": ("CT-PET v5", "pesa_fat_ctpet"),
    "dixon-v5": ("Dixon v5", "pesa_fat_dixon"),
}
STATUSES = ("PENDING", "OK", "FAIL")


def _stage_dirs(pipeline: str) -> tuple[str, str]:
    if pipeline == "ct-pet-v5":
        from nvitk.pipes.pesa_fat.ct_pet_v5 import config as cfg
    else:
        from nvitk.pipes.pesa_fat.dixon_v5 import config as cfg
    return cfg.STAGE2_DIR, cfg.STAGE3_DIR


def subject_files(pipeline: str, roots: Any, subject: str) -> tuple[list[Path], list[Path]]:
    """``(input volumes, masks)`` of *subject* for *pipeline* under the preset *roots*."""
    batch = roots.batch or ""
    nifti = Path(roots.nifti_root) / batch / subject
    results = Path(roots.results_root) / batch
    stage2, _stage3 = _stage_dirs(pipeline)
    images: list[Path] = []
    if nifti.is_dir():
        for path in sorted(nifti.rglob("*.nii*")):
            name = path.name.upper()
            if pipeline == "ct-pet-v5":
                wanted = name.startswith(("CT", "PET", "PT", "SUV"))
            else:
                # The region-named Dixon series (DIXON_<REGION>_<map>), not the raw MR_<series> ones.
                wanted = name.startswith("DIXON_")
            if wanted:
                images.append(path)
    masks_dir = results / stage2 / subject
    masks = sorted(p for p in masks_dir.rglob("*.nii*") if not p.name.split(".")[0].endswith("_VOL")) if masks_dir.is_dir() else []
    return images, masks


def subject_measurements(pipeline: str, roots: Any, subject: str) -> Any:
    """The subject's stage-3 measurement row (a one-row DataFrame, empty when not measured)."""
    import pandas as pd

    from nvitk.pipes.pesa_fat.qc.measurements_table import load_per_subject_tables

    _stage2, stage3 = _stage_dirs(pipeline)
    folder = Path(roots.results_root) / (roots.batch or "") / stage3 / "per_subject"
    try:
        return load_per_subject_tables(folder, [subject])
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


class PesaFatQcPanel(QWidget):
    """Review PESA-Fat CT-PET / Dixon results subject by subject."""

    def __init__(self, viewer: Any, parent: QWidget | None = None, *, embedded: bool = False) -> None:
        """*embedded* (the QC hub): build :attr:`picker` and :attr:`review` for the hub to place,
        instead of this panel's own two sections."""
        super().__init__(parent)
        self._viewer = viewer
        self._roots: Any = None

        self.picker = QWidget()
        pick = QVBoxLayout(self.picker)
        pick.setContentsMargins(0, 0, 0, 0)
        pick.setSpacing(design.SPACE_TIGHT)
        row = QHBoxLayout()
        self._pipeline = QComboBox()
        self._pipeline.setToolTip("PESA-Fat pipeline: CT-PET or Dixon")
        for key, (label, _preset) in PIPELINES.items():
            self._pipeline.addItem(label, key)
        self._batch = QComboBox()
        self._batch.setToolTip("Cohort / batch folder")
        row.addWidget(self._pipeline)
        row.addWidget(self._batch, 1)
        pick.addLayout(row)
        self._subjects = QListWidget()
        self._subjects.setMinimumHeight(80)
        pick.addWidget(self._subjects, 1)
        row = QHBoxLayout()
        load = QPushButton("Load subject")
        load.clicked.connect(self._load)
        row.addWidget(load)
        self._status = hint_label("")
        row.addWidget(self._status, 1)
        pick.addLayout(row)

        self.review = QWidget()
        rev = QVBoxLayout(self.review)
        rev.setContentsMargins(0, 0, 0, 0)
        rev.setSpacing(design.SPACE_TIGHT)
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Structure", "Aspect", "Status", "Comment", "Measurements"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        self._table.setMinimumHeight(120)
        rev.addWidget(self._table, 1)
        row = QHBoxLayout()
        all_ok = QPushButton("Mark pending as OK")
        all_ok.clicked.connect(self._pending_ok)
        save = QPushButton("Save reviews")
        save.clicked.connect(self._save)
        row.addWidget(all_ok)
        row.addStretch(1)
        row.addWidget(save)
        rev.addLayout(row)

        if not embedded:
            root = QVBoxLayout(self)
            root.setContentsMargins(0, 0, 0, 0)
            root.setSpacing(design.SPACE_TIGHT)
            subject = Section("Subject", icon_name="database", hue="blue", pref_key="pesafat_qc.subject")
            subject.add(self.picker)
            root.addWidget(subject)
            review = Section("Review", icon_name="qc", hue="green", pref_key="pesafat_qc.review")
            review.add(self.review)
            root.addWidget(review, 1)
        else:
            # The hub reparents picker / review; keep them owned until it does.
            self.picker.setParent(self)
            self.review.setParent(self)

        self._pipeline.currentIndexChanged.connect(lambda _i: self._load_roots())
        self._batch.currentIndexChanged.connect(lambda _i: self._list_subjects())
        self._subjects.currentTextChanged.connect(lambda _t: self._fill_review())
        self._loaded_roots = False

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        self.ensure_loaded()

    def ensure_loaded(self) -> None:
        """List the cohorts and subjects the first time they are wanted (a share may be slow)."""
        if not self._loaded_roots:
            self._loaded_roots = True
            self._load_roots()

    def pipeline(self) -> str:
        return str(self._pipeline.currentData())

    # roots and subjects
    def _load_roots(self) -> None:
        from nvitk.gui.pipeline.data_presets import list_local_cohorts, load_preset_roots

        try:
            self._roots = load_preset_roots(PIPELINES[self.pipeline()][1])
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"No pipeline roots: {exc}")
            return
        self._batch.blockSignals(True)
        self._batch.clear()
        try:
            cohorts = list_local_cohorts(nifti_root=self._roots.nifti_root, dicom_root=self._roots.dicom_root,
                                         results_root=self._roots.results_root)
        except Exception:  # noqa: BLE001 (a network share that is not mounted)
            cohorts = []
        self._batch.addItems(cohorts)
        if self._roots.batch and self._roots.batch in cohorts:
            self._batch.setCurrentText(self._roots.batch)
        self._batch.blockSignals(False)
        self._list_subjects()

    def _list_subjects(self) -> None:
        from dataclasses import replace

        from nvitk.gui.pipeline.data_presets import list_local_subjects

        self._subjects.clear()
        if self._roots is None:
            return
        self._roots = replace(self._roots, batch=self._batch.currentText() or self._roots.batch)
        try:
            subjects = list_local_subjects(self._roots, include_dicom=False)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"Could not list subjects: {exc}")
            return
        self._subjects.addItems(subjects)
        self._status.setText(f"{len(subjects)} subject(s) in {self._roots.batch or 'the root'}")

    # loading
    def _load(self) -> None:
        subject = self._subjects.currentItem().text() if self._subjects.currentItem() else ""
        if not subject or self._roots is None:
            self._status.setText("Pick a subject first.")
            return
        from nvitk.gui.core.layer_folders import _set_folder
        from nvitk.gui.io.napari_io import open_paths_with_nvitk

        images, masks = subject_files(self.pipeline(), self._roots, subject)
        if not images and not masks:
            self._status.setText("No volumes or masks found for this subject.")
            return
        folder = f"QC {PIPELINES[self.pipeline()][0]}/{subject}"
        added = []
        for path in images + masks:
            try:
                for layer in open_paths_with_nvitk(self._viewer, [path]) or []:
                    _set_folder(layer, folder + ("/masks" if path in masks else ""))
                    added.append(layer)
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Could not open {path.name}: {exc}")
        self._status.setText(f"Loaded {len(images)} volume(s) and {len(masks)} mask(s) into “{folder}”.")
        self._fill_review()

    # review
    def _fill_review(self) -> None:
        from nvitk.pipes.pesa_fat.qc.expected_ranges import EXPECTED_RANGES_CTPET, EXPECTED_RANGES_DIXON, cell_level
        from nvitk.pipes.pesa_fat.qc.review_policy import expected_review_entries

        subject = self._subjects.currentItem().text() if self._subjects.currentItem() else ""
        self._table.setRowCount(0)
        if not subject or self._roots is None:
            return
        pipeline = self.pipeline()
        recorded = self._recorded(subject, pipeline)
        row_frame = subject_measurements(pipeline, self._roots, subject)
        values = row_frame.iloc[0].to_dict() if len(row_frame) else {}
        ranges = EXPECTED_RANGES_CTPET if pipeline == "ct-pet-v5" else EXPECTED_RANGES_DIXON
        entries = expected_review_entries(pipeline)
        self._table.setRowCount(len(entries))
        for r, (structure, aspect) in enumerate(entries):
            self._table.setItem(r, 0, QTableWidgetItem(structure))
            self._table.setItem(r, 1, QTableWidgetItem(aspect))
            combo = QComboBox()
            combo.addItems(STATUSES)
            prior = recorded.get((structure, aspect), {})
            combo.setCurrentText(str(prior.get("qc_status") or "PENDING").upper())
            self._table.setCellWidget(r, 2, combo)
            self._table.setItem(r, 3, QTableWidgetItem(str(prior.get("comment") or "")))
            related = {k: v for k, v in values.items() if str(k).upper().startswith(structure.upper())} if aspect == "MEASUREMENT" else {}
            text = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in list(related.items())[:6])
            item = QTableWidgetItem(text)
            levels = {cell_level(str(k), v, ranges) for k, v in related.items()}
            level = "bad" if "bad" in levels else "warn" if "warn" in levels else None
            if level:
                item.setBackground(QBrush(QColor(design.STATUS_COLORS[level])))
                item.setToolTip("Outside the expected range" if level == "warn" else "Implausible value")
            self._table.setItem(r, 4, item)

    def _recorded(self, subject: str, pipeline: str) -> dict[tuple[str, str], dict[str, Any]]:
        try:
            from nvitk.pipes.pesa_fat.common.db_publish import fetch_qc_reviews

            rows = fetch_qc_reviews(batch=self._roots.batch, subject=subject, pipeline=pipeline)
        except Exception:  # noqa: BLE001 (no dataset, or no reviews table yet)
            return {}
        return {(str(r.get("structure")), str(r.get("review_aspect") or "MEASUREMENT")): r for r in rows}

    def _pending_ok(self) -> None:
        for r in range(self._table.rowCount()):
            combo = self._table.cellWidget(r, 2)
            if combo is not None and combo.currentText() == "PENDING":
                combo.setCurrentText("OK")

    def _save(self) -> None:
        from nvitk.pipes.pesa_fat.common.db_publish import publish_qc_review

        subject = self._subjects.currentItem().text() if self._subjects.currentItem() else ""
        if not subject:
            return
        saved = 0
        reviewer = getpass.getuser()
        for r in range(self._table.rowCount()):
            combo = self._table.cellWidget(r, 2)
            status = combo.currentText() if combo is not None else "PENDING"
            comment = self._table.item(r, 3).text() if self._table.item(r, 3) else ""
            if status == "PENDING" and not comment:
                continue
            try:
                publish_qc_review(batch=self._roots.batch or "", subject=subject, pipeline=self.pipeline(),
                                  structure=self._table.item(r, 0).text(), review_aspect=self._table.item(r, 1).text(),
                                  qc_status=status, reviewer=reviewer, comment=comment)
                saved += 1
            except Exception as exc:  # noqa: BLE001
                self._status.setText(f"Could not save: {exc}")
                return
        self._status.setText(f"Saved {saved} review(s) for {subject}.")


__all__ = ["PIPELINES", "PesaFatQcPanel", "subject_files", "subject_measurements"]
