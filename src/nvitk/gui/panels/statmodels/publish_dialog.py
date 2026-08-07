"""
Dialog for writing a derived analysis column back into the dataset.

Description
-----------
The write is an upsert into a shared dataset table, so it is the one action in this tool that
changes something other people will read. The dialog is built around that: the target table and its
key are stated up front, the reason that table was chosen is shown so a wrong guess is visible, and
*Preview* builds the exact rows without touching anything. Nothing is written until **Publish**.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
from typing import Any, Sequence

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger

from .db_publish import (
    IMAGE_TABLE,
    PUBLISH_TABLES,
    PublishRequest,
    PublishTarget,
    publish_derived_column,
)
from .theme import COLOR_ERROR, COLOR_MUTED, muted_label_style

log = Logger()


class PublishDerivedDialog(QDialog):
    """
    Edit and confirm a derived column's publication.

    Use :meth:`request` after ``exec()`` returns ``Accepted`` to get what the user settled on; the
    write itself has already happened by then, and :meth:`summary` reports what it did.
    """

    def __init__(
        self,
        parent: QWidget | None,
        *,
        repo: Any,
        frame: pd.DataFrame,
        column: str,
        target: PublishTarget,
        definition: str = "",
        region_columns: Sequence[str] = (),
    ) -> None:
        """Build the form, pre-filled from the resolved *target*."""
        super().__init__(parent)
        self.setWindowTitle(f"Publish “{column}” to the dataset")
        self.setMinimumWidth(560)
        self._repo = repo
        self._frame = frame
        self._column = column
        self._definition = definition
        self._summary: dict[str, Any] = {}

        lay = QVBoxLayout(self)
        intro = QLabel(
            f"Writes <b>{column}</b> into the dataset as a variable, so it is available to every "
            f"later session and to the rest of the toolkit — not only to this window."
        )
        intro.setWordWrap(True)
        lay.addWidget(intro)

        form = QFormLayout()

        self._variable_id = QLineEdit(column)
        self._variable_id.setToolTip(
            "The id the variable is stored and referenced under. Must be a valid identifier, since "
            "it becomes a formula term."
        )
        form.addRow("Variable id", self._variable_id)

        self._label = QLineEdit(f"{column} (derived in Statmodels)")
        form.addRow("Label", self._label)

        self._unit = QLineEdit()
        self._unit.setPlaceholderText("mL/min, ms, … — leave empty if unitless")
        form.addRow("Unit", self._unit)

        self._table = QComboBox()
        for label, table, domain in PUBLISH_TABLES:
            self._table.addItem(label, (table, domain))
        index = next(
            (i for i in range(self._table.count()) if self._table.itemData(i)[0] == target.table), 0
        )
        self._table.setCurrentIndex(index)
        self._table.currentIndexChanged.connect(self._sync_table)
        form.addRow("Table", self._table)

        self._region = QComboBox()
        for name in region_columns or ("territory",):
            self._region.addItem(str(name))
        preferred = self._region.findText(target.grouping and "territory" or "territory")
        if preferred >= 0:
            self._region.setCurrentIndex(preferred)
        self._region.setToolTip(
            "Which column identifies the region each row was measured on. For a melted frame this "
            "is the melted key, and it is stored as such — see the note below."
        )
        self._region_label = QLabel("Region column")
        form.addRow(self._region_label, self._region)

        self._register = QCheckBox("Register in the catalog (makes it selectable elsewhere)")
        self._register.setChecked(True)
        form.addRow("", self._register)
        lay.addLayout(form)

        self._reason = QLabel(f"Target chosen because it is {target.reason}." if target.reason else "")
        self._reason.setWordWrap(True)
        self._reason.setStyleSheet(muted_label_style())
        lay.addWidget(self._reason)

        self._note = QLabel("")
        self._note.setWordWrap(True)
        self._note.setStyleSheet(muted_label_style())
        lay.addWidget(self._note)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self._btn_preview = QPushButton("Preview")
        self._btn_preview.setToolTip("Build the rows and report what would be written, without writing.")
        self._btn_preview.clicked.connect(self._on_preview)
        buttons.addButton(self._btn_preview, QDialogButtonBox.ActionRole)
        self._btn_publish = QPushButton("Publish")
        self._btn_publish.setDefault(True)
        self._btn_publish.clicked.connect(self._on_publish)
        buttons.addButton(self._btn_publish, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._target = target
        self._sync_table()

    # ---- state ---------------------------------------------------------------
    def _sync_table(self) -> None:
        """Show the region picker only for the table that has regions, and explain the key."""
        table, _domain = self._table.currentData()
        is_image = table == IMAGE_TABLE
        self._region.setVisible(is_image)
        self._region_label.setVisible(is_image)
        if is_image:
            self._note.setText(
                "Rows are keyed on subject × region. If the analysis frame is melted (territory or "
                "hemisphere grouping), the melted key is stored as the region id — the value is not "
                "attributed to any single published vessel."
            )
        else:
            self._note.setText(
                "Rows are keyed on subject. A value that repeats across a subject's territories is "
                "written once; the first row per subject is used."
            )

    def request(self) -> PublishRequest:
        """The request as currently edited."""
        table, domain = self._table.currentData()
        return PublishRequest(
            column=self._column,
            variable_id=self._variable_id.text().strip(),
            label=self._label.text().strip(),
            table=table,
            domain=domain,
            unit=self._unit.text().strip(),
            modality=self._target.modality,
            pipeline_id=self._target.pipeline_id or "latest",
            grouping=self._target.grouping,
            region_column=self._region.currentText().strip() or "territory",
            definition=self._definition,
            register=self._register.isChecked(),
        )

    def summary(self) -> dict[str, Any]:
        """What the write actually did; empty until Publish succeeds."""
        return dict(self._summary)

    # ---- actions -------------------------------------------------------------
    def _run(self, *, dry_run: bool) -> dict[str, Any] | None:
        """Build (and optionally write) the rows, reporting any failure in the dialog."""
        try:
            summary = publish_derived_column(
                self._repo, self._frame, self.request(), dry_run=dry_run
            )
        except Exception as exc:
            self._status.setStyleSheet(f"color: {COLOR_ERROR};")
            self._status.setText(str(exc))
            log.debug("Publish failed: %s", exc, exc_info=True)
            return None
        self._status.setStyleSheet(f"color: {COLOR_MUTED};")
        return summary

    def _on_preview(self) -> None:
        """Report what would be written, without writing it."""
        summary = self._run(dry_run=True)
        if summary is None:
            return
        request = self.request()
        self._status.setText(
            f"Would write {summary['n_rows']} rows over {summary['n_subjects']} subjects into "
            f"{summary['table']} as {request.variable_id!r} ({summary['value_kind']} values). "
            f"Nothing has been written yet."
        )

    def _on_publish(self) -> None:
        """Write the rows and close on success."""
        summary = self._run(dry_run=False)
        if summary is None:
            return
        self._summary = summary
        self.accept()


__all__ = ["PublishDerivedDialog"]
