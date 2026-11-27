"""Editor for the XNAT scan-mapping rules (``xnat_rules.json``), with a live tester.

The rules decide which XNAT scans an import takes and what each becomes (sequence, modality,
orientation, asset slot). Here each classifier's rules are a table, edited in place:

* the common fields are columns: rule name, the series-description pattern that must match, the
  one that must not, and the resulting sequence and modality; a pattern that does not compile
  turns red as it is typed;
* everything else a rule may carry (extractors, other match fields, other result keys) is kept
  as it is, and editable as JSON with **Edit JSON…**;
* rules are added, removed and reordered (the first rule that matches wins);
* the **tester** classifies descriptions typed one per line against the rules *as edited*
  (nothing has to be saved first) and says which rule decided each one;
* **Save** validates the whole document against the schema and writes it to the configured
  ``xnat_rules.json`` (or, when none exists yet, to the configuration folder).
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QBrush, QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core import design
from nvitk.gui.core.widgets import IconButton, hint_label

#: Table columns: (title, getter, setter) over one rule dict.
_FIELD = "series_description"
COLUMNS = ("Rule", "Matches (regex)", "Not matching", "Sequence", "Modality")
_REGEX_COLUMNS = (1, 2)


def _get(rule: dict[str, Any], column: int) -> str:
    if column == 0:
        return str(rule.get("name") or "")
    if column == 1:
        return str((rule.get("match") or {}).get(_FIELD, ""))
    if column == 2:
        return str((rule.get("not_match") or {}).get(_FIELD, ""))
    result = rule.get("result") or {}
    value = result.get("sequence" if column == 3 else "modality")
    return "" if value is None else str(value)


def _set(rule: dict[str, Any], column: int, text: str) -> None:
    text = str(text)
    if column == 0:
        rule["name"] = text
    elif column in (1, 2):
        key = "match" if column == 1 else "not_match"
        part = dict(rule.get(key) or {})
        if text:
            part[_FIELD] = text
        else:
            part.pop(_FIELD, None)
        if part:
            rule[key] = part
        else:
            rule.pop(key, None)
    else:
        result = dict(rule.get("result") or {})
        result["sequence" if column == 3 else "modality"] = text or None
        rule["result"] = result


def regex_error(pattern: str) -> str:
    """Why *pattern* does not compile, or ``""``."""
    try:
        re.compile(pattern)
        return ""
    except re.error as exc:
        return str(exc)


class XnatRulesEditor(QWidget):
    """Edit, test and save the XNAT classification rules."""

    #: The document was saved (path).
    saved = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._doc: dict[str, Any] = {}
        self._path: Path | None = None
        self._dirty = False
        self._filling = False
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)

        row = QHBoxLayout()
        self._classifier = QComboBox()
        self._classifier.setToolTip("Rule set (classifier); the projects using it are listed beside it")
        self._classifier.currentIndexChanged.connect(lambda _i: self._fill())
        self._projects = hint_label("")
        row.addWidget(hint_label("Rule set"))
        row.addWidget(self._classifier)
        row.addWidget(self._projects, 1)
        for name, tip, slot in (
            ("plus", "Add a rule after the selected one", self._add),
            ("minus", "Remove the selected rule", self._remove),
            ("chevron_up", "Move the selected rule up (earlier rules win)", lambda: self._move(-1)),
            ("chevron_down", "Move the selected rule down", lambda: self._move(1)),
            ("code", "Edit the selected rule as JSON (extractors, other fields)", self._edit_json),
        ):
            button = IconButton(name, tip, icon_size=14)
            button.clicked.connect(lambda _=False, f=slot: f())
            row.addWidget(button)
        root.addLayout(row)

        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(list(COLUMNS))
        self._table.verticalHeader().setVisible(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self._table.itemChanged.connect(self._on_item_changed)
        from nvitk.gui.core.widgets import resizable  # drag the handle under it to resize

        root.addWidget(resizable(self._table, height=240, minimum=100, pref_key="xnat_rules.table"), 1)

        self._tester = QPlainTextEdit()
        self._tester.setPlaceholderText("Series descriptions to test, one per line (e.g.  4D flow AP)")
        self._quality = QComboBox()
        self._quality.addItems(["", "usable", "questionable", "unusable"])
        self._quality.setToolTip("Scan quality to test with (empty: unknown, which passes)")
        test = QPushButton("Test")
        test.clicked.connect(self.run_tests)
        trow = QHBoxLayout()
        from nvitk.gui.core.widgets import resizable  # drag the handle under it to resize
        trow.addWidget(resizable(self._tester, height=70, pref_key="xnat_rules.tester"), 1)
        col = QVBoxLayout()
        col.addWidget(self._quality)
        col.addWidget(test)
        col.addStretch(1)
        trow.addLayout(col)
        root.addLayout(trow)
        self._results = hint_label("")
        self._results.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self._results)

        row = QHBoxLayout()
        self._state = hint_label("")
        self._save = QPushButton("Save")
        self._save.clicked.connect(self.save)
        revert = QPushButton("Revert")
        revert.clicked.connect(self.reload)
        row.addWidget(self._state, 1)
        row.addWidget(revert)
        row.addWidget(self._save)
        root.addLayout(row)
        self.reload()

    # document
    def reload(self) -> None:
        """Read the rules in force again (the configured file, or the packaged default)."""
        from nvitk.db.config import xnat_rules

        self._path = xnat_rules.config_path()
        try:
            if self._path is not None:
                self._doc = json.loads(self._path.read_text(encoding="utf-8"))
            else:
                self._doc = xnat_rules.default_document()
        except Exception as exc:  # noqa: BLE001
            self._doc = xnat_rules.default_document()
            self._state.setText(f"Could not read {self._path}: {exc}; showing the packaged default.")
        current = self._classifier.currentText()
        self._classifier.blockSignals(True)
        self._classifier.clear()
        self._classifier.addItems(sorted((self._doc.get("classifiers") or {}).keys()))
        if current and self._classifier.findText(current) >= 0:
            self._classifier.setCurrentText(current)
        self._classifier.blockSignals(False)
        self._set_dirty(False)
        self._fill()

    def document(self) -> dict[str, Any]:
        """The rules document as edited (a copy)."""
        return copy.deepcopy(self._doc)

    def _rules(self) -> list[dict[str, Any]]:
        name = self._classifier.currentText()
        spec = (self._doc.get("classifiers") or {}).get(name)
        if spec is None:
            return []
        spec.setdefault("rules", [])
        return spec["rules"]

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        where = str(self._path) if self._path is not None else "the packaged empty default (Save writes a copy to the configuration folder)"
        self._state.setText(("● unsaved changes — " if dirty else "") + f"Rules from {where}")
        self._save.setEnabled(dirty or self._path is None)

    # table
    def _fill(self) -> None:
        rules = self._rules()
        users = [pid for pid, spec in (self._doc.get("projects") or {}).items()
                 if spec.get("classifier") == self._classifier.currentText()]
        self._projects.setText(("used by " + ", ".join(users)) if users else "used by no project")
        self._filling = True
        self._table.setRowCount(len(rules))
        for r, rule in enumerate(rules):
            for c in range(len(COLUMNS)):
                item = QTableWidgetItem(_get(rule, c))
                extra = sorted(set(rule.get("extract") or {}) | (set(rule.get("match") or {}) - {_FIELD}))
                if c == 0 and extra:
                    item.setToolTip("Also: " + ", ".join(extra) + " (Edit JSON… to change)")
                self._table.setItem(r, c, item)
                self._mark(item, c)
        self._filling = False

    def _mark(self, item: QTableWidgetItem, column: int) -> None:
        if column not in _REGEX_COLUMNS:
            return
        error = regex_error(item.text())
        item.setForeground(QBrush(QColor(design.accent("red") if error else design.COLOR_TEXT)))
        if error:
            item.setToolTip(f"Not a valid regular expression: {error}")

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling:
            return
        rules = self._rules()
        if item.row() >= len(rules):
            return
        _set(rules[item.row()], item.column(), item.text())
        self._filling = True
        self._mark(item, item.column())
        self._filling = False
        self._set_dirty(True)

    def _selected(self) -> int:
        rows = self._table.selectionModel().selectedRows()
        return rows[0].row() if rows else -1

    def _add(self) -> None:
        rules = self._rules()
        at = self._selected() + 1 if self._selected() >= 0 else len(rules)
        rules.insert(at, {"name": "New rule", "match": {_FIELD: ""}, "result": {"modality": "mri", "sequence": "NEW"}})
        self._fill()
        self._table.selectRow(at)
        self._set_dirty(True)

    def _remove(self) -> None:
        row = self._selected()
        rules = self._rules()
        if 0 <= row < len(rules):
            del rules[row]
            self._fill()
            self._set_dirty(True)

    def _move(self, delta: int) -> None:
        row = self._selected()
        rules = self._rules()
        target = row + delta
        if 0 <= row < len(rules) and 0 <= target < len(rules):
            rules[row], rules[target] = rules[target], rules[row]
            self._fill()
            self._table.selectRow(target)
            self._set_dirty(True)

    def _edit_json(self) -> None:
        row = self._selected()
        rules = self._rules()
        if not 0 <= row < len(rules):
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Rule: {rules[row].get('name', '')}")
        lay = QVBoxLayout(dialog)
        text = QPlainTextEdit(json.dumps(rules[row], indent=2, ensure_ascii=False))
        text.setMinimumSize(520, 380)
        lay.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        lay.addWidget(buttons)
        design.apply_theme(dialog)
        while dialog.exec():
            try:
                value = json.loads(text.toPlainText())
                if not isinstance(value, dict):
                    raise ValueError("a rule is a JSON object")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(dialog, "Not valid", str(exc))
                continue
            rules[row] = value
            self._fill()
            self._table.selectRow(row)
            self._set_dirty(True)
            break

    # testing
    def classify(self, description: str, quality: str | None = None) -> tuple[dict[str, Any] | None, str, int]:
        """(result, rule name, rule index) for *description* under the edited rules."""
        from nvitk.db.config.xnat_rules import _parse

        rules = _parse(self._doc)
        classifier = rules.classifiers.get(self._classifier.currentText())
        if classifier is None:
            return None, "", -1
        scan = {_FIELD: description, "quality": quality}
        for i, rule in enumerate(classifier.rules):
            if not classifier.quality_ok(quality) and not rule.ignore_quality:
                continue
            try:
                result = rule.apply(scan)
            except Exception:  # noqa: BLE001 (a half-typed rule)
                continue
            if result is not None:
                return result, rule.name, i
        return None, "", -1

    def run_tests(self) -> list[str]:
        """Classify every line of the tester; returns (and shows) one line of text each."""
        from nvitk.db.config.xnat_rules import _parse

        lines = [ln for ln in self._tester.toPlainText().splitlines() if ln.strip()]
        quality = self._quality.currentText() or None
        out = []
        try:
            rules = _parse(self._doc)
        except Exception as exc:  # noqa: BLE001
            self._results.setText(f"The rules do not parse: {exc}")
            return []
        hit_rows = []
        for line in lines:
            result, name, index = self.classify(line, quality)
            if result is None:
                out.append(f"{line}  →  no rule (not imported)")
                continue
            sequence = result.get("sequence")
            slot = rules.asset_slot(str(sequence)) if sequence else "—"
            out.append(f"{line}  →  {sequence or result}  · slot {slot} · rule {index + 1} “{name}”")
            hit_rows.append(index)
        self._results.setText("\n".join(out))
        if hit_rows:
            self._table.selectRow(hit_rows[-1])
        return out

    # saving
    def save(self) -> Path | None:
        """Validate and write the document; returns the path written."""
        from nvitk.core import config_paths
        from nvitk.db.config import xnat_rules

        problems = xnat_rules.validate(self._doc)
        if problems:
            QMessageBox.warning(self, "Rules not saved", "Fix these first:\n\n" + "\n".join(problems[:15]))
            return None
        target = self._path
        if target is None:
            folder = config_paths.config_dir() or config_paths.default_config_dir()
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / xnat_rules.CONFIG_NAME
        target.write_text(json.dumps(self._doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        self._path = target
        self._set_dirty(False)
        self.saved.emit(str(target))
        return target


__all__ = ["COLUMNS", "XnatRulesEditor", "regex_error"]
