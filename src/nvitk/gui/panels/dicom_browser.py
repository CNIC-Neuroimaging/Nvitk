"""The DICOM browser: look inside a DICOM folder before loading anything.

Scan a folder (or files): every study, its series — split as nvitk's loader
splits them into volumes — and each series' files are listed, from the headers
alone. Click a series or a file to read its header; tick series or single files
to load them into the viewer or to export just those as DICOM. Headers can be
edited (double-click a value, add or delete a tag) and de-identified; the changes
are kept as pending edits, shown in the header, and written only on export —
the original files are never touched.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.design import (
    COLOR_ACCENT,
    COLOR_ACCENT_DEEP,
    COLOR_BORDER,
    COLOR_BUTTON_OFF,
    COLOR_DISABLED,
    COLOR_MUTED,
    COLOR_ON_ACCENT,
    COLOR_WARN,
    SPACE,
    SPACE_TIGHT,
    Card,
)
from nvitk.io.dicom_edit import DATE_MODES, EXPORT_LAYOUTS, EXPORT_NAMES, DicomEditor, parse_tag, tag_vr
from nvitk.io.dicom_index import DicomSeries, DicomStudy, merge_studies, read_header, scan_dicom

#: Where edits go: the file shown, its series, the ticked files, or every file scanned.
SCOPES = ("this file", "this series", "ticked files", "all scanned files")
#: Values not shown or edited as text.
_BINARY_VRS = {"OB", "OW", "OF", "OD", "OL", "OV", "UN"}
_ROLE = Qt.UserRole
_PLACEHOLDER = "__files__"

_BUTTON_STYLE = (
    f"QPushButton {{ border: 1px solid {COLOR_BORDER}; border-radius: 4px; padding: 4px 8px; }}"
    f"QPushButton:disabled {{ color: {COLOR_DISABLED}; }}"
)
_ACTION_STYLE = (
    f"QPushButton {{ background-color: {COLOR_ACCENT_DEEP}; color: {COLOR_ON_ACCENT}; border-radius: 4px;"
    " padding: 6px; font-weight: bold; }"
    f"QPushButton:disabled {{ background-color: {COLOR_BUTTON_OFF}; color: {COLOR_DISABLED}; }}"
)
_INPUT_STYLE = f"border: 1px solid {COLOR_BORDER}; border-radius: 3px; padding: 2px 4px;"


def _button(text: str, tip: str = "", *, primary: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setStyleSheet(_ACTION_STYLE if primary else _BUTTON_STYLE)
    if tip:
        b.setToolTip(tip)
    return b


def _muted(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {COLOR_MUTED}; font-weight: normal;")
    return label


def _date(text: str) -> str:
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}" if len(text) >= 8 and text[:8].isdigit() else text


def _time(text: str) -> str:
    """``143706.155`` → ``14:37:06``."""
    return f"{text[:2]}:{text[2:4]}:{text[4:6]}" if len(text) >= 6 and text[:6].isdigit() else text


def value_text(elem: Any, limit: int = 400) -> str:
    """A DICOM element's value as one line of text."""
    if elem.VR == "SQ":
        return f"[{len(elem.value or [])} item(s)]"
    if elem.VR in _BINARY_VRS:
        try:
            n = len(elem.value) if elem.value is not None else 0
        except TypeError:
            n = 0
        return f"<{n:,} bytes>"
    value = elem.value
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("latin-1", errors="replace")
    elif isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        text = "\\".join(str(v) for v in value)
    else:
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


class _Relay(QObject):
    """Carries a background job's news to the GUI thread (it lives there)."""

    progress = Signal(int, int)
    done = Signal(object)
    failed = Signal(str)
    finished = Signal()


class _Job:
    """A Python thread running *fn(progress, cancelled)*; its results arrive through Qt signals.

    A plain thread rather than a ``QThread``: nothing Qt-owned is left to tear down
    when the interpreter exits, and the relay belongs to the panel.
    """

    def __init__(self, parent: QObject, fn: Any) -> None:
        self.relay = _Relay(parent)
        self._fn = fn
        self._cancel = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        try:
            result = self._fn(lambda d, t: self.relay.progress.emit(d, t), lambda: self._cancel)
            self.relay.done.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.relay.failed.emit(str(exc))
        finally:
            self.relay.finished.emit()

    def start(self) -> None:
        self._thread.start()

    def cancel(self) -> None:
        self._cancel = True

    def isRunning(self) -> bool:  # noqa: N802 — QThread-like
        return self._thread.is_alive()

    def wait(self, msecs: int | None = None) -> None:
        self._thread.join(None if msecs is None else msecs / 1000.0)


class AnonymizeDialog(QDialog):
    """De-identification options."""

    def __init__(self, parent: QWidget | None, default_scope: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("Anonymize")
        form = QFormLayout(self)
        self.scope = QComboBox()
        self.scope.addItems(list(SCOPES))
        self.scope.setCurrentText(default_scope)
        self.name = QLineEdit("ANONYMOUS")
        self.pid = QLineEdit("ANON")
        self.dates = QComboBox()
        self.dates.addItems(list(DATE_MODES))
        self.shift = QSpinBox()
        self.shift.setRange(-36500, 36500)
        self.shift.setSuffix(" days")
        self.shift.setEnabled(False)
        self.dates.currentTextChanged.connect(lambda t: self.shift.setEnabled(t == "shift"))
        self.demographics = QCheckBox("Keep sex, age, height and weight")
        self.demographics.setChecked(True)
        self.private = QCheckBox("Remove private tags")
        self.private.setChecked(True)
        self.uids = QCheckBox("New study / series / instance UIDs (kept consistent)")
        self.uids.setChecked(True)
        self.descriptions = QCheckBox("Blank study / series / protocol descriptions")
        form.addRow("Apply to", self.scope)
        form.addRow("Patient name", self.name)
        form.addRow("Patient ID", self.pid)
        form.addRow("Dates", self.dates)
        form.addRow("Shift by", self.shift)
        for box in (self.demographics, self.private, self.uids, self.descriptions):
            form.addRow(box)
        note = _muted("Removes names, IDs, addresses, physicians, institution, station and device serial "
                      "(DICOM PS3.15 basic profile, subset). Text burned into the pixels (screen captures) "
                      "is not touched.")
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def options(self) -> dict[str, Any]:
        return {
            "patient_name": self.name.text(), "patient_id": self.pid.text(), "dates": self.dates.currentText(),
            "shift_days": int(self.shift.value()), "keep_demographics": self.demographics.isChecked(),
            "remove_private": self.private.isChecked(), "new_uids": self.uids.isChecked(),
            "remove_descriptions": self.descriptions.isChecked(),
        }


class ExportDialog(QDialog):
    """Where and how to export."""

    def __init__(self, parent: QWidget | None, n_files: int, n_edited: int) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export DICOM")
        form = QFormLayout(self)
        row = QHBoxLayout()
        self.folder = QLineEdit()
        browse = _button("Choose…")
        browse.clicked.connect(self._choose)
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        form.addRow("Into folder", row)
        self.layout_box = QComboBox()
        self.layout_box.addItems(list(EXPORT_LAYOUTS))
        self.names = QComboBox()
        self.names.addItems(list(EXPORT_NAMES))
        self.names.setToolTip("auto: numbered (IM00001.dcm) when anonymized — original names often hold "
                              "UIDs — else the original names.")
        self.apply_edits = QCheckBox(f"Apply the pending edits ({n_edited:,} of these files are edited)")
        self.apply_edits.setChecked(True)
        self.apply_edits.setEnabled(n_edited > 0)
        form.addRow("Folders", self.layout_box)
        form.addRow("File names", self.names)
        form.addRow(self.apply_edits)
        form.addRow(_muted(f"{n_files:,} file(s). The originals are not modified."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _choose(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Export into")
        if folder:
            self.folder.setText(folder)

    def _accept(self) -> None:
        if not self.folder.text().strip():
            self._choose()
        if self.folder.text().strip():
            self.accept()


class DicomBrowserPanel(QWidget):
    """Browse, load, edit and export the series of a DICOM folder."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._studies: list[DicomStudy] = []
        self._series: dict[str, DicomSeries] = {}
        self._file_series: dict[str, str] = {}
        self._source: list[str] = []
        self._adding = False
        self.setAcceptDrops(True)
        self.editor = DicomEditor()
        self._worker: Any = None
        self._shown_path: str | None = None
        self._filling = False
        self._checking = False

        root = QVBoxLayout(self)
        root.setContentsMargins(SPACE_TIGHT, SPACE_TIGHT, SPACE_TIGHT, SPACE_TIGHT)
        root.setSpacing(SPACE)

        # ── source ──
        src = Card("DICOM source")
        row = QHBoxLayout()
        row.setSpacing(SPACE_TIGHT)
        self._path = QLineEdit()
        self._path.setPlaceholderText("DICOM folders or files (; between several) — or drop them here…")
        self._path.setStyleSheet(_INPUT_STYLE)
        self._path.returnPressed.connect(self.scan_current)
        b_folder = _button("Folders…", "Choose one or several DICOM folders (Ctrl / Shift-click; read recursively).")
        b_add = _button("Add…", "Add more folders to what is listed (same studies are merged; edits are kept).")
        b_files = _button("Files…", "Choose DICOM files.")
        b_add.clicked.connect(self._add_folders)
        self._b_scan = _button("Scan", "Read the headers and list studies, series and files.", primary=True)
        b_folder.clicked.connect(self._choose_folder)
        b_files.clicked.connect(self._choose_files)
        self._b_scan.clicked.connect(self._scan_or_cancel)
        row.addWidget(self._path, 1)
        row.addWidget(b_folder)
        row.addWidget(b_add)
        row.addWidget(b_files)
        row.addWidget(self._b_scan)
        src.add_layout(row)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        src.add(self._progress)
        self._status = _muted("Choose a folder: nothing is loaded until you ask for it.")
        src.add(self._status)
        root.addWidget(src)

        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)

        # ── series and files ──
        tree_box = QWidget()
        tree_lay = QVBoxLayout(tree_box)
        tree_lay.setContentsMargins(0, 0, 0, 0)
        tree_lay.setSpacing(SPACE_TIGHT)
        filt = QHBoxLayout()
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter series (number, description, modality)…")
        self._filter.setStyleSheet(_INPUT_STYLE)
        self._filter.setClearButtonEnabled(True)
        self._filter.textChanged.connect(self._apply_filter)
        b_all = _button("Tick all")
        b_none = _button("None")
        b_all.clicked.connect(lambda: self._tick_all(True))
        b_none.clicked.connect(lambda: self._tick_all(False))
        filt.addWidget(self._filter, 1)
        filt.addWidget(b_all)
        filt.addWidget(b_none)
        tree_lay.addLayout(filt)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Series / file", "Files", "Mod.", "Matrix", "Info"])
        self._tree.setUniformRowHeights(True)
        self._tree.setAlternatingRowColors(True)
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        header = self._tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3, 4):
            header.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._tree.itemExpanded.connect(self._on_expanded)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.currentItemChanged.connect(lambda cur, _prev: self._show_item(cur))
        self._tree.itemDoubleClicked.connect(self._on_tree_double_click)
        self._tree.setMinimumHeight(180)
        tree_lay.addWidget(self._tree, 1)
        actions = QHBoxLayout()
        self._b_load = _button("Load ticked into the viewer", "Each ticked series (or the ticked files of a "
                               "series) becomes a volume, through nvitk's DICOM loader.", primary=True)
        self._b_export = _button("Export ticked…", "Copy the ticked files as DICOM into a folder, with the "
                                 "pending edits (anonymization) applied.")
        self._b_load.clicked.connect(self.load_ticked)
        self._b_export.clicked.connect(self._export_dialog)
        actions.addWidget(self._b_load, 1)
        actions.addWidget(self._b_export)
        tree_lay.addLayout(actions)
        self._tick_info = _muted()
        tree_lay.addWidget(self._tick_info)
        split.addWidget(tree_box)

        # ── header ──
        head_box = QWidget()
        head_lay = QVBoxLayout(head_box)
        head_lay.setContentsMargins(0, 0, 0, 0)
        head_lay.setSpacing(SPACE_TIGHT)
        self._head_title = QLabel("Header")
        self._head_title.setStyleSheet(f"color: {COLOR_ACCENT}; font-weight: 600;")
        self._head_title.setWordWrap(True)
        head_lay.addWidget(self._head_title)
        tools = QHBoxLayout()
        tools.setSpacing(SPACE_TIGHT)
        self._head_search = QLineEdit()
        self._head_search.setPlaceholderText("Search tags…")
        self._head_search.setStyleSheet(_INPUT_STYLE)
        self._head_search.setClearButtonEnabled(True)
        self._head_search.textChanged.connect(self._filter_header)
        self._scope = QComboBox()
        self._scope.addItems(list(SCOPES))
        self._scope.setCurrentText("this series")
        self._scope.setToolTip("Which files an edit (a typed value, an added or deleted tag) goes to.")
        tools.addWidget(self._head_search, 1)
        tools.addWidget(QLabel("Edits apply to"))
        tools.addWidget(self._scope)
        head_lay.addLayout(tools)
        self._header = QTreeWidget()
        self._header.setHeaderLabels(["Tag", "Name", "VR", "Value"])
        self._header.setAlternatingRowColors(True)
        self._header.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._header.itemDoubleClicked.connect(self._on_header_double_click)
        self._header.itemChanged.connect(self._on_header_changed)
        hh = self._header.header()
        for col in (0, 1, 2):
            hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self._header.setMinimumHeight(160)
        head_lay.addWidget(self._header, 1)
        edit_row = QHBoxLayout()
        edit_row.setSpacing(SPACE_TIGHT)
        b_add = _button("Add / set tag…", "Set any tag by keyword (PatientName) or number (0010,0010).")
        b_del = _button("Delete tag", "Remove the selected tag on export.")
        b_anon = _button("Anonymize…", "De-identify: names, IDs, dates, institution, private tags, UIDs.")
        self._b_undo = _button("Undo edit")
        self._b_clear = _button("Clear edits")
        b_add.clicked.connect(self._add_tag)
        b_del.clicked.connect(self._delete_tag)
        b_anon.clicked.connect(self._anonymize_dialog)
        self._b_undo.clicked.connect(self.undo_edit)
        self._b_clear.clicked.connect(self.clear_edits)
        for b in (b_add, b_del, b_anon, self._b_undo, self._b_clear):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            edit_row.addWidget(b)
        head_lay.addLayout(edit_row)
        self._edits_info = _muted()
        head_lay.addWidget(self._edits_info)
        split.addWidget(head_box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)
        self._sync_edits()
        self._sync_ticks()
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.shutdown)

    def shutdown(self) -> None:
        """Stop a scan or export still running (the app is quitting)."""
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(10000)

    # ── scanning ──────────────────────────────────────────────────────────────

    def _start_dir(self) -> str:
        first = (self._source or [""])[0]
        return first if os.path.isdir(first) else os.path.dirname(first)

    def _choose_folder(self) -> None:
        from nvitk.gui.core.dialogs import choose_directories

        folders = choose_directories(self, "DICOM folder(s) — Ctrl / Shift-click for several", self._start_dir())
        if folders:
            self.scan(folders)

    def _add_folders(self) -> None:
        from nvitk.gui.core.dialogs import choose_directories

        folders = choose_directories(self, "Add DICOM folder(s)", self._start_dir())
        if folders:
            self.scan(folders, add=bool(self._studies))

    def dragEnterEvent(self, event: Any) -> None:  # noqa: N802 — Qt naming
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: Any) -> None:  # noqa: N802 — Qt naming
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.scan(paths, add=bool(self._studies))

    def _choose_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "DICOM files", self._path.text() or "",
                                                "DICOM (*.dcm *.DCM *.ima *.IMA);;All files (*)")
        if files:
            self._path.setText(" ; ".join(files))
            self.scan_current()

    def _scan_or_cancel(self) -> None:
        if self._worker is not None and getattr(self._worker, "kind", "") == "scan":
            self._worker.cancel()
            return
        self.scan_current()

    def scan_current(self) -> None:
        """Scan what the path field names (a folder, or files separated by ``;``)."""
        paths = [p.strip() for p in self._path.text().split(";") if p.strip()]
        if paths:
            self.scan(paths)

    def scan(self, paths: list[str] | str, *, add: bool = False, wait: bool = False) -> None:
        """Read the headers under *paths* (folders and / or files) in the background.

        With *add*, they join what is listed — the same study or series found in
        several folders is merged, and pending edits and ticks are kept. *wait*
        blocks until done (scripts, tests).
        """
        if isinstance(paths, str):
            paths = [paths]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            self._status.setText(f"Not found: {missing[0]}")
            return
        if self._worker is not None:
            self._status.setText("Still busy — wait for the current scan or export.")
            return
        new = [p for p in paths if not add or p not in self._source]
        self._source = (self._source + new) if add else list(paths)
        self._adding = add
        self._path.setText(" ; ".join(self._source))
        worker = _Job(self, lambda progress, cancelled: scan_dicom(list(new or paths), progress=progress,
                                                                    cancelled=cancelled))
        worker.relay.progress.connect(self._on_progress)
        worker.relay.done.connect(self._on_scanned)
        worker.relay.failed.connect(self._on_failed)
        worker.relay.finished.connect(self._on_worker_finished)
        worker.kind = "scan"
        self._worker = worker
        self._b_scan.setText("Cancel")
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._status.setText("Reading headers…")
        worker.start()
        if wait:
            worker.wait()
            from qtpy.QtWidgets import QApplication

            QApplication.processEvents()

    def _on_progress(self, done: int, total: int) -> None:
        self._progress.setRange(0, max(total, 1))
        self._progress.setValue(done)
        self._progress.setFormat("%v / %m files")

    def _on_failed(self, message: str) -> None:
        self._status.setText(f"Failed: {message}")

    def _on_worker_finished(self) -> None:
        worker, self._worker = self._worker, None
        if worker is not None:
            worker.relay.deleteLater()
        self._b_scan.setText("Scan")
        self._progress.setVisible(False)

    def _on_scanned(self, studies: list[DicomStudy]) -> None:
        if getattr(self, "_adding", False) and self._studies:
            ticked = self.ticked()
            full = {k for k, paths in ticked.items() if len(paths) == len(self._series[k].files)}
            self.set_studies(merge_studies(self._studies, studies), keep_edits=True)
            for key, paths in ticked.items():
                if key in self._series:
                    self.tick(key, None if key in full else paths)
        else:
            self.set_studies(studies)

    def set_studies(self, studies: list[DicomStudy], *, keep_edits: bool = False) -> None:
        """Show *studies* (from :func:`~nvitk.io.dicom_index.scan_dicom`); pending edits
        are dropped unless *keep_edits*."""
        self._studies = studies
        self._series = {s.key: s for st in studies for s in st.series}
        self._file_series = {os.path.abspath(f.path): s.key for s in self._series.values() for f in s.files}
        if not keep_edits:
            self.editor.clear()
        self._filling = True
        self._tree.clear()
        bold = QFont()
        bold.setBold(True)
        for k, st in enumerate(studies):
            sitem = QTreeWidgetItem([st.label, f"{len(st.files):,}", "", "", st.accession and f"acc. {st.accession}"])
            sitem.setData(0, _ROLE, ("study", k))
            sitem.setFont(0, bold)
            sitem.setFlags(sitem.flags() | Qt.ItemIsUserCheckable)
            sitem.setCheckState(0, Qt.Unchecked)
            self._tree.addTopLevelItem(sitem)
            for s in st.series:
                info = ", ".join(x for x in (_date(s.date), s.body_part,
                                             "" if s.loadable else f"{s.kind} — not a volume") if x)
                item = QTreeWidgetItem([s.label, f"{len(s.files):,}", s.modality, s.matrix if s.loadable else "", info])
                item.setData(0, _ROLE, ("series", s.key))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(0, Qt.Unchecked)
                item.setToolTip(0, f"{s.label}\nUID {s.uid}\n{s.manufacturer} {s.protocol}".strip())
                if not s.loadable:
                    for col in range(5):
                        item.setForeground(col, QBrush(QColor(COLOR_MUTED)))
                placeholder = QTreeWidgetItem([_PLACEHOLDER])
                item.addChild(placeholder)
                sitem.addChild(item)
            sitem.setExpanded(True)
        self._filling = False
        if not studies:
            self._status.setText(f"No DICOM files found in {' ; '.join(self._source) or 'the selection'}.")
            self._sync_ticks()
            self._sync_edits()
            return
        n_series = len(self._series)
        n_files = sum(len(st.files) for st in studies)
        n_load = sum(1 for s in self._series.values() if s.loadable)
        n_src = len(self._source)
        where = f" from {n_src} folders / files" if n_src > 1 else ""
        self._status.setText(f"{n_files:,} files · {len(studies)} stud{'y' if len(studies) == 1 else 'ies'} · "
                             f"{n_series} series ({n_load} loadable as volumes){where}. Tick series or files to load "
                             "or export; click one to see its header.")
        self._apply_filter(self._filter.text())
        self._sync_ticks()
        self._sync_edits()
        first = self._tree.topLevelItem(0)
        if first is not None and first.childCount():
            self._tree.setCurrentItem(first.child(0))

    # ── tree ──────────────────────────────────────────────────────────────────

    def _populate(self, item: QTreeWidgetItem) -> None:
        """Put the files under a series item (on first expansion)."""
        if item.childCount() != 1 or item.child(0).text(0) != _PLACEHOLDER:
            return
        kind, key = item.data(0, _ROLE)
        s = self._series[key]
        state = Qt.Checked if item.checkState(0) == Qt.Checked else Qt.Unchecked
        self._filling = True
        item.takeChild(0)
        for f in s.files:
            info = _time(f.time) if f.time else ""
            if f.kind != "image":
                info = f"{f.kind}" + (f" · {info}" if info else "")
            matrix = f.matrix if f.kind == "image" else ""
            child = QTreeWidgetItem([f.name, "" if f.instance is None else str(f.instance), "", matrix, info])
            child.setData(0, _ROLE, ("file", f.path))
            child.setFlags(child.flags() | Qt.ItemIsUserCheckable)
            child.setCheckState(0, state)
            child.setToolTip(0, f.path)
            item.addChild(child)
        self._filling = False

    def _on_expanded(self, item: QTreeWidgetItem) -> None:
        data = item.data(0, _ROLE)
        if data and data[0] == "series":
            self._populate(item)

    def _on_item_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._filling or self._checking or column != 0:
            return
        self._checking = True
        try:
            state = item.checkState(0)
            if state != Qt.PartiallyChecked:
                self._set_children(item, state)
            parent = item.parent()
            while parent is not None:
                states = {parent.child(i).checkState(0) for i in range(parent.childCount())
                          if parent.child(i).text(0) != _PLACEHOLDER}
                parent.setCheckState(0, states.pop() if len(states) == 1 else Qt.PartiallyChecked)
                parent = parent.parent()
        finally:
            self._checking = False
        self._sync_ticks()

    def _set_children(self, item: QTreeWidgetItem, state: Any) -> None:
        for i in range(item.childCount()):
            child = item.child(i)
            if child.text(0) == _PLACEHOLDER:
                continue
            child.setCheckState(0, state)
            self._set_children(child, state)

    def _tick_all(self, on: bool) -> None:
        for i in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(i)
            item.setCheckState(0, Qt.Checked if on else Qt.Unchecked)

    def _series_items(self) -> list[QTreeWidgetItem]:
        out = []
        for i in range(self._tree.topLevelItemCount()):
            st = self._tree.topLevelItem(i)
            out.extend(st.child(j) for j in range(st.childCount()))
        return out

    def tick(self, key: str, files: list[str] | None = None) -> None:
        """Tick series *key* (or only its *files*) — scripting and tests."""
        for item in self._series_items():
            if item.data(0, _ROLE)[1] != key:
                continue
            if files is None:
                item.setCheckState(0, Qt.Checked)
                return
            self._populate(item)
            wanted = {os.path.abspath(f) for f in files}
            for j in range(item.childCount()):
                child = item.child(j)
                if os.path.abspath(child.data(0, _ROLE)[1]) in wanted:
                    child.setCheckState(0, Qt.Checked)
            return

    def ticked(self) -> dict[str, list[str]]:
        """``{series key: ticked file paths}`` for every series with something ticked."""
        out: dict[str, list[str]] = {}
        for item in self._series_items():
            key = item.data(0, _ROLE)[1]
            state = item.checkState(0)
            if state == Qt.Unchecked:
                continue
            populated = not (item.childCount() == 1 and item.child(0).text(0) == _PLACEHOLDER)
            if state == Qt.Checked or not populated:
                out[key] = self._series[key].paths
            else:
                files = [item.child(j).data(0, _ROLE)[1] for j in range(item.childCount())
                         if item.child(j).checkState(0) == Qt.Checked]
                if files:
                    out[key] = files
        return out

    def _sync_ticks(self) -> None:
        ticked = self.ticked()
        n_files = sum(len(v) for v in ticked.values())
        loadable = sum(1 for k in ticked if self._series[k].loadable)
        self._b_load.setEnabled(loadable > 0)
        self._b_export.setEnabled(n_files > 0)
        self._tick_info.setText(f"Ticked: {len(ticked)} series, {n_files:,} file(s)"
                                + (f" ({len(ticked) - loadable} not loadable as volumes)" if len(ticked) > loadable else "")
                                if ticked else "Nothing ticked.")

    def _apply_filter(self, text: str) -> None:
        needle = str(text or "").strip().lower()
        for item in self._series_items():
            s = self._series[item.data(0, _ROLE)[1]]
            hay = f"{s.label} {s.modality} {s.protocol} {s.kind}".lower()
            item.setHidden(bool(needle) and needle not in hay)

    def _on_tree_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        data = item.data(0, _ROLE)
        if data and data[0] == "series" and self._series[data[1]].loadable:
            self.load({data[1]: self._series[data[1]].paths})

    # ── header ────────────────────────────────────────────────────────────────

    def _show_item(self, item: QTreeWidgetItem | None) -> None:
        if item is None:
            return
        data = item.data(0, _ROLE)
        if not data:
            return
        kind, ref = data
        if kind == "file":
            self.show_header(ref)
        elif kind == "series":
            s = self._series[ref]
            if s.files:
                self.show_header(s.files[0].path)
        elif kind == "study":
            st = self._studies[ref]
            if st.series and st.series[0].files:
                self.show_header(st.series[0].files[0].path)

    def show_header(self, path: str) -> None:
        """Show *path*'s header as it will be exported (edits marked)."""
        self._shown_path = path
        try:
            original = read_header(path)
            effective = self.editor.header(path)
        except Exception as exc:  # noqa: BLE001
            self._head_title.setText(f"{os.path.basename(path)}: {exc}")
            return
        key = self._file_series.get(os.path.abspath(path))
        series = self._series.get(key) if key else None
        self._head_title.setText(f"{os.path.basename(path)}" + (f"  —  {series.label}" if series else "")
                                 + ("  (edited)" if self.editor.edited(path) else ""))
        edited = QBrush(QColor(COLOR_WARN))
        muted = QBrush(QColor(COLOR_MUTED))
        self._filling = True
        self._header.clear()
        tags = sorted({int(e.tag) for e in effective} | {int(e.tag) for e in original})
        for tag in tags:
            elem = effective.get(tag) if tag in effective else None
            if elem is None:
                old = original[tag]
                item = QTreeWidgetItem([self._tag_text(tag), old.name, str(old.VR), "(removed on export)"])
                for col in range(4):
                    item.setForeground(col, muted)
                font = item.font(3)
                font.setStrikeOut(True)
                item.setFont(1, font)
                item.setData(0, _ROLE, (tag, False))
                self._header.addTopLevelItem(item)
                continue
            item = self._element_item(elem, editable=True)
            before = original.get(tag) if tag in original else None
            if before is None or value_text(before) != value_text(elem):
                for col in range(4):
                    item.setForeground(col, edited)
                item.setToolTip(3, "Was: " + (value_text(before) if before is not None else "(not present)"))
            self._header.addTopLevelItem(item)
        self._filling = False
        self._filter_header(self._head_search.text())

    @staticmethod
    def _tag_text(tag: int) -> str:
        return f"({tag >> 16:04X},{tag & 0xFFFF:04X})"

    def _element_item(self, elem: Any, *, editable: bool) -> QTreeWidgetItem:
        item = QTreeWidgetItem([self._tag_text(int(elem.tag)), elem.name, str(elem.VR), value_text(elem)])
        can_edit = editable and elem.VR not in _BINARY_VRS and elem.VR != "SQ"
        item.setData(0, _ROLE, (int(elem.tag), can_edit))
        if can_edit:
            item.setFlags(item.flags() | Qt.ItemIsEditable)
        if elem.VR == "SQ":
            for k, ds in enumerate(elem.value or []):
                sub = QTreeWidgetItem([f"item {k + 1}", "", "", f"[{len(ds)} tag(s)]"])
                sub.setData(0, _ROLE, (None, False))
                for child in ds:
                    sub.addChild(self._element_item(child, editable=False))
                item.addChild(sub)
        return item

    def _filter_header(self, text: str) -> None:
        needle = str(text or "").strip().lower()
        for i in range(self._header.topLevelItemCount()):
            item = self._header.topLevelItem(i)
            hay = " ".join(item.text(c) for c in range(4)).lower()
            item.setHidden(bool(needle) and needle not in hay)

    def _on_header_double_click(self, item: QTreeWidgetItem, column: int) -> None:
        data = item.data(0, _ROLE)
        if data and data[1] and item.parent() is None:
            self._header.editItem(item, 3)

    def _on_header_changed(self, item: QTreeWidgetItem, column: int) -> None:
        if self._filling or column != 3:
            return
        tag, can_edit = item.data(0, _ROLE) or (None, False)
        if tag is None or not can_edit:
            return
        self.set_tag(tag, item.text(3))

    # ── edits ─────────────────────────────────────────────────────────────────

    def scope_paths(self, scope: str | None = None) -> list[str]:
        """The files an edit with *scope* (default: the panel's choice) goes to."""
        scope = scope or self._scope.currentText()
        if scope == "all scanned files":
            return [f.path for s in self._series.values() for f in s.files]
        if scope == "ticked files":
            return [p for paths in self.ticked().values() for p in paths]
        path = self._shown_path
        if path is None:
            return []
        if scope == "this series":
            key = self._file_series.get(os.path.abspath(path))
            return self._series[key].paths if key else [path]
        return [path]

    def _require_scope(self, scope: str | None = None) -> list[str] | None:
        paths = self.scope_paths(scope)
        if not paths:
            self._edits_info.setText("Nothing to edit: tick files, or click a file to show its header.")
            return None
        return paths

    def set_tag(self, tag: str | int, value: str, *, scope: str | None = None) -> None:
        """Set *tag* to *value* on the files of *scope* (pending until export)."""
        paths = self._require_scope(scope)
        if paths is None:
            return
        t = parse_tag(tag)
        vr = ""
        if self._shown_path is not None:
            try:
                vr = tag_vr(t, read_header(self._shown_path))
            except Exception:  # noqa: BLE001
                vr = ""
        try:
            from nvitk.io.dicom_edit import coerce_value

            coerce_value(value, vr or tag_vr(t))
        except ValueError as exc:
            QMessageBox.warning(self, "DICOM tag", f"Not a valid value for VR {vr}: {exc}")
            self._refresh_header()
            return
        self.editor.set(paths, t, value, vr=vr)
        self._after_edit(f"Set {self._tag_text(t)} on {len(paths):,} file(s).")

    def delete_tag(self, tag: str | int, *, scope: str | None = None) -> None:
        paths = self._require_scope(scope)
        if paths is None:
            return
        t = parse_tag(tag)
        self.editor.delete(paths, t)
        self._after_edit(f"Deleted {self._tag_text(t)} on {len(paths):,} file(s).")

    def anonymize(self, *, scope: str | None = None, **options: Any) -> None:
        paths = self._require_scope(scope)
        if paths is None:
            return
        self.editor.anonymize(paths, **options)
        self._after_edit(f"Anonymization pending on {len(paths):,} file(s).")

    def undo_edit(self) -> None:
        if self.editor.undo() is not None:
            self._after_edit("Last edit undone.")

    def clear_edits(self) -> None:
        self.editor.clear()
        self._after_edit("All pending edits dropped.")

    def _after_edit(self, message: str) -> None:
        self._refresh_header()
        self._sync_edits(message)

    def _refresh_header(self) -> None:
        if self._shown_path is not None:
            self.show_header(self._shown_path)

    def _sync_edits(self, message: str = "") -> None:
        ops = self.editor.ops
        self._b_undo.setEnabled(bool(ops))
        self._b_clear.setEnabled(bool(ops))
        if not ops:
            self._edits_info.setText(message or "No pending edits. Double-click a value to edit it; edits are "
                                     "written only when exporting.")
            return
        files = set()
        for op in ops:
            files |= set(op.files) if op.files else set(self._file_series)
        lines = f"{len(ops)} pending edit(s) on {len(files):,} file(s) — written on export."
        self._edits_info.setText(f"{message}  {lines}" if message else lines)

    def _add_tag(self) -> None:
        tag, ok = QInputDialog.getText(self, "Add / set tag", "Tag (keyword like PatientName, or gggg,eeee):")
        if not ok or not tag.strip():
            return
        try:
            t = parse_tag(tag)
        except ValueError as exc:
            QMessageBox.warning(self, "DICOM tag", str(exc))
            return
        value, ok = QInputDialog.getText(self, "Add / set tag", f"Value of {self._tag_text(t)} "
                                         "(\\ separates multiple values):")
        if ok:
            self.set_tag(t, value)

    def _delete_tag(self) -> None:
        item = self._header.currentItem()
        data = item.data(0, _ROLE) if item is not None else None
        if not data or data[0] is None or item.parent() is not None:
            self._edits_info.setText("Select a top-level tag in the header to delete it.")
            return
        self.delete_tag(data[0])

    def _anonymize_dialog(self) -> None:
        default = "ticked files" if self.ticked() else "all scanned files"
        dlg = AnonymizeDialog(self, default)
        if dlg.exec() == QDialog.Accepted:
            self.anonymize(scope=dlg.scope.currentText(), **dlg.options())

    # ── load and export ───────────────────────────────────────────────────────

    def load_ticked(self) -> list[Any]:
        """Load each ticked series (or its ticked files) as volumes into the viewer."""
        return self.load(self.ticked())

    def load(self, selection: dict[str, list[str]]) -> list[Any]:
        from qtpy.QtWidgets import QApplication

        from nvitk.gui.io.napari_io import open_dicom_files_with_nvitk

        layers: list[Any] = []
        skipped = [self._series[k].label for k in selection if not self._series[k].loadable]
        todo = [(k, v) for k, v in selection.items() if self._series[k].loadable]
        for n, (key, paths) in enumerate(todo, 1):
            s = self._series[key]
            self._status.setText(f"Loading {s.label} ({n}/{len(todo)}, {len(paths):,} files)…")
            QApplication.processEvents()
            source = os.path.commonpath(paths) if len(paths) > 1 else os.path.dirname(paths[0])
            layers.extend(open_dicom_files_with_nvitk(self._viewer, paths, source=source))
        note = f" Skipped (not volumes): {', '.join(skipped)}." if skipped else ""
        self._status.setText(f"Loaded {len(layers)} volume(s) from {len(todo)} series.{note}")
        return layers

    def _export_dialog(self) -> None:
        paths = [p for v in self.ticked().values() for p in v]
        if not paths:
            return
        n_edited = sum(1 for p in paths if self.editor.edited(p))
        dlg = ExportDialog(self, len(paths), n_edited)
        if dlg.exec() != QDialog.Accepted:
            return
        self.export(paths, dlg.folder.text().strip(), layout=dlg.layout_box.currentText(),
                    names=dlg.names.currentText(), apply_edits=dlg.apply_edits.isChecked())

    def export(self, paths: list[str], out_dir: str, *, layout: str = "series folders", names: str = "auto",
               apply_edits: bool = True, wait: bool = False) -> None:
        """Write *paths* into *out_dir* in the background (*wait*: block until done)."""
        from nvitk.io.dicom_edit import export_dicom

        editor = self.editor if apply_edits else None
        files = list(paths)
        worker = _Job(self, lambda progress, cancelled: export_dicom(
            files, out_dir, editor=editor, layout=layout, names=names, progress=progress, cancelled=cancelled))
        worker.kind = "export"
        worker.relay.progress.connect(self._on_progress)
        worker.relay.done.connect(lambda written: self._on_exported(written, out_dir))
        worker.relay.failed.connect(lambda msg: self._status.setText(f"Export failed: {msg}"))
        worker.relay.finished.connect(self._on_worker_finished)
        self._worker = worker
        self._progress.setVisible(True)
        self._progress.setRange(0, len(paths))
        self._status.setText(f"Exporting {len(paths):,} file(s)…")
        worker.start()
        if wait:
            worker.wait()
            from qtpy.QtWidgets import QApplication

            QApplication.processEvents()

    def _on_exported(self, written: list[str], out_dir: str) -> None:
        self.last_export = written
        self._status.setText(f"Exported {len(written):,} file(s) into {out_dir}.")
        try:
            from nvitk.gui.tools.runner import notify

            notify(f"DICOM export: {len(written):,} file(s) → {out_dir}")
        except Exception:  # noqa: BLE001
            pass


def build_dicom_browser(viewer: Any) -> DicomBrowserPanel:
    """The DICOM browser dock's widget."""
    panel = DicomBrowserPanel(viewer)
    viewer._nvitk_dicom_browser = panel
    return panel


__all__ = ["AnonymizeDialog", "DicomBrowserPanel", "ExportDialog", "SCOPES", "build_dicom_browser", "value_text"]
