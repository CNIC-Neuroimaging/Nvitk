"""Dataset & XNAT setup (Data & QC module): configure, create and fill a dataset; map XNAT scans.

* **Configuration**: which ``dataset.json`` / ``xnat_rules.json`` are in force (or the packaged
  empty defaults), with buttons to write a starter, open the files and validate them.
* **New dataset**: scaffold an empty dataset folder (catalog, schemas, tables).
* **Import a spreadsheet**: opens the import editor (:mod:`nvitk.gui.panels.import_editor`): edit
  the sheet, map its columns to variables (and, for images, regions and a pipeline version), check
  what would change, import, and keep the mapping as a recipe.
* **Refresh from XNAT**: what ``nvitk-xnat-sync`` does, from here: the dataset's ``subjects``,
  ``sessions``, ``scans``, ``assets`` and ``subject_ids`` tables re-read from an XNAT project and
  classified under the scan mapping, optionally with the pipeline results (``nvitk-xnat-pipeline-sync``),
  then the query index rebuilt. It runs in the background and asks before writing.
* **XNAT scan mapping**: the classifier rules, edited in place
  (:class:`~nvitk.gui.panels.xnat_rules_editor.XnatRulesEditor`), with a tester that classifies
  typed series descriptions under the rules as edited, and Save.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qtpy.QtCore import QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core import design
from nvitk.gui.core.thread_worker import ThreadWorker
from nvitk.gui.core.widgets import Section, hint_label


class XnatRefreshWorker(ThreadWorker):
    """Runs :func:`nvitk.db.xnat.sync_xnat_project` (and the pipeline-resource index) off the GUI thread."""

    progress = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, project: str, *, subjects: str = "", sequences: str = "", password: str = "",
                 pipelines: bool = True, rebuild_index: bool = True, repo: Any = None) -> None:
        super().__init__()
        self._project = project
        self._subjects = subjects
        self._sequences = sequences
        self._password = password
        self._pipelines = pipelines
        self._rebuild_index = rebuild_index
        self._repo = repo

    def run(self) -> None:
        try:
            from nvitk.db.repo import get_repo
            from nvitk.db.xnat import sync_xnat_project
            from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection

            repo = self._repo or get_repo()
            conn = resolve_xnat_connection(load_xnat_profile(None), project=self._project,
                                           password=self._password or None)
            self.progress.emit(f"Reading {self._project} from {conn.server}…")
            frames = sync_xnat_project(
                repo, conn, subjects=self._subjects or None, requested_sequences=self._sequences or None,
                build_sqlite_index=False,
            )
            counts = {name: int(len(frame)) for name, frame in frames.items()}
            if self._pipelines:
                from nvitk.db.xnat_pipeline_resources import sync_xnat_pipeline_resources

                self.progress.emit("Indexing the pipeline results (qvtpy, eICAB…)…")
                counts["pipeline assets"] = int(len(sync_xnat_pipeline_resources(
                    repo, conn, subjects=self._subjects or None, build_sqlite_index=False)))
            if self._rebuild_index:
                self.progress.emit("Rebuilding the query index…")
                repo.build_sqlite_index()
            self.done.emit(counts)
        except Exception as exc:  # noqa: BLE001 (network, auth, a project the rules do not know)
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DatasetSetupPanel(QWidget):
    """Configuration, creation and import for the dataset and XNAT mapping."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(design.SPACE_TIGHT)

        cfg = Section("Configuration", icon_name="sliders", hue="blue", pref_key="setup.config")
        from nvitk.gui.core.dataset_root import DatasetRootButton, dataset_bus

        cfg.add_header_widget(DatasetRootButton())
        dataset_bus().changed.connect(lambda _r: self.refresh())
        self._config_text = hint_label("")
        cfg.add(self._config_text)
        row = QHBoxLayout()
        for text, tip, slot in (
            ("Write starter", "Generic starter files to fill in.", self._write_starter),
            ("Open files", "Open the configuration files in your editor.", self._open_files),
            ("Validate", "Check both files against their schemas.", self._validate),
        ):
            button = QPushButton(text)
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
        cfg.add_layout(row)
        root.addWidget(cfg)

        new = Section("New dataset", icon_name="database", hue="green", pref_key="setup.new", expanded=False)
        row = QHBoxLayout()
        self._new_root = QLineEdit()
        self._new_root.setPlaceholderText("Folder for the new dataset")
        browse = QPushButton("…")
        browse.clicked.connect(lambda: self._pick_dir(self._new_root))
        create = QPushButton("Create")
        create.clicked.connect(self._create)
        row.addWidget(self._new_root, 1)
        row.addWidget(browse)
        row.addWidget(create)
        new.add_layout(row)
        new.add(hint_label("Point settings.json “dataset.root” (or $NVITK_DATASET_ROOT) at it to use it."))
        root.addWidget(new)
        root.addWidget(self._build_from_scratch())

        imp = Section("Import a spreadsheet", icon_name="import", hue="orange", pref_key="setup.import", expanded=False)
        imp.add(hint_label(
            "Open a CSV / Excel sheet in the import editor: edit the table (rename, drop, filter rows, "
            "convert units…), say what each column is (subject id, visit, a variable, for images its "
            "region and pipeline version), check what would change, then import. The edits and mapping "
            "can be saved as a recipe for the next file of the same layout."))
        row = QHBoxLayout()
        open_editor = QPushButton("Open a spreadsheet…")
        open_editor.setToolTip("Open a file in the import editor")
        open_editor.clicked.connect(lambda: self._open_import_editor(ask=True))
        empty_editor = QPushButton("Import editor")
        empty_editor.setToolTip("Open the import editor (drop a file on it, or apply a recipe)")
        empty_editor.clicked.connect(lambda: self._open_import_editor(ask=False))
        row.addWidget(open_editor)
        row.addWidget(empty_editor)
        row.addStretch(1)
        imp.add_layout(row)
        self._import_text = hint_label("")
        imp.add(self._import_text)
        root.addWidget(imp)

        root.addWidget(self._build_refresh())

        xnat = Section("XNAT scan mapping", icon_name="server", hue="purple", pref_key="setup.xnat")
        from nvitk.gui.panels.xnat_rules_editor import XnatRulesEditor

        self._rules_editor = XnatRulesEditor()
        self._rules_editor.saved.connect(lambda _p: self.refresh())
        xnat.add(self._rules_editor)
        root.addWidget(xnat)
        root.addStretch(1)
        self.refresh()

    # building a dataset from scratch
    def _build_from_scratch(self) -> QWidget:
        """The order that keeps a new dataset's subjects single: the subject registry first, then
        what refers to subjects (XNAT, folders, measurement sheets) resolved against it."""
        box = Section("Build from scratch", icon_name="layers", hue="cyan", pref_key="setup.scratch", expanded=False)
        box.add(hint_label(
            "Each subject is kept once (subject_uid); every other id it goes by (patient id, MRI id, XNAT "
            "label, study number…) is a row of subject_ids pointing at it. Everything brought in afterwards "
            "is matched through those ids, so a sheet keyed by the MRI id lands on the same subjects."))
        steps = QFormLayout()
        steps.setContentsMargins(0, 0, 0, 0)
        registry = QPushButton("Import the subject registry…")
        registry.setToolTip("A sheet with one row per subject and one column per identifier (and sex, birth date, "
                            "cohort): opens the import editor on the 'subjects' kind")
        registry.clicked.connect(lambda: self._open_import_editor(kind="subjects"))
        steps.addRow("1. Subjects", registry)
        steps.addRow("2. XNAT", hint_label("Refresh from XNAT (below): sessions, scans and assets; XNAT labels "
                                           "already in the registry join their subject."))
        folder_row = QHBoxLayout()
        self._local_root = QLineEdit()
        self._local_root.setPlaceholderText("Folder with one sub-folder per subject")
        browse = QPushButton("…")
        browse.clicked.connect(lambda: self._pick_dir(self._local_root))
        self._local_kind = QComboBox()
        self._local_kind.addItem("NIfTI (TOF / 4DFlow)", "nifti")
        self._local_kind.addItem("DICOM", "dicom")
        index_btn = QPushButton("Index")
        index_btn.setToolTip("Add the folder's scans to assets; sub-folders named by any of a subject's ids join it")
        index_btn.clicked.connect(self._index_local)
        folder_row.addWidget(self._local_root, 1)
        folder_row.addWidget(browse)
        folder_row.addWidget(self._local_kind)
        folder_row.addWidget(index_btn)
        steps.addRow("3. Folders", folder_row)
        measures = QPushButton("Import measurements…")
        measures.setToolTip("Clinical / cognitive / image sheets: their subject column may hold any registered id")
        measures.clicked.connect(lambda: self._open_import_editor())
        steps.addRow("4. Values", measures)
        box.add_layout(steps)
        self._scratch_text = hint_label("")
        box.add(self._scratch_text)
        return box

    def _index_local(self) -> None:
        root = self._local_root.text().strip()
        if not root:
            self._scratch_text.setText("Choose the folder to index.")
            return
        kind = str(self._local_kind.currentData() or "nifti")
        self._scratch_text.setText(f"Indexing {root}…")
        import threading

        from nvitk.gui.core.main_thread import call_on_main

        def work() -> None:
            try:
                from nvitk.db.repo import get_repo

                repo = get_repo()
                if kind == "dicom":
                    from nvitk.db.local_dicom_assets import upsert_dicom_assets as upsert
                else:
                    from nvitk.db.local_nifti_assets import upsert_nifti_assets as upsert
                df = upsert(repo, root, build_sqlite_index=True)
                n_subjects = int(df["subject_uid"].nunique()) if not df.empty else 0
                text = f"Indexed {len(df):,} file(s) of {n_subjects:,} subject(s) from {root}."
            except Exception as exc:  # noqa: BLE001
                text = f"Could not index {root}: {exc}"
            call_on_main(self._scratch_text.setText, text)

        threading.Thread(target=work, name="nvitk-index-folder", daemon=True).start()

    # refresh from XNAT
    def _build_refresh(self) -> QWidget:
        from nvitk.db.xnat_projects import list_xnat_project_ids

        sync = Section("Refresh from XNAT", icon_name="refresh", hue="cyan", pref_key="setup.sync", expanded=False)
        sync.setToolTip("Same as nvitk-xnat-sync: re-read subjects, sessions, scans and assets from XNAT")
        form = QFormLayout()
        self._sync_project = QComboBox()
        try:
            projects = list_xnat_project_ids()
        except Exception:  # noqa: BLE001
            projects = []
        self._sync_project.addItems(projects)
        self._sync_project.setEditable(True)
        try:
            from nvitk.db.xnat_config import load_xnat_profile

            default = str(load_xnat_profile(None).get("default_project") or "")
            if default:
                self._sync_project.setCurrentText(default)
        except Exception:  # noqa: BLE001
            pass
        form.addRow("Project", self._sync_project)
        self._sync_subjects = QLineEdit()
        self._sync_subjects.setPlaceholderText("all subjects (or labels, comma separated)")
        form.addRow("Subjects", self._sync_subjects)
        self._sync_sequences = QLineEdit()
        self._sync_sequences.setPlaceholderText("the project's default sequences (or TOF, 4DFLOW_AP, …)")
        form.addRow("Sequences", self._sync_sequences)
        self._sync_password = QLineEdit()
        self._sync_password.setEchoMode(QLineEdit.Password)
        self._sync_password.setPlaceholderText("only if ~/.netrc has no entry (never stored)")
        form.addRow("Password", self._sync_password)
        sync.add_layout(form)
        self._sync_pipelines = QCheckBox("Also index the pipeline results (qvtpy, eICAB, 4dflows)")
        self._sync_pipelines.setChecked(True)
        self._sync_index = QCheckBox("Rebuild the query index afterwards")
        self._sync_index.setChecked(True)
        sync.add(self._sync_pipelines)
        sync.add(self._sync_index)
        row = QHBoxLayout()
        self._sync_button = QPushButton("Refresh from XNAT")
        self._sync_button.setToolTip("Re-read the project from XNAT into the dataset tables (asks first)")
        self._sync_button.clicked.connect(self._refresh_from_xnat)
        row.addWidget(self._sync_button)
        row.addStretch(1)
        sync.add_layout(row)
        self._sync_text = hint_label("Downloads no images: only the tables that list them.")
        sync.add(self._sync_text)
        self._sync_worker: XnatRefreshWorker | None = None
        return sync

    def _refresh_from_xnat(self) -> None:
        from nvitk.db.repo import get_repo

        if self._sync_worker is not None and self._sync_worker.isRunning():
            return
        project = self._sync_project.currentText().strip()
        if not project:
            self._sync_text.setText("Choose an XNAT project.")
            return
        try:
            repo = get_repo()
        except Exception as exc:  # noqa: BLE001
            self._sync_text.setText(f"No dataset: {exc}")
            return
        subjects = self._sync_subjects.text().strip()
        answer = QMessageBox.question(
            self, "Refresh from XNAT",
            f"Re-read {project} ({subjects or 'all subjects'}) from XNAT into the dataset at\n{repo.root}?\n\n"
            "The subjects, sessions, scans and assets rows of those subjects are rewritten.",
        )
        if answer != QMessageBox.Yes:
            return
        worker = XnatRefreshWorker(
            project, subjects=subjects, sequences=self._sync_sequences.text().strip(),
            password=self._sync_password.text(), pipelines=self._sync_pipelines.isChecked(),
            rebuild_index=self._sync_index.isChecked(), repo=repo,
        )
        worker.setParent(self)
        worker.progress.connect(self._sync_text.setText)
        worker.done.connect(self._sync_done)
        worker.failed.connect(lambda message: self._sync_text.setText(f"Refresh failed: {message}"))
        worker.finished.connect(lambda: self._sync_button.setEnabled(True))
        self._sync_password.clear()
        self._sync_button.setEnabled(False)
        self._sync_text.setText(f"Connecting to XNAT ({project})…")
        self._sync_worker = worker
        worker.start()

    def _sync_done(self, counts: dict) -> None:
        parts = [f"{name} {n}" for name, n in counts.items() if n]
        self._sync_text.setText("Refreshed: " + (", ".join(parts) if parts else "nothing new") + " rows.")

    # configuration
    def refresh(self) -> None:
        from nvitk.db.config import dataset_config, xnat_rules

        lines = []
        for name, module in (("dataset.json", dataset_config), ("xnat_rules.json", xnat_rules)):
            path = module.config_path()
            lines.append(f"{name}: {path if path is not None else 'not found — the packaged empty default is used'}")
        try:
            from nvitk.gui.core.dataset_root import current_root

            lines.insert(0, f"Working dataset: {current_root() or 'none'}")
        except Exception:  # noqa: BLE001
            pass
        self._config_text.setText("\n".join(lines))
        editor = getattr(self, "_rules_editor", None)
        if editor is not None and not editor._dirty:
            editor.reload()

    def _config_dir(self) -> Path:
        from nvitk.core import config_paths

        return config_paths.config_dir() or config_paths.default_config_dir()

    def _write_starter(self) -> None:
        import shutil

        templates = Path(__file__).resolve().parents[2] / "cli" / "config_templates"
        target = self._config_dir()
        target.mkdir(parents=True, exist_ok=True)
        for name in ("dataset.json", "xnat_rules.json"):
            if not (target / name).exists():
                shutil.copyfile(templates / name.replace(".json", ".example.json"), target / name)
        self.refresh()

    def _open_files(self) -> None:
        from nvitk.db.config import dataset_config, xnat_rules

        for module in (dataset_config, xnat_rules):
            path = module.config_path()
            if path is not None:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _validate(self) -> None:
        import json

        from nvitk.db.config import dataset_config, xnat_rules

        out = []
        for name, module in (("dataset.json", dataset_config), ("xnat_rules.json", xnat_rules)):
            path = module.config_path()
            if path is None:
                out.append(f"{name}: default in use")
                continue
            try:
                problems = module.validate(json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001
                problems = [str(exc)]
            out.append(f"{name}: {'ok' if not problems else '; '.join(problems[:4])}")
        self._config_text.setText("\n".join(out))

    # new dataset / import
    def _pick_dir(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a folder")
        if path:
            edit.setText(path)

    def _create(self) -> None:
        from nvitk.db.catalog import DatasetCatalog

        root = self._new_root.text().strip()
        if not root:
            return
        try:
            made = DatasetCatalog.create_scaffold(root)
            self._new_root.setText(str(made))
        except Exception as exc:  # noqa: BLE001
            self._new_root.setText(f"Could not create: {exc}")

    def _open_import_editor(self, ask: bool = True, kind: str | None = None) -> None:
        from nvitk.gui.panels.import_editor import open_import_editor

        path = ""
        if ask:
            path, _ = QFileDialog.getOpenFileName(self, "Spreadsheet", "", "Tables (*.xlsx *.xls *.csv *.tsv *.parquet)")
            if not path:
                return
        editor = open_import_editor(self.window(), path=path or None, kind=kind)
        editor.imported.connect(lambda result: self._import_text.setText(
            str(result.get("summary") or "").split("\n")[0] if result.get("summary") else
            f"Imported {result.get('written', result.get('rows'))} row(s) into {result.get('table')}."))

__all__ = ["DatasetSetupPanel"]
