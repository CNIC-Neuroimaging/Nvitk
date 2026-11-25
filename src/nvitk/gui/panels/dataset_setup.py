"""Dataset & XNAT setup (Data & QC module): configure, create and fill a dataset; map XNAT scans.

* **Configuration**: which ``dataset.json`` / ``xnat_rules.json`` are in force (or the packaged
  PESA defaults), with buttons to write a starter or the PESA configuration, open the files and
  validate them.
* **New dataset**: scaffold an empty dataset folder (catalog, schemas, tables).
* **Import a spreadsheet**: clinical or image measurements from a CSV / Excel sheet into the
  dataset, with a source batch id (the version tag of that import) and, for images, a modality.
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
        self._config_text = hint_label("")
        cfg.add(self._config_text)
        row = QHBoxLayout()
        for text, tip, slot in (
            ("Write PESA config", "dataset.json + xnat_rules.json reproducing the PESA setup, to edit.", self._write_pesa),
            ("Write starter", "Generic starter files to fill in for another project.", self._write_starter),
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
        new.add(hint_label("Point settings.json “db.root” (or $NVITK_DATASET_ROOT) at it to use it."))
        root.addWidget(new)

        imp = Section("Import a spreadsheet", icon_name="import", hue="orange", pref_key="setup.import", expanded=False)
        form = QFormLayout()
        self._file = QLineEdit()
        pick = QPushButton("…")
        pick.clicked.connect(self._pick_file)
        frow = QHBoxLayout()
        frow.addWidget(self._file, 1)
        frow.addWidget(pick)
        form.addRow("File", frow)
        self._sheet = QComboBox()
        form.addRow("Sheet", self._sheet)
        self._kind = QComboBox()
        self._kind.addItems(["clinical", "image"])
        form.addRow("Kind", self._kind)
        self._modality = QLineEdit()
        self._modality.setPlaceholderText("image modality, e.g. asl, t1, 4dflow")
        form.addRow("Modality", self._modality)
        self._batch = QLineEdit()
        self._batch.setPlaceholderText("source batch id (version tag), e.g. clinical_2026-11")
        form.addRow("Batch id", self._batch)
        imp.add_layout(form)
        go = QPushButton("Import into the dataset")
        go.clicked.connect(self._import)
        imp.add(go)
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

            default = str(load_xnat_profile(None).get("project") or "")
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
            lines.append(f"{name}: {path if path is not None else 'not found — the packaged PESA default is used'}")
        self._config_text.setText("\n".join(lines))
        editor = getattr(self, "_rules_editor", None)
        if editor is not None and not editor._dirty:
            editor.reload()

    def _config_dir(self) -> Path:
        from nvitk.core import config_paths

        return config_paths.config_dir() or config_paths.default_config_dir()

    def _write_pesa(self) -> None:
        import json

        from nvitk.db.config import dataset_config, xnat_rules

        target = self._config_dir()
        target.mkdir(parents=True, exist_ok=True)
        written = []
        for name, doc in ((dataset_config.CONFIG_NAME, dataset_config.default_document()),
                          (xnat_rules.CONFIG_NAME, xnat_rules.default_document())):
            path = target / name
            if path.exists():
                continue
            path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(str(path))
        self._config_text.setText(("Wrote " + ", ".join(written)) if written else f"Both files already exist in {target}.")
        self.refresh()

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

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Spreadsheet", "", "Tables (*.xlsx *.xls *.csv *.tsv *.parquet)")
        if not path:
            return
        self._file.setText(path)
        from nvitk.db.importers import tabular_sheet_names

        self._sheet.clear()
        try:
            self._sheet.addItems([str(s) for s in tabular_sheet_names(path)] or ["0"])
        except Exception:  # noqa: BLE001
            self._sheet.addItem("0")

    def _import(self) -> None:
        from nvitk.db.importers import import_measurements_from_source
        from nvitk.db.repo import get_repo

        path = self._file.text().strip()
        batch = self._batch.text().strip()
        if not path or not batch:
            self._import_text.setText("Choose a file and give the import a batch id.")
            return
        sheet: Any = self._sheet.currentText() or 0
        try:
            result = import_measurements_from_source(
                get_repo(), path, kind=self._kind.currentText(), source_batch_id=batch,
                modality=self._modality.text().strip() or None, sheet_name=sheet,
            )
            self._import_text.setText(f"Imported {len(result)} measurement rows from {Path(path).name} ({batch}).")
        except Exception as exc:  # noqa: BLE001
            self._import_text.setText(f"Import failed: {exc}")

__all__ = ["DatasetSetupPanel"]
