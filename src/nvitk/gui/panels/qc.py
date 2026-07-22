"""QC subwindow: pipeline/subject selector and Load into Napari."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.db.pipeline_assets import (
    XNAT_RESOURCE_EICAB,
    XNAT_RESOURCE_QVTPY,
    resource_label_to_asset_slot,
)
from nvitk.db.repo import DataRepo, get_repo_from_settings
from nvitk.db.settings_paths import _find_repo_root
from nvitk.db.xnat_pipeline_resources import list_pipeline_assets_for_subject
from nvitk.db.xnat_projects import get_xnat_project, list_xnat_project_ids
from nvitk.gui.core.log_panel import gui_log
from nvitk.gui.tools.runner import notify
from nvitk.gui.viz.qc_loader import (
    download_pipeline_resource_for_qc,
    load_eicab_qc_layers,
    load_qvtpy_qc_layers,
)
from nvitk.gui.viz.qc_review_panel import show_qc_measurements

log = Logger()

PIPELINE_QVTPY = "qvtpy"
PIPELINE_EICAB = "eicab"


class _QcLoadWorker(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        *,
        pipeline: str,
        subject_uid: str,
        project_id: str,
        config_path: Path,
        password: str,
        local_path: str,
        app_state: dict[str, Any],
        download_root: Path,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._subject_uid = subject_uid
        self._project_id = project_id
        self._config_path = config_path
        self._password = password
        self._local_path = local_path
        self._app_state = app_state
        self._download_root = download_root

    def run(self) -> None:
        try:
            resource = (
                XNAT_RESOURCE_QVTPY
                if self._pipeline == PIPELINE_QVTPY
                else XNAT_RESOURCE_EICAB
            )
            roots: list[Path] = []
            local = Path(self._local_path) if self._local_path else None
            if local is not None and local.is_dir() and any(local.rglob("*")):
                self.progress.emit(f"Using local cache {local}")
                roots.append(local)
                primary = local
            else:
                self.progress.emit(f"Downloading {resource} for {self._subject_uid}…")
                primary = download_pipeline_resource_for_qc(
                    config_path=self._config_path,
                    project_id=self._project_id,
                    subject_uid=self._subject_uid,
                    resource_label=resource,
                    password=self._password,
                    download_root=self._download_root,
                    app_state=self._app_state,
                )
                roots.append(primary)

            # Also fetch companion resources useful for phases / CD / TOF.
            companions = (
                ("4dflows",) if self._pipeline == PIPELINE_QVTPY else ("4dflows",)
            )
            for label in companions:
                try:
                    self.progress.emit(f"Downloading companion {label}…")
                    companion = download_pipeline_resource_for_qc(
                        config_path=self._config_path,
                        project_id=self._project_id,
                        subject_uid=self._subject_uid,
                        resource_label=label,
                        password=self._password,
                        download_root=self._download_root,
                        app_state=self._app_state,
                    )
                    roots.append(companion)
                except Exception as exc:
                    self.progress.emit(f"Companion {label} skipped: {exc}")

            self.finished_ok.emit(
                {
                    "pipeline": self._pipeline,
                    "subject_uid": self._subject_uid,
                    "primary": primary,
                    "roots": roots,
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class QcPanel(QWidget):
    """Right-tab QC browser for qvtpy / eicab subjects."""

    def __init__(
        self,
        viewer: Any,
        app_state: dict[str, Any],
        *,
        on_inputs_opened: Callable[[list[str]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._app_state = app_state
        self._on_inputs_opened = on_inputs_opened
        self._worker: _QcLoadWorker | None = None
        self._review_panel = None

        try:
            self._repo: DataRepo = get_repo_from_settings()
            if isinstance(self._repo, tuple):
                self._repo = self._repo[0]
        except Exception:
            from nvitk.db.repo import DataRepo as _DR, _default_dataset_root

            self._repo = _DR(_default_dataset_root(), auto_scaffold=True)

        self._status = QLabel("Select a pipeline and subject, then Load.")
        self._status.setWordWrap(True)

        self._pipeline = QComboBox()
        self._pipeline.addItem("qvtpy (4D flow)", PIPELINE_QVTPY)
        self._pipeline.addItem("eicab (TOF)", PIPELINE_EICAB)
        self._pipeline.setCurrentIndex(0)

        self._project = QComboBox()
        for pid in list_xnat_project_ids():
            self._project.addItem(get_xnat_project(pid).display_name, pid)

        self._config_path = QLineEdit()
        self._config_path.setPlaceholderText("XNAT config path")
        repo_root = _find_repo_root()
        default_cfg = (
            (repo_root / ".nvitk" / "xnat.json")
            if repo_root is not None
            else Path.cwd() / ".nvitk" / "xnat.json"
        )
        if default_cfg.is_file():
            self._config_path.setText(str(default_cfg))

        self._subject_search = QLineEdit()
        self._subject_search.setPlaceholderText("Filter subjects…")
        self._subject_search.setClearButtonEnabled(True)

        self._subjects = QListWidget()
        self._all_subjects: list[dict[str, Any]] = []

        self._btn_refresh = QPushButton("Refresh subjects")
        self._btn_load = QPushButton("Load")
        self._btn_load.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._btn_refresh)
        btn_row.addWidget(self._btn_load)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("Pipeline"))
        lay.addWidget(self._pipeline)
        lay.addWidget(QLabel("XNAT project"))
        lay.addWidget(self._project)
        lay.addWidget(QLabel("XNAT config"))
        lay.addWidget(self._config_path)
        lay.addWidget(QLabel("Subjects with indexed resources"))
        lay.addWidget(self._subject_search)
        lay.addWidget(self._subjects, stretch=1)
        lay.addLayout(btn_row)
        lay.addWidget(self._status)
        self.setLayout(lay)

        self._pipeline.currentIndexChanged.connect(self._reload_subjects)
        self._project.currentIndexChanged.connect(self._reload_subjects)
        self._subject_search.textChanged.connect(self._filter_subjects)
        self._subjects.currentItemChanged.connect(self._on_subject_selected)
        self._btn_refresh.clicked.connect(self._reload_subjects)
        self._btn_load.clicked.connect(self._on_load)

        self._reload_subjects()

    def _current_pipeline(self) -> str:
        return str(self._pipeline.currentData() or PIPELINE_QVTPY)

    def _current_project(self) -> str:
        return str(self._project.currentData() or "")

    def _required_slot(self) -> str:
        label = (
            XNAT_RESOURCE_QVTPY
            if self._current_pipeline() == PIPELINE_QVTPY
            else XNAT_RESOURCE_EICAB
        )
        return resource_label_to_asset_slot(label)

    def _reload_subjects(self) -> None:
        project_id = self._current_project()
        slot = self._required_slot()
        self._all_subjects = []
        try:
            assets = self._repo._load_table_frame("assets", use_sqlite=True)
            if not assets.empty and "asset_slot" in assets.columns:
                hit = assets[assets["asset_slot"].astype(str) == slot]
                # Also accept resource_label matches.
                if "resource_label" in assets.columns:
                    label = (
                        XNAT_RESOURCE_QVTPY
                        if self._current_pipeline() == PIPELINE_QVTPY
                        else XNAT_RESOURCE_EICAB
                    )
                    hit = pd_concat_unique(
                        hit,
                        assets[
                            assets["resource_label"].astype(str).str.lower() == label
                        ],
                    )
                subjects = sorted(
                    {
                        str(s).strip()
                        for s in hit.get("subject_uid", []).dropna().unique()
                        if str(s).strip()
                    }
                )
            else:
                subjects = []

            # Prefer subjects also present in project sessions when available.
            if project_id and self._repo.catalog.table_exists("sessions"):
                sessions = self._repo._load_table_frame(
                    "sessions",
                    filters={"project_id": project_id},
                    use_sqlite=True,
                )
                if not sessions.empty:
                    in_project = {
                        str(s).strip()
                        for s in sessions["subject_uid"].dropna().unique()
                    }
                    subjects = [s for s in subjects if s in in_project] or subjects

            for subject in subjects:
                local_path = ""
                try:
                    pdf = list_pipeline_assets_for_subject(
                        self._repo, project_id, subject
                    )
                    if not pdf.empty:
                        want = slot
                        match = pdf[pdf["asset_slot"].astype(str) == want]
                        if not match.empty:
                            local_path = str(match.iloc[0].get("asset_path") or "")
                except Exception:
                    pass
                self._all_subjects.append(
                    {
                        "subject_uid": subject,
                        "asset_path": local_path,
                        "asset_slot": slot,
                    }
                )
            self._status.setText(
                f"{len(self._all_subjects)} subject(s) with {slot} in catalog."
            )
        except Exception as exc:
            self._status.setText(f"Could not load subjects: {exc}")
        self._filter_subjects()

    def _filter_subjects(self) -> None:
        needle = self._subject_search.text().strip().lower()
        self._subjects.clear()
        for row in self._all_subjects:
            subj = row["subject_uid"]
            if needle and needle not in subj.lower():
                continue
            item = QListWidgetItem(subj)
            item.setData(Qt.UserRole, row)
            self._subjects.addItem(item)
        self._btn_load.setEnabled(False)

    def _on_subject_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        self._btn_load.setEnabled(current is not None)

    def _on_load(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            notify("QC load already in progress.", error=True)
            return
        item = self._subjects.currentItem()
        if item is None:
            notify("Select a subject.", error=True)
            return
        row = item.data(Qt.UserRole) or {}
        subject_uid = str(row.get("subject_uid") or item.text())
        config_text = self._config_path.text().strip()
        if not config_text:
            notify("Set an XNAT config path.", error=True)
            return
        config_path = Path(config_text).expanduser()
        if not config_path.is_file():
            notify(f"Config not found: {config_path}", error=True)
            return

        password, ok = QInputDialog.getText(
            self,
            "XNAT password",
            "Enter XNAT password (not stored):",
            QLineEdit.Password,
        )
        if not ok:
            return

        download_root = Path(tempfile.mkdtemp(prefix="nvitk_qc_"))
        roots = list(self._app_state.get("xnat_temp_dirs") or [])
        roots.append(str(download_root))
        self._app_state["xnat_temp_dirs"] = roots

        self._btn_load.setEnabled(False)
        self._status.setText(f"Loading {subject_uid}…")
        self._worker = _QcLoadWorker(
            pipeline=self._current_pipeline(),
            subject_uid=subject_uid,
            project_id=self._current_project(),
            config_path=config_path,
            password=str(password),
            local_path=str(row.get("asset_path") or ""),
            app_state=self._app_state,
            download_root=download_root,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_load_ok)
        self._worker.failed.connect(self._on_load_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_progress(self, message: str) -> None:
        self._status.setText(message)
        gui_log(message)

    def _on_load_ok(self, payload: object) -> None:
        assert isinstance(payload, dict)
        pipeline = str(payload["pipeline"])
        subject_uid = str(payload["subject_uid"])
        primary = Path(payload["primary"])
        roots = [Path(p) for p in payload["roots"]]
        try:
            if pipeline == PIPELINE_QVTPY:
                loaded = load_qvtpy_qc_layers(
                    self._viewer,
                    self._app_state,
                    primary,
                    extra_search_roots=roots,
                )
                stage6 = loaded.get("stage6_dir")
                if isinstance(stage6, Path):
                    self._review_panel = show_qc_measurements(
                        self._viewer,
                        subject_uid=subject_uid,
                        stage6_dir=stage6,
                    )
                notify(f"QC loaded qvtpy subject {subject_uid}.")
            else:
                load_eicab_qc_layers(
                    self._viewer, primary, tof_search_roots=roots
                )
                notify(f"QC loaded eicab subject {subject_uid}.")
            if self._on_inputs_opened is not None:
                self._on_inputs_opened([str(primary)])
            self._status.setText(f"Loaded {subject_uid} ({pipeline}).")
        except Exception as exc:
            self._status.setText(str(exc))
            notify(f"QC load failed while opening layers: {exc}", error=True)

    def _on_load_failed(self, message: str) -> None:
        self._status.setText(message)
        notify(f"QC load failed: {message}", error=True)

    def _on_worker_finished(self) -> None:
        self._btn_load.setEnabled(self._subjects.currentItem() is not None)
        self._worker = None


def pd_concat_unique(a: Any, b: Any) -> Any:
    import pandas as pd

    if a is None or getattr(a, "empty", True):
        return b
    if b is None or getattr(b, "empty", True):
        return a
    return pd.concat([a, b], ignore_index=True).drop_duplicates()
