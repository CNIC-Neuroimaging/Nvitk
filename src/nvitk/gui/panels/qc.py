"""QC subwindow: pipeline/subject selector and Load into Napari.

Supports two QC setups:
- **XNAT**: catalog subjects + download pipeline resources (existing path).
- **Local**: NIfTI root + results root on disk (no XNAT password).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nvitk.gui.core.thread_worker import ThreadWorker
from nvitk.core.logger import Logger
from nvitk.db.pipeline_assets import (
    XNAT_RESOURCE_EICAB,
    XNAT_RESOURCE_QVTPY,
    resource_label_to_asset_slot,
)
from nvitk.db.repo import DataRepo, get_repo_from_settings
from nvitk.db.xnat_projects import get_xnat_project, list_xnat_project_ids
from nvitk.gui.core.log_panel import gui_log
from nvitk.gui.tools.runner import notify
from nvitk.gui.viz.qc_loader import (
    download_phase_niftis_for_qc,
    download_pipeline_resource_for_qc,
    load_eicab_morphometrics,
    load_eicab_qc_layers,
    load_qvtpy_qc_layers,
)
from nvitk.db.qvtpy_qc import subject_qc_status_summary
from nvitk.gui.viz.qc_review_panel import show_qc_measurements
from nvitk.pipes.qvtpy import config as qvt_cfg

log = Logger()

PIPELINE_QVTPY = "qvtpy"
PIPELINE_EICAB = "eicab"

SOURCE_XNAT = 0
SOURCE_LOCAL = 1


class _QcLoadWorker(ThreadWorker):
    """Background thread that downloads (or reuses cached) pipeline resources and phase NIfTIs for a
    QC subject, so the GUI stays responsive during network I/O."""

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
        """Store the parameters needed to resolve/download the subject's pipeline resources."""
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
        """Resolve the primary pipeline resource (cached locally or downloaded from XNAT), then best-
        effort fetch companion ``4dflows`` and AP/RL/FH phase NIfTI resources; emits ``finished_ok``
        with the collected roots or ``failed`` on error."""
        try:
            resource = (
                XNAT_RESOURCE_QVTPY
                if self._pipeline == PIPELINE_QVTPY
                else XNAT_RESOURCE_EICAB
            )
            roots: list[Path] = []
            local = Path(self._local_path) if self._local_path else None
            if local is not None and local.is_dir() and any(local.rglob("*")):
                from nvitk.db.xnat_pipeline_resources import unwrap_xnat_resource_download

                # Catalog cache may still hold xnatpy ZIP nesting.
                local = unwrap_xnat_resource_download(local, resource)
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

            # Also fetch companion resources useful for phases / CD / TOF, and for eICAB the qvtpy
            # resource, which holds its stage-7 morphometrics (results tables, lesions).
            companions = ("4dflows",) if self._pipeline == PIPELINE_QVTPY else ("4dflows", XNAT_RESOURCE_QVTPY)
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

            # Scan-level AP/RL/FH phase NIfTIs (not in the 4dflows derivative bundle).
            try:
                self.progress.emit("Downloading AP/RL/FH phase NIfTIs…")
                phases_root = download_phase_niftis_for_qc(
                    config_path=self._config_path,
                    project_id=self._project_id,
                    subject_uid=self._subject_uid,
                    password=self._password,
                    download_root=self._download_root,
                    app_state=self._app_state,
                )
                roots.append(phases_root)
                # Subject root also helps find_phase_paths / CD discovery.
                roots.append(phases_root.parent)
            except Exception as exc:
                self.progress.emit(f"Phase NIfTI download skipped: {exc}")

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


def _default_local_roots() -> tuple[Path, Path]:
    """NIfTI + results defaults from qvtpy local config."""
    nifti = Path(getattr(qvt_cfg, "LOCAL_DEFAULT_NIFTI_ROOT", Path.home()))
    results = Path(getattr(qvt_cfg, "LOCAL_DEFAULT_RESULTS_ROOT", Path.home()))
    return nifti, results


def _list_subject_dirs(root: Path, prefix: str = "PESA") -> list[str]:
    """Names of subdirectories under *root* whose name starts with *prefix* (case-insensitive)."""
    if not root.is_dir():
        return []
    out: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.upper().startswith(prefix.upper()):
            out.append(child.name)
    return out


def _subject_has_pipeline_results(results_root: Path, subject: str, pipeline: str) -> bool:
    """True if *subject* has *pipeline* (qvtpy or eicab) output under *results_root*, recognizing
    bundled, unpacked-stage, and flat-export layouts."""
    subj = results_root / subject
    if not subj.is_dir():
        return False
    bundle = subj / ("qvtpy" if pipeline == PIPELINE_QVTPY else "eicab")
    if bundle.is_dir():
        return True
    # Unpacked resource or flat stage folders under the subject.
    if pipeline == PIPELINE_QVTPY:
        return any(
            (subj / name).is_dir()
            for name in (
                qvt_cfg.STAGE4_SEG_DIR,
                qvt_cfg.STAGE6_MEASURE_DIR,
                qvt_cfg.QVT_SUBDIR,
            )
        )
    if (subj / qvt_cfg.STAGE1_EICAB_DIR).is_dir():
        return True
    # Flat eICAB export: TOF_* under the subject folder.
    return any(
        p.is_file() and p.name.upper().startswith("TOF")
        for p in subj.glob("TOF*")
    )


class QcPanel(QWidget):
    """Right-tab QC browser for qvtpy / eicab subjects (XNAT or local roots)."""

    def __init__(
        self,
        viewer: Any,
        app_state: dict[str, Any],
        *,
        on_inputs_opened: Callable[[list[str]], None] | None = None,
        parent: QWidget | None = None,
        embedded: bool = False,
        review_panel: Any = None,
        tof_review: Any = None,
    ) -> None:
        """Build the source/pipeline pickers, XNAT/local stacked pages, subject list, and Load button.

        *embedded* (the QC hub, :mod:`nvitk.gui.panels.qc_hub`): the pipeline is chosen by the
        hub (:meth:`set_pipeline`), and the QVTPy / eICAB reviews go to the hub's *review_panel*
        (a :class:`~nvitk.gui.viz.qc_review_panel.QcMeasurementsPanel`) and *tof_review* (a
        :class:`~nvitk.gui.viz.tof_qc_review.TofReviewPanel`) rather than to a dock of their own.
        """
        super().__init__(parent)
        self._viewer = viewer
        self._app_state = app_state
        self._on_inputs_opened = on_inputs_opened
        self._worker: _QcLoadWorker | None = None
        self._review_panel = review_panel
        self._embedded_review = review_panel
        self._tof_review = tof_review
        self._embedded = bool(embedded)

        try:
            self._repo: DataRepo = get_repo_from_settings()
            if isinstance(self._repo, tuple):
                self._repo = self._repo[0]
        except Exception:
            from nvitk.db.repo import DataRepo as _DR, _default_dataset_root

            self._repo = _DR(_default_dataset_root(), auto_scaffold=True)

        from nvitk.gui.core import design
        from nvitk.gui.core.dataset_root import DatasetRootButton, dataset_bus
        from nvitk.gui.core.widgets import IconButton, SegmentedButtons, hint_label, primary_button

        dataset_bus().changed.connect(self._on_dataset_changed)
        self._status = hint_label("Pick a subject, then Load.")

        # The source: the combo carries the state (scripts set it), the segmented switch shows it.
        self._source = QComboBox()
        self._source.addItem("XNAT (catalog)", SOURCE_XNAT)
        self._source.addItem("Local (NIfTI + results)", SOURCE_LOCAL)
        self._source.setVisible(False)
        self._source_switch = SegmentedButtons(
            [(SOURCE_XNAT, "XNAT", "server",
              "Subjects whose pipeline results are indexed in the dataset catalog, downloaded from XNAT"),
             (SOURCE_LOCAL, "Local folders", "folder", "NIfTI and results folders on disk (no XNAT password)")],
            current=SOURCE_XNAT,
        )
        self._source_switch.changed.connect(lambda key: self._source.setCurrentIndex(self._source.findData(key)))
        self._source.currentIndexChanged.connect(
            lambda _i: self._source_switch.set_current(self._source.currentData()))

        self._pipeline = QComboBox()
        self._pipeline.addItem("qvtpy (4D flow)", PIPELINE_QVTPY)
        self._pipeline.addItem("eicab (TOF)", PIPELINE_EICAB)
        self._pipeline.setCurrentIndex(0)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_xnat_page())
        self._stack.addWidget(self._build_local_page())

        self._subject_search = QLineEdit()
        self._subject_search.setPlaceholderText("Search subjects…")
        self._subject_search.setClearButtonEnabled(True)

        self._subjects = QListWidget()
        self._subjects.setMinimumHeight(80)
        self._subjects.setAlternatingRowColors(True)
        self._all_subjects: list[dict[str, Any]] = []
        self._qc_statuses: dict[str, str] = {}  # subject_uid -> revised/partial/pending
        self._subject_count = hint_label("")
        self._subject_count.setWordWrap(False)

        self._btn_refresh = IconButton("refresh", "Reload the subject list")
        self._btn_load = primary_button("Load", "Load the selected subject's QC layers and its review")
        self._btn_load.setEnabled(False)

        lay = QVBoxLayout()
        lay.setSpacing(design.SPACE_TIGHT)
        if self._embedded:
            lay.setContentsMargins(0, 0, 0, 0)
            self._pipeline.setVisible(False)  # the hub's pipeline choice drives it
        # Top row: where the subjects come from (and, standalone, the dataset the reviews go to;
        # the hub shows that one itself).
        source_head = QHBoxLayout()
        source_head.setSpacing(design.SPACE_TIGHT)
        source_head.addWidget(self._source_switch)
        source_head.addStretch(1)
        if not self._embedded:
            source_head.addWidget(DatasetRootButton())
        lay.addLayout(source_head)
        lay.addWidget(self._source)
        if not self._embedded:
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.addRow("Pipeline", self._pipeline)
            lay.addLayout(form)
        lay.addWidget(self._stack)
        subjects_head = QHBoxLayout()
        subjects_head.setContentsMargins(0, design.SPACE_TIGHT, 0, 0)
        subjects_head.setSpacing(design.SPACE_TIGHT)
        subjects_title = QLabel("Subjects")
        subjects_title.setStyleSheet("font-weight: bold;")
        subjects_head.addWidget(subjects_title)
        subjects_head.addStretch(1)
        subjects_head.addWidget(self._subject_count)
        subjects_head.addWidget(self._btn_refresh)
        lay.addLayout(subjects_head)
        lay.addWidget(self._subject_search)
        lay.addWidget(self._subjects, stretch=1)
        lay.addWidget(self._btn_load)
        lay.addWidget(self._status)
        self.setLayout(lay)

        from nvitk.gui.core.xnat_profile import profile_bus

        profile_bus().changed.connect(self._on_profile_changed)
        self._source.currentIndexChanged.connect(self._on_source_changed)
        self._pipeline.currentIndexChanged.connect(self._reload_subjects)
        self._subject_search.textChanged.connect(self._filter_subjects)
        self._subjects.currentItemChanged.connect(self._on_subject_selected)
        self._btn_refresh.clicked.connect(self._reload_subjects)
        self._btn_load.clicked.connect(self._on_load)

        # Page setup only. Reading the subject list means querying the catalog,
        # which on a real dataset dominated GUI start-up (~6.8s of a 9.6s launch)
        # for a tab most launches never open, so it waits for showEvent.
        self._subjects_loaded = False
        self._sync_source_page()

    def _build_xnat_page(self) -> QWidget:
        """Build the XNAT-source page: the project and the connection profile (found from the
        configuration, switchable from its button)."""
        from nvitk.gui.core import design
        from nvitk.gui.core.widgets import hint_label
        from nvitk.gui.core.xnat_profile import XnatProfileButton

        page = QWidget()
        self._project = QComboBox()
        for pid in list_xnat_project_ids():
            self._project.addItem(get_xnat_project(pid).display_name, pid)
        # Prefer PESA Brain as the QC default project (the profile's own project wins below).
        pesa_idx = self._project.findData("PESA_Brain")
        if pesa_idx >= 0:
            self._project.setCurrentIndex(pesa_idx)

        # The profile (server, user, project) comes from the configuration; the button names its
        # server and switches to another profile file when there is one. Shared with the Data tab.
        self._profile_button = XnatProfileButton()

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(design.SPACE_TIGHT)
        form.addRow("Project", self._project)
        profile_row = QHBoxLayout()
        profile_row.setContentsMargins(0, 0, 0, 0)
        profile_row.addWidget(self._profile_button)
        profile_row.addStretch(1)
        form.addRow("Profile", profile_row)

        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(design.SPACE_TIGHT)
        lay.addLayout(form)
        lay.addWidget(hint_label("Results are read from the local cache when the catalog has them, "
                                 "else downloaded (the password is asked for, never stored)."))
        page.setLayout(lay)

        # The profile's own project first, before the combo is wired: the rest of the panel is
        # not built yet (the first showEvent loads the subjects).
        self._on_profile_changed()
        self._project.currentIndexChanged.connect(self._reload_subjects)
        return page

    def _on_profile_changed(self, _path: str = "") -> None:
        """A new XNAT profile: pick its default project when the catalog has it."""
        from nvitk.gui.core.xnat_profile import current_profile, profile_summary

        project = profile_summary(current_profile())["project"]
        if project:
            index = self._project.findData(project)
            if index < 0:
                index = self._project.findText(project)
            if index >= 0 and index != self._project.currentIndex():
                self._project.setCurrentIndex(index)

    def _build_local_page(self) -> QWidget:
        """Build the local-source page: NIfTI root and results root path fields with browse buttons."""
        from nvitk.gui.core import design
        from nvitk.gui.core.widgets import hint_label

        page = QWidget()
        default_nifti, default_results = _default_local_roots()

        self._nifti_root = QLineEdit()
        self._nifti_root.setPlaceholderText("Subjects / 4DFlow / …")
        if default_nifti.is_dir():
            self._nifti_root.setText(str(default_nifti))

        self._results_root = QLineEdit()
        self._results_root.setPlaceholderText("Subjects / qvtpy / …")
        if default_results.is_dir():
            self._results_root.setText(str(default_results))

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setVerticalSpacing(design.SPACE_TIGHT)
        form.addRow("NIfTI", self._path_row(self._nifti_root))
        form.addRow("Results", self._path_row(self._results_root))

        lay = QVBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(design.SPACE_TIGHT)
        lay.addLayout(form)
        lay.addWidget(hint_label("Subjects with results are listed first. Load reads the QC layers "
                                 "from results, and phases / CD / TOF from NIfTI."))
        page.setLayout(lay)
        for edit in (self._nifti_root, self._results_root):
            edit.editingFinished.connect(self._on_root_edited)
        return page

    def _on_root_edited(self) -> None:
        if not self._is_xnat() and self._subjects_loaded:
            self._reload_subjects()

    def _path_row(self, edit: QLineEdit) -> QWidget:
        """Wrap *edit* with a folder button that fills it from a directory picker."""
        from nvitk.gui.core.widgets import IconButton

        row = QWidget()
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, stretch=1)
        btn = IconButton("folder", "Choose the folder")
        btn.clicked.connect(lambda: self._browse_into(edit))
        h.addWidget(btn)
        row.setLayout(h)
        return row

    def _browse_into(self, edit: QLineEdit) -> None:
        """Open a directory picker and set *edit*'s text to the chosen path, then reload subjects."""
        path = QFileDialog.getExistingDirectory(
            self, "Select directory", edit.text() or str(Path.home())
        )
        if path:
            edit.setText(path)
            self._reload_subjects()

    def _is_xnat(self) -> bool:
        """True if the currently selected QC source is XNAT (vs. local disk)."""
        return int(self._source.currentData() or SOURCE_XNAT) == SOURCE_XNAT

    def _sync_source_page(self) -> None:
        """Point the stack and the Load button at the selected source. No I/O."""
        is_xnat = self._is_xnat()
        self._stack.setCurrentIndex(SOURCE_XNAT if is_xnat else SOURCE_LOCAL)
        self._btn_load.setText("Load" if is_xnat else "Load from disk")

    def _on_source_changed(self) -> None:
        """Switch the stacked page and Load button label for the newly selected source, then reload
        the subject list."""
        self._sync_source_page()
        self._reload_subjects()

    def showEvent(self, event: Any) -> None:
        """Load the subject list the first time the tab is looked at."""
        super().showEvent(event)
        if not self._subjects_loaded:
            self._subjects_loaded = True
            self._reload_subjects()

    def set_pipeline(self, key: str) -> None:
        """Switch to pipeline *key* (``"qvtpy"`` or ``"eicab"``); reloads the subject list."""
        index = self._pipeline.findData(key)
        if index >= 0 and index != self._pipeline.currentIndex():
            self._pipeline.setCurrentIndex(index)

    def _current_pipeline(self) -> str:
        """The currently selected pipeline id (``"qvtpy"`` or ``"eicab"``)."""
        return str(self._pipeline.currentData() or PIPELINE_QVTPY)

    def _current_project(self) -> str:
        """The currently selected XNAT project id."""
        return str(self._project.currentData() or "")

    def _required_slot(self) -> str:
        """The catalog ``asset_slot`` corresponding to the currently selected pipeline's resource."""
        label = (
            XNAT_RESOURCE_QVTPY
            if self._current_pipeline() == PIPELINE_QVTPY
            else XNAT_RESOURCE_EICAB
        )
        return resource_label_to_asset_slot(label)

    def _refresh_qc_statuses(self) -> None:
        """Query the DB for per-subject QC revision status."""
        try:
            if self._current_pipeline() == PIPELINE_EICAB:
                from nvitk.gui.viz.tof_qc_review import subject_review_status

                self._qc_statuses = subject_review_status(self._repo)
            else:
                self._qc_statuses = subject_qc_status_summary(repo=self._repo)
        except Exception:
            self._qc_statuses = {}

    def _on_dataset_changed(self, _root: str) -> None:
        """The working dataset changed (Data / Analysis / Statmodels switch): reload from it."""
        try:
            repo = get_repo_from_settings()
            self._repo = repo[0] if isinstance(repo, tuple) else repo
        except Exception:  # noqa: BLE001
            return
        try:
            self._reload_subjects()
        except Exception:  # noqa: BLE001
            pass

    def _reload_subjects(self) -> None:
        """Refresh QC statuses and reload the subject list from the currently selected source."""
        self._refresh_qc_statuses()
        if self._is_xnat():
            self._reload_xnat_subjects()
        else:
            self._reload_local_subjects()

    def _reload_xnat_subjects(self) -> None:
        """Populate the subject list from the catalog's ``assets`` table, filtered to the required
        pipeline resource slot and (if available) the currently selected XNAT project."""
        project_id = self._current_project()
        slot = self._required_slot()
        self._all_subjects = []
        try:
            assets = self._repo._load_table_frame("assets", use_sqlite=True)
            if not assets.empty and "asset_slot" in assets.columns:
                hit = assets[assets["asset_slot"].astype(str) == slot]
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

            # One pass over the frame already in hand. This used to call
            # list_pipeline_assets_for_subject per subject, which re-queries the
            # very same `assets` table filtered to that subject: 520 subjects
            # became ~1050 sqlite round-trips for a column we have loaded.
            # (_required_slot only ever names the qvtpy or eicab slot, so the
            # 4dflows bundle collapsing in that helper cannot apply here.)
            paths_by_subject: dict[str, str] = {}
            try:
                if not assets.empty and {"asset_slot", "subject_uid"} <= set(assets.columns):
                    slot_rows = assets[assets["asset_slot"].astype(str) == slot]
                    if "asset_path" in slot_rows.columns:
                        first = slot_rows.dropna(subset=["subject_uid"]).groupby(
                            slot_rows["subject_uid"].astype(str), sort=False
                        )["asset_path"].first()
                        paths_by_subject = {
                            str(k): ("" if v is None or v != v else str(v))
                            for k, v in first.items()
                        }
            except Exception:
                paths_by_subject = {}

            for subject in subjects:
                local_path = paths_by_subject.get(str(subject), "")
                self._all_subjects.append(
                    {
                        "subject_uid": subject,
                        "asset_path": local_path,
                        "asset_slot": slot,
                        "source": "xnat",
                    }
                )
            self._status.setText(
                f"{len(self._all_subjects)} subject(s) with {slot} in catalog."
            )
        except Exception as exc:
            self._status.setText(f"Could not load subjects: {exc}")
        self._filter_subjects()

    def _reload_local_subjects(self) -> None:
        """Populate the subject list by scanning the configured NIfTI/results root directories,
        ordering subjects that already have the selected pipeline's results first."""
        pipeline = self._current_pipeline()
        nifti_root = Path(self._nifti_root.text().strip()).expanduser()
        results_root = Path(self._results_root.text().strip()).expanduser()
        self._all_subjects = []

        from_results = _list_subject_dirs(results_root)
        from_nifti = _list_subject_dirs(nifti_root)
        # Prefer subjects that already have pipeline results for the selected pipeline.
        ordered: list[str] = []
        seen: set[str] = set()
        for subject in from_results:
            if _subject_has_pipeline_results(results_root, subject, pipeline):
                ordered.append(subject)
                seen.add(subject)
        for subject in from_results:
            if subject not in seen:
                ordered.append(subject)
                seen.add(subject)
        for subject in from_nifti:
            if subject not in seen:
                ordered.append(subject)
                seen.add(subject)

        for subject in ordered:
            results_path = results_root / subject
            nifti_path = nifti_root / subject
            has_results = _subject_has_pipeline_results(
                results_root, subject, pipeline
            )
            self._all_subjects.append(
                {
                    "subject_uid": subject,
                    "results_path": str(results_path) if results_path.is_dir() else "",
                    "nifti_path": str(nifti_path) if nifti_path.is_dir() else "",
                    "has_results": has_results,
                    "source": "local",
                }
            )

        n_ready = sum(1 for r in self._all_subjects if r.get("has_results"))
        self._status.setText(
            f"{len(self._all_subjects)} local subject(s) "
            f"({n_ready} with {pipeline} results)."
        )
        self._filter_subjects()

    _QC_TAG: dict[str, str] = {
        "revised": "  ✓ revised",
        "partial": "  ◑ partial",
    }
    _QC_COLOR: dict[str, str] = {
        "revised": "#4caf50",
        "partial": "#ff9800",
    }

    def _filter_subjects(self) -> None:
        """Rebuild the subject list widget from ``_all_subjects``, applying the search filter and
        QC-status tags/colors."""
        needle = self._subject_search.text().strip().lower()
        self._subjects.clear()
        shown = 0
        for row in self._all_subjects:
            subj = row["subject_uid"]
            if needle and needle not in subj.lower():
                continue
            shown += 1
            label = subj
            if row.get("source") == "local":
                if row.get("has_results"):
                    label = f"{subj}  [results]"
                elif row.get("nifti_path"):
                    label = f"{subj}  [nifti only]"
            qc = self._qc_statuses.get(subj, "")
            tag = self._QC_TAG.get(qc, "")
            if tag:
                label = f"{label}{tag}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, row)
            color = self._QC_COLOR.get(qc)
            if color:
                from qtpy.QtGui import QColor

                item.setForeground(QColor(color))
            self._subjects.addItem(item)
        self._btn_load.setEnabled(False)
        self._sync_subject_count(shown)

    def _sync_subject_count(self, shown: int) -> None:
        """The Subjects header: how many are listed (of how many) and how many are reviewed."""
        total = len(self._all_subjects)
        subjects = {str(r["subject_uid"]) for r in self._all_subjects}
        revised = sum(1 for s in subjects if self._qc_statuses.get(s) == "revised")
        partial = sum(1 for s in subjects if self._qc_statuses.get(s) == "partial")
        parts = [f"{shown} of {total}" if shown != total else f"{total}"]
        if revised:
            parts.append(f"✓ {revised}")
        if partial:
            parts.append(f"◑ {partial}")
        self._subject_count.setText(" · ".join(parts))
        self._subject_count.setToolTip(
            f"{total} subject(s) listed, {revised} reviewed, {partial} partly reviewed")

    def _on_subject_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        """Enable the Load button once a subject is selected."""
        self._btn_load.setEnabled(current is not None)

    def _on_load(self) -> None:
        """Dispatch the Load button to the XNAT or local subject loader based on the current source."""
        if self._worker is not None and self._worker.isRunning():
            notify("QC load already in progress.", error=True)
            return
        item = self._subjects.currentItem()
        if item is None:
            notify("Select a subject.", error=True)
            return
        row = item.data(Qt.UserRole) or {}
        if self._is_xnat():
            self._on_load_xnat(row, item)
        else:
            self._on_load_local(row, item)

    def _on_load_xnat(self, row: dict[str, Any], item: QListWidgetItem) -> None:
        """Prompt for the XNAT password and start a :class:`_QcLoadWorker` to download/cache the
        selected subject's pipeline resources in the background."""
        subject_uid = str(row.get("subject_uid") or item.text())
        from nvitk.gui.core.xnat_profile import current_profile

        profile = current_profile()
        if not profile:
            notify("No XNAT profile found: choose one from the Profile button.", error=True)
            return
        config_path = Path(profile)
        if not config_path.is_file():
            notify(f"XNAT profile not found: {config_path}", error=True)
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

    def _on_load_local(self, row: dict[str, Any], item: QListWidgetItem) -> None:
        """Resolve the selected subject's local results/NIfTI directories and load them directly
        (synchronously, no background worker needed)."""
        subject_uid = str(row.get("subject_uid") or item.text().split()[0])
        results_path = Path(str(row.get("results_path") or "")).expanduser()
        nifti_path = Path(str(row.get("nifti_path") or "")).expanduser()
        pipeline = self._current_pipeline()

        primary: Path | None = None
        if results_path.is_dir() and _subject_has_pipeline_results(
            results_path.parent, subject_uid, pipeline
        ):
            primary = results_path
        elif results_path.is_dir():
            primary = results_path
        elif nifti_path.is_dir():
            primary = nifti_path

        if primary is None or not primary.is_dir():
            notify(
                f"No local results/NIfTI folder for {subject_uid}. "
                "Check NIfTI / results roots.",
                error=True,
            )
            return

        roots: list[Path] = [primary]
        if nifti_path.is_dir() and nifti_path.resolve() != primary.resolve():
            roots.append(nifti_path)
        if results_path.is_dir() and results_path.resolve() not in {
            p.resolve() for p in roots
        }:
            roots.append(results_path)

        self._btn_load.setEnabled(False)
        self._status.setText(f"Loading local {subject_uid}…")
        gui_log(f"QC local load: primary={primary} roots={roots}")
        try:
            self._on_load_ok(
                {
                    "pipeline": pipeline,
                    "subject_uid": subject_uid,
                    "primary": primary,
                    "roots": roots,
                }
            )
        finally:
            self._btn_load.setEnabled(self._subjects.currentItem() is not None)

    def _on_progress(self, message: str) -> None:
        """Show a background-worker progress *message* in the status label and log panel."""
        self._status.setText(message)
        gui_log(message)

    def _on_load_ok(self, payload: object) -> None:
        """Open the resolved pipeline layers into Napari and (for qvtpy) attach the QC review and
        cohort violin docks."""
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
                if isinstance(stage6, Path) and self._embedded_review is not None:
                    self._embedded_review._on_revised = self._on_subject_revised
                    if self._embedded_review.load_from_stage6(subject_uid, stage6) == 0:
                        notify(f"No measurement CSVs found under {stage6}.", error=True)
                elif isinstance(stage6, Path):
                    self._review_panel = show_qc_measurements(
                        self._viewer,
                        subject_uid=subject_uid,
                        stage6_dir=stage6,
                        on_revised=self._on_subject_revised,
                    )
                try:
                    from nvitk.gui.viz.cohort_violin_panel import show_cohort_violin

                    self._cohort_violin_panel = show_cohort_violin(
                        self._viewer,
                        highlight_subject=subject_uid,
                    )
                except Exception as violin_exc:  # noqa: BLE001
                    notify(
                        f"Cohort violin dock skipped: {violin_exc}",
                        error=True,
                    )
                notify(f"QC loaded qvtpy subject {subject_uid}.")
            else:
                loaded = load_eicab_qc_layers(
                    self._viewer, primary, tof_search_roots=roots
                )
                try:
                    counts = load_eicab_morphometrics(self._viewer, loaded, [primary, *roots])
                except Exception as morpho_exc:  # noqa: BLE001
                    counts = {}
                    notify(f"Morphometrics tables skipped: {morpho_exc}", error=True)
                if self._tof_review is not None:
                    self._tof_review.on_revised = self._on_subject_revised
                    self._tof_review.load_subject(subject_uid, self._repo)
                shown = (f"; {counts.get('vessels', 0)} vessel(s), {counts.get('lesions', 0)} lesion(s) "
                         "in the results window") if counts else "; no stage-7 morphometrics found"
                notify(f"QC loaded eicab subject {subject_uid}{shown}.")
            if self._on_inputs_opened is not None:
                self._on_inputs_opened([str(primary)])
            self._status.setText(f"Loaded {subject_uid} ({pipeline}).")
        except Exception as exc:
            self._status.setText(str(exc))
            notify(f"QC load failed while opening layers: {exc}", error=True)

    def _on_subject_revised(self) -> None:
        """Refresh QC status tags after a subject is marked as revised."""
        self._refresh_qc_statuses()
        self._filter_subjects()

    def _on_load_failed(self, message: str) -> None:
        """Show the background worker's failure *message* in the status label and as a notification."""
        self._status.setText(message)
        notify(f"QC load failed: {message}", error=True)

    def _on_worker_finished(self) -> None:
        """Re-enable the Load button and drop the finished worker reference."""
        self._btn_load.setEnabled(self._subjects.currentItem() is not None)
        self._worker = None


def pd_concat_unique(a: Any, b: Any) -> Any:
    """Concatenate two DataFrames and drop exact-duplicate rows, short-circuiting when either is
    empty/``None``."""
    import pandas as pd

    if a is None or getattr(a, "empty", True):
        return b
    if b is None or getattr(b, "empty", True):
        return a
    return pd.concat([a, b], ignore_index=True).drop_duplicates()
