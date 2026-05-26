"""Browse and load imaging data from XNAT or local pipeline directories."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.db.repo import DataRepo, _default_dataset_root, get_repo_from_settings
from nvitk.db.xnat import (
    asset_slot_display_label,
    connect_xnat,
    download_scan_dicoms,
    filter_subjects_by_asset_slots,
    list_asset_slots_for_project,
    list_scans_for_subject,
    list_subjects_for_project,
    project_subject_asset_slots,
    resolve_xnat_scan_from_scan_row,
    xnat_sequence_to_asset_slot,
)
from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection
from nvitk.db.xnat_projects import (
    default_sequences_for_project,
    get_xnat_project,
    list_xnat_project_ids,
)
from nvitk.gui.io_napari import open_paths_with_nvitk
from nvitk.gui.log_panel import gui_log
from nvitk.gui.pipeline_data_presets import (
    LocalAsset,
    PipelineRoots,
    get_pipeline_preset,
    list_local_assets,
    list_local_cohorts,
    list_local_subjects,
    list_pipeline_preset_ids,
    load_preset_roots,
)
from nvitk.gui.tool_runner import notify

log = Logger()

_ITEM_ROLE = int(Qt.UserRole) + 1
SOURCE_XNAT = 0
SOURCE_LOCAL = 1


def _resolve_repo() -> tuple[DataRepo, Path]:
    try:
        repo = get_repo_from_settings()
        return repo, Path(repo.root)
    except Exception:
        root = _default_dataset_root()
        return DataRepo(root, auto_scaffold=True), root


class _XnatDownloadWorker(QThread):
    finished_ok = Signal(list)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        *,
        config_path: Path,
        project_id: str,
        password: str,
        scan_rows: list[dict[str, Any]],
        download_root: Path | None,
        temp_parent: Path | None,
    ) -> None:
        super().__init__()
        self._config_path = config_path
        self._project_id = project_id
        self._password = password
        self._scan_rows = scan_rows
        self._download_root = download_root
        self._temp_parent = temp_parent

    def run(self) -> None:
        try:
            profile = load_xnat_profile(self._config_path)
            conn = resolve_xnat_connection(
                profile,
                project=self._project_id,
                password=self._password or None,
            )
            opened_dirs: list[Path] = []
            with connect_xnat(conn) as session:
                for row in self._scan_rows:
                    subject_uid = str(row.get("subject_uid") or "")
                    scan_id = str(row.get("scan_id") or "")
                    sequence = str(row.get("asset_slot") or row.get("scan_id") or "scan")
                    self.progress.emit(f"Downloading {subject_uid} / {scan_id}…")
                    scan_obj = resolve_xnat_scan_from_scan_row(session, row)
                    if self._download_root is not None:
                        target = self._download_root / subject_uid / sequence
                    else:
                        parent = self._temp_parent or Path(tempfile.gettempdir())
                        target = parent / subject_uid / str(scan_id)
                    target.mkdir(parents=True, exist_ok=True)
                    download_scan_dicoms(scan_obj, target)
                    opened_dirs.append(target)
            self.finished_ok.emit(opened_dirs)
        except Exception as exc:
            self.failed.emit(str(exc))


class DataBrowserPanel(QWidget):
    """Load data from the XNAT catalog or from local pipeline directory presets."""

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
        self._repo, self._repo_root = _resolve_repo()
        self._all_subjects: list[str] = []
        self._xnat_subject_slots: dict[str, set[str]] = {}
        self._xnat_scan_filter_checks: dict[str, QCheckBox] = {}
        self._local_roots: PipelineRoots | None = None
        self._worker: _XnatDownloadWorker | None = None
        self._temp_session_root: Path | None = None

        self._source_combo = QComboBox()
        self._source_combo.addItem("XNAT (catalog)", SOURCE_XNAT)
        self._source_combo.addItem("Local pipeline", SOURCE_LOCAL)

        self._header_label = QLabel()
        self._header_label.setWordWrap(True)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_xnat_page())
        self._stack.addWidget(self._build_local_page())

        self._xnat_scan_filter_group = self._build_xnat_scan_filter_group()

        self._subject_search = QLineEdit()
        self._subject_search.setPlaceholderText("Search subjects…")
        self._subject_list = QListWidget()
        self._subject_list.setMinimumHeight(100)

        self._resource_label = QLabel("Scans (check to download)")
        self._resource_list = QListWidget()
        self._resource_list.setMinimumHeight(140)

        self._action_btn = QPushButton("Download selected scans")
        self._action_btn.clicked.connect(self._on_action)

        self._status = QLabel("Select a data source.")
        self._status.setWordWrap(True)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Data source"))
        layout.addWidget(self._source_combo)
        layout.addWidget(self._header_label)
        layout.addWidget(self._stack)
        layout.addWidget(self._xnat_scan_filter_group)
        layout.addWidget(QLabel("Subject"))
        layout.addWidget(self._subject_search)
        layout.addWidget(self._subject_list)
        layout.addWidget(self._resource_label)
        layout.addWidget(self._resource_list)
        layout.addWidget(self._action_btn)
        layout.addWidget(self._status)
        layout.addStretch(1)
        self.setLayout(layout)

        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._subject_search.textChanged.connect(self._filter_subjects)
        self._subject_list.currentItemChanged.connect(self._on_subject_selected)

        self._on_source_changed()

    def _build_xnat_page(self) -> QWidget:
        page = QWidget()
        self._project_combo = QComboBox()
        for pid in list_xnat_project_ids():
            self._project_combo.addItem(get_xnat_project(pid).display_name, pid)

        self._config_path = QLineEdit()
        self._config_path.setPlaceholderText("XNAT YAML/JSON config path")
        browse_config = QPushButton("Browse…")
        browse_config.clicked.connect(self._browse_config)

        self._download_path = QLineEdit()
        self._download_path.setPlaceholderText("Optional save folder (empty = temp, removed on exit)")
        browse_download = QPushButton("Folder…")
        browse_download.clicked.connect(self._browse_download_folder)

        self._temp_only = QCheckBox("Use temporary cache only (default)")
        self._temp_only.setChecked(True)
        self._temp_only.toggled.connect(self._on_temp_only_toggled)

        config_row = QHBoxLayout()
        config_row.addWidget(self._config_path, stretch=1)
        config_row.addWidget(browse_config)

        download_row = QHBoxLayout()
        download_row.addWidget(self._download_path, stretch=1)
        download_row.addWidget(browse_download)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("XNAT project"))
        lay.addWidget(self._project_combo)
        lay.addWidget(QLabel("XNAT config file"))
        lay.addLayout(config_row)
        lay.addWidget(self._temp_only)
        lay.addLayout(download_row)
        page.setLayout(lay)

        self._project_combo.currentIndexChanged.connect(self._reload_xnat_catalog)
        return page

    def _build_xnat_scan_filter_group(self) -> QGroupBox:
        group = QGroupBox("Filter subjects by indexed scans")
        self._xnat_filter_count_label = QLabel("Select scan types to filter the subject list.")
        self._xnat_filter_count_label.setWordWrap(True)

        self._xnat_match_all = QCheckBox("Subject must have all selected scans")
        self._xnat_match_all.setChecked(True)
        self._xnat_match_all.toggled.connect(self._filter_subjects)

        clear_btn = QPushButton("Clear scan filters")
        clear_btn.clicked.connect(self._clear_xnat_scan_filters)

        self._xnat_scan_filters_host = QWidget()
        self._xnat_scan_filters_layout = QGridLayout()
        self._xnat_scan_filters_layout.setContentsMargins(0, 0, 0, 0)
        self._xnat_scan_filters_host.setLayout(self._xnat_scan_filters_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._xnat_scan_filters_host)
        scroll.setMaximumHeight(120)

        lay = QVBoxLayout()
        lay.addWidget(self._xnat_filter_count_label)
        lay.addWidget(scroll)
        lay.addWidget(self._xnat_match_all)
        lay.addWidget(clear_btn)
        group.setLayout(lay)
        return group

    def _build_local_page(self) -> QWidget:
        page = QWidget()
        self._preset_combo = QComboBox()
        for pid in list_pipeline_preset_ids():
            spec = get_pipeline_preset(pid)
            self._preset_combo.addItem(spec.label, pid)

        self._batch_combo = QComboBox()
        self._batch_combo.setEditable(True)
        self._batch_combo.setPlaceholderText("Cohort folder (e.g. 202602_Week1)")

        self._dicom_root = QLineEdit()
        self._nifti_root = QLineEdit()
        self._results_root = QLineEdit()

        form = QFormLayout()
        form.addRow("DICOM root", self._path_row(self._dicom_root))
        form.addRow("NIfTI root", self._path_row(self._nifti_root))
        form.addRow("Results root", self._path_row(self._results_root))

        self._include_dicom = QCheckBox("DICOM series")
        self._include_dicom.setChecked(True)
        self._include_nifti = QCheckBox("NIfTI volumes")
        self._include_nifti.setChecked(True)
        self._include_results = QCheckBox("Results masks")
        self._include_results.setChecked(True)

        refresh_btn = QPushButton("Apply roots / refresh subjects")
        refresh_btn.clicked.connect(self._apply_local_roots)

        lay = QVBoxLayout()
        lay.addWidget(QLabel("Pipeline preset"))
        lay.addWidget(self._preset_combo)
        lay.addWidget(QLabel("Cohort / batch folder"))
        lay.addWidget(self._batch_combo)
        lay.addLayout(form)
        types_row = QHBoxLayout()
        types_row.addWidget(self._include_dicom)
        types_row.addWidget(self._include_nifti)
        types_row.addWidget(self._include_results)
        lay.addLayout(types_row)
        lay.addWidget(refresh_btn)
        page.setLayout(lay)

        self._preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self._include_dicom.toggled.connect(self._on_local_resource_filter_changed)
        self._include_nifti.toggled.connect(self._on_local_resource_filter_changed)
        self._include_results.toggled.connect(self._on_local_resource_filter_changed)
        return page

    def _path_row(self, edit: QLineEdit) -> QWidget:
        row = QWidget()
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, stretch=1)
        btn = QPushButton("…")
        btn.setFixedWidth(28)
        btn.clicked.connect(lambda: self._browse_into(edit))
        h.addWidget(btn)
        row.setLayout(h)
        return row

    def _browse_into(self, edit: QLineEdit) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select directory", edit.text() or str(Path.home()))
        if path:
            edit.setText(path)

    def cleanup_temp_dirs(self) -> None:
        roots = self._app_state.get("xnat_temp_dirs") or []
        for path in list(roots):
            p = Path(path)
            if p.exists():
                try:
                    shutil.rmtree(p, ignore_errors=True)
                except Exception as exc:
                    log.warning(f"Could not remove temp dir {p}: {exc}")
        self._app_state["xnat_temp_dirs"] = []
        self._temp_session_root = None

    def _is_xnat(self) -> bool:
        return int(self._source_combo.currentData() or SOURCE_XNAT) == SOURCE_XNAT

    def _on_source_changed(self) -> None:
        is_xnat = self._is_xnat()
        self._stack.setCurrentIndex(SOURCE_XNAT if is_xnat else SOURCE_LOCAL)
        self._xnat_scan_filter_group.setVisible(is_xnat)
        if is_xnat:
            self._header_label.setText(f"Dataset catalog: {self._repo_root}")
            self._resource_label.setText("Scans (check to download)")
            self._action_btn.setText("Download selected scans")
            self._reload_xnat_catalog()
        else:
            self._header_label.setText("Browse pipeline directories on disk.")
            self._resource_label.setText("Assets (check to load)")
            self._action_btn.setText("Load selected in Napari")
            self._on_preset_changed()

    def _on_preset_changed(self) -> None:
        if self._is_xnat():
            return
        preset_id = str(self._preset_combo.currentData() or "")
        try:
            spec = get_pipeline_preset(preset_id)
            roots = load_preset_roots(preset_id)
            self._batch_combo.setVisible(spec.show_batch)
            self._dicom_root.setText(str(roots.dicom_root))
            self._nifti_root.setText(str(roots.nifti_root))
            self._results_root.setText(str(roots.results_root))
            self._refresh_cohort_combo(spec.pesa_fat_layout)
            if roots.batch:
                self._set_batch_combo_text(roots.batch)
            self._local_roots = roots
        except Exception as exc:
            self._status.setText(f"Could not load preset: {exc}")
            return
        self._apply_local_roots()

    def _set_batch_combo_text(self, batch: str) -> None:
        idx = self._batch_combo.findText(batch)
        if idx >= 0:
            self._batch_combo.setCurrentIndex(idx)
        else:
            self._batch_combo.setEditText(batch)

    def _refresh_cohort_combo(self, pesa_fat_layout: bool) -> None:
        """Populate cohort dropdown from NIfTI/DICOM/RESULTS roots (PESA-Fat)."""
        if not pesa_fat_layout:
            return
        current = self._batch_combo.currentText().strip()
        cohorts = list_local_cohorts(
            nifti_root=self._nifti_root.text().strip() or ".",
            dicom_root=self._dicom_root.text().strip() or None,
            results_root=self._results_root.text().strip() or None,
        )
        self._batch_combo.blockSignals(True)
        self._batch_combo.clear()
        for name in cohorts:
            self._batch_combo.addItem(name)
        if current:
            self._set_batch_combo_text(current)
        elif cohorts:
            self._batch_combo.setCurrentIndex(0)
        self._batch_combo.blockSignals(False)

    def _apply_local_roots(self) -> None:
        if self._is_xnat():
            return
        preset_id = str(self._preset_combo.currentData() or "")
        spec = get_pipeline_preset(preset_id)
        try:
            self._refresh_cohort_combo(spec.pesa_fat_layout)
            self._local_roots = load_preset_roots(
                preset_id,
                dicom_root=self._dicom_root.text().strip() or None,
                nifti_root=self._nifti_root.text().strip() or None,
                results_root=self._results_root.text().strip() or None,
                batch=self._batch_combo.currentText().strip() or None,
            )
            if self._local_roots.batch:
                self._set_batch_combo_text(self._local_roots.batch)
        except Exception as exc:
            self._status.setText(str(exc))
            return
        try:
            self._all_subjects = list_local_subjects(
                self._local_roots,
                include_dicom=self._include_dicom.isChecked(),
                include_nifti=self._include_nifti.isChecked(),
                include_results=self._include_results.isChecked(),
            )
        except Exception as exc:
            self._all_subjects = []
            self._status.setText(f"Could not scan subjects: {exc}")
            return
        self._filter_subjects()
        self._resource_list.clear()
        self._status.setText(self._local_subjects_status_text(preset_id))

    def _local_subjects_status_text(self, preset_id: str) -> str:
        assert self._local_roots is not None
        sources: list[str] = []
        if self._include_nifti.isChecked():
            sources.append("NIfTI")
        if self._include_dicom.isChecked():
            sources.append("DICOM")
        if self._include_results.isChecked():
            sources.append("results")
        src = ", ".join(sources) if sources else "no roots selected"
        batch = self._local_roots.batch or ""
        batch_part = f" · batch={batch}" if batch else ""
        return f"{len(self._all_subjects)} subject(s) from {src} ({preset_id}{batch_part})"

    def _on_local_resource_filter_changed(self) -> None:
        if self._is_xnat():
            return
        if self._local_roots is None:
            return
        preset_id = str(self._preset_combo.currentData() or "")
        try:
            self._all_subjects = list_local_subjects(
                self._local_roots,
                include_dicom=self._include_dicom.isChecked(),
                include_nifti=self._include_nifti.isChecked(),
                include_results=self._include_results.isChecked(),
            )
        except Exception as exc:
            self._status.setText(f"Could not scan subjects: {exc}")
            return
        self._filter_subjects()
        self._status.setText(self._local_subjects_status_text(preset_id))
        item = self._subject_list.currentItem()
        if item is not None:
            self._refresh_resources_for_subject(item.text())
        else:
            self._resource_list.clear()

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select XNAT config",
            str(Path.home()),
            "Config (*.yaml *.yml *.json);;All (*)",
        )
        if path:
            self._config_path.setText(path)

    def _browse_download_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Download folder", str(Path.home()))
        if path:
            self._download_path.setText(path)
            self._temp_only.setChecked(False)

    def _on_temp_only_toggled(self, checked: bool) -> None:
        self._download_path.setEnabled(not checked)
        if checked:
            self._download_path.clear()

    def _current_project_id(self) -> str:
        return str(self._project_combo.currentData() or "")

    def _reload_xnat_catalog(self) -> None:
        if not self._is_xnat():
            return
        project_id = self._current_project_id()
        try:
            self._xnat_subject_slots = project_subject_asset_slots(self._repo, project_id)
            catalog_subjects = list_subjects_for_project(self._repo, project_id)
            self._all_subjects = sorted(
                set(catalog_subjects) | set(self._xnat_subject_slots.keys())
            )
            slots = list_asset_slots_for_project(self._repo, project_id)
            if not slots and self._xnat_subject_slots:
                slots = sorted({s for ss in self._xnat_subject_slots.values() for s in ss})
            self._rebuild_xnat_scan_filter_checkboxes(project_id, slots)
        except Exception as exc:
            self._all_subjects = []
            self._xnat_subject_slots = {}
            self._status.setText(f"Could not load catalog: {exc}")
        self._filter_subjects()
        self._resource_list.clear()

    def _rebuild_xnat_scan_filter_checkboxes(self, project_id: str, slots: list[str]) -> None:
        while self._xnat_scan_filters_layout.count():
            item = self._xnat_scan_filters_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._xnat_scan_filter_checks.clear()

        if not slots:
            hint = QLabel("(No indexed scans — run nvitk-xnat-sync for this project.)")
            hint.setWordWrap(True)
            self._xnat_scan_filters_layout.addWidget(hint, 0, 0)
            return

        preferred = {
            xnat_sequence_to_asset_slot(seq)
            for seq in default_sequences_for_project(project_id)
        }
        ordered = [s for s in slots if s in preferred]
        ordered.extend(s for s in slots if s not in preferred)

        columns = 3
        for idx, slot in enumerate(ordered):
            cb = QCheckBox(asset_slot_display_label(slot))
            cb.toggled.connect(self._filter_subjects)
            row, col = divmod(idx, columns)
            self._xnat_scan_filters_layout.addWidget(cb, row, col)
            self._xnat_scan_filter_checks[slot] = cb

    def _selected_xnat_scan_slots(self) -> set[str]:
        return {
            slot
            for slot, cb in self._xnat_scan_filter_checks.items()
            if cb.isChecked()
        }

    def _clear_xnat_scan_filters(self) -> None:
        for cb in self._xnat_scan_filter_checks.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self._filter_subjects()

    def _xnat_subjects_matching_scan_filter(self) -> list[str]:
        required = self._selected_xnat_scan_slots()
        if not required:
            return list(self._all_subjects)
        return filter_subjects_by_asset_slots(
            self._xnat_subject_slots,
            required,
            match_all=self._xnat_match_all.isChecked(),
        )

    def _filter_subjects(self) -> None:
        needle = self._subject_search.text().strip().lower()
        if self._is_xnat():
            candidates = self._xnat_subjects_matching_scan_filter()
            total = len(self._all_subjects)
        else:
            candidates = self._all_subjects
            total = len(candidates)

        self._subject_list.clear()
        for subj in candidates:
            if needle and needle not in subj.lower():
                continue
            self._subject_list.addItem(subj)
        shown = self._subject_list.count()

        if self._is_xnat():
            required = self._selected_xnat_scan_slots()
            if required:
                mode = "all" if self._xnat_match_all.isChecked() else "any"
                labels = ", ".join(asset_slot_display_label(s) for s in sorted(required))
                self._xnat_filter_count_label.setText(
                    f"{shown} of {total} subject(s) have {mode} of: {labels}"
                )
                self._status.setText(
                    f"{shown} / {total} subject(s) in {self._current_project_id()} "
                    f"(scan filter: {mode} of {len(required)} type(s))."
                )
            else:
                self._xnat_filter_count_label.setText(
                    f"{total} subject(s) in catalog — check scan types above to filter."
                )
                self._status.setText(f"{shown} subject(s) in {self._current_project_id()}.")
        elif self._local_roots is not None:
            self._status.setText(f"{shown} local subject(s).")

    def _on_subject_selected(
        self,
        current: QListWidgetItem | None,
        _previous: QListWidgetItem | None,
    ) -> None:
        self._resource_list.clear()
        if current is None:
            return
        subject = current.text()
        if self._is_xnat():
            self._populate_xnat_resources(subject)
        else:
            self._populate_local_resources(subject)

    def _populate_xnat_resources(self, subject_uid: str) -> None:
        project_id = self._current_project_id()
        try:
            df = list_scans_for_subject(self._repo, project_id, subject_uid)
        except Exception as exc:
            self._status.setText(f"Could not load scans: {exc}")
            return
        if df.empty:
            self._status.setText(f"No scans indexed for {subject_uid}. Run nvitk-xnat-sync first.")
            return
        for _, row in df.iterrows():
            scan_id = str(row.get("scan_id") or "")
            desc = str(row.get("series_description") or "")
            exp = str(row.get("experiment_label") or "")
            label = f"{scan_id} — {desc} ({exp})"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(_ITEM_ROLE, row.to_dict())
            self._resource_list.addItem(item)
        self._status.setText(f"{len(df)} scan(s) for {subject_uid}.")

    def _populate_local_resources(self, subject: str) -> None:
        if self._local_roots is None:
            self._status.setText("Apply pipeline roots first.")
            return
        self._refresh_resources_for_subject(subject)

    def _refresh_resources(self) -> None:
        item = self._subject_list.currentItem()
        if item is not None and not self._is_xnat():
            self._refresh_resources_for_subject(item.text())

    def _refresh_resources_for_subject(self, subject: str) -> None:
        assert self._local_roots is not None
        self._resource_list.clear()
        try:
            assets = list_local_assets(
                self._local_roots,
                subject,
                include_dicom=self._include_dicom.isChecked(),
                include_nifti=self._include_nifti.isChecked(),
                include_results=self._include_results.isChecked(),
            )
        except Exception as exc:
            self._status.setText(f"Could not list assets: {exc}")
            return
        for asset in assets:
            label = f"[{asset.kind}] {asset.label}"
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            item.setData(_ITEM_ROLE, asset)
            self._resource_list.addItem(item)
        self._status.setText(f"{len(assets)} asset(s) for {subject}.")

    def _selected_xnat_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i in range(self._resource_list.count()):
            item = self._resource_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            data = item.data(_ITEM_ROLE)
            if isinstance(data, dict):
                rows.append(dict(data))
        return rows

    def _selected_local_assets(self) -> list[LocalAsset]:
        assets: list[LocalAsset] = []
        for i in range(self._resource_list.count()):
            item = self._resource_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            data = item.data(_ITEM_ROLE)
            if isinstance(data, LocalAsset):
                assets.append(data)
        return assets

    def _on_action(self) -> None:
        if self._is_xnat():
            self._on_xnat_download()
        else:
            self._on_local_load()

    def _on_local_load(self) -> None:
        assets = self._selected_local_assets()
        if not assets:
            notify("Check at least one asset to load.", error=True)
            return
        opened = 0
        paths: list[str] = []
        for asset in assets:
            path = asset.path
            if not path.exists():
                notify(f"Missing: {path}", error=True)
                continue
            try:
                open_paths_with_nvitk(self._viewer, path)
                opened += 1
                paths.append(str(path))
            except Exception as exc:
                log.warning(f"Could not open {path}: {exc}")
        self._record_opened(paths)
        notify(f"Opened {opened} asset(s) in Napari.")

    def _on_xnat_download(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            notify("Download already in progress.", error=True)
            return

        config_text = self._config_path.text().strip()
        if not config_text:
            notify("Select an XNAT config file.", error=True)
            return
        config_path = Path(config_text).expanduser()
        if not config_path.is_file():
            notify(f"Config not found: {config_path}", error=True)
            return

        scan_rows = self._selected_xnat_rows()
        if not scan_rows:
            notify("Check at least one scan to download.", error=True)
            return

        password, ok = QInputDialog.getText(
            self,
            "XNAT password",
            "Enter XNAT password (not stored):",
            QLineEdit.Password,
        )
        if not ok:
            return

        use_temp = self._temp_only.isChecked() or not self._download_path.text().strip()
        download_root: Path | None = None
        temp_parent: Path | None = None
        if use_temp:
            if self._temp_session_root is None:
                self._temp_session_root = Path(tempfile.mkdtemp(prefix="nvitk_xnat_gui_"))
                roots = list(self._app_state.get("xnat_temp_dirs") or [])
                roots.append(str(self._temp_session_root))
                self._app_state["xnat_temp_dirs"] = roots
            temp_parent = self._temp_session_root
        else:
            download_root = Path(self._download_path.text().strip()).expanduser()
            download_root.mkdir(parents=True, exist_ok=True)

        self._action_btn.setEnabled(False)
        self._status.setText("Connecting to XNAT…")

        self._worker = _XnatDownloadWorker(
            config_path=config_path,
            project_id=self._current_project_id(),
            password=str(password),
            scan_rows=scan_rows,
            download_root=download_root,
            temp_parent=temp_parent,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_download_ok)
        self._worker.failed.connect(self._on_download_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_progress(self, message: str) -> None:
        self._status.setText(message)
        gui_log(message)

    def _record_opened(self, paths: list[str]) -> None:
        if self._on_inputs_opened is not None:
            self._on_inputs_opened(paths)
        else:
            inputs = list(self._app_state.get("inputs") or [])
            for p in paths:
                inputs.append({"path": p, "name": Path(p).name})
            self._app_state["inputs"] = inputs

    def _on_download_ok(self, dirs: list) -> None:
        paths: list[str] = []
        opened = 0
        for d in dirs:
            path = Path(d)
            if not path.is_dir():
                continue
            try:
                open_paths_with_nvitk(self._viewer, path)
                opened += 1
                paths.append(str(path))
            except Exception as exc:
                log.warning(f"Could not open {path}: {exc}")
        self._record_opened(paths)
        notify(f"Downloaded and opened {opened} DICOM series in Napari.")

    def _on_download_failed(self, message: str) -> None:
        self._status.setText(message)
        notify(f"XNAT download failed: {message}", error=True)

    def _on_worker_finished(self) -> None:
        self._action_btn.setEnabled(True)
        self._worker = None


# Backward-compatible alias
XnatDownloadPanel = DataBrowserPanel
