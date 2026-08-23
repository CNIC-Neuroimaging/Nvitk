"""
Configure-and-run (or load) dialog for voxelwise analysis with FSL ``randomise``.

The tool needs a dozen inputs — a directory, an include pattern, an exclusion list, a cohort, a
list of EVs, a contrast table, a mask, a permutation count — which is far past what the Tools
panel's flat ``ParamSpec`` list can carry, so it opens its own window (the same escape the mouse
TOF CoW workflow takes).

The window's job beyond collecting those inputs is the **intersection readout**:

    412 images matched · 388 in cohort 4dflow_v3 · 371 complete cases

That is the single most common thing to get wrong, it is cheap to compute before anything is
merged, and once ``randomise`` has run for an hour a wrong count is indistinguishable from a right
one. Nothing here re-implements the analysis: the buttons build a ``nvitk-voxelwise`` command line
and stream it, so the GUI path and the CLI path cannot drift apart.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import shlex
from pathlib import Path
from typing import Any, Sequence

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.logger import Logger
from nvitk.gui.core.log_panel import gui_log, run_subprocess_logged

log = Logger()

DIALOG_OBJECT_NAME = "nvitk_voxelwise_dialog"

#: Measurement families the design frame can be built from, and a sensible feature for each.
PIPELINE_KINDS: tuple[tuple[str, str, str], ...] = (
    ("qvtpy", "flow_mean", "vessel"),
    ("asl", "mean_cbf", "territory"),
    ("t1", "t1_gray_volume", "region"),
    ("flair", "wmh_volume", "region"),
    ("tof", "diameter_mean", "vessel"),
)


# ──────────────────────────────────────────────────────────────────────────────
# Workers
# ──────────────────────────────────────────────────────────────────────────────
class _IntersectionWorker(QThread):
    """Compute ``images ∩ cohort ∩ complete cases`` without merging anything.

    Off the GUI thread because the design frame is several ``repo.image()`` queries — seconds on
    this dataset — and a frozen window during a routine parameter change trains people not to
    press the button that keeps them honest.
    """

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, params: dict[str, Any]) -> None:
        """Store a plain dict of parameters; the repo is only touched from :meth:`run`."""
        super().__init__()
        self._p = dict(params)

    def run(self) -> None:
        """Resolve, filter, and build the frame; emit ``(counts, frame_columns)``."""
        try:
            from nvitk.db.repo import get_repo
            from nvitk.measure.voxelwise import (
                DEFAULT_ID_NAMESPACES,
                apply_prefilters,
                build_design_frame,
                cohort_subjects,
                parse_prefilters,
                resolve_cohort_images,
            )

            repo = get_repo()
            images = resolve_cohort_images(
                self._p["image_dir"],
                repo=repo,
                include=self._p["include"] or "*",
                exclude_csv=self._p["exclude_csv"] or None,
                namespaces=DEFAULT_ID_NAMESPACES,
                on_duplicate=self._p.get("on_duplicate", "error"),
            )
            n_found = len(images)

            cohort = str(self._p["cohort"] or "").strip()
            if cohort:
                allowed = cohort_subjects(repo, cohort)
                images = [im for im in images if im.subject_uid in allowed]
            n_cohort = len(images)

            columns: list[str] = []
            n_complete = n_cohort
            frame = None
            rules = parse_prefilters(self._p.get("prefilters") or [])
            if cohort:
                frame, _meta = build_design_frame(
                    repo,
                    pipeline=cohort,
                    pipeline_kind=self._p["pipeline_kind"],
                    feature=self._p["feature"],
                    grouping=self._p["grouping"],
                    covariates=list(self._p["covariates"]) + [r.column for r in rules],
                )
                # The readout has to reflect the rules, or it promises a cohort the run will not
                # produce — which is the one thing this preview exists to prevent.
                frame = apply_prefilters(frame, rules)
                columns = sorted(str(c) for c in frame.columns)
                subjects = set(frame["subject_uid"].astype(str))
                n_complete = sum(1 for im in images if im.subject_uid in subjects)

            evs = [e for e in self._p["evs"] if e]
            problems = ""
            if evs and frame is not None:
                from nvitk.measure.voxelwise import VoxelwiseDesign, align_design_to_images

                design = VoxelwiseDesign(evs=tuple(evs), contrasts=())
                problems = design.validate(frame)
                if not problems or "EV(s) not in" not in problems:
                    try:
                        kept, _aligned = align_design_to_images(images, frame, design)
                        n_complete = len(kept)
                    except Exception as exc:  # noqa: BLE001
                        log.debug("Alignment preview failed: %s", exc)
        except Exception as exc:  # noqa: BLE001
            log.debug("Intersection preview failed: %s", exc, exc_info=True)
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(
            {
                "n_found": n_found,
                "n_cohort": n_cohort,
                "n_complete": n_complete,
                "cohort": cohort,
                "columns": columns,
                "problems": problems,
            }
        )


class _RunWorker(QThread):
    """Run one ``nvitk-voxelwise`` command line, streaming its output into the GUI log."""

    finished_ok = Signal(int)

    def __init__(self, argv: Sequence[str], env: dict[str, str] | None = None) -> None:
        """Store the argv and any extra environment; the subprocess starts in :meth:`run`."""
        super().__init__()
        self._argv = [str(a) for a in argv]
        self._env = dict(env) if env else None

    def run(self) -> None:
        """Stream the command and report its exit code."""
        self.finished_ok.emit(int(run_subprocess_logged(self._argv, env=self._env)))


# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────
class VoxelwisePanel(QDialog):
    """Configure and launch a voxelwise analysis, or load a finished results folder."""

    def __init__(self, viewer: Any, parent: QWidget | None = None) -> None:
        """Build the form, populate the cohort picker, and wire the buttons."""
        super().__init__(parent)
        self.setObjectName(DIALOG_OBJECT_NAME)
        self.setWindowTitle("Voxelwise analysis (FSL randomise)")
        self.setMinimumWidth(640)
        self._viewer = viewer
        self._worker: QThread | None = None
        self._runner: _RunWorker | None = None
        self._out_dir: Path | None = None

        outer = QVBoxLayout(self)
        outer.addWidget(self._build_source_group())
        outer.addWidget(self._build_design_group())
        outer.addWidget(self._build_analysis_group())

        self._intersection = QLabel("Press “Check cohort” to see how many subjects survive.")
        self._intersection.setWordWrap(True)
        self._intersection.setTextFormat(Qt.PlainText)
        outer.addWidget(self._intersection)

        self._status = QLabel(self._backend_line())
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        outer.addLayout(self._build_buttons())
        self._populate_cohorts()

    # -- construction ---------------------------------------------------------
    def _build_source_group(self) -> QGroupBox:
        """Image directory, include pattern, exclusion list and cohort."""
        box = QGroupBox("Images and cohort")
        form = QFormLayout(box)

        self._image_dir = QLineEdit()
        self._image_dir.setPlaceholderText("Flat directory of MNI-normalised volumes")
        form.addRow("Image directory", self._with_browse(self._image_dir, directory=True))

        self._include = QLineEdit("*")
        self._include.setToolTip("Glob on the file name, e.g. *_s8_* for 8 mm-smoothed volumes.")
        form.addRow("Include pattern", self._include)

        self._exclude_csv = QLineEdit()
        self._exclude_csv.setPlaceholderText("Optional: one id-glob per line (BMRI12345*)")
        form.addRow("Exclusion list", self._with_browse(self._exclude_csv, file_filter="*.csv *.txt"))

        self._from_source = QComboBox()
        for label, key in (
            ("local — upload the selected images", "local"),
            ("cluster — already there, upload nothing", "sge"),
        ):
            self._from_source.addItem(label, key)
        self._from_source.setToolTip(
            "Where the image directory and mask live, for a cluster submission.\n\n"
            "'local' resolves the cohort here and uploads only the volumes the design keeps — a "
            "few hundred out of however many are in the folder — into <output>/inputs/.\n"
            "'cluster' treats the paths as already visible to the nodes and sends nothing."
        )

        self._on_duplicate = QComboBox()
        for label, key in (
            ("error — list the conflicts", "error"),
            ("skip — drop those subjects", "skip"),
            ("first — keep the first by filename", "first"),
            ("last — keep the last by filename", "last"),
        ):
            self._on_duplicate.addItem(label, key)
        self._on_duplicate.setToolTip(
            "A voxelwise design has one row per subject. When two files resolve to the same "
            "subject — a repeat scan, or one session exported under two id namespaces — this "
            "decides what happens.\n"
            "Which session is kept changes the result, so the default reports the conflicts "
            "instead of choosing for you."
        )
        form.addRow("Duplicate subjects", self._on_duplicate)
        form.addRow("Input location (SGE)", self._from_source)

        self._cohort = QComboBox()
        self._cohort.setToolTip(
            "Subjects with measurements from this pipeline. Also the design matrix's source."
        )
        self._cohort.currentIndexChanged.connect(self._on_cohort_changed)
        form.addRow("Cohort (pipeline)", self._cohort)

        kinds = QHBoxLayout()
        self._pipeline_kind = QComboBox()
        for kind, _feature, _grouping in PIPELINE_KINDS:
            self._pipeline_kind.addItem(kind, kind)
        self._pipeline_kind.currentIndexChanged.connect(self._on_kind_changed)
        self._feature = QLineEdit("flow_mean")
        self._grouping = QLineEdit("vessel")
        for widget, label in ((self._pipeline_kind, "kind"), (self._feature, "feature"),
                              (self._grouping, "grouping")):
            kinds.addWidget(QLabel(label))
            kinds.addWidget(widget)
        form.addRow("Measurement", kinds)
        return box

    def _build_design_group(self) -> QGroupBox:
        """EV picker and contrast list."""
        box = QGroupBox("Design")
        layout = QVBoxLayout(box)

        layout.addWidget(QLabel("Explanatory variables (order is the design's):"))
        self._evs = QListWidget()
        self._evs.setSelectionMode(QAbstractItemView.MultiSelection)
        self._evs.setMaximumHeight(120)
        self._evs.itemSelectionChanged.connect(self._on_evs_changed)
        layout.addWidget(self._evs)

        self._ev_hint = QLabel("Pick a cohort and press “Check cohort” to list the columns.")
        self._ev_hint.setWordWrap(True)
        layout.addWidget(self._ev_hint)

        layout.addWidget(QLabel("Prefilters (subject inclusion rules, e.g. flow_mean__LICA>=15):"))
        pf_row = QHBoxLayout()
        self._prefilter_edit = QLineEdit()
        self._prefilter_edit.setPlaceholderText("flow_mean__LICA>=15")
        self._prefilter_edit.setToolTip(
            "Keep only subjects matching this rule. Repeatable; rules combine with AND.\n"
            "Operators: >=, <=, !=, ==, >, <. The column need not be an EV — filtering on a "
            "vessel you are not modelling is a cohort decision, not a covariate."
        )
        self._prefilter_edit.returnPressed.connect(self._add_prefilter)
        pf_add = QPushButton("Add")
        pf_add.clicked.connect(self._add_prefilter)
        pf_remove = QPushButton("Remove")
        pf_remove.clicked.connect(self._remove_prefilter)
        pf_row.addWidget(self._prefilter_edit, 1)
        pf_row.addWidget(pf_add)
        pf_row.addWidget(pf_remove)
        layout.addLayout(pf_row)

        self._prefilters = QListWidget()
        self._prefilters.setMaximumHeight(70)
        layout.addWidget(self._prefilters)

        layout.addWidget(QLabel("Contrasts (+ev:name, -ev:name, or weights 0,1,0:name):"))
        row = QHBoxLayout()
        self._contrast_edit = QLineEdit()
        self._contrast_edit.setPlaceholderText("+flow_mean__LMCA:mca_positive")
        self._contrast_edit.returnPressed.connect(self._add_contrast)
        add = QPushButton("Add")
        add.clicked.connect(self._add_contrast)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_contrast)
        row.addWidget(self._contrast_edit, 1)
        row.addWidget(add)
        row.addWidget(remove)
        layout.addLayout(row)

        self._contrasts = QListWidget()
        self._contrasts.setMaximumHeight(90)
        layout.addWidget(self._contrasts)
        return box

    def _build_analysis_group(self) -> QGroupBox:
        """Mask, permutations, TFCE and output folder."""
        box = QGroupBox("Analysis")
        form = QFormLayout(box)

        self._mask = QLineEdit()
        self._mask.setPlaceholderText("Optional — defaults to the MNI152 brain mask")
        form.addRow("Mask", self._with_browse(self._mask, file_filter="*.nii *.nii.gz"))

        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("Results folder")
        form.addRow("Output folder", self._with_browse(self._out_dir_edit, directory=True))

        row = QHBoxLayout()
        self._n_perm = QSpinBox()
        self._n_perm.setRange(0, 100000)
        self._n_perm.setSingleStep(500)
        self._n_perm.setValue(5000)
        self._n_perm.setToolTip("0 runs the exhaustive set of permutations.")
        self._tfce = QCheckBox("TFCE")
        self._tfce.setChecked(True)
        self._parallel = QCheckBox("randomise_parallel")
        row.addWidget(QLabel("permutations"))
        row.addWidget(self._n_perm)
        row.addWidget(self._tfce)
        row.addWidget(self._parallel)
        row.addStretch(1)
        form.addRow("", row)
        return box

    def _build_buttons(self) -> QHBoxLayout:
        """The action row: check, run, submit, load, close."""
        row = QHBoxLayout()
        self._check_btn = QPushButton("Check cohort")
        self._check_btn.clicked.connect(self._check_intersection)
        self._run_btn = QPushButton("Run locally")
        self._run_btn.clicked.connect(lambda: self._launch("local"))
        self._sge_btn = QPushButton("Submit to SGE")
        self._sge_btn.clicked.connect(lambda: self._launch("sge"))
        load = QPushButton("Load results folder…")
        load.clicked.connect(self._load_results)
        self._show_3d = QPushButton("3D scene…")
        self._show_3d.setToolTip(
            "Open the 3-D window: suprathreshold voxels inside a translucent brain shell.\n"
            "The flat map layers answer whether there is an effect; this answers where it is.\n"
            "Map, contrasts, threshold and appearance are all configured there."
        )
        self._show_3d.clicked.connect(self._show_results_3d)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        for widget in (self._check_btn, self._run_btn, self._sge_btn, load, self._show_3d):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(close)
        return row

    def _with_browse(self, edit: QLineEdit, *, directory: bool = False,
                     file_filter: str = "") -> QWidget:
        """Wrap *edit* with a Browse… button that fills it in."""
        container = QWidget()
        row = QHBoxLayout(container)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        button = QPushButton("Browse…")

        def pick() -> None:
            """Open the right chooser for this field and write the result back."""
            if directory:
                chosen = QFileDialog.getExistingDirectory(self, "Select folder", edit.text())
            else:
                chosen, _ = QFileDialog.getOpenFileName(
                    self, "Select file", edit.text(), file_filter or "All files (*)"
                )
            if chosen:
                edit.setText(chosen)

        button.clicked.connect(pick)
        row.addWidget(button)
        return container

    # -- state ----------------------------------------------------------------
    def _backend_line(self) -> str:
        """One line describing whether FSL can run here."""
        from nvitk.measure.voxelwise import fsl_backend_status

        status = fsl_backend_status()
        if status.available:
            return f"FSL: {status.summary()}"
        return f"FSL unavailable — {status.reason}. Local runs will fail; SGE may still work."

    def _populate_cohorts(self) -> None:
        """Fill the cohort combo with the selectable pipeline ids and their subject counts."""
        try:
            from nvitk.db.repo import get_repo
            from nvitk.measure.voxelwise import available_cohorts

            options = [o for o in available_cohorts(get_repo()) if o.registered]
        except Exception as exc:  # noqa: BLE001
            log.debug("Cohort listing failed: %s", exc)
            self._cohort.addItem("(dataset unavailable)", "")
            return
        self._cohort.addItem("(none)", "")
        for option in options:
            self._cohort.addItem(
                f"{option.pipeline_id} — {option.n_subjects} subject(s)", option.pipeline_id
            )

    def _on_cohort_changed(self) -> None:
        """A new cohort invalidates the EV list, which came from the old one's frame."""
        self._evs.clear()
        self._ev_hint.setText("Cohort changed — press “Check cohort” to reload the columns.")

    def _on_kind_changed(self) -> None:
        """Default the feature and grouping to something valid for the chosen family."""
        kind = str(self._pipeline_kind.currentData() or "")
        for name, feature, grouping in PIPELINE_KINDS:
            if name == kind:
                self._feature.setText(feature)
                self._grouping.setText(grouping)
                return

    def _on_evs_changed(self) -> None:
        """Keep the hint showing the design's column order as it is built."""
        evs = self.selected_evs()
        if evs:
            self._ev_hint.setText("Design columns: intercept, " + ", ".join(evs))

    def selected_evs(self) -> list[str]:
        """The chosen EV names, in the list's order."""
        return [item.text() for item in self._evs.selectedItems()]

    def contrasts(self) -> list[str]:
        """The contrast tokens currently in the list."""
        return [self._contrasts.item(i).text() for i in range(self._contrasts.count())]

    def prefilters(self) -> list[str]:
        """The prefilter rules currently in the list."""
        return [self._prefilters.item(i).text() for i in range(self._prefilters.count())]

    def _add_prefilter(self) -> None:
        """Append the edit's text as a prefilter, rejecting it here if it will not parse.

        Validated on entry rather than at run time: a typo'd rule is far cheaper to fix while the
        user is still looking at the field than after the intersection has been recomputed.
        """
        text = self._prefilter_edit.text().strip()
        if not text:
            return
        try:
            from nvitk.measure.voxelwise import PreFilter

            PreFilter.parse(text)
        except ValueError as exc:
            QMessageBox.warning(self, "Voxelwise", str(exc))
            return
        self._prefilters.addItem(QListWidgetItem(text))
        self._prefilter_edit.clear()

    def _remove_prefilter(self) -> None:
        """Drop the selected prefilter rule(s)."""
        for item in self._prefilters.selectedItems():
            self._prefilters.takeItem(self._prefilters.row(item))

    def _add_contrast(self) -> None:
        """Append the edit's text as a contrast token."""
        text = self._contrast_edit.text().strip()
        if text:
            self._contrasts.addItem(QListWidgetItem(text))
            self._contrast_edit.clear()

    def _remove_contrast(self) -> None:
        """Drop the selected contrast token(s)."""
        for item in self._contrasts.selectedItems():
            self._contrasts.takeItem(self._contrasts.row(item))

    # -- intersection ---------------------------------------------------------
    def _check_intersection(self) -> None:
        """Recompute the images ∩ cohort ∩ complete-cases counts in the background."""
        image_dir = self._image_dir.text().strip()
        if not image_dir:
            QMessageBox.warning(self, "Voxelwise", "Choose an image directory first.")
            return
        self._check_btn.setEnabled(False)
        self._intersection.setText("Checking…")
        worker = _IntersectionWorker(
            {
                "image_dir": image_dir,
                "include": self._include.text().strip(),
                "exclude_csv": self._exclude_csv.text().strip(),
                "cohort": self._cohort.currentData(),
                "pipeline_kind": str(self._pipeline_kind.currentData() or "qvtpy"),
                "feature": self._feature.text().strip() or "flow_mean",
                "grouping": self._grouping.text().strip() or "vessel",
                "covariates": self.selected_evs(),
                "evs": self.selected_evs(),
                "on_duplicate": str(self._on_duplicate.currentData() or "error"),
                "prefilters": self.prefilters(),
            }
        )
        worker.finished_ok.connect(self._on_intersection)
        worker.failed.connect(self._on_intersection_failed)
        worker.finished.connect(lambda: self._check_btn.setEnabled(True))
        self._worker = worker
        worker.start()

    def _on_intersection(self, payload: dict[str, Any]) -> None:
        """Show the counts and refresh the EV list from the frame's columns."""
        cohort = payload["cohort"] or "no cohort filter"
        self._intersection.setText(
            f"{payload['n_found']} images matched · {payload['n_cohort']} in cohort {cohort} · "
            f"{payload['n_complete']} complete cases"
            + (f"\n{payload['problems']}" if payload["problems"] else "")
        )
        columns = payload["columns"]
        if columns:
            chosen = set(self.selected_evs())
            self._evs.clear()
            for column in columns:
                if column in {"subject_uid", "group_key", "patient_id"}:
                    continue
                item = QListWidgetItem(column)
                self._evs.addItem(item)
                if column in chosen:
                    item.setSelected(True)
            self._ev_hint.setText(f"{self._evs.count()} column(s) available as EVs.")

    def _on_intersection_failed(self, message: str) -> None:
        """Report why the preview could not be computed."""
        self._intersection.setText(f"Could not check: {message}")

    # -- launching ------------------------------------------------------------
    def build_argv(self, submit: str) -> list[str]:
        """The ``nvitk-voxelwise run`` command line for the current form state.

        Built rather than calling the library directly so that what the GUI runs is exactly what a
        user could paste into a terminal — and so the log shows them that command.
        """
        image_dir = self._image_dir.text().strip()
        out_dir = self._out_dir_edit.text().strip()
        evs = self.selected_evs()
        if not image_dir:
            raise ValueError("Choose an image directory.")
        if not out_dir:
            raise ValueError("Choose an output folder.")
        if not evs:
            raise ValueError("Select at least one EV.")
        cohort = str(self._cohort.currentData() or "")
        if not cohort:
            raise ValueError("Choose a cohort — it is the design matrix's measurement source.")

        argv = [
            "nvitk-voxelwise", "run",
            "--image-dir", image_dir,
            "--include", self._include.text().strip() or "*",
            "--cohort", cohort,
            "--pipeline-kind", str(self._pipeline_kind.currentData() or "qvtpy"),
            "--feature", self._feature.text().strip() or "flow_mean",
            "--grouping", self._grouping.text().strip() or "vessel",
            "-o", out_dir,
            "--n-perm", str(int(self._n_perm.value())),
            "--submit", submit,
        ]
        argv += ["--on-duplicate", str(self._on_duplicate.currentData() or "error")]
        if submit == "sge":
            argv += ["--from-source", str(self._from_source.currentData() or "local")]
        if self._exclude_csv.text().strip():
            argv += ["--exclude-csv", self._exclude_csv.text().strip()]
        if self._mask.text().strip():
            argv += ["--mask", self._mask.text().strip()]
        for ev in evs:
            argv += ["--ev", ev]
        for contrast in self.contrasts():
            argv += ["--contrast", contrast]
        for rule in self.prefilters():
            argv += ["--prefilter", rule]
        argv.append("--tfce" if self._tfce.isChecked() else "--no-tfce")
        if self._parallel.isChecked():
            argv.append("--parallel")
        return argv

    def _launch(self, submit: str) -> None:
        """Start the analysis (or the submission) and stream it into the log dock."""
        try:
            argv = self.build_argv(submit)
        except ValueError as exc:
            QMessageBox.warning(self, "Voxelwise", str(exc))
            return
        if self._runner is not None and self._runner.isRunning():
            QMessageBox.information(self, "Voxelwise", "A run is already in progress.")
            return

        # The CLI would prompt for a password; a subprocess has no tty, so it would hang with
        # nothing on screen. Collect the credentials here and hand them over as environment
        # variables, which the CLI reads before it considers prompting.
        env: dict[str, str] | None = None
        if submit == "sge":
            env = self._collect_ssh_env()
            if env is None:
                return

        self._out_dir = Path(self._out_dir_edit.text().strip())
        gui_log(f"Voxelwise: {' '.join(shlex.quote(a) for a in argv)}")
        self._run_btn.setEnabled(False)
        self._sge_btn.setEnabled(False)
        self._status.setText(
            "Running…" if submit == "local" else "Submitting…"
        )
        runner = _RunWorker(argv, env=env)
        runner.finished_ok.connect(lambda rc: self._on_run_finished(rc, submit))
        self._runner = runner
        runner.start()

    def _collect_ssh_env(self) -> dict[str, str] | None:
        """Ask for cluster credentials and return the child's environment, or ``None`` if cancelled.

        Reuses the GUI's existing SSH form rather than growing a second one. Already-set
        ``NVITK_SGE_SSH_*`` variables are inherited and pre-fill it, so a session that exported
        them is not asked twice.
        """
        import os

        try:
            from nvitk.gui.sge.dialog import SgeSubmitDialog
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Voxelwise", f"The SSH credentials form is unavailable:\n{exc}"
            )
            return None

        dialog = SgeSubmitDialog(self)
        if os.environ.get("NVITK_SGE_SSH_HOST"):
            dialog.host.setText(os.environ["NVITK_SGE_SSH_HOST"])
        if os.environ.get("NVITK_SGE_SSH_USER"):
            dialog.user.setText(os.environ["NVITK_SGE_SSH_USER"])
        if not dialog.exec_():
            return None

        host = dialog.host.text().strip()
        user = dialog.user.text().strip()
        password = dialog.password.text()
        if not (host and user and password):
            QMessageBox.warning(self, "Voxelwise", "Host, user and password are all required.")
            return None
        return {
            **os.environ,
            "NVITK_SGE_SSH_HOST": host,
            "NVITK_SGE_SSH_USER": user,
            "NVITK_SGE_SSH_PASSWORD": password,
        }

    def _on_run_finished(self, returncode: int, submit: str) -> None:
        """Re-enable the buttons and, for a finished local run, show the maps."""
        self._run_btn.setEnabled(True)
        self._sge_btn.setEnabled(True)
        if returncode != 0:
            self._status.setText(f"Failed (exit {returncode}) — see the log.")
            return
        if submit != "local":
            self._status.setText("Submitted. Load the results folder when the job finishes.")
            return
        self._status.setText("Finished.")
        if self._out_dir is not None:
            self._show_results(self._out_dir)

    # -- results --------------------------------------------------------------
    def _load_results(self) -> None:
        """Pick a finished results folder and display it — no run, no database."""
        chosen = QFileDialog.getExistingDirectory(
            self, "Select a voxelwise results folder", self._out_dir_edit.text()
        )
        if chosen:
            self._show_results(Path(chosen))

    def _show_results_3d(self, *_args: Any) -> None:
        """Open the 3-D configuration window, pointed at the current results folder if there is one.

        No separate load step: the window reads the folder as soon as it has one, fills its pickers
        from what is in it, and draws on demand.
        """
        directory = self._out_dir or (self._out_dir_edit.text().strip() or None)
        try:
            from nvitk.gui.viz.voxelwise_3d_panel import start_voxelwise_3d

            start_voxelwise_3d(self._viewer, directory)
        except Exception as exc:  # noqa: BLE001
            log.debug("Voxelwise 3-D window failed: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Voxelwise", f"Could not open the 3-D window:\n{exc}")

    def _show_results(self, out_dir: Path) -> None:
        """Add the corrected maps to the viewer over the MNI template."""
        try:
            add_result_layers(self._viewer, out_dir)
        except Exception as exc:  # noqa: BLE001
            log.debug("Loading voxelwise result failed: %s", exc, exc_info=True)
            QMessageBox.critical(self, "Voxelwise", f"Could not load results:\n{exc}")
            return
        self._status.setText(f"Loaded {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Napari layers
# ──────────────────────────────────────────────────────────────────────────────
def add_result_layers(
    viewer: Any,
    out_dir: str | Path,
    *,
    kind: str = "",
    alpha: float = 0.05,
    with_template: bool = True,
) -> list[Any]:
    """Add a result's corrected maps to *viewer*, named by contrast, over the MNI template.

    Each map is shown with its contrast limits starting at the significance threshold, so the
    colour ramp spans *only* the range that means anything: a voxel at 1 − p = 0.6 is not a faint
    result, it is no result, and giving it a faint colour would say otherwise.
    """
    from nvitk.gui.io.napari_io import open_paths_with_nvitk
    from nvitk.measure.voxelwise import load_voxelwise_result
    from nvitk.stats.voxelwise_map import alpha_to_map_threshold, is_corrp_kind

    result = load_voxelwise_result(out_dir)
    kind = str(kind or result.primary_kind())
    added: list[Any] = []

    if with_template and not any(
        getattr(layer, "name", "") == "MNI152 template" for layer in viewer.layers
    ):
        try:
            from nilearn.datasets import load_mni152_template

            template_path = Path(load_mni152_template().get_filename() or "")
            if template_path.is_file():
                for layer in open_paths_with_nvitk(viewer, template_path):
                    layer.name = "MNI152 template"
                    layer.colormap = "gray"
                    added.append(layer)
        except Exception as exc:  # noqa: BLE001
            log.debug("MNI template unavailable as a base layer: %s", exc)

    cut = alpha_to_map_threshold(alpha)
    for name in result.contrast_names:
        try:
            path = result.map_path(kind, name)
        except KeyError:
            continue
        for layer in open_paths_with_nvitk(viewer, path):
            layer.name = f"{name} ({kind})"
            if is_corrp_kind(kind):
                layer.colormap = "hot"
                layer.contrast_limits = (cut, 1.0)
                layer.opacity = 0.85
            added.append(layer)

    gui_log(
        f"Voxelwise: {len(result.contrast_names)} contrast(s) from {Path(out_dir).name} "
        f"as {kind} map(s), thresholded at p < {alpha:g}."
    )
    return added


def start_voxelwise(viewer: Any) -> VoxelwisePanel:
    """Open the voxelwise dialog for *viewer* (reusing an open one)."""
    for widget in viewer.window._qt_window.findChildren(QDialog):
        if widget.objectName() == DIALOG_OBJECT_NAME:
            widget.show()
            widget.raise_()
            return widget  # type: ignore[return-value]
    panel = VoxelwisePanel(viewer, parent=viewer.window._qt_window)
    panel.show()
    return panel


__all__ = [
    "DIALOG_OBJECT_NAME",
    "VoxelwisePanel",
    "add_result_layers",
    "start_voxelwise",
]
