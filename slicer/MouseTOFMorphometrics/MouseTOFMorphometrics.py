"""3D Slicer ScriptedLoadableModule: TOF Circle-of-Willis morphometrics.

Runs the nvitk morphometrics pipeline (centerlines, radii, tortuosity,
stenosis/enlargement, volumetry) on a multilabel labelmap node and shows the
results as tables plus centerline/surface models.

Pairs with the **Mouse TOF CoW** module: its ``{volume}_tof_cow_trees`` output
uses labels 1/2/3, exactly what ``mouse_root_topology.json`` describes.

**Does not import nvitk.** The pipeline is vendored under
``MouseTOFMorphometricsLib/nvitk_vendor/`` — copied verbatim from the nvitk
source with only the root package renamed, so the measurements are exactly those
of the upstream pipeline. See
``MouseTOFMorphometricsLib/nvitk_vendor/VENDORED.md``.

Optional pip deps (Slicer already has numpy / scipy / vtk / matplotlib):
  pandas, nibabel, scikit-image, openpyxl
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile

import ctk
import numpy as np
import qt
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleWidget,
)
from slicer.util import VTKObservationMixin

_module_dir = os.path.dirname(os.path.abspath(__file__))
if _module_dir not in sys.path:
    sys.path.insert(0, _module_dir)

from MouseTOFMorphometricsLib import deps, mrml_io, results  # noqa: E402
from MouseTOFMorphometricsLib import morphometrics as morpho  # noqa: E402

#
# MouseTOFMorphometrics
#


class MouseTOFMorphometrics(ScriptedLoadableModule):
    """Module metadata for Additional Module Paths discovery."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "TOF CoW Morphometrics"
        self.parent.categories = ["nvitk"]
        self.parent.dependencies = []
        self.parent.contributors = ["nvitk"]
        self.parent.helpText = (
            "Run TOF Circle-of-Willis morphometrics on a multilabel labelmap: "
            "Taubin smoothing, skeletonisation, centerlines, radius/tortuosity, "
            "stenosis/enlargement and per-label volumetry.\n\n"
            "Pick a topology JSON to make the pipeline topology-aware. "
            "'mouse_root_topology.json' matches the Mouse TOF CoW output "
            "(1=Left ICA, 2=Right ICA, 3=Basilar) and declares species 'mouse', "
            "so each vessel's proximal end is found along the caudal direction "
            "(scanner A/P for a quadruped) rather than inferior.\n\n"
            "Self-contained: the pipeline is vendored, so no nvitk installation "
            "or source checkout is needed. Requires pandas, nibabel, "
            "scikit-image and openpyxl in Slicer's Python — use the "
            "'Install dependencies' button once.\n\n"
            "Caliber thresholds (stenosis, enlargement, taper) are calibrated on "
            "human anatomy — treat those columns as uncalibrated for mouse data."
        )
        self.parent.acknowledgementText = "nvitk TOF morphometrics (Slicer port)."
        iconPath = os.path.join(
            os.path.dirname(__file__), "Resources", "Icons", "MouseTOFMorphometrics.png"
        )
        if os.path.isfile(iconPath):
            self.parent.icon = qt.QIcon(iconPath)


#
# Logic
#


class MouseTOFMorphometricsLogic(ScriptedLoadableModuleLogic):
    """MRML marshalling plus the call into the vendored morphometrics pipeline."""

    # -- run ----------------------------------------------------------------
    def export_labelmap(self, labelNode, out_dir: str, *, bridge_gaps: bool = True):
        """Write the labelmap to a NIfTI for the pipeline; returns ``(path, n_bridged)``.

        Reconnects same-label fragments with MST tubes first, so a vessel split
        into fragments by segmentation noise is still measured as one tree.
        """
        seg = mrml_io.array_from_labelmap(labelNode)
        if seg.ndim != 3:
            raise ValueError(f"Morphometrics expects a 3D labelmap, got shape {seg.shape}.")
        if not np.any(seg):
            raise ValueError(f"Labelmap '{labelNode.GetName()}' is empty.")

        data = None
        n_bridged = 0
        if bridge_gaps:
            bridged = np.asarray(morpho.bridge_same_label_components(seg, max_gap=24), dtype=np.int32)
            n_bridged = int(np.count_nonzero(bridged != seg))
            if n_bridged:
                data = bridged

        # Named after the node so it becomes the workbook's case_id, rather than
        # every case being called "seg".
        stem = mrml_io.safe_filename(labelNode.GetName() or "labelmap")
        seg_path = os.path.join(out_dir, f"{stem}.nii.gz")
        mrml_io.write_labelmap_nifti(labelNode, seg_path, data=data)
        return seg_path, n_bridged

    def run(
        self,
        labelNode,
        *,
        output_dir: str = "",
        topology: str = "",
        species: str = "auto",
        input_already_smoothed: bool = True,
        skip_if_excel_exists: bool = False,
        bridge_gaps: bool = True,
        progress=None,
    ) -> dict:
        """Run morphometrics for one labelmap node. Returns a result summary dict."""
        deps.ensure()

        def _say(text: str) -> None:
            logging.info("MouseTOFMorphometrics: %s", text)
            if progress is not None:
                progress(text)
            slicer.app.processEvents()

        case_dir = (
            os.path.abspath(os.path.expanduser(output_dir))
            if str(output_dir).strip()
            else tempfile.mkdtemp(prefix="slicer_tof_morphometrics_")
        )
        os.makedirs(case_dir, exist_ok=True)

        _say("Exporting labelmap …")
        # Kept beside the results: it is the exact input that was measured, and it
        # carries the geometry needed to place the result models back in RAS.
        seg_path, n_bridged = self.export_labelmap(labelNode, case_dir, bridge_gaps=bridge_gaps)
        if n_bridged:
            _say(f"Bridged {n_bridged} voxel(s) to reconnect same-label fragments.")

        _say(f"Running morphometrics (topology={topology!r}, species={species!r}) …")
        excel = morpho.run_case(
            seg_path,
            case_dir,
            mapping_json=topology or morpho.none_topology(),
            case_out_dir_override=case_dir,
            # run_case parallelises with spawned subprocesses, which are not
            # reliable inside Slicer's embedded Python — always serial here.
            n_workers=1,
            input_already_smoothed=bool(input_already_smoothed),
            skip_if_excel_exists=bool(skip_if_excel_exists),
            species=species or "auto",
        )

        n_centerlines, n_surfaces = results.count_result_vtps(case_dir)
        return {
            "case_dir": case_dir,
            "vtp_to_ras": mrml_io.vtp_to_ras_matrix(seg_path),
            "excel": str(excel),
            "n_bridged": n_bridged,
            "n_centerlines": n_centerlines,
            "n_surfaces": n_surfaces,
            "persisted": bool(str(output_dir).strip()),
            "provenance": results.anatomy_provenance(case_dir),
        }


#
# Widget
#


class MouseTOFMorphometricsWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """UI: labelmap + topology/species selection, run, results tables, 3D models."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic: MouseTOFMorphometricsLogic | None = None
        self._resultNodes: list = []
        self._lastResult: dict | None = None

    # -- setup --------------------------------------------------------------
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = MouseTOFMorphometricsLogic()

        self._setupDependenciesSection()
        self._setupInputSection()
        self._setupOptionsSection()
        self._setupRunSection()
        self._setupResultsSection()

        self.layout.addStretch(1)
        self._refreshChoices()
        self._syncEnabled()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)

    def _setupDependenciesSection(self):
        self.depsCollapsible = ctk.ctkCollapsibleButton()
        self.depsCollapsible.text = "Dependencies"
        self.layout.addWidget(self.depsCollapsible)
        depsLayout = qt.QVBoxLayout(self.depsCollapsible)

        self.depsLabel = qt.QLabel("Checking …")
        self.depsLabel.setWordWrap(True)
        depsLayout.addWidget(self.depsLabel)

        self.installDepsButton = qt.QPushButton("Install dependencies")
        self.installDepsButton.toolTip = (
            "pip-install the required packages into Slicer's own Python "
            f"({deps.PIP_INSTALL_ARGS}). Needed once per Slicer installation. "
            "nvitk itself is not installed — the pipeline is vendored with this module."
        )
        depsLayout.addWidget(self.installDepsButton)
        self.installDepsButton.connect("clicked(bool)", self.onInstallDeps)

        self.recheckDepsButton = qt.QPushButton("Re-check")
        self.recheckDepsButton.toolTip = "Re-run the import check after installing or restarting."
        depsLayout.addWidget(self.recheckDepsButton)
        self.recheckDepsButton.connect("clicked(bool)", self.onRecheckDeps)

    def _setupInputSection(self):
        inputCollapsible = ctk.ctkCollapsibleButton()
        inputCollapsible.text = "Input"
        self.layout.addWidget(inputCollapsible)
        inputLayout = qt.QFormLayout(inputCollapsible)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLLabelMapVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = True
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip(
            "Multilabel vessel segmentation, e.g. '<volume>_tof_cow_trees' from Mouse TOF CoW."
        )
        inputLayout.addRow("Labelmap:", self.inputSelector)

        self.topologyCombo = qt.QComboBox()
        self.topologyCombo.setToolTip(
            "Vessel topology JSON. 'mouse_root_topology.json' matches Mouse TOF CoW "
            "labels 1/2/3 and declares species 'mouse'."
        )
        inputLayout.addRow("Topology JSON:", self.topologyCombo)

        self.speciesCombo = qt.QComboBox()
        self.speciesCombo.setToolTip(
            "Anatomical frame used to find each vessel's proximal end. 'auto' reads it "
            "from the topology's _meta block."
        )
        inputLayout.addRow("Species:", self.speciesCombo)

        self.outputDirEdit = ctk.ctkPathLineEdit()
        self.outputDirEdit.filters = ctk.ctkPathLineEdit.Dirs
        self.outputDirEdit.setToolTip(
            "Where the workbook, CSV and VTPs are written. Leave empty for a temporary folder."
        )
        inputLayout.addRow("Output directory:", self.outputDirEdit)

    def _setupOptionsSection(self):
        optionsCollapsible = ctk.ctkCollapsibleButton()
        optionsCollapsible.text = "Options"
        optionsCollapsible.collapsed = True
        self.layout.addWidget(optionsCollapsible)
        optionsLayout = qt.QFormLayout(optionsCollapsible)

        self.smoothedCheck = qt.QCheckBox()
        self.smoothedCheck.setChecked(True)
        self.smoothedCheck.setToolTip(
            "Measure the labelmap as-is. Uncheck to Taubin-smooth it first, which "
            "rounds off staircasing but also shrinks the mask."
        )
        optionsLayout.addRow("Input already Taubin-smoothed:", self.smoothedCheck)

        self.skipExistingCheck = qt.QCheckBox()
        self.skipExistingCheck.setToolTip("Reuse an existing workbook in the output directory.")
        optionsLayout.addRow("Skip if Excel exists:", self.skipExistingCheck)

        self.bridgeCheck = qt.QCheckBox()
        self.bridgeCheck.setChecked(True)
        self.bridgeCheck.setToolTip(
            "Reconnect same-label fragments with MST tubes before measuring, so a vessel\n"
            "broken into pieces by segmentation noise is still measured as one tree."
        )
        optionsLayout.addRow("Bridge same-label gaps:", self.bridgeCheck)

        self.loadSurfacesCheck = qt.QCheckBox()
        self.loadSurfacesCheck.setChecked(True)
        self.loadSurfacesCheck.setToolTip("Also load the vessel surface meshes (semi-transparent).")
        optionsLayout.addRow("Load surfaces in 3D:", self.loadSurfacesCheck)

    def _setupRunSection(self):
        runCollapsible = ctk.ctkCollapsibleButton()
        runCollapsible.text = "Run"
        self.layout.addWidget(runCollapsible)
        runLayout = qt.QVBoxLayout(runCollapsible)

        self.runButton = qt.QPushButton("Run morphometrics")
        self.runButton.toolTip = "Taubin smooth → skeletonise → centerlines → metrics → volumetry."
        runLayout.addWidget(self.runButton)

        buttonRow = qt.QHBoxLayout()
        self.showModelsButton = qt.QPushButton("Show centerlines in 3D")
        self.clearModelsButton = qt.QPushButton("Clear loaded models")
        self.openFolderButton = qt.QPushButton("Copy output path")
        for button in (self.showModelsButton, self.clearModelsButton, self.openFolderButton):
            buttonRow.addWidget(button)
        runLayout.addLayout(buttonRow)

        self.statusLabel = qt.QLabel("Idle — select a labelmap and run.")
        self.statusLabel.setWordWrap(True)
        runLayout.addWidget(self.statusLabel)

        self.runButton.connect("clicked(bool)", self.onRun)
        self.showModelsButton.connect("clicked(bool)", self.onShowModels)
        self.clearModelsButton.connect("clicked(bool)", self.onClearModels)
        self.openFolderButton.connect("clicked(bool)", self.onCopyOutputPath)

    def _setupResultsSection(self):
        resultsCollapsible = ctk.ctkCollapsibleButton()
        resultsCollapsible.text = "Results"
        self.layout.addWidget(resultsCollapsible)
        resultsLayout = qt.QVBoxLayout(resultsCollapsible)

        self.resultsTabs = qt.QTabWidget()
        self.vesselTable = self._makeTable()
        self.volumetryTable = self._makeTable()
        self.pathTable = self._makeTable()
        self.resultsTabs.addTab(self.vesselTable, "Per vessel")
        self.resultsTabs.addTab(self.volumetryTable, "Volumetry")
        self.resultsTabs.addTab(self.pathTable, "Per segment")
        resultsLayout.addWidget(self.resultsTabs)

        helpLabel = qt.QLabel(
            "Per vessel: mask volumetry (voxel count, volume, surface area) beside "
            "length-weighted centerline metrics. Per segment: one row per "
            "non-overlapping centerline segment — each piece of vessel is measured "
            "exactly once, so lengths sum to the real tree length. "
            "Volumetry is measured on the Taubin-smoothed mask, with the pipeline "
            "input volume alongside. Stenosis / enlargement / taper thresholds are "
            "human-calibrated — treat those columns as uncalibrated for mouse data."
        )
        helpLabel.setWordWrap(True)
        helpLabel.setStyleSheet("color: gray;")
        resultsLayout.addWidget(helpLabel)

    @staticmethod
    def _makeTable():
        table = qt.QTableWidget()
        table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.setMinimumHeight(180)
        return table

    def cleanup(self):
        self.onClearModels()
        self.removeObservers()

    def onSceneStartClose(self, caller, event):
        self._resultNodes = []
        self._lastResult = None

    # -- helpers ------------------------------------------------------------
    def _setStatus(self, text: str) -> None:
        self.statusLabel.setText(text)

    def _syncEnabled(self) -> None:
        ready = not deps.missing()
        has_result = bool(self._lastResult)
        self.runButton.setEnabled(ready)
        self.topologyCombo.setEnabled(ready)
        self.speciesCombo.setEnabled(ready)
        self.showModelsButton.setEnabled(has_result)
        self.openFolderButton.setEnabled(has_result)
        self.clearModelsButton.setEnabled(bool(self._resultNodes))
        self.installDepsButton.setEnabled(not ready)
        self.depsCollapsible.collapsed = ready

    def _refreshDepsLabel(self) -> None:
        absent = deps.missing()
        self.depsLabel.setText(
            deps.status_text()
            + ("" if absent else " The morphometrics pipeline is vendored with this module — nvitk is not required.")
        )
        self.depsLabel.setStyleSheet("color: #c0392b;" if absent else "color: gray;")
        self.installDepsButton.setText(
            "Install dependencies (" + ", ".join(absent) + ")" if absent else "Dependencies installed"
        )

    def _refreshChoices(self) -> None:
        """Populate the topology/species combos once dependencies are importable."""
        self._refreshDepsLabel()
        if deps.missing():
            self.topologyCombo.clear()
            self.speciesCombo.clear()
            self._setStatus("Install the missing Python packages to enable the pipeline.")
            return

        try:
            topologies = list(morpho.topology_choices())
            species = list(morpho.species_choices())
            default_topology = morpho.default_topology()
        except Exception as exc:  # noqa: BLE001
            self._setStatus(f"Vendored pipeline could not be loaded: {exc}")
            return

        previous = self.topologyCombo.currentText
        self.topologyCombo.clear()
        self.topologyCombo.addItems(topologies)
        wanted = previous if previous in topologies else default_topology
        self.topologyCombo.setCurrentIndex(max(0, topologies.index(wanted)))

        self.speciesCombo.clear()
        self.speciesCombo.addItems(species)
        self._setStatus("Ready — select a labelmap and run.")

    def _fillTable(self, table, headers, rows) -> None:
        table.clear()
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))
        table.setHorizontalHeaderLabels(list(headers))
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                table.setItem(r, c, qt.QTableWidgetItem(str(value)))
        table.resizeColumnsToContents()

    def _loadResultTables(self, case_dir: str) -> None:
        self._fillTable(self.vesselTable, *results.vessel_table(case_dir))
        self._fillTable(self.volumetryTable, *results.volumetry_table(case_dir))
        self._fillTable(self.pathTable, *results.path_summary_table(case_dir))

    # -- handlers -----------------------------------------------------------
    def onInstallDeps(self):
        installed: list[str] = []
        with slicer.util.tryWithErrorDisplay("Dependency install failed.", waitCursor=True):
            installed = deps.install()
        if installed:
            slicer.util.infoDisplay(
                "Installed: " + ", ".join(installed) + "\nRestart Slicer if imports still fail.",
                windowTitle="TOF CoW Morphometrics",
            )
        self._refreshChoices()
        self._syncEnabled()

    def onRecheckDeps(self):
        self._refreshChoices()
        self._syncEnabled()

    def onRun(self):
        labelNode = self.inputSelector.currentNode()
        if labelNode is None:
            slicer.util.warningDisplay(
                "Select a multilabel labelmap volume.", windowTitle="TOF CoW Morphometrics"
            )
            return

        try:
            with slicer.util.tryWithErrorDisplay("Morphometrics failed.", waitCursor=True):
                result = self.logic.run(
                    labelNode,
                    output_dir=str(self.outputDirEdit.currentPath or ""),
                    topology=self.topologyCombo.currentText,
                    species=self.speciesCombo.currentText or "auto",
                    input_already_smoothed=bool(self.smoothedCheck.checked),
                    skip_if_excel_exists=bool(self.skipExistingCheck.checked),
                    bridge_gaps=bool(self.bridgeCheck.checked),
                    progress=self._setStatus,
                )
        except Exception:
            self._setStatus("Failed — see the error dialog and the Python console.")
            return

        self._lastResult = result
        self._loadResultTables(result["case_dir"])
        self.onShowModels()

        provenance = result.get("provenance") or {}
        anatomy = ""
        if provenance:
            anatomy = (
                f" | species={provenance.get('species', '?')} "
                f"axcodes={provenance.get('orientation_axcodes', '?')} "
                f"length_scale={provenance.get('length_scale', '?')}"
            )
        where = result["case_dir"] if result["persisted"] else f"{result['case_dir']} (temporary)"
        self._setStatus(
            f"Done: {result['n_centerlines']} centerline(s), {result['n_surfaces']} surface(s) "
            f"→ {where}{anatomy}"
        )
        self._syncEnabled()

    def onShowModels(self):
        if not self._lastResult:
            return
        self.onClearModels()
        with slicer.util.tryWithErrorDisplay("Could not load result models.", waitCursor=True):
            self._resultNodes = mrml_io.load_result_models(
                self._lastResult["case_dir"],
                load_centerlines=True,
                load_surfaces=bool(self.loadSurfacesCheck.checked),
                matrix=self._lastResult.get("vtp_to_ras"),
            )
        self._syncEnabled()

    def onClearModels(self):
        mrml_io.remove_nodes(self._resultNodes)
        self._resultNodes = []
        self._syncEnabled()

    def onCopyOutputPath(self):
        if not self._lastResult:
            return
        path = self._lastResult["case_dir"]
        try:
            qt.QApplication.clipboard().setText(path)
            self._setStatus(f"Copied to clipboard: {path}")
        except Exception:  # noqa: BLE001
            self._setStatus(f"Output: {path}")
