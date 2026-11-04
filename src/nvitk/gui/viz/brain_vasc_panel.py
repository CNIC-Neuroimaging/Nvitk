"""
Dock for the brain-vascular map: inputs, optional registration, and the clicked vessel's regions.

The pipeline, from the top of the dock down:

1. **Vessel mask**: a multilabel segmentation (eICAB / TopBrain / QVTPy). Its label vocabulary
   names the arteries; the guess from the layer can be overridden.
2. **Parcellation** of an input image (ANTsPyNet DKT or FastSurfer on a T1w MRI,
   TotalSegmentator's brain structures on a CT), or an existing label layer; as Desikan-
   Killiany(-Tourville) regions or collapsed to lobes.
3. **Registration** (off by default): the input image registered to the image the vessels were
   segmented from (FLIRT or ANTs, linear), and the parcellation carried across as a world-space
   matrix, so a whole-brain parcellation is not cropped to an angiography slab.
4. **Territories** (:func:`nvitk.measure.vascular_territory.map_territories`) and the 3D scene
   (:mod:`nvitk.gui.viz.brain_vasc_map`).

The heavy steps run on a worker thread (:func:`~nvitk.gui.core.activity.run_in_background`): the
window keeps repainting and the status bar shows which step is running. Layers are read and
added on the GUI thread, inside :func:`~nvitk.gui.viz.main_canvas.world_geometry`, so the
geometry is the real one whatever the canvas resampling mode.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.gui.core.design import COLOR_MUTED, Card
from nvitk.gui.core.layer_folders import FOLDER_KEY
from nvitk.gui.viz.brain_vasc_map import (
    DEFAULT_DIM_OPACITY,
    DEFAULT_MIN_FRACTION,
    DEFAULT_REGION_OPACITY,
    DEFAULT_SHELL_OPACITY,
    FOLDER,
    REGION_TYPES,
    BrainVascScene,
    build_scene_data,
    current_scene,
)

log = Logger()

DOCK_OBJECT_NAME = "nvitk_brain_vasc_dock"
DOCK_TITLE = "Brain-vascular map"

#: Parcellation sources: (label, key).
SOURCES: tuple[tuple[str, str], ...] = (
    ("ANTsPyNet DKT — T1w MRI", "dkt"),
    ("FastSurfer — T1w MRI", "fastsurfer"),
    ("TotalSegmentator brain structures — CT (licensed)", "totalseg"),
    ("Existing label layer", "existing"),
)
#: Region schemes offered per source: (label, scheme key).
_REGIONS_APARC = (("Desikan-Killiany-Tourville regions", "aparc"), ("Lobes", "lobes"))
_REGIONS_BY_SOURCE: dict[str, tuple[tuple[str, str], ...]] = {
    "dkt": _REGIONS_APARC,
    "fastsurfer": _REGIONS_APARC,
    "totalseg": (("Brain structures (lobes + deep grey)", "brain_structures"),),
    "existing": (
        ("Desikan-Killiany (aparc+aseg ids)", "aparc"),
        ("Lobes (from aparc+aseg ids)", "lobes"),
        ("As labelled", "generic"),
    ),
}
FLIRT_DOF_CHOICES: tuple[tuple[str, int], ...] = (
    ("6 — rigid", 6), ("7 — rigid + scale", 7), ("9 — traditional", 9), ("12 — affine", 12),
)
ANTS_LINEAR_TRANSFORMS: tuple[str, ...] = ("Rigid", "Similarity", "Affine")
_AUTO = "auto"


# ──────────────────────────────────────────────────────────────────────────────
# Layer helpers
# ──────────────────────────────────────────────────────────────────────────────


def _data_to_world(layer: Any) -> np.ndarray:
    """The layer's full data -> world 4x4 (scale, translate, affine...), spatial block."""
    from nvitk.gui.tools.interpolation_tools import layer_data_to_world

    return np.asarray(layer_data_to_world(layer), dtype=float)[-4:, -4:]


def _is_label_layer(layer: Any) -> bool:
    from nvitk.gui.labels.visibility import is_label_like_layer

    return type(layer).__name__ == "Labels" or is_label_like_layer(layer)


def _is_image_layer(layer: Any) -> bool:
    return type(layer).__name__ == "Image" and not _is_label_layer(layer)


def _is_parcellation_candidate(layer: Any) -> bool:
    """Labels, label-like images, and integer images: a DKT / aparc+aseg map has ~100 ids, more
    than nvitk calls "label-like", so it opens as an integer Image layer."""
    if _is_label_layer(layer):
        return True
    dtype = getattr(getattr(layer, "data", None), "dtype", None)
    return type(layer).__name__ == "Image" and dtype is not None and np.issubdtype(dtype, np.integer)


def _label_array(layer: Any) -> np.ndarray:
    data = np.asarray(to_numpy(layer.data))
    if not np.issubdtype(data.dtype, np.integer):
        data = np.rint(data)
    return data.astype(np.int32)


def _vessel_names(layer: Any, ids: list[int], schema_key: str) -> dict[int, str]:
    """Label names: hand-given ones first, then *schema_key* (or the layer's own vocabulary)."""
    from nvitk.gui.labels.catalog import custom_label_names, get_schema, label_name

    custom = custom_label_names(layer)
    schema = get_schema(schema_key) if schema_key and schema_key != _AUTO else None
    out: dict[int, str] = {}
    for lid in ids:
        name = custom.get(lid) or (schema.name_for(lid) if schema is not None else label_name(layer, lid))
        if name:
            out[lid] = str(name)
    return out


def _label_colours(layer: Any, ids: list[int]) -> dict[int, tuple[float, float, float, float]]:
    """Each label's colour on *layer* (napari Labels), else a qualitative palette."""
    import matplotlib

    palette = matplotlib.colormaps["tab20"]
    out: dict[int, tuple[float, float, float, float]] = {}
    for i, lid in enumerate(ids):
        colour = None
        if type(layer).__name__ == "Labels":
            try:
                colour = layer.get_color(int(lid))
            except Exception:  # noqa: BLE001
                colour = None
        if colour is None or np.asarray(colour).size < 3:
            colour = palette(i % 20)
        c = np.asarray(colour, dtype=float).ravel()
        out[int(lid)] = (float(c[0]), float(c[1]), float(c[2]), 1.0)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class MapRequest:
    """Everything the dock's form says, resolved to layers."""

    vessel_layer: Any
    vessel_schema: str
    source: str
    scheme: str
    input_layer: Any | None
    parcellation_layer: Any | None
    register: bool
    algorithm: str
    cost: str
    dof: int
    ants_transform: str
    target_layer: Any | None
    target_mm: float
    follow_collaterals: bool
    respect_hemispheres: bool


def _add_labels_like(viewer: Any, data: np.ndarray, like: Any, name: str) -> Any:
    """A hidden Labels layer of *data* placed like *like*, in the map's folder."""
    from nvitk.gui.core.spatial import layer_spatial_kwargs

    return viewer.add_labels(np.asarray(data, dtype=np.int32), name=name, visible=False, opacity=0.6,
                             metadata={FOLDER_KEY: FOLDER}, **layer_spatial_kwargs(like))


def _file_labels(viewer: Any, path: Path, name: str) -> Any:
    """A label file opened with nvitk's reader (its own placement), hidden, in the map's folder."""
    from nvitk.gui.tools.runner import _add_volume_file

    layer = _add_volume_file(viewer, path, name, labels=True)
    layer.visible = False
    layer.metadata = {**dict(layer.metadata or {}), FOLDER_KEY: FOLDER}
    return layer


def _run_parcellation(viewer: Any, req: MapRequest, task: Any, scratch: Path) -> tuple[Any, str]:
    """The parcellation layer the request asks for, computed when needed; ``(layer, scheme)``."""
    from nvitk.gui.core.activity import run_in_background
    from nvitk.gui.tools.runner import _ensure_nifti_path

    if req.source == "existing":
        if req.parcellation_layer is None:
            raise ValueError("Pick the existing parcellation layer.")
        # "Lobes" of an existing layer means lobes collapsed from its aparc+aseg ids.
        return req.parcellation_layer, ("aparc" if req.scheme == "lobes" else req.scheme)

    image = req.input_layer
    if image is None:
        raise ValueError("Pick the image to parcellate.")

    if req.source == "dkt":
        from nvitk.core.backend import using
        from nvitk.gui.core.spatial import layer_to_image
        from nvitk.segmentation.dkt import desikan_killiany_tourville_labeling

        task.update(message="Brain-vascular map: ANTsPyNet DKT parcellation…")
        img = layer_to_image(image)

        def dkt() -> np.ndarray:
            with using("numpy"):
                return np.asarray(to_numpy(desikan_killiany_tourville_labeling(img)))

        labels = run_in_background(dkt)
        return _add_labels_like(viewer, labels, image, f"{image.name}_dkt"), "aparc"

    if req.source == "fastsurfer":
        from nvitk.segmentation.fastsurfer import FastSurferRequest, fastsurfer_argv, run

        task.update(message="Brain-vascular map: FastSurfer segmentation…")
        t1 = _ensure_nifti_path(image, prefix="bvm_fastsurfer")
        # FastSurfer's "min" conforms to the input's finest spacing, and it is only trained down to
        # 0.7 mm: a 0.25 mm TOF became an 800³ volume (an ~80 GB GPU allocation). Finer inputs are
        # segmented at 1 mm; the parcellation only needs ~1 mm anyway (the working grid).
        finest = float(np.linalg.norm(_data_to_world(image)[:3, :3], axis=0).min())
        vox_size = "min" if finest >= 0.7 else "1"
        from nvitk.gui.tools.fastsurfer_tools import _gpu_device

        device = _gpu_device()  # follows nvitk's GPU switch, like the FastSurfer tools
        argv, env = fastsurfer_argv(FastSurferRequest(
            sd=scratch, sid="subject", t1=t1, cereb=False, hypothal=False, cc=False,
            device=device, viewagg_device=device,
            # Only aparc.DKTatlas+aseg is used: no bias field correction or partial-volume stats.
            biasfield=False, vox_size=vox_size,
        ))
        tail: list[str] = []

        def keep(line: str) -> None:
            tail.append(line)
            del tail[:-20]
            log.info(f"FastSurfer: {line}")

        def fatal(line: str) -> bool:
            """The segmentation failed: stop instead of letting the script run on without it."""
            return "[CRITICAL" in line or "INSUFFICIENT GPU MEMORY" in line or "Traceback (most recent" in line

        code = run_in_background(run, argv, env, on_line=keep, stop_when=fatal)
        out = scratch / "subject" / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
        if code != 0 or not out.is_file():
            hint = ""
            if any("GPU MEMORY" in line for line in tail):
                hint = "\n\nOut of GPU memory: switch nvitk's GPU off to segment on the CPU (slower, but it fits)."
            raise RuntimeError(f"FastSurfer failed (exit {code}):\n" + "\n".join(tail[-8:]) + hint)
        if any("may be corrupted" in line or "too small" in line for line in tail):
            # FastSurfer's own volume QC: expected on an angiography slab or a non-T1w contrast.
            from nvitk.gui.tools.runner import notify

            notify("FastSurfer QC: the segmented brain volume is too small (partial field of view or a "
                   "non-T1w image?). The map will only cover what was segmented; a whole-brain T1w with "
                   "registration gives a complete parcellation.", error=True)
        return _file_labels(viewer, out, f"{image.name}_fastsurfer_aparc"), "aparc"

    if req.source == "totalseg":
        from nvitk.gui.core.backend import gpu_enabled
        from nvitk.gui.tools.runner import _totalseg_model_dir
        from nvitk.segmentation.brain_parcellation import TOTALSEG_BILATERAL, lateralize_by_midline
        from nvitk.segmentation.total_segmentator import run_totalsegmentator

        task.update(message="Brain-vascular map: TotalSegmentator brain structures…")
        inp = _ensure_nifti_path(image, prefix="bvm_totalseg")
        out = scratch / "brain_structures.nii.gz"
        proc = run_in_background(
            run_totalsegmentator, inp, out, "brain_structures", device="gpu" if gpu_enabled() else "cpu",
            multilabel=True, statistics=False, model_dir=_totalseg_model_dir(), check=False, capture_output=True,
        )
        if getattr(proc, "returncode", 1) != 0 or not out.is_file():
            detail = str(getattr(proc, "stderr", "") or getattr(proc, "stdout", "") or "").strip().splitlines()[-8:]
            raise RuntimeError("TotalSegmentator brain_structures failed (it needs a licence):\n" + "\n".join(detail))
        layer = _file_labels(viewer, out, f"{image.name}_brain_structures")
        lateral = run_in_background(lambda: np.asarray(to_numpy(lateralize_by_midline(
            _label_array(layer), _data_to_world(layer), sorted(TOTALSEG_BILATERAL)))))
        layer.data = lateral.astype(np.int32)
        return layer, "brain_structures"

    raise ValueError(f"Unknown parcellation source {req.source!r}.")


def _run_registration(req: MapRequest, task: Any, scratch: Path, viewer: Any) -> np.ndarray:
    """Input-image world -> vessel world, as a 4x4 (registration on its own grids)."""
    from nvitk.gui.core.activity import run_in_background
    from nvitk.gui.tools.runner import _add_volume_file, _ensure_nifti_path

    if req.input_layer is None:
        raise ValueError("Registration needs the input image (the one the parcellation is on).")
    if req.target_layer is None:
        raise ValueError("Pick the image to register to: the one the vessels were segmented from.")
    moving = _ensure_nifti_path(req.input_layer, prefix="bvm_moving")
    fixed = _ensure_nifti_path(req.target_layer, prefix="bvm_fixed")
    out_dir = scratch / "registration"
    if req.algorithm == "flirt":
        from nvitk.registration.fsl.flirt import flirt_register_rigid
        from nvitk.registration.world import flirt_matrix_to_world

        task.update(message=f"Brain-vascular map: FLIRT ({req.dof} DOF, {req.cost})…")
        res = run_in_background(flirt_register_rigid, moving, fixed, out_dir, dof=req.dof, cost=req.cost)
        world = flirt_matrix_to_world(res.matrix_path, moving, fixed)
        warped = res.warped_path
    else:
        from nvitk.registration.ants import ants_register
        from nvitk.registration.world import ants_linear_to_world

        task.update(message=f"Brain-vascular map: ANTs {req.ants_transform} ({req.cost})…")
        res = run_in_background(
            ants_register, fixed_path=fixed, moving_path=moving, out_dir=out_dir,
            type_of_transform=req.ants_transform, aff_metric=req.cost,
        )
        world = ants_linear_to_world(res.fwd_transforms)
        warped = res.warped_moving_path
    # The registered image, for checking the alignment against the vessels' image.
    if warped is not None and Path(warped).is_file():
        layer = _add_volume_file(viewer, Path(warped), f"{req.input_layer.name}_registered")
        layer.visible = False
        layer.metadata = {**dict(layer.metadata or {}), FOLDER_KEY: FOLDER}
    return np.asarray(world, dtype=float)


def build_brain_vascular_map(viewer: Any, req: MapRequest, *, on_select: Any = None,
                             scene_kwargs: dict[str, Any] | None = None) -> BrainVascScene:
    """Run the whole pipeline for *req* and draw the scene (GUI thread; heavy steps threaded)."""
    from nvitk.core.backend import get_global_backend, using
    from nvitk.gui.core.activity import activity, run_in_background
    from nvitk.gui.core.log_panel import gui_log
    from nvitk.gui.labels.catalog import label_name
    from nvitk.gui.tools.runner import _TEMP_INPUT_DIRS, _remove_temp_inputs
    from nvitk.gui.viz.main_canvas import world_geometry
    from nvitk.measure.vascular_territory import map_territories
    from nvitk.segmentation.brain_parcellation import (
        lobes_from_aparc,
        non_parenchyma_ids,
        region_side,
        scheme_label_names,
    )

    since = len(_TEMP_INPUT_DIRS)
    scratch = Path(tempfile.mkdtemp(prefix="nvitk_bvm_"))
    try:
        with activity("Brain-vascular map…") as task, world_geometry(viewer):
            # ---- 1. Vessels ----------------------------------------------------------
            vessels = _label_array(req.vessel_layer)
            if vessels.ndim != 3:
                raise ValueError(f"The vessel mask must be a 3D label map (it is {vessels.ndim}D).")
            m_v = _data_to_world(req.vessel_layer)
            ids = sorted(int(v) for v in np.unique(vessels) if int(v) > 0)
            names = _vessel_names(req.vessel_layer, ids, req.vessel_schema)
            colours = _label_colours(req.vessel_layer, ids)

            # ---- 2. Parcellation (computed, or picked) --------------------------------
            parc_layer, scheme = _run_parcellation(viewer, req, task, scratch)
            parc = _label_array(parc_layer)
            if parc.ndim != 3:
                raise ValueError(f"The parcellation must be a 3D label map (it is {parc.ndim}D).")
            m_p = _data_to_world(parc_layer)
            # Lobes are collapsed on the working grid (map_territories' relabel), not here: the
            # white-matter fill is a distance transform, too heavy on a sub-millimetre T1 grid.
            relabel = None
            if req.scheme == "lobes" and scheme == "aparc":
                relabel, scheme = lobes_from_aparc, "lobes"
                region_ids = sorted(scheme_label_names(scheme))
            else:
                region_ids = sorted(int(v) for v in np.unique(parc) if int(v) > 0)
            table = scheme_label_names(scheme)
            region_names = {
                r: table.get(r) or label_name(parc_layer, r) or f"Region {r}" for r in region_ids
            }
            excluded = non_parenchyma_ids(scheme)
            tissue = None if relabel is not None else [r for r in region_ids if r not in excluded]

            # ---- 3. Registration (optional) --------------------------------------------
            world_map = _run_registration(req, task, scratch, viewer) if req.register else None

            # ---- 4. Territories and meshes -----------------------------------------------
            task.update(message="Brain-vascular map: vascular territories…")
            territory_args = (vessels, m_v, names, parc, m_p,
                              {r: region_side(n) for r, n in region_names.items()}, tissue)
            territory_kwargs = dict(
                world_map=world_map, target_mm=req.target_mm, relabel=relabel,
                follow_collaterals=req.follow_collaterals, respect_hemispheres=req.respect_hemispheres,
            )
            try:
                result = run_in_background(map_territories, *territory_args, **territory_kwargs)
            except Exception as exc:  # noqa: BLE001 (GPU only: retried on the CPU, and said so)
                if get_global_backend() == "numpy":
                    raise
                log.warning(f"Territories failed on the GPU backend ({exc}); recomputing on the CPU.")
                gui_log(f"Brain-vascular map: GPU failed ({exc}); territories recomputed on the CPU.")

                def on_cpu() -> Any:
                    with using("numpy"):
                        return map_territories(*territory_args, **territory_kwargs)

                result = run_in_background(on_cpu)
            task.update(message="Brain-vascular map: meshes…")
            data = run_in_background(build_scene_data, result, vessels, m_v, names, colours, region_names)

            scene = BrainVascScene(viewer, data, on_select=on_select, **dict(scene_kwargs or {}))
            scene.build()
        return scene
    finally:
        _remove_temp_inputs(since)
        shutil.rmtree(scratch, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# The dock
# ──────────────────────────────────────────────────────────────────────────────


def _spin(lo: float, hi: float, value: float, step: float, decimals: int = 2) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    return box


class BrainVascPanel(QWidget):
    """Form, run button and the selected vessel's regions."""

    def __init__(self, viewer: Any, vessel_layer: Any | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._viewer = viewer
        self._scene: BrainVascScene | None = None

        body = QWidget()
        lay = QVBoxLayout(body)
        lay.addWidget(self._build_inputs_card())
        lay.addWidget(self._build_registration_card())
        lay.addWidget(self._build_territory_card())
        lay.addWidget(self._build_selection_card())
        lay.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.refresh_layers()
        if vessel_layer is not None:
            self.set_vessel_layer(vessel_layer)
        self._on_source_changed()
        self._sync_registration()
        self._set_selection_enabled(False)
        viewer.layers.events.inserted.connect(self._on_layers_changed)
        viewer.layers.events.removed.connect(self._on_layers_changed)

    # ---- construction ---------------------------------------------------------------

    def _build_inputs_card(self) -> Card:
        card = Card("Inputs")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)

        self._vessel = QComboBox()
        self._vessel.setToolTip("Multilabel vessel segmentation (eICAB, TopBrain, QVTPy…).")
        self._vessel.currentIndexChanged.connect(self._on_vessel_changed)
        form.addRow("Vessel mask", self._vessel)

        self._vocab = QComboBox()
        self._vocab.setToolTip(
            "Which artery each label is. 'auto' uses the layer's own vocabulary (guessed from its\n"
            "name / source, or chosen with the layer's label ▾ chip). Labels whose names are not\n"
            "arteries (veins, extracranial branches) are drawn but have no territory."
        )
        form.addRow("Vessel labels", self._vocab)

        self._source = QComboBox()
        for label, key in SOURCES:
            self._source.addItem(label, key)
        self._source.setToolTip(
            "How the brain is parcellated.\n\n"
            "ANTsPyNet DKT and FastSurfer expect a T1-weighted MRI; TotalSegmentator's brain\n"
            "structures a head CT (and a TotalSegmentator licence). An existing label layer is\n"
            "used as it is."
        )
        self._source.currentIndexChanged.connect(self._on_source_changed)
        form.addRow("Parcellation", self._source)

        self._input_label = QLabel("Image to parcellate")
        self._input = QComboBox()
        form.addRow(self._input_label, self._input)

        self._parc_label = QLabel("Parcellation layer")
        self._parc = QComboBox()
        form.addRow(self._parc_label, self._parc)

        self._regions = QComboBox()
        self._regions.setToolTip(
            "Desikan-Killiany-Tourville: cortical gyri + deep structures.\n"
            "Lobes: frontal / parietal / temporal / occipital / cingulate / insula / basal ganglia /\n"
            "thalamus / cerebellum per hemisphere + brainstem, white matter included."
        )
        form.addRow("Regions", self._regions)
        card.add_layout(form)
        return card

    def _build_registration_card(self) -> Card:
        card = Card("Registration")
        self._register = QCheckBox("Register the image into the vessel space")
        self._register.setToolTip(
            "Off when the parcellation is already in the vessels' space (same session, same\n"
            "scanner coordinates). On to align the input image to the image the vessels were\n"
            "segmented from; the parcellation follows the same (linear) transform."
        )
        self._register.toggled.connect(self._sync_registration)
        card.add(self._register)

        self._reg_box = QWidget()
        form = QFormLayout(self._reg_box)
        form.setContentsMargins(0, 0, 0, 0)
        self._algorithm = QComboBox()
        self._algorithm.addItem("FLIRT (FSL)", "flirt")
        self._algorithm.addItem("ANTs (ANTsPy)", "ants")
        self._algorithm.currentIndexChanged.connect(self._on_algorithm_changed)
        form.addRow("Algorithm", self._algorithm)
        self._cost = QComboBox()
        self._cost.setToolTip("Similarity metric. Across contrasts (T1 → TOF / CT) use a mutual-information "
                              "or correlation-ratio cost.")
        form.addRow("Cost", self._cost)
        self._transform = QComboBox()
        form.addRow("Transform", self._transform)
        self._target = QComboBox()
        self._target.setToolTip("The raw image the vessel segmentation was made from (TOF, CTA…).")
        form.addRow("Register to", self._target)
        card.add(self._reg_box)
        self._on_algorithm_changed()
        return card

    def _build_territory_card(self) -> Card:
        card = Card("Territories")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self._resolution = _spin(0.5, 4.0, 1.0, 0.25)
        self._resolution.setToolTip("Voxel size (mm) the territories and region surfaces are computed at.")
        form.addRow("Working voxel (mm)", self._resolution)
        card.add_layout(form)
        self._collaterals = QCheckBox("Follow the communicating arteries")
        self._collaterals.setToolTip(
            "Off: the ACom / PCom are territories of their own and pass nothing on (normal circle).\n"
            "On: the PCom also feeds the PCA beyond P1 and the ACom both ACAs beyond A1\n"
            "(fetal PCA, hypoplastic A1)."
        )
        card.add(self._collaterals)
        self._hemispheres = QCheckBox("Keep each hemisphere to its own vessels")
        self._hemispheres.setChecked(True)
        self._hemispheres.setToolTip("A left-hemisphere region is only supplied by left-sided or unpaired vessels.")
        card.add(self._hemispheres)
        row = QHBoxLayout()
        self._run_btn = QPushButton("Build map")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self.run)
        self._clear_btn = QPushButton("Remove map layers")
        self._clear_btn.clicked.connect(self.remove_scene)
        row.addWidget(self._run_btn)
        row.addWidget(self._clear_btn)
        row.addStretch(1)
        card.add_layout(row)
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"color: {COLOR_MUTED};")
        card.add(self._status)
        return card

    def _build_selection_card(self) -> Card:
        card = Card("Selected vessel")
        hint = QLabel("Click a vessel in the 3D view, or pick it here. Its regions show in 3D and in the "
                      "orthogonal views; click a region in the table to focus it.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLOR_MUTED};")
        card.add(hint)
        row = QHBoxLayout()
        self._pick = QComboBox()
        self._pick.currentIndexChanged.connect(self._on_pick_changed)
        clear = QPushButton("Clear")
        clear.clicked.connect(lambda: self._scene.select(None) if self._scene else None)
        row.addWidget(self._pick, 1)
        row.addWidget(clear)
        card.add_layout(row)
        self._downstream = QLabel("")
        self._downstream.setWordWrap(True)
        card.add(self._downstream)

        types = QHBoxLayout()
        types.addWidget(QLabel("Region types"))
        self._types = QComboBox()
        self._types.setToolTip(
            "Which regions the table, the 3D view and the orthogonal views show: cortical parcels,\n"
            "everything else, or one kind of tissue. Shares are unchanged; this only filters."
        )
        self._types.currentIndexChanged.connect(self._on_types_changed)
        types.addWidget(self._types, 1)
        card.add_layout(types)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Region", "Supplied"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setToolTip("Click a region to focus it (the others fade); click it again to show all.")
        self._table.cellClicked.connect(self._on_region_clicked)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.setMinimumHeight(160)
        card.add(self._table)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        self._min_share = _spin(1.0, 100.0, DEFAULT_MIN_FRACTION * 100.0, 5.0, 0)
        self._min_share.setSuffix(" %")
        self._min_share.setToolTip("A region is drawn when at least this share of it lies in the vessel's territory.")
        self._min_share.valueChanged.connect(lambda v: self._scene and self._scene.set_min_fraction(v / 100.0))
        form.addRow("Show regions from", self._min_share)
        self._shell_opacity = _spin(0.0, 1.0, DEFAULT_SHELL_OPACITY, 0.02)
        self._shell_opacity.valueChanged.connect(lambda v: self._scene and self._scene.set_shell_opacity(v))
        form.addRow("Brain opacity", self._shell_opacity)
        self._dim_opacity = _spin(0.0, 1.0, DEFAULT_DIM_OPACITY, 0.05)
        self._dim_opacity.valueChanged.connect(lambda v: self._scene and self._scene.set_dim_opacity(v))
        form.addRow("Other vessels' opacity", self._dim_opacity)
        self._region_opacity = _spin(0.05, 1.0, DEFAULT_REGION_OPACITY, 0.05)
        self._region_opacity.valueChanged.connect(lambda v: self._scene and self._scene.set_region_opacity(v))
        form.addRow("Region opacity", self._region_opacity)
        card.add_layout(form)
        self._selection_card = card
        return card

    # ---- layer pickers --------------------------------------------------------------

    def _fill(self, combo: QComboBox, layers: list[Any], *, none: str | None = None, prefer: str = "") -> None:
        current = combo.currentText() or prefer
        combo.blockSignals(True)
        combo.clear()
        if none is not None:
            combo.addItem(none, None)
        for layer in layers:
            combo.addItem(layer.name, layer.name)
        index = combo.findText(current)
        if index < 0 and prefer:
            index = next((i for i in range(combo.count()) if prefer.lower() in combo.itemText(i).lower()), -1)
        combo.setCurrentIndex(max(index, 0))
        combo.blockSignals(False)

    def refresh_layers(self) -> None:
        """Re-list the viewer's layers in every picker, keeping the current choices."""
        from nvitk.gui.viz.brain_vasc_map import PREFIX

        layers = [lyr for lyr in self._viewer.layers if not str(lyr.name).startswith(PREFIX)]
        labels = [lyr for lyr in layers if _is_label_layer(lyr)]
        images = [lyr for lyr in layers if _is_image_layer(lyr)]
        self._fill(self._vessel, labels, prefer="eicab")
        self._fill(self._parc, [lyr for lyr in layers if _is_parcellation_candidate(lyr)], prefer="aparc")
        self._fill(self._input, images, prefer="t1")
        self._fill(self._target, images, prefer="tof")
        self._fill_vocab()

    def _on_layers_changed(self, _event: Any = None) -> None:
        try:
            self.refresh_layers()
        except RuntimeError:  # the dock is being torn down
            pass

    def _layer(self, combo: QComboBox) -> Any | None:
        name = combo.currentData()
        if not name:
            return None
        try:
            return self._viewer.layers[name]
        except KeyError:
            return None

    def set_vessel_layer(self, layer: Any) -> None:
        index = self._vessel.findText(str(getattr(layer, "name", "")))
        if index >= 0:
            self._vessel.setCurrentIndex(index)
        self._fill_vocab()

    def _fill_vocab(self) -> None:
        from nvitk.gui.labels.catalog import all_schemas, layer_schema_key

        layer = self._layer(self._vessel)
        guess = layer_schema_key(layer) if layer is not None else None
        current = self._vocab.currentData()
        self._vocab.blockSignals(True)
        self._vocab.clear()
        self._vocab.addItem(f"auto ({guess})" if guess else "auto (none found)", _AUTO)
        for key, schema in all_schemas().items():
            if schema.group.startswith(("Segmentation / vessels", "Pipelines / QVTpy")) or key.startswith("topbrain"):
                self._vocab.addItem(schema.title, key)
        index = self._vocab.findData(current) if current else 0
        self._vocab.setCurrentIndex(max(index, 0))
        self._vocab.blockSignals(False)

    def _on_vessel_changed(self) -> None:
        self._fill_vocab()

    # ---- dependent widgets ----------------------------------------------------------

    def _on_source_changed(self) -> None:
        source = str(self._source.currentData() or "dkt")
        existing = source == "existing"
        self._parc.setVisible(existing)
        self._parc_label.setVisible(existing)
        self._input_label.setText("Image the parcellation is on" if existing else "Image to parcellate")
        previous = self._regions.currentData()
        self._regions.clear()
        for label, key in _REGIONS_BY_SOURCE.get(source, _REGIONS_APARC):
            self._regions.addItem(label, key)
        index = self._regions.findData(previous)
        if index < 0 and existing:
            layer = self._layer(self._parc)
            aparc = layer is not None and float(np.max(to_numpy(layer.data))) > 1000
            index = self._regions.findData("aparc" if aparc else "generic")
        self._regions.setCurrentIndex(max(index, 0))
        self._sync_registration()

    def _on_algorithm_changed(self) -> None:
        from nvitk.registration.ants import ANTSPY_AFF_METRICS
        from nvitk.registration.fsl.flirt import FLIRT_COSTS

        flirt = self._algorithm.currentData() == "flirt"
        self._cost.clear()
        self._transform.clear()
        if flirt:
            for cost in FLIRT_COSTS:
                self._cost.addItem(cost, cost)
            self._cost.setCurrentIndex(max(self._cost.findData("normmi"), 0))
            for label, dof in FLIRT_DOF_CHOICES:
                self._transform.addItem(label, dof)
        else:
            for metric in ANTSPY_AFF_METRICS:
                self._cost.addItem(metric, metric)
            self._cost.setCurrentIndex(max(self._cost.findData("mattes"), 0))
            for name in ANTS_LINEAR_TRANSFORMS:
                self._transform.addItem(name, name)

    def _sync_registration(self) -> None:
        self._reg_box.setVisible(self._register.isChecked())

    def _set_selection_enabled(self, enabled: bool) -> None:
        self._selection_card.setEnabled(enabled)

    # ---- run ----------------------------------------------------------------------

    def request(self) -> MapRequest:
        """The form as a :class:`MapRequest` (raises ``ValueError`` on a missing input)."""
        vessel = self._layer(self._vessel)
        if vessel is None:
            raise ValueError("Pick the vessel mask.")
        source = str(self._source.currentData() or "dkt")
        register = self._register.isChecked()
        algorithm = str(self._algorithm.currentData() or "flirt")
        transform = self._transform.currentData()
        return MapRequest(
            vessel_layer=vessel,
            vessel_schema=str(self._vocab.currentData() or _AUTO),
            source=source,
            scheme=str(self._regions.currentData() or "aparc"),
            input_layer=self._layer(self._input),
            parcellation_layer=self._layer(self._parc) if source == "existing" else None,
            register=register,
            algorithm=algorithm,
            cost=str(self._cost.currentData() or ("normmi" if algorithm == "flirt" else "mattes")),
            dof=int(transform) if algorithm == "flirt" and transform is not None else 6,
            ants_transform=str(transform) if algorithm == "ants" and transform else "Rigid",
            target_layer=self._layer(self._target) if register else None,
            target_mm=float(self._resolution.value()),
            follow_collaterals=self._collaterals.isChecked(),
            respect_hemispheres=self._hemispheres.isChecked(),
        )

    def run(self) -> BrainVascScene | None:
        """Build the map from the form."""
        try:
            req = self.request()
        except ValueError as exc:
            QMessageBox.warning(self, DOCK_TITLE, str(exc))
            return None
        self._run_btn.setEnabled(False)
        self._status.setText("Running…")
        try:
            scene = build_brain_vascular_map(
                self._viewer, req, on_select=self._on_selected,
                scene_kwargs={
                    "shell_opacity": self._shell_opacity.value(),
                    "dim_opacity": self._dim_opacity.value(),
                    "region_opacity": self._region_opacity.value(),
                    "min_fraction": self._min_share.value() / 100.0,
                },
            )
        except Exception as exc:  # noqa: BLE001 (reported, not raised into Qt)
            log.debug("Brain-vascular map failed: %s", exc, exc_info=True)
            self._status.setText(f"Failed: {exc}")
            QMessageBox.critical(self, DOCK_TITLE, f"Could not build the map:\n{exc}")
            return None
        finally:
            self._run_btn.setEnabled(True)
        self._scene = scene
        self._after_build(scene)
        return scene

    def _after_build(self, scene: BrainVascScene) -> None:
        result = scene.data.result
        self._pick.blockSignals(True)
        self._pick.clear()
        self._pick.addItem("(none)", None)
        for lid in scene.vessel_ids():
            suffix = "" if scene.has_territory(lid) else "  — no territory"
            self._pick.addItem(f"{scene.vessel_name(lid)}{suffix}", lid)
        self._pick.blockSignals(False)
        self._fill_types(scene)
        self._set_selection_enabled(True)
        self._on_selected(None)
        seeds = len(result.table.seed_ids)
        named = sum(1 for c in result.classes.values() if c.family)
        lines = [
            f"{len(scene.vessel_ids())} vessel(s), {seeds} with a territory of their own; "
            f"{len(result.table.region_ids)} region(s) on a {'×'.join(str(s) for s in result.shape)} working grid.",
        ]
        if not named:
            lines.append("No label is named after an artery: every label is its own territory, with no "
                         "supply tree. Set 'Vessel labels' to the segmentation's vocabulary.")
        if result.unclassified:
            lines.append("Drawn without a territory (not arteries): "
                         + ", ".join(scene.vessel_name(v) for v in result.unclassified) + ".")
        self._status.setText("\n".join(lines))

    def _fill_types(self, scene: BrainVascScene) -> None:
        """Offer the region-type filters this parcellation has something for."""
        present = scene.categories_present()
        previous = self._types.currentText()
        self._types.blockSignals(True)
        self._types.clear()
        for label, types in REGION_TYPES:
            if types is None or (types & present):
                self._types.addItem(label, label)
        index = self._types.findText(previous)
        self._types.setCurrentIndex(max(index, 0))
        self._types.blockSignals(False)
        self._on_types_changed()

    def _on_types_changed(self) -> None:
        scene = self._scene
        if scene is None:
            return
        chosen = dict(REGION_TYPES).get(self._types.currentText())
        scene.set_region_types(chosen)

    def _on_region_clicked(self, row: int, _column: int) -> None:
        item = self._table.item(row, 0)
        if self._scene is not None and item is not None:
            self._scene.focus_region(item.data(Qt.UserRole))

    def _on_pick_changed(self) -> None:
        if self._scene is not None:
            self._scene.select(self._pick.currentData())

    def _on_selected(self, lid: int | None) -> None:
        """Mirror the scene's selection in the picker, the downstream line and the table."""
        scene = self._scene or current_scene(self._viewer)
        index = self._pick.findData(lid) if lid is not None else 0
        self._pick.blockSignals(True)
        self._pick.setCurrentIndex(max(index, 0))
        self._pick.blockSignals(False)
        self._table.setRowCount(0)
        if scene is None or lid is None:
            self._downstream.setText("")
            return
        below = scene.downstream(lid)
        if not scene.has_territory(lid):
            self._downstream.setText("Not an artery in the supply tree: no territory.")
            return
        self._downstream.setText(
            "Feeds: " + ", ".join(scene.vessel_name(v) for v in below) if below else "Feeds no other segmented vessel."
        )
        rows = scene.irrigated(lid)
        self._table.setRowCount(len(rows))
        focused_row = -1
        for r, (rid, share) in enumerate(rows):
            name = QTableWidgetItem(scene.region_name(rid))
            name.setData(Qt.UserRole, int(rid))
            if rid == scene.focused:
                focused_row = r
            colour = scene.region_colours.get(rid)
            if colour is not None:
                from qtpy.QtGui import QColor

                name.setData(Qt.DecorationRole, QColor.fromRgbF(*colour[:3]))
            value = QTableWidgetItem(f"{100.0 * share:.0f} %")
            value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(r, 0, name)
            self._table.setItem(r, 1, value)
        self._table.blockSignals(True)
        if focused_row >= 0:
            self._table.selectRow(focused_row)
        else:
            self._table.clearSelection()
        self._table.blockSignals(False)

    def remove_scene(self) -> None:
        scene = self._scene or current_scene(self._viewer)
        if scene is not None:
            scene.remove()
        self._scene = None
        self._table.setRowCount(0)
        self._pick.clear()
        self._downstream.setText("")
        self._set_selection_enabled(False)
        self._status.setText("Map layers removed.")


def start_brain_vascular_map(viewer: Any, vessel_layer: Any | None = None) -> BrainVascPanel | None:
    """Open (or re-show) the dock, with *vessel_layer* as the vessel mask."""
    from nvitk.gui.viz.left_dock import attach_left_inspection_dock

    state = getattr(viewer, "_nvitk_brain_vasc_panel", None)
    panel = state() if callable(state) else None
    try:
        alive = panel is not None and panel.isVisible() is not None
    except RuntimeError:
        alive = False
    if not alive:
        import weakref

        panel = BrainVascPanel(viewer, vessel_layer)
        setattr(viewer, "_nvitk_brain_vasc_panel", weakref.ref(panel))
    elif vessel_layer is not None:
        panel.refresh_layers()
        panel.set_vessel_layer(vessel_layer)
    attach_left_inspection_dock(viewer, panel, object_name=DOCK_OBJECT_NAME, title=DOCK_TITLE, minimum_width=360)
    return panel


__all__ = [
    "BrainVascPanel",
    "DOCK_OBJECT_NAME",
    "MapRequest",
    "build_brain_vascular_map",
    "start_brain_vascular_map",
]
