"""Locate and load QC assets for qvtpy / eicab pipelines into Napari."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.gui.io.napari_io import open_paths_with_nvitk
from nvitk.pipes.qvtpy import config as cfg

log = Logger()


def _find_first(root: Path, patterns: tuple[str, ...]) -> Path | None:
    if not root.is_dir():
        return None
    for pattern in patterns:
        hits = sorted(root.rglob(pattern))
        if hits:
            return hits[0]
    return None


def _set_layer_visible(viewer: Any, name: str, visible: bool) -> Any | None:
    for layer in viewer.layers:
        if layer.name == name:
            layer.visible = bool(visible)
            return layer
    return None


def _layer_by_name(viewer: Any, name: str) -> Any | None:
    for layer in viewer.layers:
        if layer.name == name:
            return layer
    return None


def _open_nifti(viewer: Any, path: Path, *, name: str, visible: bool) -> Any | None:
    before = {lyr.name for lyr in viewer.layers}
    open_paths_with_nvitk(viewer, path)
    # Prefer rename of newly added layer(s).
    for lyr in viewer.layers:
        if lyr.name not in before:
            lyr.name = name
            lyr.visible = bool(visible)
            return lyr
    # Fallback: match by stem.
    stem = path.name.replace(".nii.gz", "").replace(".nii", "")
    for lyr in viewer.layers:
        if stem in lyr.name or lyr.name.endswith(stem):
            lyr.name = name
            lyr.visible = bool(visible)
            return lyr
    return _set_layer_visible(viewer, name, visible)


def resolve_qvtpy_tree(resource_root: Path) -> Path:
    """Return the directory that contains ``qvtpy/`` stage folders."""
    root = Path(resource_root)
    if (root / cfg.QVT_SUBDIR).is_dir():
        return root
    if root.name == cfg.QVT_SUBDIR:
        return root.parent
    nested = _find_first(root, (f"**/{cfg.QVT_SUBDIR}",))
    if nested is not None and nested.is_dir():
        return nested.parent
    return root


def resolve_eicab_tree(resource_root: Path) -> Path:
    root = Path(resource_root)
    if (root / cfg.STAGE1_EICAB_DIR).is_dir():
        return root / cfg.STAGE1_EICAB_DIR
    if root.name == cfg.STAGE1_EICAB_DIR:
        return root
    nested = _find_first(root, (f"**/{cfg.STAGE1_EICAB_DIR}",))
    if nested is not None and nested.is_dir():
        return nested
    return root


def find_phase_paths(search_roots: list[Path]) -> dict[str, Path]:
    """Discover AP/RL/FH phase NIfTIs under download trees."""
    out: dict[str, Path] = {}
    for axis, folder in (("ap", "AP"), ("rl", "RL"), ("fh", "FH")):
        for root in search_roots:
            # Standard NIFTI layout: .../4DFlow/AP/*_ph.nii.gz
            axis_dir = _find_first(root, (f"**/4DFlow/{folder}", f"**/{folder}"))
            if axis_dir is not None and axis_dir.is_dir():
                hit = _find_first(axis_dir, ("*_ph.nii.gz", "*_ph.nii", "*phase*.nii.gz"))
                if hit is not None:
                    out[axis] = hit
                    break
            # Flat 4dflows resource naming.
            hit = _find_first(
                root,
                (
                    f"**/*{folder.lower()}*_ph.nii.gz",
                    f"**/*{axis.upper()}*_ph.nii.gz",
                    f"**/*{axis}*_ph.nii.gz",
                ),
            )
            if hit is not None:
                out[axis] = hit
                break
    return out


def find_cd_path(search_roots: list[Path]) -> Path | None:
    for root in search_roots:
        hit = _find_first(
            root,
            (
                "**/ComplexDifference_3D.nii.gz",
                "**/ComplexDifference_3D.nii",
                "**/complexdifference3d.nii.gz",
            ),
        )
        if hit is not None:
            return hit
    return None


def load_eicab_qc_layers(
    viewer: Any,
    resource_root: Path,
    *,
    tof_search_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Load TOF + eICAB resampled / CW / WB outputs."""
    eicab_dir = resolve_eicab_tree(resource_root)
    roots = [eicab_dir, *(tof_search_roots or [])]
    loaded: dict[str, Any] = {"eicab_dir": eicab_dir}

    tof = None
    for root in roots:
        tof = _find_first(root, ("**/TOF.nii.gz", "**/TOF.nii", "**/tof.nii.gz"))
        if tof is not None:
            break
    if tof is not None:
        loaded["tof"] = _open_nifti(viewer, tof, name="TOF", visible=True)

    resampled = _find_first(eicab_dir, ("**/TOF_resampled.nii.gz", "**/TOF_resampled.nii"))
    if resampled is not None:
        loaded["tof_resampled"] = _open_nifti(
            viewer, resampled, name="TOF_resampled", visible=True
        )

    cw = _find_first(
        eicab_dir,
        ("**/*_eICAB_CW_pp.nii.gz", "**/*_eICAB_CW.nii.gz", "**/*eICAB_CW*.nii.gz"),
    )
    if cw is not None:
        loaded["eicab_cw"] = _open_nifti(viewer, cw, name="eICAB_CW", visible=True)

    wb = _find_first(
        eicab_dir,
        ("**/*_eICAB_WB_pp.nii.gz", "**/*_eICAB_WB.nii.gz", "**/*eICAB_WB*.nii.gz"),
    )
    if wb is not None:
        loaded["eicab_wb"] = _open_nifti(viewer, wb, name="eICAB_WB", visible=True)

    return loaded


def load_qvtpy_qc_layers(
    viewer: Any,
    app_state: dict[str, Any],
    resource_root: Path,
    *,
    extra_search_roots: list[Path] | None = None,
) -> dict[str, Any]:
    """Load qvtpy QC layers and auxiliary docks for one subject."""
    from nvitk.gui.viz.loc_points import add_locs_layer, load_locs_csv
    from nvitk.gui.viz.vessel_cross_sections import install_vessel_cross_sections
    from nvitk.gui.core.spatial import layer_to_image

    subject_root = resolve_qvtpy_tree(resource_root)
    qvt = subject_root / cfg.QVT_SUBDIR
    if not qvt.is_dir():
        # Resource may unpack directly as stage folders.
        qvt = subject_root

    search_roots = [subject_root, qvt, *(extra_search_roots or [])]
    loaded: dict[str, Any] = {
        "subject_root": subject_root,
        "qvt_dir": qvt,
        "stage6_dir": qvt / cfg.STAGE6_MEASURE_DIR,
    }

    # --- Phase images (default visibility off) ---
    phases = find_phase_paths(search_roots)
    for axis, label in (("ap", "AP phase"), ("rl", "RL phase"), ("fh", "FH phase")):
        path = phases.get(axis)
        if path is not None:
            loaded[f"phase_{axis}"] = _open_nifti(
                viewer, path, name=label, visible=False
            )

    # --- Complex Difference (on) ---
    cd_path = find_cd_path(search_roots)
    if cd_path is not None:
        loaded["cd"] = _open_nifti(
            viewer, cd_path, name="Complex Difference 3D", visible=True
        )

    # --- Segmentation + centerlines (on) ---
    stage4 = qvt / cfg.STAGE4_SEG_DIR
    seg_path = _find_first(stage4, ("seg_4dflow.nii.gz", "seg_4dflow.nii"))
    if seg_path is None:
        seg_path = _find_first(qvt, ("**/seg_4dflow.nii.gz",))
    if seg_path is not None:
        loaded["seg"] = _open_nifti(viewer, seg_path, name="seg_4dflow", visible=True)

    cl_path = _find_first(
        stage4,
        ("centerlines_mask_4dflow.nii.gz", "centerlines_mask_4dflow.nii"),
    )
    if cl_path is None:
        stage3 = qvt / cfg.STAGE3_CENTERLINE_DIR
        cl_path = _find_first(stage3, ("centerlines_mask.nii.gz", "centerlines_mask.nii"))
    if cl_path is not None:
        loaded["centerlines"] = _open_nifti(
            viewer, cl_path, name="centerlines_mask_4dflow", visible=True
        )

    # --- LOCs (on) ---
    locs_csv = _find_first(
        qvt / cfg.STAGE5_LOC_DIR,
        ("locs.csv",),
    )
    if locs_csv is None:
        locs_csv = _find_first(qvt, ("**/locs.csv",))
    if locs_csv is not None:
        try:
            rows = load_locs_csv(locs_csv)
            ref = loaded.get("seg") or loaded.get("cd") or (
                viewer.layers[-1] if viewer.layers else None
            )
            add_locs_layer(viewer, rows, reference_layer=ref, name="LOCs")
            _set_layer_visible(viewer, "LOCs", True)
            loaded["locs_csv"] = locs_csv
            loaded["locs"] = _layer_by_name(viewer, "LOCs")
        except Exception as exc:
            log.warning("QC LOCs load failed: %s", exc)

    # --- Flow velocity vectors (off) ---
    ap_lyr = loaded.get("phase_ap")
    rl_lyr = loaded.get("phase_rl")
    fh_lyr = loaded.get("phase_fh")
    seg_lyr = loaded.get("seg")
    if ap_lyr is not None and rl_lyr is not None and fh_lyr is not None and seg_lyr is not None:
        try:
            from nvitk.gui.tools.runner import _run_viz_flowshow_napari

            _run_viz_flowshow_napari(
                viewer,
                seg_lyr,
                {
                    "ap_layer": ap_lyr.name,
                    "rl_layer": rl_lyr.name,
                    "fh_layer": fh_lyr.name,
                    "reference_layer": (loaded.get("cd") or ap_lyr).name,
                    "length_scale": 5.0,
                    "sync_dims": True,
                },
                label_ids=None,
            )
            flow_lyr = _layer_by_name(viewer, "Flow velocity")
            if flow_lyr is not None:
                flow_lyr.visible = False
                loaded["flow_velocity"] = flow_lyr
        except Exception as exc:
            log.warning("QC flow vectors failed: %s", exc)

    # --- Vessel cross-sections (on), pick-off by default ---
    cd_lyr = loaded.get("cd")
    cl_lyr = loaded.get("centerlines")
    if cd_lyr is not None and cl_lyr is not None:
        try:
            from nvitk.core.array import to_numpy as _to_np

            cl_img = layer_to_image(cl_lyr)
            centerline_mask = _to_np(cl_img.data).astype(np.int32, copy=False)
            seg_arr = None
            if seg_lyr is not None:
                seg_arr = _to_np(layer_to_image(seg_lyr).data).astype(np.int32, copy=False)

            vx = vy = vz = None
            if ap_lyr is not None and rl_lyr is not None and fh_lyr is not None:
                from nvitk.measure.hemodynamics import velocity_mm_s_from_phases

                vx, vy, vz = velocity_mm_s_from_phases(
                    _to_np(layer_to_image(ap_lyr).data),
                    _to_np(layer_to_image(rl_lyr).data),
                    _to_np(layer_to_image(fh_lyr).data),
                )

            install_vessel_cross_sections(
                viewer,
                app_state,
                intensity_layer=cd_lyr,
                centerline_mask=centerline_mask,
                segmentation=seg_arr,
                params={
                    "cross_section_radius_vox": 10.0,
                    "measure_resegment": True,
                    "thr_algorithm": "otsu",
                    "show_segmentation_3d": True,
                },
                vx=vx,
                vy=vy,
                vz=vz,
            )
            xs_state = app_state.get("vessel_xs") or {}
            panel = xs_state.get("panel")
            if panel is not None and hasattr(panel, "set_picking_enabled"):
                panel.set_picking_enabled(False)
            for name in ("Vessel centerlines (xs)", "Centerline points (xs)"):
                lyr = _set_layer_visible(viewer, name, True)
                if lyr is not None:
                    loaded[name] = lyr
            # Attach named bifurcation branches (LACA-A1, LMCA-M2a, …) when the
            # stage-4 sidecar is present so QC can surface them beside the mask.
            try:
                from nvitk.pipes.qvtpy.util.centerline_io import (
                    CENTERLINE_SEG_BRANCHES_JSON,
                    flatten_branches,
                    load_arterial_branches,
                )

                if (stage4 / CENTERLINE_SEG_BRANCHES_JSON).is_file():
                    arterial_br = load_arterial_branches(
                        stage4, min_points=3, from_segmentation=True
                    )
                    flat = flatten_branches(arterial_br)
                    loaded["arterial_branches"] = {
                        name: pts for name, pts in flat.items()
                    }
                    xs_state = app_state.get("vessel_xs") or {}
                    xs_state["branch_names"] = sorted(flat.keys())
                    app_state["vessel_xs"] = xs_state
                    # Tip points labeled by branch name for visual QC.
                    tips = []
                    tip_names: list[str] = []
                    for name, pts in sorted(flat.items()):
                        arr = _to_np(pts)
                        if arr.ndim == 2 and arr.shape[0] >= 1:
                            tips.append(arr[-1])
                            tip_names.append(str(name))
                    if tips:
                        tip_layer = viewer.add_points(
                            np.asarray(tips, dtype=float),
                            name="Branch tips (named)",
                            size=8,
                            face_color="yellow",
                            border_color="black",
                            text={
                                "string": "{branch_name}",
                                "size": 9,
                                "color": "white",
                                "anchor": "upper_left",
                            },
                            properties={"branch_name": tip_names},
                            visible=True,
                        )
                        loaded["branch_tips"] = tip_layer
            except Exception as br_exc:
                log.warning("QC branch-name overlay failed: %s", br_exc)
        except Exception as exc:
            log.warning("QC vessel cross-sections failed: %s", exc)

    # --- PITC / PWV overlays + diagnostics (off) ---
    stage6 = loaded["stage6_dir"]
    if isinstance(stage6, Path) and stage6.is_dir() and cl_lyr is not None:
        try:
            from nvitk.gui.tools.runner import _run_viz_pitc, _run_viz_pwv

            hemo_params = {
                "centerline_layer": cl_lyr.name,
                "segmentation_layer": seg_lyr.name if seg_lyr is not None else "",
                "ap_layer": ap_lyr.name if ap_lyr is not None else "",
                "rl_layer": rl_lyr.name if rl_lyr is not None else "",
                "fh_layer": fh_lyr.name if fh_lyr is not None else "",
                "cross_section_radius_vox": 10.0,
                "measure_resegment": True,
                "stride": 1,
                "root_region": "All",
                "station_point_size": 2.5,
            }
            # Use CD layer as active for hemo prep.
            active = cd_lyr or cl_lyr
            _run_viz_pitc(viewer, active, hemo_params)
            _run_viz_pwv(viewer, active, hemo_params)
            # Hide hemo overlay layers by default.
            for lyr in viewer.layers:
                name = str(lyr.name)
                if any(
                    key in name.lower()
                    for key in (
                        "pitc",
                        "pwv",
                        "root init",
                        "station",
                        "hemo",
                    )
                ):
                    lyr.visible = False
        except Exception as exc:
            log.warning("QC PITC/PWV overlays failed: %s", exc)

    return loaded


def download_pipeline_resource_for_qc(
    *,
    config_path: Path,
    project_id: str,
    subject_uid: str,
    resource_label: str,
    password: str,
    download_root: Path | None = None,
    app_state: dict[str, Any] | None = None,
) -> Path:
    """Download an XNAT experiment resource into a cache directory."""
    from nvitk.db.xnat import connect_xnat
    from nvitk.db.xnat_config import load_xnat_profile, resolve_xnat_connection
    from nvitk.db.xnat_pipeline_resources import download_experiment_resource
    from nvitk.db.xnat_upload import resolve_subject_experiment

    if download_root is None:
        download_root = Path(tempfile.mkdtemp(prefix="nvitk_qc_"))
        if app_state is not None:
            roots = list(app_state.get("xnat_temp_dirs") or [])
            roots.append(str(download_root))
            app_state["xnat_temp_dirs"] = roots

    profile = load_xnat_profile(config_path)
    conn = resolve_xnat_connection(
        profile, project=project_id, password=password or None
    )
    target = download_root / subject_uid / resource_label
    target.mkdir(parents=True, exist_ok=True)
    with connect_xnat(conn) as session:
        project = session.projects[project_id]
        experiment, _ = resolve_subject_experiment(project, subject_uid)
        download_experiment_resource(experiment, resource_label, target)
    return target
