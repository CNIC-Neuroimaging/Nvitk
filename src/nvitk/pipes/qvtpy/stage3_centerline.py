"""qvtpy stage 3: centerlines (eICAB in 4D-flow space + venous branches from CD).

**Inputs**

- Stage-2 FLIRT transform, eICAB CW/WB mask, ``ComplexDifference_3D``.

**Outputs**

- ``eicab_in_4dflow.nii.gz``, ``centerlines_mask.nii.gz``, ``centerline_meta.json``,
  ``cd_vessel_binary_qc.nii.gz`` under ``stage3_centerline/``.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, TextIO

import click

import nvitk
from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_stage,
)
from nvitk.core.click_backend import backend_click_option
from nvitk.pipes.qvtpy.util.io.sge_backend import (
    sge_backend_cli_args,
    sge_qvtpy_stage_resources,
    sge_stage_extra_env,
    sge_stage_use_nv,
)
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread, imsave
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.labels import (
    QVTPY_ARTERIAL_LABEL_IDS,
    QVTPY_SMALL_ARTERIAL_IDS,
    VENOUS_UNKNOWN_LABEL,
    relabel_eicab_mask_to_qvtpy,
)
from nvitk.pipes.qvtpy.util.centerline.centerline_io import (
    CENTERLINES_MASK_NIFTI,
    arterial_main_paths,
    centerlines_from_segmentation,
)
from nvitk.pipes.qvtpy.util.eicab.brain_mask import brain_mask_for_reference
from nvitk.pipes.qvtpy.util.io.paths import resolve_totalseg_model_dir
from nvitk.pipes.qvtpy.util.eicab.eicab_masks import EicabMaskKind, resolve_eicab_mask
from nvitk.pipes.qvtpy.util.centerline.flow_volume_masks import (
    arterial_exclusion_mask,
    binary_vessel_segment_cd,
    venous_search_region,
)
from nvitk.pipes.qvtpy.util.centerline.mask_cleaning import (
    clean_multilabel_islands,
    clean_venous_slab_mask,
)
from nvitk.pipes.qvtpy.util.centerline.venous_heuristics import assign_venous_branches, venous_name_to_label_id
from nvitk.registration.fsl.flirt import flirt_apply_rigid

setup(globals())

log = Logger()


def _load_wb_arterial_labels_for_venous_exclusion(
    *,
    eicab_dir: Path,
    fixed: Path,
    mat: Path,
    out_dir: Path,
    eicab_prefer_pp: bool,
    cw_labels_fallback: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Warp WB eICAB (preferred) for venous arterial exclusion; else use CW labels.

    CW Circle-of-Willis masks omit distal MCA/ACA/PCA territory that often enters
    the superior venous slab on CD; WB covers those for exclusion only.
    """
    wb_res = resolve_eicab_mask(
        eicab_dir,
        preference="wb",
        prefer_postprocessed=bool(eicab_prefer_pp),
    )
    if wb_res.used != "wb":
        log.info(
            "venous arterial exclusion: WB eICAB not available "
            f"({wb_res.fallback_reason or 'fallback'}); using CW labels"
        )
        return np.asarray(cw_labels_fallback, dtype=np.int32), "cw_fallback"

    warped_wb = out_dir / "eicab_wb_in_4dflow_venous_excl.nii.gz"
    log.step(f"warp eICAB WB ({wb_res.path.name}) for venous arterial exclusion")
    flirt_apply_rigid(
        wb_res.path,
        fixed,
        mat,
        warped_wb,
        interp="nearestneighbour",
    )
    wb_img = imread(warped_wb)
    wb_lab = as_backend_array(wb_img.data).astype(np.int32, copy=False)
    wb_lab = relabel_eicab_mask_to_qvtpy(wb_lab)
    wb_lab = clean_multilabel_islands(wb_lab, min_fraction=0.005, bridge_open_radius=0)
    n_art = sum(int(np.count_nonzero(wb_lab == lid)) for lid in QVTPY_ARTERIAL_LABEL_IDS)
    log.info(
        f"venous arterial exclusion: WB in 4D-flow arterial voxels={n_art} "
        f"(pp={bool(wb_res.postprocessed)})"
    )
    return np.asarray(wb_lab, dtype=np.int32), "wb"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _stage2_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE2_REGISTRATION_DIR


def _stage3_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE3_CENTERLINE_DIR


def _load_stage2_meta(output_root: Path, subject: str) -> dict[str, Any]:
    p = _stage2_dir(output_root, subject) / "registration_meta.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing stage2 meta: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


# ---- Centerline mask rasterization -------------------------------------------


def _rasterize_centerlines_mask(
    shape: tuple[int, int, int],
    arterial: dict[int, Any],
    venous: dict[str, np.ndarray],
    *,
    venous_label_by_name: dict[str, int],
) -> np.ndarray:
    """Voxel mask: qvtpy arterial id on arterial points; venous id on venous polylines."""
    mask = np.zeros(shape, dtype=np.int32)
    for vid, pts in sorted(arterial.items()):
        p = as_backend_array(pts)
        for row in p:
            i, j, k = (
                int(round(float(row[0]))),
                int(round(float(row[1]))),
                int(round(float(row[2]))),
            )
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                mask[i, j, k] = int(vid)
    for name, pts in venous.items():
        lid = int(venous_label_by_name.get(name, VENOUS_UNKNOWN_LABEL))
        p = as_backend_array(pts)
        for row in p:
            i, j, k = (
                int(round(float(row[0]))),
                int(round(float(row[1]))),
                int(round(float(row[2]))),
            )
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2] and mask[i, j, k] == 0:
                mask[i, j, k] = lid
    return mask


def _cd_path(nifti_root: Path, subject: str) -> Path:
    p = nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii.gz"
    if p.is_file():
        return p
    p2 = nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii"
    if p2.is_file():
        return p2
    raise FileNotFoundError(f"Missing ComplexDifference_3D for {subject}")


# ---------------------------------------------------------------------------
# Stage 3: eICAB warp, CD threshold, arterial + venous centerlines
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_subdir: str | None = None,
    eicab_mask: EicabMaskKind = "cw",
    skip_existing: bool = False,
    cd_up_thresh: float | None = None,
    cd_shift_hm: bool | None = None,
    venous_min_component_frac: float = 0.005,
    eicab_min_island_fraction: float = 0.005,
    eicab_bridge_open_radius: int = 0,
    venous_min_branch_points: int = 12,
    eicab_prefer_pp: bool = True,
    venous_brain_mask: bool = True,
    totalseg_device: str = "gpu",
    totalseg_model_dir: Path | None = None,
) -> Path:
    """Warp eICAB, extract arterial/venous centerlines; return stage-3 output directory."""
    # ---- Inputs: stage2 registration + eICAB mask resolution -----------------
    meta = _load_stage2_meta(output_root, subject)
    mat = Path(meta["matrix"])
    fixed = Path(meta["fixed"])
    subdir = (eicab_subdir or cfg.STAGE1_EICAB_DIR).strip() or "eicab"
    eicab_dir = output_root / subject / subdir
    eicab_res = resolve_eicab_mask(
        eicab_dir,
        preference=eicab_mask,
        prefer_postprocessed=bool(eicab_prefer_pp),
    )

    out_dir = _stage3_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = out_dir / "centerline_meta.json"
    if skip_existing and done.is_file():
        log.info(f"[{subject}] stage3 centerline: skip -> {out_dir}")
        return out_dir

    # ---- Warp eICAB labels into 4D-flow space (nearest neighbour) ----------
    pp_tag = " pp" if eicab_res.postprocessed else ""
    log.step(f"warp eICAB ({eicab_res.used}{pp_tag}) into 4D-flow space")
    warped_labels = out_dir / "eicab_in_4dflow.nii.gz"
    flirt_apply_rigid(
        eicab_res.path,
        fixed,
        mat,
        warped_labels,
        interp="nearestneighbour",
    )

    # ---- Global CD vessel mask (sliding threshold + area opening) ------------
    log.step("CD slding-threshold vessel mask for venous search")
    lab_img = imread(warped_labels)
    labels_arr = as_backend_array(lab_img.data)
    shape3 = tuple(int(x) for x in labels_arr.shape[:3])

    cd_img = imread(_cd_path(nifti_root, subject))
    cd = as_backend_array(cd_img.data).astype(np.float64)

    up_thresh = 0.8 if cd_up_thresh is None else float(cd_up_thresh)
    shift_hm = True if cd_shift_hm is None else bool(cd_shift_hm)
    vessel_bin, sliding_opt_thresh = binary_vessel_segment_cd(
        cd,
        up_thresh=up_thresh,
        shift_hm_flag=shift_hm,
    )
    vessel_bin = as_backend_array(vessel_bin.astype(np.uint8, copy=False))

    # ---- Arterial labels: clean eICAB, clear venous slab, island filter ------
    # Per-label islands only (no global arterial CC wipe): a global min-size
    # keyed off total arterial voxels was dropping whole small vessels (PCA/comm).
    # Centerlines use the same extractor as stage 4 (min_points=3, 2 for MCA/ACA/PCA).
    log.step("clean arterial eICAB labels + island filter")
    venous_region = venous_search_region(shape3)
    labels_np = as_backend_array(labels_arr).astype(np.int32, copy=False)
    native_eicab_path = out_dir / "eicab_in_4dflow_eicab_ids.nii.gz"
    imsave(native_eicab_path, labels_np, metadata=dict(lab_img.metadata or {}))
    labels_np = relabel_eicab_mask_to_qvtpy(labels_np)
    labels_np = clean_multilabel_islands(
        labels_np,
        min_fraction=eicab_min_island_fraction,
        bridge_open_radius=eicab_bridge_open_radius,
        min_fraction_for_label={int(lid): 0.0 for lid in QVTPY_SMALL_ARTERIAL_IDS},
    )
    # Keep CW labels as fallback; prefer WB for venous arterial exclusion.
    arterial_labels_full = np.asarray(labels_np, dtype=np.int32).copy()

    arterial_vol = np.where(venous_region, 0, labels_np).astype(np.int32, copy=False)
    imsave(warped_labels, arterial_vol, metadata=dict(lab_img.metadata or {}))

    arterial_branches = centerlines_from_segmentation(arterial_vol, min_points=3)
    arterial = arterial_main_paths(arterial_branches)
    log.step(f"arterial centerlines: {len(arterial)} label(s)")

    # ---- Venous: CD ∧ slab [∧ brain] → clean → bifurcate/split skeleton → name/label
    log.step("venous branch detection from CD + slab region")
    brain_region: np.ndarray | None = None
    if venous_brain_mask:
        log.step("TotalSegmentator brain mask (total_mr / brain on Angiography_3D)")
        brain_region = brain_mask_for_reference(
            nifti_root,
            subject,
            out_dir,
            cd_img,
            device=totalseg_device,
            model_dir=totalseg_model_dir,
            overwrite=not skip_existing,
        )
        imsave(
            out_dir / "brain_mask_on_cd.nii.gz",
            brain_region.astype(np.uint8),
            metadata=dict(cd_img.metadata or {}),
        )
        log.info(
            f"[{subject}] venous: brain mask voxels={int(np.count_nonzero(brain_region))}"
        )

    n_vessel = int(np.count_nonzero(vessel_bin))
    n_slab = int(np.count_nonzero(venous_region))
    venous_mask = vessel_bin.astype(bool) & venous_region
    n_after_slab = int(np.count_nonzero(venous_mask))
    log.info(
        f"[{subject}] venous: CD vessel={n_vessel}, slab={n_slab}, "
        f"CD∧slab={n_after_slab}"
    )
    # Subtract distal arteries via WB eICAB (CW often misses territory in the slab).
    excl_labels, excl_src = _load_wb_arterial_labels_for_venous_exclusion(
        eicab_dir=eicab_dir,
        fixed=fixed,
        mat=mat,
        out_dir=out_dir,
        eicab_prefer_pp=bool(eicab_prefer_pp),
        cw_labels_fallback=arterial_labels_full,
    )
    art_excl = arterial_exclusion_mask(excl_labels, dilate_vox=2)
    n_art_hit = int(np.count_nonzero(venous_mask & art_excl))
    venous_mask = venous_mask & ~art_excl
    log.info(
        f"[{subject}] venous: excluded arterial overlap "
        f"(source={excl_src}, dilate=2) removed={n_art_hit}, "
        f"remaining={int(np.count_nonzero(venous_mask))}"
    )
    if brain_region is not None:
        venous_mask = venous_mask & brain_region
        log.info(
            f"[{subject}] venous: after brain mask voxels={int(np.count_nonzero(venous_mask))}"
        )
    venous_clean = as_backend_array(
        clean_venous_slab_mask(venous_mask, min_fraction=venous_min_component_frac)
    )
    log.info(
        f"[{subject}] venous: after area-open "
        f"(min_frac={float(venous_min_component_frac)}) "
        f"voxels={int(np.count_nonzero(venous_clean))}"
    )
    venous_branches = assign_venous_branches(
        venous_clean,
        min_points=venous_min_branch_points,
        affine=as_backend_array(cd_img.affine) if getattr(cd_img, "affine", None) is not None else None,
    )
    venous_label_by_name = {
        name: venous_name_to_label_id(name) for name in venous_branches
    }
    log.info(
        f"[{subject}] venous labeled: "
        + (
            ", ".join(
                f"{n}=id{venous_label_by_name[n]}(n={int(venous_branches[n].shape[0])})"
                for n in venous_branches
            )
            or "none"
        )
    )

    # ---- Outputs: centerlines mask, CD QC NIfTI, centerline_meta.json --------
    cl_mask = _rasterize_centerlines_mask(
        shape3,
        arterial,
        venous_branches,
        venous_label_by_name=venous_label_by_name,
    )
    mask_path = out_dir / CENTERLINES_MASK_NIFTI
    imsave(mask_path, cl_mask, metadata=dict(lab_img.metadata or {}))

    qc_path = out_dir / "cd_vessel_binary_qc.nii.gz"
    imsave(qc_path, vessel_bin.astype(np.uint8), metadata=dict(lab_img.metadata or {}))

    meta_out = {
        "subject": subject,
        "eicab_in_4dflow": str(warped_labels),
        "eicab_in_4dflow_eicab_ids": str(native_eicab_path),
        "eicab_mask_requested": eicab_res.requested,
        "eicab_mask_used": eicab_res.used,
        "eicab_mask_fallback": eicab_res.fallback,
        "eicab_mask_fallback_reason": eicab_res.fallback_reason,
        "eicab_mask_postprocessed": bool(eicab_res.postprocessed),
        "eicab_mask_original": (
            str(eicab_res.original_path) if eicab_res.original_path is not None else None
        ),
        "eicab_labels_source": str(eicab_res.path),
        "arterial_label_scheme": "qvtpy",
        "arterial_labels": [int(k) for k in sorted(arterial.keys())],
        "venous_vessels": list(venous_branches.keys()),
        "venous_label_by_name": venous_label_by_name,
        "n_venous_points": int(sum(p.shape[0] for p in venous_branches.values())),
        "centerlines_mask_nifti": str(mask_path),
        "cd_vessel_binary_qc": str(qc_path),
        "binary_segmentation_sliding_threshold": True,
        "sliding_threshold_up_thresh": up_thresh,
        "sliding_threshold_shift_hm": shift_hm,
        "sliding_threshold_opt_absolute": float(sliding_opt_thresh),
        "global_bwareaopen_fraction_of_foreground": 0.005,
        "venous_min_component_frac": float(venous_min_component_frac),
        "eicab_min_island_fraction": float(eicab_min_island_fraction),
        "eicab_bridge_open_radius": int(eicab_bridge_open_radius),
        "venous_region_axis1_third": int(max(1, round(shape3[1] / 3.0))),
        "min_points_per_vessel": 5,
        "venous_min_branch_points": int(venous_min_branch_points),
        "venous_brain_mask": bool(venous_brain_mask),
        "totalseg_task": "total_mr",
        "totalseg_roi_subset": ["brain"],
        "totalseg_input": "Angiography_3D",
        "totalseg_model_dir": str(resolve_totalseg_model_dir(model_dir=totalseg_model_dir)),
        "brain_mask_on_cd": str(out_dir / "brain_mask_on_cd.nii.gz") if brain_region is not None else None,
        "venous_voxels_after_brain_mask": int(np.count_nonzero(venous_mask)),
        "venous_arterial_exclusion_source": excl_src,
        "venous_arterial_exclusion_dilate_vox": 2,
    }
    done.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    log.info(f"[{subject}] stage3 centerline -> {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _stage3_cli_options(func):
    func = click.option("--subject", required=True)(func)
    func = click.option("--nifti-root", type=click.Path(path_type=Path), required=True)(func)
    func = click.option("--output-root", type=click.Path(path_type=Path), required=True)(func)
    func = click.option("--skip-existing", is_flag=True, default=False)(func)
    func = click.option(
        "--eicab-mask",
        type=click.Choice(["cw", "wb"], case_sensitive=False),
        default="cw",
        show_default=True,
    )(func)
    func = click.option("--cd-up-thresh", type=float, default=None, help="CD sliding-threshold upper fraction.")(func)
    func = click.option(
        "--cd-shift-hm/--no-cd-shift-hm",
        default=None,
        help="FWHM shift along threshold curve (default on).",
    )(func)
    func = click.option("--venous-min-component-frac", type=float, default=0.005, show_default=True)(func)
    func = click.option("--eicab-min-island-fraction", type=float, default=0.005, show_default=True)(func)
    func = click.option("--eicab-bridge-open-radius", type=int, default=1, show_default=True)(func)
    func = click.option("--venous-min-branch-points", type=int, default=12, show_default=True)(func)
    func = click.option(
        "--eicab-prefer-pp/--no-eicab-prefer-pp",
        default=True,
        show_default=True,
        help="Prefer stage1 *_pp eICAB mask when present.",
    )(func)
    func = click.option(
        "--venous-brain-mask/--no-venous-brain-mask",
        default=True,
        show_default=True,
        help="Restrict venous candidates to TotalSegmentator brain on Angiography_3D.",
    )(func)
    func = click.option(
        "--totalseg-device",
        type=click.Choice(["gpu", "cpu"], case_sensitive=False),
        default="gpu",
        show_default=True,
        help="Device for TotalSegmentator brain segmentation.",
    )(func)
    func = click.option(
        "--totalseg-model-dir",
        type=click.Path(path_type=Path),
        default=None,
        help="TotalSegmentator weights (default: qvtpy config cluster/local model_root).",
    )(func)
    return func


def _subject_sge_spec(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    eicab_mask: str = "cw",
    cd_up_thresh: float | None = None,
    cd_shift_hm: bool | None = None,
    venous_min_component_frac: float = 0.005,
    eicab_min_island_fraction: float = 0.005,
    eicab_bridge_open_radius: int = 1,
    venous_min_branch_points: int = 12,
    eicab_prefer_pp: bool = True,
    venous_brain_mask: bool = True,
    totalseg_device: str | None = None,
    totalseg_model_dir: Path | None = None,
    backend: str = "gpu",
) -> tuple[StageSpec, ClusterPaths]:
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    parts: list[str] = [
        *python_module_argv("nvitk.pipes.qvtpy.stage3_centerline"),
        *sge_backend_cli_args(backend),
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
        "--output-root",
        shlex.quote(binds.output),
        "--eicab-mask",
        shlex.quote(str(eicab_mask).lower()),
        "--venous-min-component-frac",
        str(float(venous_min_component_frac)),
        "--eicab-min-island-fraction",
        str(float(eicab_min_island_fraction)),
        "--eicab-bridge-open-radius",
        str(int(eicab_bridge_open_radius)),
        "--venous-min-branch-points",
        str(int(venous_min_branch_points)),
    ]
    if skip_existing:
        parts.append("--skip-existing")
    if not eicab_prefer_pp:
        parts.append("--no-eicab-prefer-pp")
    if cd_up_thresh is not None:
        parts.extend(["--cd-up-thresh", str(float(cd_up_thresh))])
    if cd_shift_hm is not None:
        parts.append("--cd-shift-hm" if cd_shift_hm else "--no-cd-shift-hm")
    if not venous_brain_mask:
        parts.append("--no-venous-brain-mask")
    ts_dev = totalseg_device or backend
    parts.extend(["--totalseg-device", shlex.quote(str(ts_dev).strip().lower())])
    # Host weights path for Singularity -B; in-container CLI/env must use binds.models.
    model_root = resolve_totalseg_model_dir(model_dir=totalseg_model_dir) if venous_brain_mask else None
    if venous_brain_mask and model_root is not None:
        parts.extend(["--totalseg-model-dir", shlex.quote(binds.models)])
    python_cmd = " ".join(parts)

    extra_env = dict(sge_stage_extra_env(binds.src, backend))
    use_nv = sge_stage_use_nv(backend)
    if venous_brain_mask and model_root is not None:
        extra_env["TOTALSEG_HOME_DIR"] = str(binds.models)
        use_nv = use_nv or str(ts_dev).strip().lower() == "gpu"
    paths = ClusterPaths(
        src=src_p,
        container=container,
        models=model_root,
        data_root=nifti_root,
        output_root=output_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )
    spec = StageSpec(
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage3_{subject}",
        python_cmd=python_cmd,
        resources=sge_qvtpy_stage_resources(backend),
        binds=binds,
        use_nv=use_nv,
        extra_env=extra_env,
    )
    return spec, paths


def build_subject_sge_command(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    eicab_mask: str = "cw",
    cd_up_thresh: float | None = None,
    cd_shift_hm: bool | None = None,
    venous_min_component_frac: float = 0.005,
    eicab_min_island_fraction: float = 0.005,
    eicab_bridge_open_radius: int = 1,
    venous_min_branch_points: int = 12,
    eicab_prefer_pp: bool = True,
    venous_brain_mask: bool = True,
    totalseg_device: str | None = None,
    totalseg_model_dir: Path | None = None,
    backend: str = "gpu",
) -> str:
    """Return the host shell command for one stage3 array/SGE task."""
    from nvitk.cluster.sge import build_singularity_command

    spec, paths = _subject_sge_spec(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        container=container,
        src_dir=src_dir,
        skip_existing=skip_existing,
        eicab_mask=eicab_mask,
        cd_up_thresh=cd_up_thresh,
        cd_shift_hm=cd_shift_hm,
        venous_min_component_frac=venous_min_component_frac,
        eicab_min_island_fraction=eicab_min_island_fraction,
        eicab_bridge_open_radius=eicab_bridge_open_radius,
        venous_min_branch_points=venous_min_branch_points,
        eicab_prefer_pp=eicab_prefer_pp,
        venous_brain_mask=venous_brain_mask,
        totalseg_device=totalseg_device,
        totalseg_model_dir=totalseg_model_dir,
        backend=backend,
    )
    return build_singularity_command(spec, paths)


def submit_subject_sge(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    hold_jid: str | None = None,
    emit: TextIO | None = None,
    eicab_mask: str = "cw",
    cd_up_thresh: float | None = None,
    cd_shift_hm: bool | None = None,
    venous_min_component_frac: float = 0.005,
    eicab_min_island_fraction: float = 0.005,
    eicab_bridge_open_radius: int = 1,
    venous_min_branch_points: int = 12,
    eicab_prefer_pp: bool = True,
    venous_brain_mask: bool = True,
    totalseg_device: str | None = None,
    totalseg_model_dir: Path | None = None,
    backend: str = "gpu",
) -> str:
    """Emit or submit one stage-3 SGE job. Returns qsub job id."""
    spec, paths = _subject_sge_spec(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        container=container,
        src_dir=src_dir,
        skip_existing=skip_existing,
        eicab_mask=eicab_mask,
        cd_up_thresh=cd_up_thresh,
        cd_shift_hm=cd_shift_hm,
        venous_min_component_frac=venous_min_component_frac,
        eicab_min_island_fraction=eicab_min_island_fraction,
        eicab_bridge_open_radius=eicab_bridge_open_radius,
        venous_min_branch_points=venous_min_branch_points,
        eicab_prefer_pp=eicab_prefer_pp,
        venous_brain_mask=venous_brain_mask,
        totalseg_device=totalseg_device,
        totalseg_model_dir=totalseg_model_dir,
        backend=backend,
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage3-centerline")
@backend_click_option()
@_stage3_cli_options
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    eicab_mask: str,
    cd_up_thresh: float | None,
    cd_shift_hm: bool | None,
    venous_min_component_frac: float,
    eicab_min_island_fraction: float,
    eicab_bridge_open_radius: int,
    venous_min_branch_points: int,
    eicab_prefer_pp: bool,
    venous_brain_mask: bool,
    totalseg_device: str,
    totalseg_model_dir: Path | None,
) -> None:
    Logger()
    run_subject(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        skip_existing=skip_existing,
        eicab_mask=eicab_mask.lower(),
        cd_up_thresh=cd_up_thresh,
        cd_shift_hm=cd_shift_hm,
        venous_min_component_frac=venous_min_component_frac,
        eicab_min_island_fraction=eicab_min_island_fraction,
        eicab_bridge_open_radius=eicab_bridge_open_radius,
        venous_min_branch_points=venous_min_branch_points,
        eicab_prefer_pp=eicab_prefer_pp,
        venous_brain_mask=venous_brain_mask,
        totalseg_device=totalseg_device,
        totalseg_model_dir=totalseg_model_dir,
    )


__all__ = ["main", "run_subject", "build_subject_sge_command", "submit_subject_sge"]


if __name__ == "__main__":
    main()
