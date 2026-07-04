"""qvtpy stage 5: per-vessel LOC (voxel + tangent) from centerlines.

**Inputs**

- Stage-3/4 centerlines, contrast volumes (CD, angio, velocity magnitude).

**Outputs**

- ``locs.csv``, ``loc_meta.json`` under ``stage5_loc_generation/``.
"""

from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import TextIO

import click
import pandas as pd

import nvitk
from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.cluster.sge import (
    ClusterPaths,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_stage,
)
from nvitk.core.click_backend import backend_click_option
from nvitk.pipes.qvtpy.util.sge_backend import (
    sge_backend_cli_args,
    sge_qvtpy_stage_resources,
    sge_stage_extra_env,
    sge_stage_use_nv,
)
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.util.flow_volume_masks import binary_vessel_segment_cd, venous_search_region
from nvitk.pipes.qvtpy.util.loc_selection import (
    loc_record_to_dict,
    select_arterial_locs,
    select_venous_locs,
)
from nvitk.pipes.qvtpy.util.centerline_io import load_centerline_meta, load_centerlines
from nvitk.pipes.qvtpy.util.mask_cleaning import clean_venous_slab_mask

setup(globals())

log = Logger()


# ---------------------------------------------------------------------------
# Path helpers + contrast volume I/O
# ---------------------------------------------------------------------------


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _stage3_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE3_CENTERLINE_DIR


def _stage5_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE5_LOC_DIR


def _cd_path(nifti_root: Path, subject: str) -> Path:
    p = nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii.gz"
    if p.is_file():
        return p
    return nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii"


def _voxel_spacing_from_meta(meta: dict) -> tuple[float, float, float]:
    sp = meta.get("spacing")
    if sp and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    return (1.0, 1.0, 1.0)


def _load_contrast_volumes(nifti_root: Path, subject: str) -> tuple[Any, Any, Any, tuple[float, float, float]]:
    sub = nifti_root / subject / "4DFlow"
    cd_p = _cd_path(nifti_root, subject)
    cd_img = imread(cd_p)
    cd = as_backend_array(cd_img.data).astype(np.float64)
    sp = _voxel_spacing_from_meta(dict(cd_img.metadata or {}))

    mag = cd
    angio = sub / "Angiography_3D.nii.gz"
    if angio.is_file():
        mag = as_backend_array(imread(angio).data).astype(np.float64)

    vel = np.abs(cd)
    vmag = sub / "VelocityMagnitude_3D.nii.gz"
    if vmag.is_file():
        vel = as_backend_array(imread(vmag).data).astype(np.float64)
    return mag, cd, vel, sp


# ---------------------------------------------------------------------------
# Stage 5: LOC generation (arterial + venous)
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool = False,
    tangent_k_half: int = 2,
    loc_arterial_strategy: str = "qvtpy",
    cross_section_radius_vox: float = 10.0,
    venous_min_component_frac: float = 0.005,
    loc_endpoint_inset_frac: float = 0.08,
) -> Path:
    """Place arterial and venous LOCs; return stage-5 output directory."""
    del tangent_k_half  # tangents computed inside loc_selection
    # ---- Prerequisites: stage3 centerlines -----------------------------------
    s3 = _stage3_dir(output_root, subject)
    from nvitk.pipes.qvtpy.util.centerline_io import centerlines_mask_path

    if not centerlines_mask_path(s3).is_file():
        raise FileNotFoundError(f"Missing {centerlines_mask_path(s3)} (run stage3)")
    out_dir = _stage5_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    loc_csv = out_dir / "locs.csv"
    if skip_existing and loc_csv.is_file():
        log.info(f"[{subject}] stage5 loc: skip -> {out_dir}")
        return out_dir

    s4 = output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE4_SEG_DIR
    arterial, venous, cl_meta = load_centerlines(s3, min_points=5, stage4_dir=s4)
    meta = load_centerline_meta(s3)

    arterial_seg = None
    seg_path = s4 / "seg_4dflow.nii.gz"
    if seg_path.is_file():
        arterial_seg = as_backend_array(imread(seg_path).data).astype(np.int32, copy=False)

    mag, cd, vel_mag, voxel_spacing = _load_contrast_volumes(nifti_root, subject)

    # ---- Venous mask for LOC heuristics (CD binary ∧ superior slab) ----------
    venous_mask = None
    shape3 = cd.shape
    venous_region = venous_search_region(shape3)
    vessel_bin, _ = binary_vessel_segment_cd(cd)
    venous_mask = as_backend_array(
        clean_venous_slab_mask(
            vessel_bin.astype(bool) & venous_region,
            min_fraction=venous_min_component_frac,
        )
    )

    name_to_id = {str(k): int(v) for k, v in (meta.get("venous_label_by_name") or {}).items()}

    # ---- QVTplus-style LOC selection (arterial + venous) ---------------------
    rows: list[dict[str, float | int | str]] = []
    arterial_recs, arterial_meta = select_arterial_locs(
        arterial,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        voxel_spacing=voxel_spacing,
        radius_vox=cross_section_radius_vox,
        strategy=loc_arterial_strategy,
        endpoint_inset_frac=loc_endpoint_inset_frac,
        arterial_seg=arterial_seg,
    )
    for rec in arterial_recs:
        rows.append(loc_record_to_dict(rec))
    for rec in select_venous_locs(
        venous,
        venous_mask=venous_mask,
        name_to_id=name_to_id,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        voxel_spacing=voxel_spacing,
        radius_vox=cross_section_radius_vox,
    ):
        rows.append(loc_record_to_dict(rec))

    # ---- Write locs.csv, locs.xlsx, loc_meta.json ----------------------------
    fieldnames = [
        "vessel_id",
        "vessel_name",
        "segment_id",
        "loc_role",
        "centerline_index",
        "i",
        "j",
        "k",
        "centerline_x",
        "centerline_y",
        "centerline_z",
        "tangent_x",
        "tangent_y",
        "tangent_z",
        "loc_circularity",
        "loc_cross_section_area_mm2",
    ]
    with loc_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    pd.DataFrame(rows).to_excel(out_dir / "locs.xlsx", index=False)
    (out_dir / "loc_meta.json").write_text(
        json.dumps(
            {
                "n_locs": len(rows),
                "loc_arterial_strategy": loc_arterial_strategy,
                "loc_endpoint_inset_frac": float(loc_endpoint_inset_frac),
                "cross_section_radius_vox": float(cross_section_radius_vox),
                "centerline_meta_source": cl_meta.get("source", "stage3"),
                **arterial_meta,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"[{subject}] stage5 loc -> {loc_csv}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


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
    loc_arterial_strategy: str = "qvtpy",
    cross_section_radius_vox: float = 10.0,
    venous_min_component_frac: float = 0.005,
    loc_endpoint_inset_frac: float = 0.08,
    backend: str = "gpu",
) -> str:
    """Emit or submit one stage-5 SGE job. Returns qsub job id."""
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    parts = [
        *python_module_argv("nvitk.pipes.qvtpy.stage5_loc_generation"),
        *sge_backend_cli_args(backend),
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
        "--output-root",
        shlex.quote(binds.output),
        "--loc-arterial-strategy",
        shlex.quote(loc_arterial_strategy),
        "--cross-section-radius-vox",
        str(float(cross_section_radius_vox)),
        "--venous-min-component-frac",
        str(float(venous_min_component_frac)),
        "--loc-endpoint-inset-frac",
        str(float(loc_endpoint_inset_frac)),
    ]
    if skip_existing:
        parts.append("--skip-existing")
    python_cmd = " ".join(parts)
    paths = ClusterPaths(
        src=src_p,
        container=container,
        models=None,
        data_root=nifti_root,
        output_root=output_root,
        log_dir=cfg.SGE_LOG_DIR,
        err_dir=cfg.SGE_ERR_DIR,
    )
    spec = StageSpec(
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage5_{subject}",
        python_cmd=python_cmd,
        resources=sge_qvtpy_stage_resources(backend),
        binds=binds,
        use_nv=sge_stage_use_nv(backend),
        extra_env=sge_stage_extra_env(binds.src, backend),
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage5-loc")
@backend_click_option()
@click.option("--subject", required=True)
@click.option("--nifti-root", type=click.Path(path_type=Path), required=True)
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--tangent-k-half", type=int, default=2, show_default=True)
@click.option(
    "--loc-arterial-strategy",
    type=click.Choice(["qvtpy", "midpoint"]),
    default="qvtpy",
    show_default=True,
)
@click.option("--cross-section-radius-vox", type=float, default=10.0, show_default=True)
@click.option("--venous-min-component-frac", type=float, default=0.005, show_default=True)
@click.option("--loc-endpoint-inset-frac", type=float, default=0.08, show_default=True)
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    tangent_k_half: int,
    loc_arterial_strategy: str,
    cross_section_radius_vox: float,
    venous_min_component_frac: float,
    loc_endpoint_inset_frac: float,
) -> None:
    Logger()
    run_subject(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        skip_existing=skip_existing,
        tangent_k_half=tangent_k_half,
        loc_arterial_strategy=loc_arterial_strategy,
        cross_section_radius_vox=cross_section_radius_vox,
        venous_min_component_frac=venous_min_component_frac,
        loc_endpoint_inset_frac=loc_endpoint_inset_frac,
    )


__all__ = ["main", "run_subject", "submit_subject_sge"]


if __name__ == "__main__":
    main()
