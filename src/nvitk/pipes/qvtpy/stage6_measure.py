"""qvtpy stage 6: LOC-wise velocity / PI / RI from PC phase volumes.

**Inputs**

- Stage-5 ``locs.csv``, AP/RL/FH phase NIfTIs (:func:`~nvitk.io.conversors.phase2volume.discover_phase_inputs`),
  optional stage-4 ``seg_4dflow`` when not re-segmenting in-plane.

**Outputs**

- ``loc_measurements.csv``, optional cross-section QC PNGs under ``stage6_measure/``.
"""

from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import Literal, TextIO

import click
import numpy as np

import nvitk
from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_stage,
)
from nvitk.core.click_backend import backend_click_option
from nvitk.pipes.qvtpy.util.sge_backend import sge_backend_cli_args, sge_stage_extra_env
from nvitk.core.logger import Logger
from nvitk.io.conversors.phase2volume import discover_phase_inputs
from nvitk.io.imageio import imread
from nvitk.measure.hemodynamics import velocity_mm_s_from_phases
from nvitk.pipes.qvtpy import config as cfg
from nvitk.measure.cross_section import ThrAlgorithm
from nvitk.pipes.qvtpy.util.centerline_io import load_arterial_centerlines
from nvitk.pipes.qvtpy.util.loc_measure import run_loc_measurements
from nvitk.pipes.qvtpy.util.measure_qc import save_loc_cross_section_qc_png
from nvitk.pipes.qvtpy.labels import qvtpy_vessel_name

setup(globals())

log = Logger()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_RADIUS_VOX = 10.0


# ---------------------------------------------------------------------------
# Path helpers + contrast / phase I/O
# ---------------------------------------------------------------------------


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _stage3_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE3_CENTERLINE_DIR


def _stage5_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE5_LOC_DIR


def _stage4_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE4_SEG_DIR


def _stage6_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE6_MEASURE_DIR


def _load_seg_4dflow(output_root: Path, subject: str) -> np.ndarray:
    seg_p = _stage4_dir(output_root, subject) / "seg_4dflow.nii.gz"
    if not seg_p.is_file():
        raise FileNotFoundError(f"Missing {seg_p} (run stage4; required when --no-measure-resegment)")
    return as_backend_array(imread(seg_p).data).astype(np.int32)


def _voxel_spacing(ap_img_path: Path) -> tuple[float, float, float]:
    ap_img = imread(ap_img_path)
    sp = ap_img.spacing
    if sp is not None and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    aff = ap_img.affine
    if aff is not None:
        a = np.asarray(aff, dtype=np.float64)
        return (
            float(np.linalg.norm(a[:3, 0])),
            float(np.linalg.norm(a[:3, 1])),
            float(np.linalg.norm(a[:3, 2])),
        )
    return (1.0, 1.0, 1.0)


def _load_contrast(nifti_root: Path, subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = nifti_root / subject / "4DFlow"
    cd_p = sub / "ComplexDifference_3D.nii.gz"
    if not cd_p.is_file():
        cd_p = sub / "ComplexDifference_3D.nii"
    cd = as_backend_array(imread(cd_p).data).astype(np.float64)
    mag = cd
    angio = sub / "Angiography_3D.nii.gz"
    if angio.is_file():
        mag = as_backend_array(imread(angio).data).astype(np.float64)
    vel = np.abs(cd)
    vmag = sub / "VelocityMagnitude_3D.nii.gz"
    if vmag.is_file():
        vel = as_backend_array(imread(vmag).data).astype(np.float64)
    return mag, cd, vel


# ---------------------------------------------------------------------------
# Stage 6: per-LOC flow, PI, RI
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool = False,
    cross_section_radius_vox: float = _DEFAULT_RADIUS_VOX,
    measure_resegment: bool = True,
    measure_thr_algorithm: ThrAlgorithm = "lsthr",
    cross_section_res: int = 0,
    cross_section_plane_interp: int = 1,
    write_cross_section_qc: bool = True,
) -> Path:
    """Measure flow metrics at each LOC; return stage-6 output directory."""
    # ---- Prerequisites: stage5 locs.csv --------------------------------------
    loc_csv = _stage5_dir(output_root, subject) / "locs.csv"
    if not loc_csv.is_file():
        raise FileNotFoundError(f"Missing {loc_csv} (run stage5)")
    out_dir = _stage6_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    meas_csv = out_dir / "loc_measurements.csv"
    if skip_existing and meas_csv.is_file():
        log.info(f"[{subject}] stage6 measure: skip -> {out_dir}")
        return out_dir

    # ---- Phase volumes → mm/s velocity time series ---------------------------
    inputs = discover_phase_inputs(nifti_root / subject)
    voxel_spacing = _voxel_spacing(inputs.ap_phase_path)
    mag, cd, vel_mag = _load_contrast(nifti_root, subject)
    volume_seg: np.ndarray | None = None
    if not measure_resegment:
        volume_seg = _load_seg_4dflow(output_root, subject)

    ap = as_backend_array(imread(inputs.ap_phase_path).data).astype(np.float64)
    rl = as_backend_array(imread(inputs.rl_phase_path).data).astype(np.float64)
    fh = as_backend_array(imread(inputs.fh_phase_path).data).astype(np.float64)
    vx, vy, vz = velocity_mm_s_from_phases(ap, rl, fh)
    if vx.ndim != 4:
        raise ValueError("Expected 4D phase volumes for LOC-wise time series.")
    nt = int(vx.shape[3])

    vel_cols = [f"loc_velocity_mm_s_t{t}" for t in range(nt)]
    flow_cols = [f"loc_flow_ml_s_t{t}" for t in range(nt)]
    fieldnames = [
        "vessel_id",
        "vessel_name",
        "loc_cross_section_radius_vox",
        "loc_cross_section_area_mm2",
        "loc_mean_velocity_mm_s",
        "loc_mean_flow_ml_s",
        "loc_pi",
        "loc_ri",
        *vel_cols,
        *flow_cols,
    ]

    # ---- Optional centerlines for QC -------------------------------------------
    arterial_cls: dict[int, np.ndarray] | None = None
    qc_dir = out_dir / "cross-sections"
    if write_cross_section_qc:
        s3 = _stage3_dir(output_root, subject)
        try:
            arterial_cls = {
                int(k): np.asarray(v)
                for k, v in load_arterial_centerlines(s3, min_points=3).items()
            }
        except FileNotFoundError:
            log.warning(f"[{subject}] stage6 QC: missing stage3 centerlines, skipping PNGs")

    # ---- Per-LOC: resegment, masked-plane flow, PI / RI ----------------------
    loc_rows: list[dict[str, str]] = []
    with loc_csv.open(newline="", encoding="utf-8") as fh:
        loc_rows = list(csv.DictReader(fh))

    rows_out = run_loc_measurements(
        loc_rows,
        mag=mag,
        cd=cd,
        vel_mag=vel_mag,
        vx=vx,
        vy=vy,
        vz=vz,
        voxel_spacing=voxel_spacing,
        cross_section_radius_vox=cross_section_radius_vox,
        measure_resegment=measure_resegment,
        thr_algorithm=measure_thr_algorithm,
        cross_section_res=cross_section_res,
        cross_section_plane_interp=cross_section_plane_interp,
        volume_seg=volume_seg,
    )
    qc_paths: list[str] = []
    for row, _rec in zip(loc_rows, rows_out):
        vid = int(row["vessel_id"])
        vname = (row.get("vessel_name") or "").strip() or qvtpy_vessel_name(vid)

        if arterial_cls is not None and vid in arterial_cls:
            seg_id = int(row.get("segment_id") or 0)
            loc_role = str(row.get("loc_role") or "mid")
            cl_idx = int(row.get("centerline_index") or 0)
            qc_name = f"{vname}_seg{seg_id}.png"
            qc_path = qc_dir / qc_name
            try:
                save_loc_cross_section_qc_png(
                    qc_path,
                    cd=cd,
                    mag=mag,
                    vel_mag=vel_mag,
                    centerline_pts=arterial_cls[vid],
                    loc_index=cl_idx,
                    vessel_name=vname,
                    segment_id=seg_id,
                    loc_role=loc_role,
                    voxel_spacing=voxel_spacing,
                    radius_vox=cross_section_radius_vox,
                    cross_section_res=cross_section_res,
                    plane_interp_order=int(cross_section_plane_interp),
                    measure_resegment=measure_resegment,
                    thr_algorithm=measure_thr_algorithm,
                    volume_seg=volume_seg,
                    volume_label_id=vid,
                )
                qc_paths.append(str(qc_path.relative_to(out_dir)))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                log.warning(f"[{subject}] QC PNG failed for {vname}: {exc}")

    # ---- Write loc_measurements.csv + measure_meta.json ----------------------
    with meas_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    (out_dir / "measure_meta.json").write_text(
        json.dumps(
            {
                "n_rows": len(rows_out),
                "n_timepoints": nt,
                "measure_resegment": bool(measure_resegment),
                "measure_thr_algorithm": str(measure_thr_algorithm),
                "cross_section_radius_vox": float(cross_section_radius_vox),
                "cross_section_plane_interp": int(cross_section_plane_interp),
                "reported_flow_velocity_as_magnitude": True,
                "cross_section_qc_dir": "cross-sections",
                "cross_section_qc_pngs": qc_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"[{subject}] stage6 measure -> {meas_csv}")
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
    cross_section_radius_vox: float = _DEFAULT_RADIUS_VOX,
    measure_resegment: bool = True,
    measure_thr_algorithm: str = "lsthr",
    cross_section_res: int = 0,
    cross_section_plane_interp: int = 1,
    backend: str = "gpu",
) -> str:
    """Emit or submit one stage-6 SGE job. Returns qsub job id."""
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    parts = [
        *python_module_argv("nvitk.pipes.qvtpy.stage6_measure"),
        *sge_backend_cli_args(backend),
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
        "--output-root",
        shlex.quote(binds.output),
        "--cross-section-radius-vox",
        str(float(cross_section_radius_vox)),
        "--cross-section-res",
        str(int(cross_section_res)),
        "--cross-section-plane-interp",
        str(int(cross_section_plane_interp)),
        "--measure-thr-algorithm",
        shlex.quote(str(measure_thr_algorithm)),
    ]
    if skip_existing:
        parts.append("--skip-existing")
    if not measure_resegment:
        parts.append("--no-measure-resegment")
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
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage6_{subject}",
        python_cmd=python_cmd,
        resources=SgeResources(
            project=cfg.SGE_PROJECT,
            account=cfg.SGE_ACCOUNT,
            ngpu=0,
            h_vmem=cfg.SGE_H_VMEM,
            queue=cfg.SGE_QUEUE,
        ),
        binds=binds,
        use_nv=False,
        extra_env=sge_stage_extra_env(binds.src, backend),
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage6-measure")
@backend_click_option()
@click.option("--subject", required=True)
@click.option("--nifti-root", type=click.Path(path_type=Path), required=True)
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--cross-section-radius-vox", type=float, default=_DEFAULT_RADIUS_VOX, show_default=True)
@click.option(
    "--measure-resegment/--no-measure-resegment",
    default=True,
    show_default=True,
    help="Recompute in-plane segmentation at each LOC (default on).",
)
@click.option(
    "--measure-thr-algorithm",
    type=click.Choice(["lsthr", "lthr", "otsu"], case_sensitive=False),
    default="lsthr",
    show_default=True,
    help="In-plane threshold when --measure-resegment (ignored if resegment off).",
)
@click.option("--cross-section-res", type=int, default=0, show_default=True)
@click.option("--cross-section-plane-interp", type=int, default=1, show_default=True)
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    cross_section_radius_vox: float,
    measure_resegment: bool,
    measure_thr_algorithm: str,
    cross_section_res: int,
    cross_section_plane_interp: int,
) -> None:
    Logger()
    run_subject(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        skip_existing=skip_existing,
        cross_section_radius_vox=cross_section_radius_vox,
        measure_resegment=measure_resegment,
        measure_thr_algorithm=measure_thr_algorithm.lower(),  # type: ignore[arg-type]
        cross_section_res=cross_section_res,
        cross_section_plane_interp=cross_section_plane_interp,
    )


__all__ = ["main", "run_subject", "submit_subject_sge"]


if __name__ == "__main__":
    main()
