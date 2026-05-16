"""qvtpy stage 6: LOC-wise velocity / PI / RI from PC phase volumes."""

from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import TextIO

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
    submit_stage,
)
from nvitk.core.logger import Logger
from nvitk.io.conversors.phase2volume import discover_phase_inputs
from nvitk.io.imageio import imread
from nvitk.measure.hemodynamics import (
    mean_velocity_mm_s,
    pulsatility_index,
    resistivity_index,
    velocity_mm_s_from_phases,
)
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.util.cross_section import (
    flow_series_ml_s,
    masked_plane_velocity_series,
    segment_at_point,
)
from nvitk.pipes.qvtpy.labels import eicab_vessel_name

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


def _stage5_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE5_LOC_DIR


def _stage6_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE6_MEASURE_DIR


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
    cross_section_res: int = 0,
) -> Path:
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

    # ---- Per-LOC: resegment, masked-plane flow, PI / RI ----------------------
    rows_out: list[dict[str, float | int | str]] = []
    with loc_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vid = int(row["vessel_id"])
            vname = (row.get("vessel_name") or "").strip() or eicab_vessel_name(vid)
            center = np.array(
                [
                    float(row["centerline_x"]),
                    float(row["centerline_y"]),
                    float(row["centerline_z"]),
                ],
                dtype=np.float64,
            )
            tang = np.array(
                [float(row["tangent_x"]), float(row["tangent_y"]), float(row["tangent_z"])],
                dtype=np.float64,
            )
            area_csv = row.get("loc_cross_section_area_mm2")
            xs = segment_at_point(
                center,
                tang,
                mag=mag,
                cd=cd,
                vel_mag=vel_mag,
                voxel_spacing=voxel_spacing,
                radius_vox=cross_section_radius_vox,
                cross_section_res=cross_section_res,
            )
            if measure_resegment or not area_csv:
                area_mm2 = float(xs.area_mm2)
            else:
                area_mm2 = float(area_csv)

            vel_ts = masked_plane_velocity_series(vx, vy, vz, xs)
            flow_ts = flow_series_ml_s(vel_ts, area_mm2)
            flow_2d = flow_ts.reshape(1, -1)

            rec: dict[str, float | int | str] = {
                "vessel_id": vid,
                "vessel_name": vname,
                "loc_cross_section_radius_vox": float(cross_section_radius_vox),
                "loc_cross_section_area_mm2": float(area_mm2),
                "loc_mean_velocity_mm_s": float(mean_velocity_mm_s(vel_ts)),
                "loc_mean_flow_ml_s": float(np.mean(flow_ts)),
                "loc_pi": float(pulsatility_index(flow_2d)[0]),
                "loc_ri": float(resistivity_index(flow_2d)[0]),
            }
            for t in range(nt):
                rec[f"loc_velocity_mm_s_t{t}"] = float(vel_ts[t])
                rec[f"loc_flow_ml_s_t{t}"] = float(flow_ts[t])
            rows_out.append(rec)

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
                "cross_section_radius_vox": float(cross_section_radius_vox),
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
) -> str:
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    script = f"{binds.src}nvitk/pipes/qvtpy/stage6_measure.py"
    parts = [
        "python",
        shlex.quote(script),
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
        "--output-root",
        shlex.quote(binds.output),
        "--cross-section-radius-vox",
        str(float(cross_section_radius_vox)),
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
        extra_env={"PYTHONPATH": str(binds.src)},
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage6-measure")
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
@click.option("--cross-section-res", type=int, default=0, show_default=True)
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    cross_section_radius_vox: float,
    measure_resegment: bool,
    cross_section_res: int,
) -> None:
    Logger()
    run_subject(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        skip_existing=skip_existing,
        cross_section_radius_vox=cross_section_radius_vox,
        measure_resegment=measure_resegment,
        cross_section_res=cross_section_res,
    )


__all__ = ["main", "run_subject", "submit_subject_sge"]
