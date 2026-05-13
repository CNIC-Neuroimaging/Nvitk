"""qvtpy stage 6: LOC-wise velocity / PI / RI from PC phase volumes."""

from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import TextIO

import click

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
    mean_flow_ml_s,
    pulsatility_index,
    resistivity_index,
    through_plane_velocity_series,
    velocity_mm_s_from_phases,
)
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.labels import eicab_vessel_name

setup(globals())

log = Logger()

# Disk radius (voxels) for cross-sectional flow proxy, consistent with MATLAB QVT-style planes.
_LOC_CROSS_SECTION_RADIUS_VOX: float = 10.0


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _stage5_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE5_LOC_DIR


def _stage6_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE6_MEASURE_DIR


def _min_voxel_spacing_mm(ap_img_path: Path) -> float:
    ap_img = imread(ap_img_path)
    sp = ap_img.spacing
    if sp is not None and len(sp) >= 3:
        return min(float(sp[0]), float(sp[1]), float(sp[2]))
    aff = ap_img.affine
    if aff is not None:
        a = np.asarray(aff, dtype=np.float64)
        return float(min(np.linalg.norm(a[:3, 0]), np.linalg.norm(a[:3, 1]), np.linalg.norm(a[:3, 2])))
    return 1.0


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool = False,
) -> Path:
    loc_csv = _stage5_dir(output_root, subject) / "locs.csv"
    if not loc_csv.is_file():
        raise FileNotFoundError(f"Missing {loc_csv} (run stage5)")
    out_dir = _stage6_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    meas_csv = out_dir / "loc_measurements.csv"
    if skip_existing and meas_csv.is_file():
        log.info(f"[{subject}] stage6 measure: skip -> {out_dir}")
        return out_dir

    inputs = discover_phase_inputs(nifti_root / subject)
    min_sp = _min_voxel_spacing_mm(inputs.ap_phase_path)
    radius_mm = float(_LOC_CROSS_SECTION_RADIUS_VOX) * min_sp
    area_mm2 = float(np.pi) * radius_mm * radius_mm

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

    rows_out: list[dict[str, float | int | str]] = []
    with loc_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            i = int(row["i"])
            j = int(row["j"])
            k = int(row["k"])
            vid = int(row["vessel_id"])
            vname = (row.get("vessel_name") or "").strip() or eicab_vessel_name(vid)
            tang = np.array(
                [float(row["tangent_x"]), float(row["tangent_y"]), float(row["tangent_z"])],
                dtype=np.float64,
            )
            ts = through_plane_velocity_series(vx, vy, vz, i=i, j=j, k=k, tangent=tang)
            ts_np = as_backend_array(ts).astype(np.float64).reshape(-1)
            ts_pos = np.abs(ts_np)
            ts2 = ts_pos.reshape(1, -1)
            q_series = ts_pos * (area_mm2 / 1000.0)

            rec: dict[str, float | int | str] = {
                "vessel_id": vid,
                "vessel_name": vname,
                "loc_cross_section_radius_vox": float(_LOC_CROSS_SECTION_RADIUS_VOX),
                "loc_cross_section_area_mm2": float(area_mm2),
                "loc_mean_velocity_mm_s": float(mean_flow_ml_s(ts2, None)[0]),
                "loc_mean_flow_ml_s": float(np.mean(q_series)),
                "loc_pi": float(pulsatility_index(ts2)[0]),
                "loc_ri": float(resistivity_index(ts2)[0]),
            }
            for t in range(nt):
                rec[f"loc_velocity_mm_s_t{t}"] = float(ts_pos[t])
                rec[f"loc_flow_ml_s_t{t}"] = float(q_series[t])
            rows_out.append(rec)

    with meas_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    (out_dir / "measure_meta.json").write_text(
        json.dumps({"n_rows": len(rows_out), "n_timepoints": nt}, indent=2), encoding="utf-8"
    )
    log.info(f"[{subject}] stage6 measure -> {meas_csv}")
    return out_dir


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
def main(subject: str, nifti_root: Path, output_root: Path, skip_existing: bool) -> None:
    Logger()
    run_subject(subject, nifti_root=nifti_root, output_root=output_root, skip_existing=skip_existing)


__all__ = ["main", "run_subject", "submit_subject_sge"]
