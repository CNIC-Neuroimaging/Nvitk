"""qvtpy stage 5: per-vessel LOC (voxel + tangent) from centerlines."""

from __future__ import annotations

import csv
import json
import shlex
from pathlib import Path
from typing import TextIO

import click

import nvitk
from nvitk.core.backend import setup
from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    submit_stage,
)
from nvitk.core.logger import Logger
from nvitk.morphology.centerline import centerline_tangents
from nvitk.pipes.qvtpy import config as cfg

setup(globals())

log = Logger()


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _stage3_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE3_CENTERLINE_DIR


def _stage5_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE5_LOC_DIR


def run_subject(
    subject: str,
    *,
    output_root: Path,
    skip_existing: bool = False,
    tangent_k_half: int = 2,
) -> Path:
    s3 = _stage3_dir(output_root, subject)
    npz_path = s3 / "centerlines.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing {npz_path}")
    out_dir = _stage5_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    loc_csv = out_dir / "locs.csv"
    if skip_existing and loc_csv.is_file():
        log.info(f"[{subject}] stage5 loc: skip -> {out_dir}")
        return out_dir

    z = np.load(npz_path)
    rows: list[dict[str, float | int]] = []
    for key in sorted(z.files):
        if not key.startswith("arterial_"):
            continue
        label = int(key.split("_", 1)[1])
        pts = z[key]
        if pts.shape[0] < 3:
            continue
        mid = pts.shape[0] // 2
        tangents = centerline_tangents(pts, k_half=tangent_k_half)
        i, j, k = (int(round(pts[mid, 0])), int(round(pts[mid, 1])), int(round(pts[mid, 2])))
        tx, ty, tz = (float(tangents[mid, 0]), float(tangents[mid, 1]), float(tangents[mid, 2]))
        rows.append(
            {
                "vessel_id": label,
                "i": i,
                "j": j,
                "k": k,
                "tangent_x": tx,
                "tangent_y": ty,
                "tangent_z": tz,
            }
        )

    with loc_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["vessel_id", "i", "j", "k", "tangent_x", "tangent_y", "tangent_z"],
        )
        w.writeheader()
        w.writerows(rows)

    (out_dir / "loc_meta.json").write_text(json.dumps({"n_locs": len(rows)}, indent=2), encoding="utf-8")
    log.info(f"[{subject}] stage5 loc -> {loc_csv}")
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
    script = f"{binds.src}nvitk/pipes/qvtpy/stage5_loc_generation.py"
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
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage5_{subject}",
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


@click.command("qvtpy-stage5-loc")
@click.option("--subject", required=True)
@click.option("--nifti-root", type=click.Path(path_type=Path), required=True)
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--tangent-k-half", type=int, default=2, show_default=True)
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    tangent_k_half: int,
) -> None:
    Logger()
    run_subject(
        subject,
        output_root=output_root,
        skip_existing=skip_existing,
        tangent_k_half=tangent_k_half,
    )


__all__ = ["main", "run_subject", "submit_subject_sge"]
