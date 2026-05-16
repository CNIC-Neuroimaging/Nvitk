"""qvtpy stage 4: per-vessel local CD threshold + optional region growing → ``seg_4dflow``."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import TextIO

import click

import nvitk
from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.cluster.sge import (
    ClusterPaths,
    SgeResources,
    SingularityBinds,
    StageSpec,
    submit_stage,
)
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread, imsave
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.util.centerline_io import centerlines_mask_path
from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import (
    ThrAlgorithm,
    build_seg_4dflow_local,
    vessel_stats_to_dict,
)

setup(globals())

log = Logger()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _default_nvitk_src_dir() -> Path:
    return Path(nvitk.__file__).resolve().parent.parent


def _stage3_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE3_CENTERLINE_DIR


def _stage4_out(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE4_SEG_DIR


def _cd_path(nifti_root: Path, subject: str) -> Path:
    p = nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii.gz"
    if p.is_file():
        return p
    p2 = nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii"
    if p2.is_file():
        return p2
    raise FileNotFoundError(f"Missing ComplexDifference_3D for {subject}")


# ---------------------------------------------------------------------------
# Stage 4: local CD crop + threshold + region growing
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool = False,
    crop_padding_bbox: int = 3,
    thr_algorithm: ThrAlgorithm = "lsthr",
    region_growing: bool = True,
    rg_intensity_frac: float = 0.5,
) -> Path:
    s3 = _stage3_dir(output_root, subject)
    cl_path = centerlines_mask_path(s3)
    if not cl_path.is_file():
        raise FileNotFoundError(f"Missing {cl_path} (run stage3)")

    out_dir = _stage4_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / "seg_4dflow.nii.gz"
    meta_path = out_dir / "segmentation_meta.json"
    if skip_existing and seg_path.is_file() and meta_path.is_file():
        log.info(f"[{subject}] stage4 seg: skip -> {out_dir}")
        return out_dir

    cd_img = imread(_cd_path(nifti_root, subject))
    cd = as_backend_array(cd_img.data).astype(np.float64)
    ref_meta = dict(cd_img.metadata or {})

    cl_img = imread(cl_path)
    centerlines_mask = as_backend_array(cl_img.data).astype(np.int32, copy=False)

    result = build_seg_4dflow_local(
        cd,
        centerlines_mask,
        crop_padding_bbox=int(crop_padding_bbox),
        thr_algorithm=thr_algorithm,
        region_growing=bool(region_growing),
        rg_intensity_frac=float(rg_intensity_frac),
    )

    imsave(seg_path, result.segmentation, metadata=ref_meta)
    meta_path.write_text(
        json.dumps(
            {
                "subject": subject,
                "complex_difference": str(_cd_path(nifti_root, subject)),
                "centerlines_mask": str(cl_path),
                "crop_padding_bbox": int(crop_padding_bbox),
                "thr_algorithm": thr_algorithm,
                "region_growing": bool(region_growing),
                "rg_intensity_frac": float(rg_intensity_frac),
                "vessels": [vessel_stats_to_dict(st) for st in result.vessel_stats],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"[{subject}] stage4 segmentation -> {seg_path}")
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
    crop_padding_bbox: int = 3,
    thr_algorithm: str = "lsthr",
    region_growing: bool = True,
    rg_intensity_frac: float = 0.5,
) -> str:
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    script = f"{binds.src}nvitk/pipes/qvtpy/stage4_4dflow_segmentation.py"
    parts = [
        "python",
        shlex.quote(script),
        "--subject",
        shlex.quote(subject),
        "--nifti-root",
        shlex.quote(binds.data),
        "--output-root",
        shlex.quote(binds.output),
        "--crop-padding-bbox",
        str(int(crop_padding_bbox)),
        "--4dflow-thr-algorithm",
        shlex.quote(str(thr_algorithm).lower()),
        "--rg-intensity-frac",
        str(float(rg_intensity_frac)),
    ]
    if region_growing:
        parts.append("--region-growing")
    else:
        parts.append("--no-region-growing")
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
        job_name=f"{cfg.SGE_JOB_PREFIX}_stage4_{subject}",
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


@click.command("qvtpy-stage4-seg")
@click.option("--subject", required=True)
@click.option("--nifti-root", type=click.Path(path_type=Path), required=True)
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--crop-padding-bbox", type=int, default=0, show_default=True)
@click.option(
    "--4dflow-thr-algorithm",
    "thr_algorithm",
    type=click.Choice(["lsthr", "lthr", "otsu"], case_sensitive=False),
    default="lsthr",
    show_default=True,
    help="lsthr: local sliding threshold (no FWHM); lthr: with FWHM; otsu: Otsu on crop.",
)
@click.option(
    "--region-growing/--no-region-growing",
    default=True,
    show_default=True,
    help="Expand each label along high CD into unlabeled voxels only.",
)
@click.option("--rg-intensity-frac", type=float, default=0.5, show_default=True)
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    crop_padding_bbox: int,
    thr_algorithm: str,
    region_growing: bool,
    rg_intensity_frac: float,
) -> None:
    Logger()
    run_subject(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        skip_existing=skip_existing,
        crop_padding_bbox=crop_padding_bbox,
        thr_algorithm=thr_algorithm.lower(),  # type: ignore[arg-type]
        region_growing=region_growing,
        rg_intensity_frac=rg_intensity_frac,
    )


__all__ = ["main", "run_subject", "submit_subject_sge"]
