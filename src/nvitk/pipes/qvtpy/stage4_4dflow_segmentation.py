"""qvtpy stage 4: multilabel ``seg_4dflow`` (arterial eICAB + four venous regions in the CD venous slab)."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any, TextIO

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
from nvitk.io.imageio import imread, imsave
from skimage.morphology import remove_small_objects  # type: ignore

from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.flow_volume_masks import (
    binary_vessel_segment_cd,
    venous_four_region_labels,
    venous_search_region,
)
from nvitk.pipes.qvtpy.labels import (
    MATLAB_QVT_VENOUS_VESSEL_NAMES,
    VENOUS_REGION_BASE,
)

setup(globals())

log = Logger()


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
    return nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii"


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool = False,
) -> Path:
    s3 = _stage3_dir(output_root, subject)
    eicab_in = s3 / "eicab_in_4dflow.nii.gz"
    if not eicab_in.is_file():
        raise FileNotFoundError(f"Missing {eicab_in} (run stage3)")
    cd_p = _cd_path(nifti_root, subject)
    out_dir = _stage4_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / "seg_4dflow.nii.gz"
    meta_path = out_dir / "segmentation_meta.json"
    if skip_existing and seg_path.is_file() and meta_path.is_file():
        log.info(f"[{subject}] stage4 seg: skip -> {out_dir}")
        return out_dir

    e_lab = as_backend_array(imread(eicab_in).data).astype(np.int32, copy=False)
    cd = as_backend_array(imread(cd_p).data).astype(np.float64)
    shape3 = e_lab.shape

    vessel_bin, sliding_opt_thresh = binary_vessel_segment_cd(cd)
    ven_slab = venous_search_region(shape3)

    # Arterial: keep warped eICAB where positive.
    seg = np.where(e_lab > 0, e_lab, 0).astype(np.int32, copy=False)

    # Venous: CD-based foreground ∩ venous slab ∩ no arterial label; clean; four largest CCs → 31–34.
    ven_raw = vessel_bin & ven_slab & (e_lab == 0)
    n_raw = int(np.count_nonzero(ven_raw))
    min_ven = max(1, int(round(0.005 * max(n_raw, 1))))

    ven_clean = as_backend_array(remove_small_objects(to_numpy(ven_raw), min_size=min_ven, connectivity=1))
    ven_labels = venous_four_region_labels(ven_clean, region_label_base=VENOUS_REGION_BASE, n_regions=4)
    seg = np.where(ven_labels > 0, ven_labels, seg).astype(np.int32, copy=False)

    ref_img = imread(eicab_in)
    imsave(seg_path, seg, metadata=dict(ref_img.metadata or {}))
    meta_path.write_text(
        json.dumps(
            {
                "subject": subject,
                "binary_vessel_sliding_threshold_opt": float(sliding_opt_thresh),
                "venous_slab_axis1_third": int(max(1, round(shape3[1] / 3.0))),
                "venous_area_open_fraction_of_raw": 0.005,
                "venous_label_ids": [int(VENOUS_REGION_BASE + i) for i in range(4)],
                "venous_name_hints_size_order": list(MATLAB_QVT_VENOUS_VESSEL_NAMES),
                "note": "Venous IDs 31–34 are the four largest CCs in the venous slab (size rank), not atlas-matched to name hints.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"[{subject}] stage4 segmentation -> {seg_path}")
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
def main(subject: str, nifti_root: Path, output_root: Path, skip_existing: bool) -> None:
    Logger()
    run_subject(subject, nifti_root=nifti_root, output_root=output_root, skip_existing=skip_existing)


__all__ = ["main", "run_subject", "submit_subject_sge"]
