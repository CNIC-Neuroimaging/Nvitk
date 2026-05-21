"""qvtpy stage 4: per-vessel local CD threshold + region growing → ``seg_4dflow``.

**Inputs**

- ``ComplexDifference_3D``, stage-3 ``centerlines_mask``, optional ``eicab_in_4dflow``.

**Outputs**

- ``seg_4dflow.nii.gz``, ``segmentation_meta.json``, stage-4 centerline exports.
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
    SgeResources,
    SingularityBinds,
    StageSpec,
    python_module_argv,
    submit_stage,
)
from nvitk.core.click_backend import backend_click_option
from nvitk.pipes.qvtpy.util.sge_backend import sge_backend_cli_args, sge_stage_extra_env
from nvitk.core.logger import Logger
from nvitk.io.imageio import imread, imsave
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.util.centerline_io import (
    centerline_meta_path,
    centerlines_mask_path,
    export_centerlines_from_segmentation,
    load_venous_centerlines,
)
from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import (
    ThrAlgorithm,
    VESSEL_EXTRA_PADDING,
    _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    _ACOMM_JUNCTION_RADIUS_DEFAULT,
    _DEFAULT_RG_INTENSITY_FRAC,
    _RG_INTENSITY_FRAC_EXPLORE,
    build_seg_4dflow_local,
    vessel_stats_to_dict,
)

setup(globals())

log = Logger()

EICAB_IN_4DFLOW_NIFTI = "eicab_in_4dflow.nii.gz"


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


def _segmentation_meta(
    *,
    subject: str,
    nifti_root: Path,
    cl_path: Path,
    eicab_in_4dflow: str | None,
    crop_padding_bbox: int,
    thr_algorithm: str,
    region_growing: bool,
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    aca_sequential_grow: bool,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
    aca_sequential_grow_info: dict[str, Any] | None,
    vessels: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "subject": subject,
        "complex_difference": str(_cd_path(nifti_root, subject)),
        "centerlines_mask": str(cl_path),
        "eicab_in_4dflow": eicab_in_4dflow,
        "crop_padding_bbox": int(crop_padding_bbox),
        "vessel_extra_padding": int(VESSEL_EXTRA_PADDING),
        "thr_algorithm": thr_algorithm,
        "region_growing": bool(region_growing),
        "rg_intensity_frac": float(rg_intensity_frac),
        "rg_intensity_frac_explore": float(rg_intensity_frac_explore),
        "rg_intensity_frac_explore_labels": "ACA,MCA,PCA (ids 4-9)",
        "rg_skip_labels": "all venous (ids 31-34)",
        "venous_region_growing": False,
        "comm_segmentation_strategy": "centerline_rg_only",
        "post_threshold_clean": "largest_cc_per_label",
        "centerlines_from_seg": "centerlines_mask_4dflow.nii.gz",
        "cl_barrier_radius": int(cl_barrier_radius),
        "rg_barrier_radius": int(rg_barrier_radius),
        "aca_sequential_grow": bool(aca_sequential_grow),
        "aca_overlap_min_voxels": int(aca_overlap_min_voxels),
        "acomm_junction_radius": int(acomm_junction_radius),
        "aca_sequential_grow_info": aca_sequential_grow_info,
        "vessels": vessels,
    }


# ---------------------------------------------------------------------------
# Stage 4: local CD crop + threshold + largest-CC clean + region growing
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
    rg_intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
    rg_intensity_frac_explore: float = _RG_INTENSITY_FRAC_EXPLORE,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
) -> Path:
    """Build multilabel 4D-flow segmentation locally; return stage-4 output directory."""
    s3 = _stage3_dir(output_root, subject)
    cl_path = centerlines_mask_path(s3)
    if not cl_path.is_file():
        raise FileNotFoundError(f"Missing {cl_path} (run stage3)")

    eicab_path = s3 / EICAB_IN_4DFLOW_NIFTI
    eicab_qvtpy = None
    if eicab_path.is_file():
        eicab_qvtpy = as_backend_array(imread(eicab_path).data).astype(np.int32, copy=False)
    else:
        log.warning(f"[{subject}] stage4: missing {eicab_path}; ACA Voronoi uses centerlines only")

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

    log.step(
        f"per-vessel local CD seg (thr={thr_algorithm}, "
        f"RG={region_growing}, rg_frac={rg_intensity_frac})"
    )
    result = build_seg_4dflow_local(
        cd,
        centerlines_mask,
        eicab_qvtpy=eicab_qvtpy,
        crop_padding_bbox=int(crop_padding_bbox),
        thr_algorithm=thr_algorithm,
        region_growing=bool(region_growing),
        rg_intensity_frac=float(rg_intensity_frac),
        rg_intensity_frac_explore=float(rg_intensity_frac_explore),
        cl_barrier_radius=int(cl_barrier_radius),
        rg_barrier_radius=int(rg_barrier_radius),
        aca_sequential_grow=bool(aca_sequential_grow),
        aca_overlap_min_voxels=int(aca_overlap_min_voxels),
        acomm_junction_radius=int(acomm_junction_radius),
    )

    imsave(seg_path, result.segmentation, metadata=ref_meta)
    if result.vertebral_split is not None:
        (out_dir / "vertebral_split.json").write_text(
            json.dumps(result.vertebral_split.as_dict(), indent=2),
            encoding="utf-8",
        )

    venous_polylines: dict[str, Any] = {}
    venous_label_by_name: dict[str, int] = {}
    if centerline_meta_path(s3).is_file():
        meta3 = json.loads(centerline_meta_path(s3).read_text(encoding="utf-8"))
        venous_label_by_name = {
            str(k): int(v) for k, v in (meta3.get("venous_label_by_name") or {}).items()
        }
        venous_polylines = load_venous_centerlines(s3, min_points=3, meta=meta3)
    export_centerlines_from_segmentation(
        result.segmentation,
        out_dir,
        metadata=ref_meta,
        venous_polylines=venous_polylines,
        venous_label_by_name=venous_label_by_name,
    )

    meta_path.write_text(
        json.dumps(
            _segmentation_meta(
                subject=subject,
                nifti_root=nifti_root,
                cl_path=cl_path,
                eicab_in_4dflow=str(eicab_path) if eicab_path.is_file() else None,
                crop_padding_bbox=crop_padding_bbox,
                thr_algorithm=thr_algorithm,
                region_growing=region_growing,
                rg_intensity_frac=rg_intensity_frac,
                rg_intensity_frac_explore=rg_intensity_frac_explore,
                cl_barrier_radius=cl_barrier_radius,
                rg_barrier_radius=rg_barrier_radius,
                aca_sequential_grow=aca_sequential_grow,
                aca_overlap_min_voxels=aca_overlap_min_voxels,
                acomm_junction_radius=acomm_junction_radius,
                aca_sequential_grow_info=(
                    None
                    if result.aca_sequential_grow is None
                    else result.aca_sequential_grow.as_dict()
                ),
                vessels=[vessel_stats_to_dict(st) for st in result.vessel_stats],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(f"[{subject}] stage4 segmentation -> {seg_path}")
    return out_dir


# ---------------------------------------------------------------------------
# CLI + SGE submission
# ---------------------------------------------------------------------------


def _stage4_cli_options(func):  # type: ignore[no-untyped-def]
    func = click.option("--subject", required=True)(func)
    func = click.option("--nifti-root", type=click.Path(path_type=Path), required=True)(func)
    func = click.option("--output-root", type=click.Path(path_type=Path), required=True)(func)
    func = click.option("--skip-existing", is_flag=True, default=False)(func)
    func = click.option("--crop-padding-bbox", type=int, default=3, show_default=True)(func)
    func = click.option(
        "--4dflow-thr-algorithm",
        "thr_algorithm",
        type=click.Choice(["lsthr", "lthr", "otsu"], case_sensitive=False),
        default="lsthr",
        show_default=True,
    )(func)
    func = click.option(
        "--region-growing/--no-region-growing",
        default=True,
        show_default=True,
    )(func)
    func = click.option(
        "--rg-intensity-frac",
        type=float,
        default=_DEFAULT_RG_INTENSITY_FRAC,
        show_default=True,
        help="RG gate: grow_thresh = max(mean(CD_seeds)*frac, local_thr). Lower = more growth.",
    )(func)
    func = click.option(
        "--rg-intensity-frac-explore",
        type=float,
        default=_RG_INTENSITY_FRAC_EXPLORE,
        show_default=True,
        help="RG frac for ACA/MCA/PCA (lower = explore more).",
    )(func)
    func = click.option("--cl-barrier-radius", type=int, default=2, show_default=True)(func)
    func = click.option("--rg-barrier-radius", type=int, default=3, show_default=True)(func)
    func = click.option(
        "--aca-sequential-grow/--no-aca-sequential-grow",
        default=True,
        show_default=True,
        help="Grow LACA then RACA; second ACA ignores first as RG barrier; fix overlap at junction.",
    )(func)
    func = click.option(
        "--aca-overlap-min-voxels",
        type=int,
        default=_ACA_OVERLAP_MIN_VOXELS_DEFAULT,
        show_default=True,
        help="Min LACA∩RACA voxels to apply convergence correction at AComm.",
    )(func)
    func = click.option(
        "--acomm-junction-radius",
        type=int,
        default=_ACOMM_JUNCTION_RADIUS_DEFAULT,
        show_default=True,
        help="Vox: only overlap within this distance of AComm junction is Voronoi-split.",
    )(func)
    return func


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
    rg_intensity_frac: float = _DEFAULT_RG_INTENSITY_FRAC,
    rg_intensity_frac_explore: float = _RG_INTENSITY_FRAC_EXPLORE,
    cl_barrier_radius: int = 2,
    rg_barrier_radius: int = 3,
    aca_sequential_grow: bool = True,
    aca_overlap_min_voxels: int = _ACA_OVERLAP_MIN_VOXELS_DEFAULT,
    acomm_junction_radius: int = _ACOMM_JUNCTION_RADIUS_DEFAULT,
    backend: str = "gpu",
) -> str:
    """Emit or submit one stage-4 SGE job. Returns qsub job id."""
    src_p = Path(src_dir) if src_dir is not None else _default_nvitk_src_dir()
    binds = SingularityBinds()
    parts = [
        *python_module_argv("nvitk.pipes.qvtpy.stage4_4dflow_segmentation"),
        *sge_backend_cli_args(backend),
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
        "--rg-intensity-frac-explore",
        str(float(rg_intensity_frac_explore)),
        "--cl-barrier-radius",
        str(int(cl_barrier_radius)),
        "--rg-barrier-radius",
        str(int(rg_barrier_radius)),
        "--aca-overlap-min-voxels",
        str(int(aca_overlap_min_voxels)),
        "--acomm-junction-radius",
        str(int(acomm_junction_radius)),
    ]
    if region_growing:
        parts.append("--region-growing")
    else:
        parts.append("--no-region-growing")
    if aca_sequential_grow:
        parts.append("--aca-sequential-grow")
    else:
        parts.append("--no-aca-sequential-grow")
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
        extra_env=sge_stage_extra_env(binds.src, backend),
    )
    return submit_stage(spec, paths, hold_jid=hold_jid, emit=emit)


@click.command("qvtpy-stage4-seg")
@backend_click_option()
@_stage4_cli_options
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    crop_padding_bbox: int,
    thr_algorithm: str,
    region_growing: bool,
    rg_intensity_frac: float,
    rg_intensity_frac_explore: float,
    cl_barrier_radius: int,
    rg_barrier_radius: int,
    aca_sequential_grow: bool,
    aca_overlap_min_voxels: int,
    acomm_junction_radius: int,
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
        rg_intensity_frac_explore=rg_intensity_frac_explore,
        cl_barrier_radius=cl_barrier_radius,
        rg_barrier_radius=rg_barrier_radius,
        aca_sequential_grow=aca_sequential_grow,
        aca_overlap_min_voxels=aca_overlap_min_voxels,
        acomm_junction_radius=acomm_junction_radius,
    )


__all__ = ["EICAB_IN_4DFLOW_NIFTI", "main", "run_subject", "submit_subject_sge"]


if __name__ == "__main__":
    main()
