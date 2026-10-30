"""qvtpy stage 4: per-vessel local CD threshold + blood flood -> ``seg_4dflow``.

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
from nvitk.pipes.qvtpy.util.centerline.centerline_io import (
    centerline_meta_path,
    centerlines_mask_path,
    export_centerlines_from_segmentation,
    load_arterial_centerlines,
    load_venous_centerlines,
)
from nvitk.pipes.qvtpy.util.segmentation.flood_options import (
    FLOOD_DEFAULTS,
    flood_cli_args,
    flood_click_options,
    flood_settings,
)
from nvitk.pipes.qvtpy.util.segmentation.vessel_cd_segmentation import (
    ThrAlgorithm,
    VESSEL_EXTRA_PADDING,
    build_seg_4dflow_local,
    vessel_stats_to_dict,
)

setup(globals())

log = Logger()

EICAB_IN_4DFLOW_NIFTI = "eicab_in_4dflow.nii.gz"
EICAB_IN_4DFLOW_EICAB_IDS_NIFTI = "eicab_in_4dflow_eicab_ids.nii.gz"


# Path helpers


def _default_nvitk_src_dir() -> Path:
    """Repo ``src/`` directory inferred from the installed ``nvitk`` package location."""
    return Path(nvitk.__file__).resolve().parent.parent


def _stage3_dir(output_root: Path, subject: str) -> Path:
    """Stage 3 (centerline) output directory for *subject* under *output_root*."""
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE3_CENTERLINE_DIR


def _stage4_out(output_root: Path, subject: str) -> Path:
    """Stage 4 (4D-flow segmentation) output directory for *subject* under *output_root*."""
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE4_SEG_DIR


def _cd_path(nifti_root: Path, subject: str) -> Path:
    """Path to *subject*'s ``ComplexDifference_3D`` NIfTI (``.nii.gz`` preferred over ``.nii``)."""
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
    cl_barrier_radius: int,
    segment_acomm: bool,
    flood: dict[str, Any],
    vessels: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the stage-4 ``segmentation_meta.json`` payload: inputs, threshold and flood
    settings, and per-vessel results."""
    return {
        "subject": subject,
        "complex_difference": str(_cd_path(nifti_root, subject)),
        "centerlines_mask": str(cl_path),
        "eicab_in_4dflow": eicab_in_4dflow,
        "crop_padding_bbox": int(crop_padding_bbox),
        "vessel_extra_padding": int(VESSEL_EXTRA_PADDING),
        "thr_algorithm": thr_algorithm,
        "cl_barrier_radius": int(cl_barrier_radius),
        "segment_acomm": bool(segment_acomm),
        "comm_segmentation_strategy": "centerline_plus_threshold_then_flood_near_field",
        "post_threshold_clean": "largest_cc_per_label",
        "growth": "region_growing (ICAs, basilar) + blood_flood (ACA / MCA / PCA)",
        **{k: v for k, v in flood.items()},
        "centerlines_from_seg": "centerlines_mask_4dflow.nii.gz",
        "vessels": vessels,
    }


# Stage 4: local CD crop + threshold + largest-CC clean + blood flood


def load_stage3_eicab(s3: Path, subject: str, stage: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Stage 3's registered eICAB labels: ``(qvtpy ids, eICAB's own ids)``, ``None`` when missing.
    The qvtpy ones anchor the ACAs' components and the AComm; the native ones wall the basilar's
    region growing off the eICAB SCAs (not segmented here)."""
    out: list[np.ndarray | None] = []
    for name, missing in ((EICAB_IN_4DFLOW_NIFTI, "the ACAs' components are not anchored on eICAB"),
                          (EICAB_IN_4DFLOW_EICAB_IDS_NIFTI, "the basilar's growth has no eICAB SCA walls")):
        path = s3 / name
        if path.is_file():
            out.append(as_backend_array(imread(path).data).astype(np.int32, copy=False))
        else:
            out.append(None)
            log.warning(f"[{subject}] {stage}: missing {path}; {missing}")
    return out[0], out[1]


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool = False,
    crop_padding_bbox: int = 3,
    thr_algorithm: ThrAlgorithm = "lsthr",
    cl_barrier_radius: int = 2,
    segment_acomm: bool = False,
    **flood: Any,
) -> Path:
    """Build the multilabel 4D-flow segmentation locally; return the stage-4 output directory.
    *flood*: the blood-flood settings (:data:`FLOOD_DEFAULTS`)."""
    flood_kw = flood_settings(**flood)
    s3 = _stage3_dir(output_root, subject)
    cl_path = centerlines_mask_path(s3)
    if not cl_path.is_file():
        raise FileNotFoundError(f"Missing {cl_path} (run stage3)")

    eicab_path = s3 / EICAB_IN_4DFLOW_NIFTI
    eicab_qvtpy, eicab_native = load_stage3_eicab(s3, subject, "stage4")

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

    log.step(f"[{subject}] per-vessel local CD seg (thr={thr_algorithm}) + blood flood "
             f"({'on' if flood_kw['flood_expand'] else 'off'})")
    result = build_seg_4dflow_local(
        cd,
        centerlines_mask,
        eicab_qvtpy=eicab_qvtpy,
        eicab_native=eicab_native,
        spacing=tuple(float(v) for v in cd_img.spacing[:3]),
        crop_padding_bbox=int(crop_padding_bbox),
        thr_algorithm=thr_algorithm,
        cl_barrier_radius=int(cl_barrier_radius),
        segment_acomm=bool(segment_acomm),
        **flood_kw,
    )

    imsave(seg_path, result.segmentation, metadata=ref_meta)
    if result.vertebral_split is not None:
        (out_dir / "vertebral_split.json").write_text(
            json.dumps(result.vertebral_split.as_dict(), indent=2),
            encoding="utf-8",
        )

    venous_polylines: dict[str, Any] = {}
    venous_label_by_name: dict[str, int] = {}
    prefer_arterial: dict[int, Any] = {}
    if centerline_meta_path(s3).is_file():
        meta3 = json.loads(centerline_meta_path(s3).read_text(encoding="utf-8"))
        venous_label_by_name = {
            str(k): int(v) for k, v in (meta3.get("venous_label_by_name") or {}).items()
        }
        venous_polylines = load_venous_centerlines(s3, min_points=3, meta=meta3)
        if centerlines_mask_path(s3).is_file():
            prefer_arterial = load_arterial_centerlines(s3, min_points=3, meta=meta3)
        stage3_arterial_labels = {
            int(x) for x in (meta3.get("arterial_labels") or [])
        }
        # Honor stage-3 PCOMM drops: never keep a mask/CL for a PCOMM that was
        # filtered by pcomm_min_points.
        from nvitk.pipes.qvtpy.labels import QVTPY_PCOMM_IDS

        seg_arr = as_backend_array(result.segmentation)
        cleared = []
        for lid in QVTPY_PCOMM_IDS:
            if int(lid) in stage3_arterial_labels:
                continue
            if int(np.count_nonzero(seg_arr == int(lid))) == 0:
                continue
            seg_arr[seg_arr == int(lid)] = 0
            cleared.append(int(lid))
        if cleared:
            result.segmentation = seg_arr
            imsave(seg_path, seg_arr, metadata=ref_meta)
            log.info(
                f"[{subject}] stage4: cleared dropped PCOMM mask label(s): {cleared}"
            )
    export_centerlines_from_segmentation(
        result.segmentation,
        out_dir,
        metadata=ref_meta,
        venous_polylines=venous_polylines,
        venous_label_by_name=venous_label_by_name,
        prefer_polylines=prefer_arterial,
    )

    meta_doc = _segmentation_meta(
        subject=subject,
        nifti_root=nifti_root,
        cl_path=cl_path,
        eicab_in_4dflow=str(eicab_path) if eicab_path.is_file() else None,
        crop_padding_bbox=crop_padding_bbox,
        thr_algorithm=thr_algorithm,
        cl_barrier_radius=cl_barrier_radius,
        segment_acomm=segment_acomm,
        flood=flood_kw,
        vessels=[vessel_stats_to_dict(st) for st in result.vessel_stats],
    )
    if result.flood is not None:
        meta_doc["flood_result"] = result.flood
    if result.siphon is not None:
        meta_doc["siphon_correction"] = result.siphon
    meta_path.write_text(
        json.dumps(meta_doc, indent=2),
        encoding="utf-8",
    )
    log.info(f"[{subject}] stage4 segmentation -> {seg_path}")
    return out_dir


# CLI + SGE submission


def _stage4_cli_options(func):
    """Decorator applying all shared stage-4 click options (subject / paths, threshold, blood flood)."""
    func = flood_click_options()(func)
    func = click.option(
        "--segment-acomm/--no-segment-acomm",
        default=False,
        show_default=True,
        help="Segment the AComm as its own label (seeded and lumen-filled like the PComms).",
    )(func)
    func = click.option("--cl-barrier-radius", type=int, default=2, show_default=True)(func)
    func = click.option(
        "--4dflow-thr-algorithm",
        "thr_algorithm",
        type=click.Choice(["lsthr", "lthr", "otsu"], case_sensitive=False),
        default="lsthr",
        show_default=True,
    )(func)
    func = click.option("--crop-padding-bbox", type=int, default=3, show_default=True)(func)
    func = click.option("--skip-existing", is_flag=True, default=False)(func)
    func = click.option("--output-root", type=click.Path(path_type=Path), required=True)(func)
    func = click.option("--nifti-root", type=click.Path(path_type=Path), required=True)(func)
    func = click.option("--subject", required=True)(func)
    return func


def _subject_sge_spec(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    container: Path,
    src_dir: Path | None = None,
    skip_existing: bool = False,
    crop_padding_bbox: int = 3,
    thr_algorithm: str = "lsthr",
    cl_barrier_radius: int = 2,
    segment_acomm: bool = False,
    backend: str = "gpu",
    **flood: Any,
) -> tuple[StageSpec, ClusterPaths]:
    """Build the SGE ``StageSpec``/``ClusterPaths`` pair for one subject's stage 4 segmentation task."""
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
        "--cl-barrier-radius",
        str(int(cl_barrier_radius)),
        "--segment-acomm" if segment_acomm else "--no-segment-acomm",
        *flood_cli_args(flood),
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
        resources=sge_qvtpy_stage_resources(backend),
        binds=binds,
        use_nv=sge_stage_use_nv(backend),
        extra_env=sge_stage_extra_env(binds.src, backend),
    )
    return spec, paths


def build_subject_sge_command(subject: str, **kwargs: Any) -> str:
    """Return the host shell command for one stage4 array/SGE task (*kwargs*: as
    :func:`_subject_sge_spec`)."""
    from nvitk.cluster.sge import build_singularity_command

    spec, paths = _subject_sge_spec(subject, **kwargs)
    return build_singularity_command(spec, paths)


def submit_subject_sge(
    subject: str,
    *,
    hold_jid: str | None = None,
    emit: TextIO | None = None,
    **kwargs: Any,
) -> str:
    """Emit or submit one stage-4 SGE job (*kwargs*: as :func:`_subject_sge_spec`). Returns the
    qsub job id."""
    spec, paths = _subject_sge_spec(subject, **kwargs)
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
    cl_barrier_radius: int,
    segment_acomm: bool,
    **flood: Any,
) -> None:
    """CLI entry point (``qvtpy-stage4-seg``): run stage 4 4D-flow vessel segmentation for one
    subject."""
    Logger()
    flood.pop("backend", None)
    run_subject(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        skip_existing=skip_existing,
        crop_padding_bbox=crop_padding_bbox,
        thr_algorithm=thr_algorithm.lower(),
        cl_barrier_radius=cl_barrier_radius,
        segment_acomm=segment_acomm,
        **flood,
    )


__all__ = [
    "FLOOD_DEFAULTS",
    "flood_cli_args",
    "flood_click_options",
    "flood_settings",
    "EICAB_IN_4DFLOW_EICAB_IDS_NIFTI",
    "load_stage3_eicab",
    "EICAB_IN_4DFLOW_NIFTI",
    "main",
    "run_subject",
    "build_subject_sge_command",
    "submit_subject_sge",
]


if __name__ == "__main__":
    main()
