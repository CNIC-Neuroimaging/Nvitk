"""qvtpy stage 4: centerline-backbone ``seg_4dflow`` in 4D-flow space."""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Literal, TextIO

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
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.util.centerline_segmentation import (
    assemble_mesh_segmentation,
    assemble_voxel_segmentation,
    postprocess_segmentation,
)
from nvitk.pipes.qvtpy.util.centerline_io import load_centerlines
from nvitk.pipes.qvtpy.util.venous_heuristics import venous_name_to_label_id

setup(globals())

log = Logger()

# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

SegAssembly = Literal["voxel", "mesh"]


# ---------------------------------------------------------------------------
# Path helpers + volume I/O
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
    return nifti_root / subject / "4DFlow" / "ComplexDifference_3D.nii"


def _load_volumes(nifti_root: Path, subject: str) -> tuple[Any, Any, Any, tuple[float, float, float], dict]:
    sub = nifti_root / subject / "4DFlow"
    cd_img = imread(_cd_path(nifti_root, subject))
    cd = as_backend_array(cd_img.data).astype(np.float64)
    meta = dict(cd_img.metadata or {})
    sp = meta.get("spacing")
    if sp and len(sp) >= 3:
        voxel_spacing = (float(sp[0]), float(sp[1]), float(sp[2]))
    else:
        voxel_spacing = (1.0, 1.0, 1.0)
    mag = cd
    angio = sub / "Angiography_3D.nii.gz"
    if angio.is_file():
        mag = as_backend_array(imread(angio).data).astype(np.float64)
    vel = np.abs(cd)
    vmag = sub / "VelocityMagnitude_3D.nii.gz"
    if vmag.is_file():
        vel = as_backend_array(imread(vmag).data).astype(np.float64)
    return mag, cd, vel, voxel_spacing, meta


# ---------------------------------------------------------------------------
# Stage 4: centerline-backbone seg_4dflow
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool = False,
    seg_assembly: SegAssembly = "voxel",
    seg_interp_level: int = 0,
    seg_stride: int = 1,
    cross_section_res: int = 0,
    cross_section_plane_interp: int = 1,
    radius_vox: float = 10.0,
) -> Path:
    # ---- Prerequisites: stage3 centerlines + output paths --------------------
    s3 = _stage3_dir(output_root, subject)
    meta_s3 = s3 / "centerline_meta.json"
    from nvitk.pipes.qvtpy.util.centerline_io import centerlines_mask_path

    if not centerlines_mask_path(s3).is_file():
        raise FileNotFoundError(f"Missing {centerlines_mask_path(s3)} (run stage3)")

    out_dir = _stage4_out(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg_path = out_dir / "seg_4dflow.nii.gz"
    meta_path = out_dir / "segmentation_meta.json"
    if skip_existing and seg_path.is_file() and meta_path.is_file():
        log.info(f"[{subject}] stage4 seg: skip -> {out_dir}")
        return out_dir

    # ---- Load polylines + contrast volumes -----------------------------------
    arterial, venous, _ = load_centerlines(s3, min_points=5)
    mag, cd, vel, voxel_spacing, ref_meta = _load_volumes(nifti_root, subject)
    shape3 = cd.shape

    name_to_id: dict[str, int] = {}
    if meta_s3.is_file():
        meta_json = json.loads(meta_s3.read_text(encoding="utf-8"))
        name_to_id = {str(k): int(v) for k, v in (meta_json.get("venous_label_by_name") or {}).items()}

    # ---- Merge arterial + venous vessels for multilabel assembly -------------
    vessels: dict[int | str, np.ndarray] = {int(k): v for k, v in arterial.items()}
    label_for_key: dict[int | str, int] = {int(k): int(k) for k in arterial}
    for name, poly in venous.items():
        vessels[name] = poly
        label_for_key[name] = venous_name_to_label_id(name, name_to_id)

    # ---- Cross-section stamp (voxel or mesh) + island cleanup ----------------
    assemble_fn = assemble_voxel_segmentation if seg_assembly == "voxel" else assemble_mesh_segmentation
    seg, asm_stats = assemble_fn(
        shape3,
        vessels,
        mag=mag,
        cd=cd,
        vel_mag=vel,
        voxel_spacing=voxel_spacing,
        label_for_key=label_for_key,
        stride=seg_stride,
        radius_vox=radius_vox,
        seg_interp_level=seg_interp_level if seg_assembly == "voxel" else 0,
        cross_section_res=cross_section_res,
        plane_interp_order=cross_section_plane_interp,
    )

    # ---- Write seg_4dflow.nii.gz + segmentation_meta.json --------------------
    seg = postprocess_segmentation(seg)
    imsave(seg_path, seg, metadata=ref_meta)
    meta_path.write_text(
        json.dumps(
            {
                "subject": subject,
                "seg_assembly": seg_assembly,
                "seg_interp_level": int(seg_interp_level),
                "seg_stride": int(seg_stride),
                "cross_section_res": int(cross_section_res),
                "cross_section_plane_interp": int(cross_section_plane_interp),
                "radius_vox": float(radius_vox),
                "assembly_stats": asm_stats,
                "note": "seg_4dflow built from centerline cross-sections; venous from stage3 geometry.",
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
    seg_assembly: str = "voxel",
    seg_interp_level: int = 0,
    seg_stride: int = 1,
    cross_section_res: int = 0,
    cross_section_plane_interp: int = 1,
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
        "--seg-assembly",
        shlex.quote(seg_assembly),
        "--seg-interp-level",
        str(int(seg_interp_level)),
        "--seg-stride",
        str(int(seg_stride)),
        "--cross-section-res",
        str(int(cross_section_res)),
        "--cross-section-plane-interp",
        str(int(cross_section_plane_interp)),
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
@click.option(
    "--seg-assembly",
    type=click.Choice(["voxel", "mesh"]),
    default="voxel",
    show_default=True,
)
@click.option("--seg-interp-level", type=int, default=0, show_default=True)
@click.option("--seg-stride", type=int, default=1, show_default=True)
@click.option("--cross-section-res", type=int, default=0, show_default=True)
@click.option("--cross-section-plane-interp", type=int, default=1, show_default=True)
@click.option("--cross-section-radius-vox", type=float, default=10.0, show_default=True)
def main(
    subject: str,
    nifti_root: Path,
    output_root: Path,
    skip_existing: bool,
    seg_assembly: str,
    seg_interp_level: int,
    seg_stride: int,
    cross_section_res: int,
    cross_section_plane_interp: int,
    cross_section_radius_vox: float,
) -> None:
    Logger()
    run_subject(
        subject,
        nifti_root=nifti_root,
        output_root=output_root,
        skip_existing=skip_existing,
        seg_assembly=seg_assembly,  # type: ignore[arg-type]
        seg_interp_level=seg_interp_level,
        seg_stride=seg_stride,
        cross_section_res=cross_section_res,
        cross_section_plane_interp=cross_section_plane_interp,
        radius_vox=cross_section_radius_vox,
    )


__all__ = ["main", "run_subject", "submit_subject_sge"]
