"""Black-blood stage 2: centerlines from eICAB + BB artery segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.io.imageio import imread
from nvitk.pipes.pesa_brain.black_blood import config as cfg
from nvitk.pipes.pesa_brain.black_blood.util import paths
from nvitk.pipes.pesa_brain.black_blood.util.bb_vessel_segmentation import (
    SegStrategy,
    ThrAlgorithm,
    run_bb_segmentation,
)
from nvitk.pipes.pesa_brain.black_blood.util.centerlines_from_eicab import (
    build_centerlines_from_eicab,
)

log = Logger()


def _load_stage1_meta(output_root: Path, subject: str) -> dict:
    meta_path = paths.registration_meta_path(output_root, subject)
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Missing stage1 registration_meta.json for {subject}: {meta_path}"
        )
    return json.loads(meta_path.read_text(encoding="utf-8"))


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_results_root: Path,
    seg_strategy: SegStrategy,
    skip_existing: bool = False,
    vwi_bb_rel: str | None = None,
    eicab_subdir: str | None = None,
    thr_algorithm: ThrAlgorithm = "otsu",
    crop_padding_bbox: int = 3,
    cl_barrier_radius: int = 2,
    min_component_frac: float = 0.005,
    rg_intensity_frac: float = 0.45,
    rg_barrier_radius: int = 2,
    min_centerline_points: int = 5,
) -> Path:
    """Build centerlines and segment vwi_bb in TOF space."""
    meta = _load_stage1_meta(output_root, subject)
    fixed = Path(meta["fixed"])
    mat = Path(meta["matrix"])
    vwi_warped = paths.vwi_bb_warped_path(output_root, subject)
    if not vwi_warped.is_file():
        vwi_warped = Path(meta.get("warped") or "")
    if not vwi_warped.is_file():
        raise FileNotFoundError(f"Missing warped vwi_bb for {subject}: {vwi_warped}")

    eicab_cw = paths.eicab_cw_mask_path(
        eicab_results_root, subject, eicab_subdir=eicab_subdir
    )
    out_dir = paths.stage2_dir(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"pesa_brain stage2 | subject={subject} strategy={seg_strategy}")

    build_centerlines_from_eicab(
        eicab_cw,
        fixed,
        out_dir,
        transform_mat=mat,
        min_points=min_centerline_points,
        skip_existing=skip_existing,
    )

    vwi_img = imread(vwi_warped)
    cl_img = imread(out_dir / "centerlines_mask.nii.gz")

    run_bb_segmentation(
        vwi_img.data,
        cl_img.data,
        out_dir,
        strategy=seg_strategy,
        thr_algorithm=thr_algorithm,
        crop_padding_bbox=crop_padding_bbox,
        cl_barrier_radius=cl_barrier_radius,
        min_component_frac=min_component_frac,
        rg_intensity_frac=rg_intensity_frac,
        rg_barrier_radius=rg_barrier_radius,
        skip_existing=skip_existing,
    )
    return out_dir


@click.command("nvitk-pesa-brain-bb-seg")
@click.option("--subject", required=True)
@click.option(
    "--seg-strategy",
    type=click.Choice(["crop-resegment", "centerline-growth"]),
    required=True,
)
@click.option(
    "--nifti-root",
    type=click.Path(path_type=Path, exists=True),
    default=None,
)
@click.option(
    "--output-root",
    type=click.Path(path_type=Path, exists=True),
    default=None,
)
@click.option(
    "--eicab-results-root",
    type=click.Path(path_type=Path, exists=True),
    default=None,
)
@click.option("--vwi-bb-rel-path", default=None)
@click.option("--wvi-rel-path", default=None, hidden=True)
@click.option("--eicab-subdir", default=None)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option(
    "--thr-algorithm",
    type=click.Choice(["otsu", "lsthr", "lthr"]),
    default="otsu",
)
@click.option("--crop-padding-bbox", type=int, default=3)
@click.option("--cl-barrier-radius", type=int, default=2)
@click.option("--min-component-frac", type=float, default=0.005)
@click.option("--rg-intensity-frac", type=float, default=0.45)
@click.option("--rg-barrier-radius", type=int, default=2)
def main(
    subject: str,
    seg_strategy: str,
    nifti_root: Path | None,
    output_root: Path | None,
    eicab_results_root: Path | None,
    vwi_bb_rel_path: str | None,
    wvi_rel_path: str | None,
    eicab_subdir: str | None,
    skip_existing: bool,
    thr_algorithm: str,
    crop_padding_bbox: int,
    cl_barrier_radius: int,
    min_component_frac: float,
    rg_intensity_frac: float,
    rg_barrier_radius: int,
) -> None:
    """CLI: BB segmentation in TOF space."""
    nifti = paths.require_path(nifti_root or cfg.DEFAULT_NIFTI_ROOT, "nifti_root")
    out = paths.require_path(output_root or cfg.DEFAULT_RESULTS_ROOT, "output_root")
    eicab = paths.resolve_eicab_results_root(eicab_results_root)
    rel = vwi_bb_rel_path or wvi_rel_path
    run_subject(
        subject,
        nifti_root=nifti,
        output_root=out,
        eicab_results_root=eicab,
        seg_strategy=seg_strategy,  # type: ignore[arg-type]
        skip_existing=skip_existing,
        vwi_bb_rel=rel,
        eicab_subdir=eicab_subdir,
        thr_algorithm=thr_algorithm,  # type: ignore[arg-type]
        crop_padding_bbox=crop_padding_bbox,
        cl_barrier_radius=cl_barrier_radius,
        min_component_frac=min_component_frac,
        rg_intensity_frac=rg_intensity_frac,
        rg_barrier_radius=rg_barrier_radius,
    )


if __name__ == "__main__":
    main()
