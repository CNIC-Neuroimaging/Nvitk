"""Black-blood sub-pipeline runner (local execution; SGE not implemented in v1)."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_brain.black_blood import config as cfg
from nvitk.pipes.pesa_brain.black_blood import stage1_registration, stage2_bb_segmentation
from nvitk.pipes.pesa_brain.black_blood.util import paths
from nvitk.pipes.pesa_brain.black_blood.util.bb_vessel_segmentation import SegStrategy

log = Logger()

STAGE_REG = "stage1"
STAGE_SEG = "stage2"

_STAGE_ALIASES: dict[str, str] = {
    "stage1": STAGE_REG,
    "stage1_registration": STAGE_REG,
    "registration": STAGE_REG,
    "stage2": STAGE_SEG,
    "stage2_bb_segmentation": STAGE_SEG,
    "segmentation": STAGE_SEG,
}

_DEFAULT_STAGES = f"{STAGE_REG},{STAGE_SEG}"


def _normalize_stages(stages: str) -> list[str]:
    out: list[str] = []
    for raw in stages.split(","):
        s = raw.strip().lower()
        if not s:
            continue
        key = _STAGE_ALIASES.get(s, s)
        if key not in (STAGE_REG, STAGE_SEG):
            raise click.BadParameter(
                f"Unknown stage {raw!r}. Valid: stage1, stage2 (aliases: registration, segmentation)."
            )
        if key not in out:
            out.append(key)
    return out or [STAGE_REG, STAGE_SEG]


def _parse_subjects(subjects: str | None, nifti_root: Path) -> list[str]:
    if subjects:
        return [s.strip() for s in subjects.split(",") if s.strip()]
    if not nifti_root.is_dir():
        return []
    return sorted(
        p.name
        for p in nifti_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


@click.command("nvitk-pesa-brain-black-blood")
@click.option("--subjects", default=None, help="Comma-separated subject ids.")
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--output-root", type=click.Path(path_type=Path), default=None)
@click.option("--eicab-results-root", type=click.Path(path_type=Path), default=None)
@click.option("--wvi-rel-path", default=None)
@click.option("--eicab-subdir", default=None)
@click.option("--stages", default=_DEFAULT_STAGES)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--dof", type=int, default=6)
@click.option("--cost", default="normmi")
@click.option(
    "--seg-strategy",
    type=click.Choice(["crop-resegment", "centerline-growth"]),
    default="crop-resegment",
)
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
    subjects: str | None,
    nifti_root: Path | None,
    output_root: Path | None,
    eicab_results_root: Path | None,
    wvi_rel_path: str | None,
    eicab_subdir: str | None,
    stages: str,
    skip_existing: bool,
    dof: int,
    cost: str,
    seg_strategy: str,
    thr_algorithm: str,
    crop_padding_bbox: int,
    cl_barrier_radius: int,
    min_component_frac: float,
    rg_intensity_frac: float,
    rg_barrier_radius: int,
) -> None:
    """Run black-blood stages locally (stage1 registration, stage2 segmentation)."""
    nifti = paths.require_path(nifti_root or cfg.DEFAULT_NIFTI_ROOT, "nifti_root")
    out = paths.require_path(output_root or cfg.DEFAULT_RESULTS_ROOT, "output_root")
    eicab = paths.require_path(
        eicab_results_root or cfg.DEFAULT_EICAB_RESULTS_ROOT, "eicab_results_root"
    )
    paths.require_wvi_rel_path(wvi_rel_path)

    stages_sel = _normalize_stages(stages)
    subj_list = _parse_subjects(subjects, nifti)
    if not subj_list:
        raise click.ClickException(
            f"No subjects found under {nifti} (use --subjects)."
        )

    strategy: SegStrategy = seg_strategy  # type: ignore[assignment]

    for subj in subj_list:
        log.info(f"=== black_blood | subject={subj} | stages={stages_sel} ===")
        try:
            if STAGE_REG in stages_sel:
                stage1_registration.run_subject(
                    subj,
                    nifti_root=nifti,
                    output_root=out,
                    eicab_results_root=eicab,
                    skip_existing=skip_existing,
                    wvi_rel=wvi_rel_path,
                    eicab_subdir=eicab_subdir,
                    dof=dof,
                    cost=cost,
                )
            if STAGE_SEG in stages_sel:
                stage2_bb_segmentation.run_subject(
                    subj,
                    nifti_root=nifti,
                    output_root=out,
                    eicab_results_root=eicab,
                    seg_strategy=strategy,
                    skip_existing=skip_existing,
                    wvi_rel=wvi_rel_path,
                    eicab_subdir=eicab_subdir,
                    thr_algorithm=thr_algorithm,  # type: ignore[arg-type]
                    crop_padding_bbox=crop_padding_bbox,
                    cl_barrier_radius=cl_barrier_radius,
                    min_component_frac=min_component_frac,
                    rg_intensity_frac=rg_intensity_frac,
                    rg_barrier_radius=rg_barrier_radius,
                )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            log.error(f"[{subj}] failed: {exc}")


if __name__ == "__main__":
    main()
