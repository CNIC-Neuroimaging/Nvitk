"""Black-blood stage 1: rigid vwi_bb → eICAB TOF_resampled (FSL FLIRT)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_brain.black_blood import config as cfg
from nvitk.pipes.pesa_brain.black_blood.util import paths
from nvitk.registration.fsl.flirt import flirt_register_rigid

log = Logger()


def run_subject(
    subject: str,
    *,
    nifti_root: Path,
    output_root: Path,
    eicab_results_root: Path,
    skip_existing: bool = False,
    vwi_bb_rel: str | None = None,
    eicab_subdir: str | None = None,
    dof: int = 6,
    cost: str = "normmi",
) -> Path:
    """Register vwi_bb (moving) to TOF_resampled (fixed)."""
    moving = paths.vwi_bb_path(nifti_root, subject, vwi_bb_rel=vwi_bb_rel)
    fixed = paths.tof_resampled_path(
        eicab_results_root, subject, eicab_subdir=eicab_subdir
    )
    out_dir = paths.stage1_dir(output_root, subject)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = paths.registration_meta_path(output_root, subject)

    if skip_existing and meta_path.is_file():
        log.info(f"[{subject}] stage1 registration: skip existing -> {out_dir}")
        return out_dir

    log.info(f"pesa_brain stage1 FLIRT | subject={subject}")
    log.info(f"  moving (vwi_bb): {moving}")
    log.info(f"  fixed (TOF_resampled): {fixed}")

    res = flirt_register_rigid(
        moving,
        fixed,
        out_dir,
        dof=dof,
        cost=cost,
        warped_name="vwi_bb_warped_to_tof.nii.gz",
        matrix_name="vwi_bb_to_tof.mat",
    )
    meta: dict[str, Any] = {
        "subject": subject,
        "moving": str(moving),
        "moving_kind": "black_blood_vwi_bb",
        "fixed": str(fixed),
        "fixed_kind": "eicab_tof_resampled",
        "matrix": str(res.matrix_path),
        "warped": str(res.warped_path) if res.warped_path else None,
        "dof": dof,
        "cost": cost,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


@click.command("nvitk-pesa-brain-bb-reg")
@click.option("--subject", required=True)
@click.option("--nifti-root", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--output-root", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--eicab-results-root", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--vwi-bb-rel-path", default=None)
@click.option("--wvi-rel-path", default=None, hidden=True)
@click.option("--eicab-subdir", default=None)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--dof", type=int, default=6)
@click.option("--cost", default="normmi")
def main(
    subject: str,
    nifti_root: Path | None,
    output_root: Path | None,
    eicab_results_root: Path | None,
    vwi_bb_rel_path: str | None,
    wvi_rel_path: str | None,
    eicab_subdir: str | None,
    skip_existing: bool,
    dof: int,
    cost: str,
) -> None:
    """CLI: vwi_bb → TOF_resampled registration."""
    nifti = paths.require_path(nifti_root or cfg.DEFAULT_NIFTI_ROOT, "nifti_root")
    out = paths.require_path(output_root or cfg.DEFAULT_RESULTS_ROOT, "output_root")
    eicab = paths.resolve_eicab_results_root(eicab_results_root)
    rel = vwi_bb_rel_path or wvi_rel_path
    run_subject(
        subject,
        nifti_root=nifti,
        output_root=out,
        eicab_results_root=eicab,
        skip_existing=skip_existing,
        vwi_bb_rel=rel,
        eicab_subdir=eicab_subdir,
        dof=dof,
        cost=cost,
    )


if __name__ == "__main__":
    main()
