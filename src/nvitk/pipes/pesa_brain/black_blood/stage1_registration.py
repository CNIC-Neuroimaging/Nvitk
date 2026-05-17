"""Black-blood stage 1: rigid vwi_bb → eICAB TOF_resampled (FSL FLIRT)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

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
