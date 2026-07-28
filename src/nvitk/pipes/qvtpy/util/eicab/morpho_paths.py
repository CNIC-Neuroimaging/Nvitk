"""Path helpers for qvtpy stage-7 TOF morphometrics.

Resolves inputs and outputs from :mod:`nvitk.pipes.qvtpy.config` layout:

- eICAB masks: ``<output_root>/<subject>/eicab/``
- stage-7 products: ``<output_root>/<subject>/qvtpy/stage7_morphometrics/``
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.util.eicab.eicab_masks import EicabMaskResolution, resolve_eicab_mask

EicabMaskPreference = Literal["cw", "wb"]

STAGE7_SKIP_MARKER = "case_metrics_donut_tree.xlsx"


def eicab_dir(output_root: Path, subject: str) -> Path:
    """Directory containing eICAB NIfTI outputs for *subject*."""
    return Path(output_root) / subject / cfg.STAGE1_EICAB_DIR


def stage7_dir(output_root: Path, subject: str) -> Path:
    """Stage-7 morphometrics output directory for *subject*."""
    return Path(output_root) / subject / cfg.QVT_SUBDIR / cfg.STAGE7_MORPHOMETRICS_DIR


def stage7_excel_path(output_root: Path, subject: str) -> Path:
    """Path to the stage-7 skip-marker / primary metrics workbook."""
    return stage7_dir(output_root, subject) / STAGE7_SKIP_MARKER


def resolve_stage7_seg_mask(
    output_root: Path,
    subject: str,
    *,
    preference: EicabMaskPreference = "wb",
    prefer_postprocessed: bool = True,
) -> EicabMaskResolution:
    """Return the eICAB multilabel mask used as stage-7 morphometrics input."""
    return resolve_eicab_mask(
        eicab_dir(output_root, subject),
        preference=preference,
        prefer_postprocessed=prefer_postprocessed,
    )


__all__ = [
    "EicabMaskPreference",
    "STAGE7_SKIP_MARKER",
    "eicab_dir",
    "resolve_stage7_seg_mask",
    "stage7_dir",
    "stage7_excel_path",
]
