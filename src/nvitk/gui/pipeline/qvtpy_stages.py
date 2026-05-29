"""Build subprocess argv for individual QVTpy pipeline stage modules."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QvtpyStageSpec:
    tool_id: str
    label: str
    module: str
    needs_subject: bool = True
    needs_nifti_root: bool = True
    needs_output_root: bool = True
    needs_dicom_root: bool = False
    is_download: bool = False


_QVT_COMMON_REQUIRED = ("subject", "nifti_root", "output_root")

STAGE_SPECS: tuple[QvtpyStageSpec, ...] = (
    QvtpyStageSpec(
        "qvtpy_stage0_convert",
        "QVTpy stage 0 (convert DICOM)",
        "nvitk.pipes.qvtpy.stage0_convert",
        needs_dicom_root=True,
    ),
    QvtpyStageSpec(
        "qvtpy_stage0_download",
        "QVTpy stage 0 (XNAT download)",
        "nvitk.pipes.qvtpy.stage0_download",
        needs_nifti_root=False,
        needs_output_root=False,
        needs_dicom_root=True,
        is_download=True,
    ),
    QvtpyStageSpec("qvtpy_stage1_eicab", "QVTpy stage 1 (eICAB)", "nvitk.pipes.qvtpy.stage1_eicab"),
    QvtpyStageSpec(
        "qvtpy_stage2_registration",
        "QVTpy stage 2 (registration)",
        "nvitk.pipes.qvtpy.stage2_registration",
    ),
    QvtpyStageSpec(
        "qvtpy_stage3_centerline",
        "QVTpy stage 3 (centerlines)",
        "nvitk.pipes.qvtpy.stage3_centerline",
    ),
    QvtpyStageSpec(
        "qvtpy_stage4_4dflow",
        "QVTpy stage 4 (4D flow seg)",
        "nvitk.pipes.qvtpy.stage4_4dflow_segmentation",
    ),
    QvtpyStageSpec(
        "qvtpy_stage4t_4dflow_t",
        "QVTpy stage 4t (per-phase seg)",
        "nvitk.pipes.qvtpy.stage4t_4dflow_t_segmentation",
    ),
    QvtpyStageSpec(
        "qvtpy_stage5_loc",
        "QVTpy stage 5 (LOCs)",
        "nvitk.pipes.qvtpy.stage5_loc_generation",
    ),
    QvtpyStageSpec(
        "qvtpy_stage6_measure",
        "QVTpy stage 6 (measure)",
        "nvitk.pipes.qvtpy.stage6_measure",
    ),
)

STAGE_BY_ID: dict[str, QvtpyStageSpec] = {s.tool_id: s for s in STAGE_SPECS}


def _req(params: dict[str, Any], key: str) -> str:
    val = str(params.get(key) or "").strip()
    if not val:
        raise ValueError(f"Missing required parameter: {key}")
    return val


def build_qvtpy_stage_argv(tool_id: str, params: dict[str, Any]) -> list[str]:
    """Return argv for ``python -m <stage_module>``."""
    spec = STAGE_BY_ID.get(tool_id)
    if spec is None:
        raise ValueError(f"Unknown QVTpy stage tool: {tool_id}")

    argv = [sys.executable, "-m", spec.module]

    if spec.is_download:
        argv.extend(["--dicom-root", _req(params, "dicom_root")])
        argv.extend(["--subjects", _req(params, "subject")])
        if bool(params.get("skip_existing", True)):
            argv.append("--skip-existing")
        return argv

    if spec.needs_dicom_root:
        argv.extend(["--dicom-root", _req(params, "dicom_root")])
    if spec.needs_nifti_root:
        argv.extend(["--nifti-root", _req(params, "nifti_root")])
    if spec.needs_output_root:
        argv.extend(["--output-root", _req(params, "output_root")])
    if spec.needs_subject:
        argv.extend(["--subject", _req(params, "subject")])

    if bool(params.get("skip_existing")):
        argv.append("--skip-existing")

    if tool_id == "qvtpy_stage0_convert":
        if bool(params.get("compute_phase_derived", True)):
            argv.append("--compute-phase-derived")
        if params.get("phase_background_correction") is False:
            argv.append("--no-phase-background-correction")
        if bool(params.get("no_cd_4d_background_correction")):
            argv.append("--no-cd-4d-background-correction")

    if tool_id == "qvtpy_stage3_centerline":
        eicab = str(params.get("eicab_mask") or "cw").strip()
        argv.extend(["--eicab-mask", eicab])

    return argv


def default_dicom_root() -> str:
    try:
        from nvitk.pipes.qvtpy import config as cfg

        return str(cfg.DEFAULT_DICOM_ROOT)
    except Exception:
        return ""
