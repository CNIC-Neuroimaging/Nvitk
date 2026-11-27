"""Pipeline CLI entries for the nvitk GUI (from ``src/nvitk/pipes``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineGuiSpec:
    """GUI-facing metadata for one installed pipeline CLI command (id, label, command, description)."""

    id: str
    label: str
    cli_command: str
    description: str = ""


PIPELINE_TOOLS: tuple[PipelineGuiSpec, ...] = (
    PipelineGuiSpec(
        "pipeline_pesa_health_ctpet",
        "PESA-Health CT-PET",
        "nvitk-pesa-health-ctpet",
        "CT-PET v5 segmentation, post-processing, and measurement.",
    ),
    PipelineGuiSpec(
        "pipeline_pesa_health_dixon",
        "PESA-Health DIXON",
        "nvitk-pesa-health-dixon",
        "Dixon v5 fat segmentation, post-processing, and measurement.",
    ),
    PipelineGuiSpec(
        "pipeline_qvtpy",
        "QVTPy (4DFlows)",
        "nvitk-qvtpy",
        "XNAT/DICOM → centerlines → 4D flow segmentation.",
    ),
)
