"""Pipeline CLI entries for the nvitk GUI (from ``src/nvitk/pipes``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineGuiSpec:
    id: str
    label: str
    cli_command: str
    description: str = ""


PIPELINE_TOOLS: tuple[PipelineGuiSpec, ...] = (
    PipelineGuiSpec(
        "pipeline_pesa_fat_ctpet",
        "PESA-Fat CT-PET",
        "nvitk-pesa-fat-ctpet",
        "CT-PET v5 segmentation, post-processing, and measurement.",
    ),
    PipelineGuiSpec(
        "pipeline_pesa_fat_dixon",
        "PESA-Fat DIXON",
        "nvitk-pesa-fat-dixon",
        "Dixon v5 fat segmentation, post-processing, and measurement.",
    ),
    PipelineGuiSpec(
        "pipeline_qvtpy",
        "QVTPy (4DFlows)",
        "nvitk-qvtpy",
        "XNAT/DICOM → centerlines → 4D flow segmentation.",
    ),
    PipelineGuiSpec(
        "pipeline_bbtpy",
        "BBTPy",
        "nvitk-bbtpy",
        "Black-blood TOF registration and segmentation.",
    ),
    PipelineGuiSpec(
        "pipeline_gpetpy",
        "GPETPy",
        "nvitk-gpetpy",
        "PET brain crop pipeline.",
    ),
)
