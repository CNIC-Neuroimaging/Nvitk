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
        "pipeline_pesa_fat",
        "PESA-FAT batch",
        "nvitk-pesa-fat",
        "Full PESA-FAT batch driver (Dixon + CT-PET pipelines).",
    ),
    PipelineGuiSpec(
        "pipeline_pesa_fat_ctpet",
        "PESA-FAT CT-PET v5",
        "nvitk-pesa-fat-ctpet",
    ),
    PipelineGuiSpec(
        "pipeline_pesa_fat_dixon",
        "PESA-FAT Dixon v5",
        "nvitk-pesa-fat-dixon",
    ),
    PipelineGuiSpec(
        "pipeline_pesa_fat_hotspot",
        "PESA-FAT hotspot QC",
        "nvitk-pesa-fat-hotspot",
    ),
    PipelineGuiSpec(
        "pipeline_pesa_fat_qc",
        "PESA-FAT stage4 QC",
        "nvitk-pesa-fat-qc",
    ),
    PipelineGuiSpec(
        "pipeline_qvtpy",
        "QVTpy (4D flow)",
        "nvitk-qvtpy",
        "XNAT/DICOM → centerlines → 4D flow segmentation.",
    ),
    PipelineGuiSpec(
        "pipeline_qvtpy_flowshow",
        "QVTpy FlowShow",
        "nvitk-qvtpy-flowshow",
    ),
    PipelineGuiSpec(
        "pipeline_bbtpy",
        "BBTpy (brain TOF)",
        "nvitk-bbtpy",
    ),
    PipelineGuiSpec(
        "pipeline_gpetpy",
        "GPETpy",
        "nvitk-gpetpy",
    ),
)
