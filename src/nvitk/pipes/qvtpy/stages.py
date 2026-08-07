"""QVTpy pipeline stage identifiers, aliases, and parsing."""

from __future__ import annotations

import click

STAGE_DOWNLOAD = "stage0_d"
STAGE_CONVERT = "stage0_c"
STAGE_EICAB = "stage1"
STAGE_REG = "stage2"
STAGE_CENTERLINE = "stage3"
STAGE_SEG = "stage4"
STAGE_SEG_T = "stage4t"
STAGE_LOC = "stage5"
STAGE_MEASURE = "stage6"
STAGE_MORPHOMETRICS = "stage7"
STAGE_XNAT_UPLOAD = "stage8_xnat_upload"
STAGE_AUTOQC = "stage9_autoqc"

STAGE_ALIASES: dict[str, str] = {
    "stage0_d": STAGE_DOWNLOAD,
    "stage0d": STAGE_DOWNLOAD,
    "stage0_download": STAGE_DOWNLOAD,
    "download": STAGE_DOWNLOAD,
    "stage0_c": STAGE_CONVERT,
    "stage0c": STAGE_CONVERT,
    "stage0_convert": STAGE_CONVERT,
    "stage0": STAGE_CONVERT,
    "convert": STAGE_CONVERT,
    "stage1": STAGE_EICAB,
    "stage1_eicab": STAGE_EICAB,
    "eicab": STAGE_EICAB,
    "stage2": STAGE_REG,
    "stage2_registration": STAGE_REG,
    "registration": STAGE_REG,
    "stage3": STAGE_CENTERLINE,
    "stage3_centerline": STAGE_CENTERLINE,
    "centerline": STAGE_CENTERLINE,
    "stage4": STAGE_SEG,
    "stage4_4dflow_segmentation": STAGE_SEG,
    "segmentation": STAGE_SEG,
    "stage4t": STAGE_SEG_T,
    "stage4t_4dflow_t_segmentation": STAGE_SEG_T,
    "segmentation_t": STAGE_SEG_T,
    "seg_t": STAGE_SEG_T,
    "stage5": STAGE_LOC,
    "stage5_loc_generation": STAGE_LOC,
    "loc": STAGE_LOC,
    "stage6": STAGE_MEASURE,
    "stage6_measure": STAGE_MEASURE,
    "measure": STAGE_MEASURE,
    "stage7": STAGE_MORPHOMETRICS,
    "stage7_morphometrics": STAGE_MORPHOMETRICS,
    "morphometrics": STAGE_MORPHOMETRICS,
    "morpho": STAGE_MORPHOMETRICS,
    "stage8": STAGE_XNAT_UPLOAD,
    "stage8_xnat_upload": STAGE_XNAT_UPLOAD,
    "xnat_upload": STAGE_XNAT_UPLOAD,
    "upload_xnat": STAGE_XNAT_UPLOAD,
    "stage9": STAGE_AUTOQC,
    "stage9_autoqc": STAGE_AUTOQC,
    "autoqc": STAGE_AUTOQC,
    "qc": STAGE_AUTOQC,
}

STAGES_ORDERED: tuple[str, ...] = (
    STAGE_DOWNLOAD,
    STAGE_CONVERT,
    STAGE_EICAB,
    STAGE_REG,
    STAGE_CENTERLINE,
    STAGE_SEG,
    STAGE_SEG_T,
    STAGE_LOC,
    STAGE_MEASURE,
    STAGE_MORPHOMETRICS,
    STAGE_XNAT_UPLOAD,
    STAGE_AUTOQC,
)

ALL_STAGES: tuple[str, ...] = STAGES_ORDERED
DEFAULT_STAGES: str = (
    f"{STAGE_CONVERT},{STAGE_EICAB},{STAGE_REG},{STAGE_CENTERLINE},"
    f"{STAGE_SEG},{STAGE_LOC},{STAGE_MEASURE}"
)

STAGE_LABELS: dict[str, str] = {
    STAGE_DOWNLOAD: "XNAT download",
    STAGE_CONVERT: "DICOM → NIfTI",
    STAGE_EICAB: "eICAB (TOF)",
    STAGE_REG: "FLIRT TOF → 4D flow",
    STAGE_CENTERLINE: "centerlines + venous",
    STAGE_SEG: "CD segmentation (4D)",
    STAGE_SEG_T: "CD segmentation (4D+t)",
    STAGE_LOC: "LOC generation",
    STAGE_MEASURE: "flow measurement",
    STAGE_MORPHOMETRICS: "TOF morphometrics",
    STAGE_XNAT_UPLOAD: "XNAT results upload",
}


def parse_stages(spec: str) -> list[str]:
    """Parse ``--stages`` comma list into canonical stage ids in pipeline order."""
    tokens = [t.strip().lower() for t in spec.split(",") if t.strip()]
    if not tokens:
        raise click.ClickException("--stages cannot be empty.")
    canonical: set[str] = set()
    for tok in tokens:
        key = tok.replace("-", "_")
        if key not in STAGE_ALIASES:
            raise click.ClickException(
                f"Unknown stage {tok!r}. Valid: {', '.join(sorted(set(STAGE_ALIASES.keys())))}."
            )
        canonical.add(STAGE_ALIASES[key])
    return [s for s in STAGES_ORDERED if s in canonical]


__all__ = [
    "ALL_STAGES",
    "DEFAULT_STAGES",
    "STAGE_ALIASES",
    "STAGE_CENTERLINE",
    "STAGE_CONVERT",
    "STAGE_DOWNLOAD",
    "STAGE_EICAB",
    "STAGE_LABELS",
    "STAGE_LOC",
    "STAGE_MEASURE",
    "STAGE_MORPHOMETRICS",
    "STAGE_REG",
    "STAGE_SEG",
    "STAGE_SEG_T",
    "STAGE_XNAT_UPLOAD",
    "STAGE_AUTOQC",
    "STAGES_ORDERED",
    "parse_stages",
]
