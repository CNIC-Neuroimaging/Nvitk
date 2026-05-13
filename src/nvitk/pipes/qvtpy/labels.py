"""Vessel label IDs and names aligned with eICAB / MATLAB QVTplus usage.

The integer values **EICAB_*** match voxel labels in ``*_eICAB_CW.nii(.gz)`` (eICAB
convention: left/right pairs and extras BAS, AComm). See ``EICAB_ID_TO_NAME`` for
canonical string names used in qvtpy.

Venous **names** (``SSSV``, …) are QVTplus LOC / branch identifiers, not eICAB mask
values. ``QVTPY_VENOUS_UNKNOWN_LABEL`` is reserved for qvtpy-derived venous voxels
outside the eICAB arterial mask (e.g. stage 3 ``centerlines_mask`` venous skeleton).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# eICAB multilabel integers (eICAB table: Left / Right columns)
# ---------------------------------------------------------------------------

EICAB_BACKGROUND: int = 0

# ICA
EICAB_LICA: int = 1
EICAB_RICA: int = 2
# Extras (single labels)
EICAB_BASILAR: int = 3
EICAB_ACOMM: int = 4
# ACA-A1
EICAB_LACA: int = 5
EICAB_RACA: int = 6
# MCA-M1
EICAB_LMCA: int = 7
EICAB_RMCA: int = 8
# PComm
EICAB_LPCOMM: int = 9
EICAB_RPCOMM: int = 10
# PCA-P1
EICAB_LPCA_P1: int = 11
EICAB_RPCA_P1: int = 12
# PCA-P2
EICAB_LPCA_P2: int = 13
EICAB_RPCA_P2: int = 14
# SCA
EICAB_LSCA: int = 15
EICAB_RSCA: int = 16
# AChA
EICAB_LACHA: int = 17
EICAB_RACHA: int = 18

# Backward compatibility (older qvtpy / QVTplus wording)
EICAB_BASI: int = EICAB_BASILAR
EICAB_LABEL_4_UNDOCUMENTED: int = EICAB_ACOMM

EICAB_ID_TO_NAME: dict[int, str] = {
    EICAB_BACKGROUND: "BACKGROUND",
    EICAB_LICA: "LICA",
    EICAB_RICA: "RICA",
    EICAB_BASILAR: "BASILAR",
    EICAB_ACOMM: "ACOMM",
    EICAB_LACA: "LACA",
    EICAB_RACA: "RACA",
    EICAB_LMCA: "LMCA",
    EICAB_RMCA: "RMCA",
    EICAB_LPCOMM: "LPCOMM",
    EICAB_RPCOMM: "RPCOMM",
    EICAB_LPCA_P1: "LPCA_P1",
    EICAB_RPCA_P1: "RPCA_P1",
    EICAB_LPCA_P2: "LPCA_P2",
    EICAB_RPCA_P2: "RPCA_P2",
    EICAB_LSCA: "LSCA",
    EICAB_RSCA: "RSCA",
    EICAB_LACHA: "LACHA",
    EICAB_RACHA: "RACHA",
}

# Primary CoW labels often referenced in QVTplus excerpts (subset of full table).
EICAB_QVTPLUS_DOCUMENTED_COW: frozenset[int] = frozenset(
    {
        EICAB_LICA,
        EICAB_RICA,
        EICAB_BASILAR,
        EICAB_LACA,
        EICAB_RACA,
        EICAB_LMCA,
        EICAB_RMCA,
    }
)

# All named eICAB vessel IDs (excluding background).
EICAB_VESSEL_LABEL_IDS: frozenset[int] = frozenset(range(1, 19))

# Human-readable name → id (uppercased keys). Ambiguous short forms map to the
# most common segment (e.g. LPCA → P1).
_EICAB_NAME_TO_ID_RAW: dict[str, int] = {
    "LICA": EICAB_LICA,
    "RICA": EICAB_RICA,
    "BASILAR": EICAB_BASILAR,
    "BASI": EICAB_BASILAR,
    "BAS": EICAB_BASILAR,
    "ACOMM": EICAB_ACOMM,
    "ACOM": EICAB_ACOMM,
    "LACA": EICAB_LACA,
    "RACA": EICAB_RACA,
    "LMCA": EICAB_LMCA,
    "RMCA": EICAB_RMCA,
    "LPCOMM": EICAB_LPCOMM,
    "RPCOMM": EICAB_RPCOMM,
    "LPComm": EICAB_LPCOMM,
    "RPComm": EICAB_RPCOMM,
    "LPCA_P1": EICAB_LPCA_P1,
    "RPCA_P1": EICAB_RPCA_P1,
    "LPCA_P2": EICAB_LPCA_P2,
    "RPCA_P2": EICAB_RPCA_P2,
    "LPCA": EICAB_LPCA_P1,
    "RPCA": EICAB_RPCA_P1,
    "LSCA": EICAB_LSCA,
    "RSCA": EICAB_RSCA,
    "LACHA": EICAB_LACHA,
    "RACHA": EICAB_RACHA,
}

EICAB_NAME_TO_ID: dict[str, int] = {k.upper(): v for k, v in _EICAB_NAME_TO_ID_RAW.items()}

# ---------------------------------------------------------------------------
# QVTplus naming (MATLAB correspondence / LOC), not eICAB voxel integers
# ---------------------------------------------------------------------------

NAME_RCOMM: str = "RCOMM"
NAME_LCOMM: str = "LCOMM"
NAME_COMM: str = "COMM"

NAME_SSSV: str = "SSSV"
NAME_LTSV: str = "LTSV"
NAME_RTSV: str = "RTSV"
NAME_STRV: str = "STRV"

MATLAB_QVT_VENOUS_VESSEL_NAMES: tuple[str, ...] = (NAME_SSSV, NAME_LTSV, NAME_RTSV, NAME_STRV)

# ---------------------------------------------------------------------------
# qvtpy-specific extensions (do not collide with typical eICAB small integers)
# ---------------------------------------------------------------------------

QVTPY_VENOUS_UNKNOWN_LABEL: int = 30
QVTPY_VENOUS_REGION_BASE: int = 31  #: optional 31..34 for four venous components
QVTPY_UNKNOWN_LABEL: int = 35

VENOUS_UNKNOWN_LABEL: int = QVTPY_VENOUS_UNKNOWN_LABEL
VENOUS_REGION_BASE: int = QVTPY_VENOUS_REGION_BASE
UNKNOWN_LABEL: int = QVTPY_UNKNOWN_LABEL


def eicab_vessel_name(label_id: int) -> str:
    """Human-readable name for an eICAB mask integer, or a stable fallback string."""
    lid = int(label_id)
    if lid in EICAB_ID_TO_NAME:
        return EICAB_ID_TO_NAME[lid]
    if lid == QVTPY_VENOUS_UNKNOWN_LABEL:
        return "QVTPY_VENOUS_UNKNOWN"
    if lid == QVTPY_UNKNOWN_LABEL:
        return "QVTPY_UNKNOWN"
    if QVTPY_VENOUS_REGION_BASE <= lid < QVTPY_VENOUS_REGION_BASE + 4:
        return f"QVTPY_VENOUS_REGION_{lid - QVTPY_VENOUS_REGION_BASE + 1}"
    return f"EICAB_OR_EXTENDED_{lid}"


def eicab_label_from_name(name: str) -> int | None:
    """Return the eICAB integer label for a known vessel *name*, or None if unknown."""
    key = name.strip().upper()
    return EICAB_NAME_TO_ID.get(key)


__all__ = [
    "EICAB_ACOMM",
    "EICAB_BACKGROUND",
    "EICAB_BASI",
    "EICAB_BASILAR",
    "EICAB_ID_TO_NAME",
    "EICAB_LABEL_4_UNDOCUMENTED",
    "EICAB_LACA",
    "EICAB_LACHA",
    "EICAB_LICA",
    "EICAB_LMCA",
    "EICAB_LPCA_P1",
    "EICAB_LPCA_P2",
    "EICAB_LPCOMM",
    "EICAB_LSCA",
    "EICAB_NAME_TO_ID",
    "EICAB_QVTPLUS_DOCUMENTED_COW",
    "EICAB_RACA",
    "EICAB_RACHA",
    "EICAB_RICA",
    "EICAB_RMCA",
    "EICAB_RPCA_P1",
    "EICAB_RPCA_P2",
    "EICAB_RPCOMM",
    "EICAB_RSCA",
    "EICAB_VESSEL_LABEL_IDS",
    "MATLAB_QVT_VENOUS_VESSEL_NAMES",
    "NAME_COMM",
    "NAME_LCOMM",
    "NAME_LTSV",
    "NAME_RCOMM",
    "NAME_RTSV",
    "NAME_SSSV",
    "NAME_STRV",
    "QVTPY_UNKNOWN_LABEL",
    "QVTPY_VENOUS_REGION_BASE",
    "QVTPY_VENOUS_UNKNOWN_LABEL",
    "UNKNOWN_LABEL",
    "VENOUS_REGION_BASE",
    "VENOUS_UNKNOWN_LABEL",
    "eicab_label_from_name",
    "eicab_vessel_name",
]
