"""Vessel label IDs: eICAB input masks vs qvtpy pipeline outputs.

**eICAB** integers match ``*_eICAB_CW.nii(.gz)`` (labels 0–18).

**qvtpy** integers are used in stage 3 ``centerlines_mask``, stage 4
``seg_4dflow``, and downstream LOC / flow tables. Arterial labels are derived
from eICAB via :func:`relabel_eicab_mask_to_qvtpy` (PCA segments merged;
SCA/AChA dropped). Venous sinuses use fixed ids 31–34.

See :data:`QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID` for the combined id→name table.
"""

from __future__ import annotations

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup

setup(globals())

# =============================================================================
# eICAB — input multilabel mask (0–18)
# =============================================================================

EICAB_BACKGROUND: int = 0

EICAB_LICA: int = 1
EICAB_RICA: int = 2
EICAB_BASILAR: int = 3
EICAB_ACOMM: int = 4
EICAB_LACA: int = 5
EICAB_RACA: int = 6
EICAB_LMCA: int = 7
EICAB_RMCA: int = 8
EICAB_LPCOMM: int = 9
EICAB_RPCOMM: int = 10
EICAB_LPCA_P1: int = 11
EICAB_RPCA_P1: int = 12
EICAB_LPCA_P2: int = 13
EICAB_RPCA_P2: int = 14
EICAB_LSCA: int = 15
EICAB_RSCA: int = 16
EICAB_LACHA: int = 17
EICAB_RACHA: int = 18

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

EICAB_VESSEL_LABEL_IDS: frozenset[int] = frozenset(range(1, 19))

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

# =============================================================================
# qvtpy — arterial labels (centerline backbone + seg_4dflow)
# =============================================================================

QVTPY_BACKGROUND: int = 0

QVTPY_LICA: int = 1
QVTPY_RICA: int = 2
QVTPY_BASILAR: int = 3
QVTPY_LACA: int = 4
QVTPY_RACA: int = 5
QVTPY_LMCA: int = 6
QVTPY_RMCA: int = 7
QVTPY_LPCA: int = 8
QVTPY_RPCA: int = 9
QVTPY_LPCOMM: int = 10
QVTPY_RPCOMM: int = 11
QVTPY_ACOMM: int = 12

QVTPY_ARTERIAL_ID_TO_NAME: dict[int, str] = {
    QVTPY_BACKGROUND: "BACKGROUND",
    QVTPY_LICA: "LICA",
    QVTPY_RICA: "RICA",
    QVTPY_BASILAR: "BASILAR",
    QVTPY_LACA: "LACA",
    QVTPY_RACA: "RACA",
    QVTPY_LMCA: "LMCA",
    QVTPY_RMCA: "RMCA",
    QVTPY_LPCA: "LPCA",
    QVTPY_RPCA: "RPCA",
    QVTPY_LPCOMM: "LPCOMM",
    QVTPY_RPCOMM: "RPCOMM",
    QVTPY_ACOMM: "ACOMM",
}

QVTPY_ARTERIAL_NAME_TO_ID: dict[str, int] = {
    name: int(lid) for lid, name in QVTPY_ARTERIAL_ID_TO_NAME.items() if int(lid) > 0
}

QVTPY_ARTERIAL_LABEL_IDS: frozenset[int] = frozenset(
    lid for lid in QVTPY_ARTERIAL_ID_TO_NAME if int(lid) > 0
)

# Vessel groups for stage-4 bbox padding (array axes i=X, j=Y, k=Z).
QVTPY_ICA_BASILAR_IDS: frozenset[int] = frozenset({QVTPY_LICA, QVTPY_RICA, QVTPY_BASILAR})
QVTPY_ACA_IDS: frozenset[int] = frozenset({QVTPY_LACA, QVTPY_RACA})
QVTPY_MCA_IDS: frozenset[int] = frozenset({QVTPY_LMCA, QVTPY_RMCA})
QVTPY_PCA_IDS: frozenset[int] = frozenset({QVTPY_LPCA, QVTPY_RPCA})
QVTPY_COMM_IDS: frozenset[int] = frozenset({QVTPY_LPCOMM, QVTPY_RPCOMM, QVTPY_ACOMM})

# Stage 4: gentler crop filtering (PCA/PComm/AComm) — see QVTPY_SMALL_ARTERIAL_IDS below.
QVTPY_SMALL_ARTERIAL_IDS: frozenset[int] = QVTPY_PCA_IDS | QVTPY_COMM_IDS

# =============================================================================
# qvtpy — venous labels (fixed 31–34)
# =============================================================================

NAME_SSSV: str = "SSSV"
NAME_STRV: str = "STRV"
NAME_LTSV: str = "LTSV"
NAME_RTSV: str = "RTSV"

NAME_RCOMM: str = "RCOMM"
NAME_LCOMM: str = "LCOMM"
NAME_COMM: str = "COMM"

QVTPY_SSSV: int = 31
QVTPY_STRV: int = 32
QVTPY_LTSV: int = 33
QVTPY_RTSV: int = 34

VENOUS_LABEL_SSSV: int = QVTPY_SSSV
VENOUS_LABEL_STRV: int = QVTPY_STRV
VENOUS_LABEL_LTSV: int = QVTPY_LTSV
VENOUS_LABEL_RTSV: int = QVTPY_RTSV

VENOUS_LABEL_BY_NAME: dict[str, int] = {
    NAME_SSSV: VENOUS_LABEL_SSSV,
    NAME_STRV: VENOUS_LABEL_STRV,
    NAME_LTSV: VENOUS_LABEL_LTSV,
    NAME_RTSV: VENOUS_LABEL_RTSV,
}

VENOUS_NAME_BY_LABEL: dict[int, str] = {v: k for k, v in VENOUS_LABEL_BY_NAME.items()}

MATLAB_QVT_VENOUS_VESSEL_NAMES: tuple[str, ...] = (NAME_SSSV, NAME_STRV, NAME_LTSV, NAME_RTSV)

QVTPY_VENOUS_LABEL_IDS: frozenset[int] = frozenset(VENOUS_NAME_BY_LABEL.keys())

# Stage 4 region growing: more exploration on ACA/MCA/PCA; STRV never grows.
QVTPY_RG_EXPLORE_MORE_IDS: frozenset[int] = QVTPY_ACA_IDS | QVTPY_MCA_IDS | QVTPY_PCA_IDS
QVTPY_RG_SKIP_LABEL_IDS: frozenset[int] = frozenset({QVTPY_STRV})

# Per-sinus RG intensity fractions (lower → more growth). STRV omitted (RG disabled).
QVTPY_RG_INTENSITY_FRAC_VENOUS: dict[int, float] = {
    QVTPY_SSSV: 0.40,
    QVTPY_LTSV: 0.38,
    QVTPY_RTSV: 0.38,
}

# =============================================================================
# qvtpy — reserved / unused on backbone+seg
# =============================================================================

QVTPY_VENOUS_UNKNOWN_LABEL: int = 30
QVTPY_VENOUS_REGION_BASE: int = 31
QVTPY_UNKNOWN_LABEL: int = 35

VENOUS_UNKNOWN_LABEL: int = QVTPY_VENOUS_UNKNOWN_LABEL
VENOUS_REGION_BASE: int = QVTPY_VENOUS_REGION_BASE
UNKNOWN_LABEL: int = QVTPY_UNKNOWN_LABEL

QVTPY_RESERVED_LABEL_BY_ID: dict[int, str] = {
    QVTPY_VENOUS_UNKNOWN_LABEL: "QVTPY_VENOUS_UNKNOWN",
    QVTPY_UNKNOWN_LABEL: "QVTPY_UNKNOWN",
}

# =============================================================================
# eICAB → qvtpy mapping (stage 3 relabel)
# =============================================================================

EICAB_LABEL_IDS_OMITTED: frozenset[int] = frozenset(
    {EICAB_LSCA, EICAB_RSCA, EICAB_LACHA, EICAB_RACHA}
)

EICAB_TO_QVTPY_LABEL: dict[int, int] = {
    EICAB_BACKGROUND: QVTPY_BACKGROUND,
    EICAB_LICA: QVTPY_LICA,
    EICAB_RICA: QVTPY_RICA,
    EICAB_BASILAR: QVTPY_BASILAR,
    EICAB_ACOMM: QVTPY_ACOMM,
    EICAB_LACA: QVTPY_LACA,
    EICAB_RACA: QVTPY_RACA,
    EICAB_LMCA: QVTPY_LMCA,
    EICAB_RMCA: QVTPY_RMCA,
    EICAB_LPCOMM: QVTPY_LPCOMM,
    EICAB_RPCOMM: QVTPY_RPCOMM,
    EICAB_LPCA_P1: QVTPY_LPCA,
    EICAB_RPCA_P1: QVTPY_RPCA,
    EICAB_LPCA_P2: QVTPY_LPCA,
    EICAB_RPCA_P2: QVTPY_RPCA,
    EICAB_LSCA: QVTPY_BACKGROUND,
    EICAB_RSCA: QVTPY_BACKGROUND,
    EICAB_LACHA: QVTPY_BACKGROUND,
    EICAB_RACHA: QVTPY_BACKGROUND,
}

QVTPY_TO_EICAB_LABELS: dict[int, tuple[int, ...]] = {
    QVTPY_LICA: (EICAB_LICA,),
    QVTPY_RICA: (EICAB_RICA,),
    QVTPY_BASILAR: (EICAB_BASILAR,),
    QVTPY_LACA: (EICAB_LACA,),
    QVTPY_RACA: (EICAB_RACA,),
    QVTPY_LMCA: (EICAB_LMCA,),
    QVTPY_RMCA: (EICAB_RMCA,),
    QVTPY_LPCA: (EICAB_LPCA_P1, EICAB_LPCA_P2),
    QVTPY_RPCA: (EICAB_RPCA_P1, EICAB_RPCA_P2),
    QVTPY_LPCOMM: (EICAB_LPCOMM,),
    QVTPY_RPCOMM: (EICAB_RPCOMM,),
    QVTPY_ACOMM: (EICAB_ACOMM,),
}

# =============================================================================
# Combined tables (centerline backbone + seg_4dflow)
# =============================================================================

QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID: dict[int, str] = {
    **QVTPY_ARTERIAL_ID_TO_NAME,
    **VENOUS_NAME_BY_LABEL,
}
QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID.pop(QVTPY_BACKGROUND, None)

QVTPY_CENTERLINE_BACKBONE_LABEL_BY_ID: dict[int, str] = QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID
QVTPY_SEG_4DFLOW_LABEL_BY_ID: dict[int, str] = QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID

QVTPY_CENTERLINE_AND_SEG_NAME_TO_LABEL: dict[str, int] = {
    **QVTPY_ARTERIAL_NAME_TO_ID,
    **{str(k).upper(): int(v) for k, v in VENOUS_LABEL_BY_NAME.items()},
}

QVTPY_CENTERLINE_BACKBONE_NAME_TO_LABEL: dict[str, int] = QVTPY_CENTERLINE_AND_SEG_NAME_TO_LABEL
QVTPY_SEG_4DFLOW_NAME_TO_LABEL: dict[str, int] = QVTPY_CENTERLINE_AND_SEG_NAME_TO_LABEL

QVTPY_CENTERLINE_AND_SEG_LABEL_IDS: frozenset[int] = frozenset(
    QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID.keys()
)

# =============================================================================
# Relabel + lookup helpers
# =============================================================================


def relabel_eicab_mask_to_qvtpy(volume: np.ndarray) -> np.ndarray:
    """Map warped eICAB labels to qvtpy arterial ids (merge PCA; drop SCA/AChA)."""
    vol = as_backend_array(volume)
    out = np.zeros_like(vol, dtype=np.int32)
    for eicab_id, qvt_id in EICAB_TO_QVTPY_LABEL.items():
        if int(eicab_id) <= 0 or int(qvt_id) <= 0:
            continue
        out[vol == int(eicab_id)] = int(qvt_id)
    return out


def eicab_vessel_name(label_id: int) -> str:
    """Name for an **eICAB** mask integer."""
    lid = int(label_id)
    if lid in EICAB_ID_TO_NAME:
        return EICAB_ID_TO_NAME[lid]
    return f"EICAB_UNKNOWN_{lid}"


def eicab_label_from_name(name: str) -> int | None:
    """eICAB label id for a known input-mask vessel name, or None."""
    key = name.strip().upper()
    out = EICAB_NAME_TO_ID.get(key)
    return None if out is None else int(out)


def qvtpy_vessel_name(label_id: int) -> str:
    """Name for a qvtpy label in ``centerlines_mask`` / ``seg_4dflow``."""
    lid = int(label_id)
    if lid in QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID:
        return QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID[lid]
    if lid in QVTPY_RESERVED_LABEL_BY_ID:
        return QVTPY_RESERVED_LABEL_BY_ID[lid]
    return f"QVTPY_UNKNOWN_{lid}"


def qvtpy_label_from_name(name: str) -> int | None:
    """qvtpy label id for a pipeline vessel name (arterial or venous), or None."""
    key = name.strip().upper()
    out = QVTPY_CENTERLINE_AND_SEG_NAME_TO_LABEL.get(key)
    return None if out is None else int(out)


__all__ = [
    "EICAB_ACOMM",
    "EICAB_BACKGROUND",
    "EICAB_BASI",
    "EICAB_BASILAR",
    "EICAB_ID_TO_NAME",
    "EICAB_LABEL_4_UNDOCUMENTED",
    "EICAB_LABEL_IDS_OMITTED",
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
    "EICAB_TO_QVTPY_LABEL",
    "EICAB_VESSEL_LABEL_IDS",
    "MATLAB_QVT_VENOUS_VESSEL_NAMES",
    "NAME_COMM",
    "NAME_LCOMM",
    "NAME_LTSV",
    "NAME_RCOMM",
    "NAME_RTSV",
    "NAME_SSSV",
    "NAME_STRV",
    "QVTPY_ACOMM",
    "QVTPY_ACA_IDS",
    "QVTPY_ARTERIAL_ID_TO_NAME",
    "QVTPY_ARTERIAL_LABEL_IDS",
    "QVTPY_ARTERIAL_NAME_TO_ID",
    "QVTPY_COMM_IDS",
    "QVTPY_ICA_BASILAR_IDS",
    "QVTPY_MCA_IDS",
    "QVTPY_PCA_IDS",
    "QVTPY_RG_EXPLORE_MORE_IDS",
    "QVTPY_RG_INTENSITY_FRAC_VENOUS",
    "QVTPY_RG_SKIP_LABEL_IDS",
    "QVTPY_SMALL_ARTERIAL_IDS",
    "QVTPY_BACKGROUND",
    "QVTPY_BASILAR",
    "QVTPY_CENTERLINE_AND_SEG_LABEL_BY_ID",
    "QVTPY_CENTERLINE_AND_SEG_LABEL_IDS",
    "QVTPY_CENTERLINE_AND_SEG_NAME_TO_LABEL",
    "QVTPY_CENTERLINE_BACKBONE_LABEL_BY_ID",
    "QVTPY_CENTERLINE_BACKBONE_NAME_TO_LABEL",
    "QVTPY_LACA",
    "QVTPY_LICA",
    "QVTPY_LMCA",
    "QVTPY_LPCA",
    "QVTPY_LPCOMM",
    "QVTPY_LTSV",
    "QVTPY_RESERVED_LABEL_BY_ID",
    "QVTPY_RACA",
    "QVTPY_RICA",
    "QVTPY_RMCA",
    "QVTPY_RPCA",
    "QVTPY_RPCOMM",
    "QVTPY_RTSV",
    "QVTPY_SEG_4DFLOW_LABEL_BY_ID",
    "QVTPY_SEG_4DFLOW_NAME_TO_LABEL",
    "QVTPY_SSSV",
    "QVTPY_STRV",
    "QVTPY_TO_EICAB_LABELS",
    "QVTPY_UNKNOWN_LABEL",
    "QVTPY_VENOUS_LABEL_IDS",
    "QVTPY_VENOUS_REGION_BASE",
    "QVTPY_VENOUS_UNKNOWN_LABEL",
    "UNKNOWN_LABEL",
    "VENOUS_LABEL_BY_NAME",
    "VENOUS_LABEL_LTSV",
    "VENOUS_LABEL_RTSV",
    "VENOUS_LABEL_SSSV",
    "VENOUS_LABEL_STRV",
    "VENOUS_NAME_BY_LABEL",
    "VENOUS_REGION_BASE",
    "VENOUS_UNKNOWN_LABEL",
    "eicab_label_from_name",
    "eicab_vessel_name",
    "qvtpy_label_from_name",
    "qvtpy_vessel_name",
    "relabel_eicab_mask_to_qvtpy",
]
