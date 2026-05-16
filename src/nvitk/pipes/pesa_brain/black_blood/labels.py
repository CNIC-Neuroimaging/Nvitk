"""eICAB → black-blood arterial label table (local copy; no qvtpy import)."""

from __future__ import annotations

import numpy as np

from nvitk.core.array import as_backend_array

# eICAB integers (Circle-of-Willis multilabel).
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

# Black-blood arterial ids (1–12).
BB_BACKGROUND: int = 0
BB_LICA: int = 1
BB_RICA: int = 2
BB_BASILAR: int = 3
BB_LACA: int = 4
BB_RACA: int = 5
BB_LMCA: int = 6
BB_RMCA: int = 7
BB_LPCA: int = 8
BB_RPCA: int = 9
BB_LPCOMM: int = 10
BB_RPCOMM: int = 11
BB_ACOMM: int = 12

BB_ARTERIAL_ID_TO_NAME: dict[int, str] = {
    BB_BACKGROUND: "BACKGROUND",
    BB_LICA: "LICA",
    BB_RICA: "RICA",
    BB_BASILAR: "BASILAR",
    BB_LACA: "LACA",
    BB_RACA: "RACA",
    BB_LMCA: "LMCA",
    BB_RMCA: "RMCA",
    BB_LPCA: "LPCA",
    BB_RPCA: "RPCA",
    BB_LPCOMM: "LPCOMM",
    BB_RPCOMM: "RPCOMM",
    BB_ACOMM: "ACOMM",
}

BB_ARTERIAL_LABEL_IDS: frozenset[int] = frozenset(
    lid for lid in BB_ARTERIAL_ID_TO_NAME if int(lid) > 0
)

EICAB_TO_BB_LABEL: dict[int, int] = {
    EICAB_BACKGROUND: BB_BACKGROUND,
    EICAB_LICA: BB_LICA,
    EICAB_RICA: BB_RICA,
    EICAB_BASILAR: BB_BASILAR,
    EICAB_ACOMM: BB_ACOMM,
    EICAB_LACA: BB_LACA,
    EICAB_RACA: BB_RACA,
    EICAB_LMCA: BB_LMCA,
    EICAB_RMCA: BB_RMCA,
    EICAB_LPCOMM: BB_LPCOMM,
    EICAB_RPCOMM: BB_RPCOMM,
    EICAB_LPCA_P1: BB_LPCA,
    EICAB_RPCA_P1: BB_RPCA,
    EICAB_LPCA_P2: BB_LPCA,
    EICAB_RPCA_P2: BB_RPCA,
    EICAB_LSCA: BB_BACKGROUND,
    EICAB_RSCA: BB_BACKGROUND,
    EICAB_LACHA: BB_BACKGROUND,
    EICAB_RACHA: BB_BACKGROUND,
}


def relabel_eicab_to_bb(volume: np.ndarray) -> np.ndarray:
    """Map eICAB labels to black-blood arterial ids (merge PCA; drop SCA/AChA)."""
    vol = as_backend_array(volume)
    out = np.zeros_like(vol, dtype=np.int32)
    for eicab_id, bb_id in EICAB_TO_BB_LABEL.items():
        if int(eicab_id) <= 0 or int(bb_id) <= 0:
            continue
        out[vol == int(eicab_id)] = int(bb_id)
    return out


def bb_vessel_name(label_id: int) -> str:
    """Name for a black-blood arterial label id."""
    lid = int(label_id)
    if lid in BB_ARTERIAL_ID_TO_NAME:
        return BB_ARTERIAL_ID_TO_NAME[lid]
    return f"LABEL_{lid}"
