"""eICAB label topology for TOF morphometrics (replaces external topology_eICAB.json)."""

from __future__ import annotations

from nvitk.measure.morpho.models import VesselInfo
from nvitk.pipes.qvtpy.labels import (
    EICAB_ACOMM,
    EICAB_BASILAR,
    EICAB_ID_TO_NAME,
    EICAB_LACA,
    EICAB_LACHA,
    EICAB_LICA,
    EICAB_LMCA,
    EICAB_LPCOMM,
    EICAB_LPCA_P1,
    EICAB_LPCA_P2,
    EICAB_LSCA,
    EICAB_RACA,
    EICAB_RACHA,
    EICAB_RICA,
    EICAB_RMCA,
    EICAB_RPCOMM,
    EICAB_RPCA_P1,
    EICAB_RPCA_P2,
    EICAB_RSCA,
    EICAB_VESSEL_LABEL_IDS,
)

_SIDE_BY_PREFIX = {
    "L": "L",
    "R": "R",
}

_PAIR_BY_NAME: dict[str, str] = {
    "LICA": "ICA",
    "RICA": "ICA",
    "LACA": "ACA",
    "RACA": "ACA",
    "LMCA": "MCA",
    "RMCA": "MCA",
    "LPCOMM": "PCOMM",
    "RPCOMM": "PCOMM",
    "LPCA_P1": "PCA",
    "RPCA_P1": "PCA",
    "LPCA_P2": "PCA",
    "RPCA_P2": "PCA",
    "LSCA": "SCA",
    "RSCA": "SCA",
    "LACHA": "ACHA",
    "RACHA": "ACHA",
}

_TERRITORY_BY_NAME: dict[str, str] = {
    "LICA": "anterior",
    "RICA": "anterior",
    "BASILAR": "posterior",
    "ACOMM": "anterior",
    "LACA": "anterior",
    "RACA": "anterior",
    "LMCA": "anterior",
    "RMCA": "anterior",
    "LPCOMM": "anterior",
    "RPCOMM": "anterior",
    "LPCA_P1": "posterior",
    "RPCA_P1": "posterior",
    "LPCA_P2": "posterior",
    "RPCA_P2": "posterior",
    "LSCA": "posterior",
    "RSCA": "posterior",
}

_FLOW_FROM: dict[str, str] = {
    "LICA": "systemic",
    "RICA": "systemic",
    "BASILAR": "systemic",
    "LMCA": "LICA",
    "RMCA": "RICA",
    "LACA": "LICA",
    "RACA": "RICA",
    "LPCOMM": "LICA",
    "RPCOMM": "RICA",
    "LPCA_P1": "BASILAR",
    "RPCA_P1": "BASILAR",
    "LPCA_P2": "LPCA_P1",
    "RPCA_P2": "RPCA_P1",
    "LSCA": "BASILAR",
    "RSCA": "BASILAR",
    "ACOMM": "LICA",
}

_FLOW_TO: dict[str, list[str]] = {
    "LICA": ["LMCA", "LACA", "LPCOMM"],
    "RICA": ["RMCA", "RACA", "RPCOMM"],
    "BASILAR": ["LPCA_P1", "RPCA_P1", "LSCA", "RSCA"],
    "LMCA": [],
    "RMCA": [],
    "LACA": [],
    "RACA": [],
    "LPCOMM": ["LPCA_P1"],
    "RPCOMM": ["RPCA_P1"],
    "LPCA_P1": ["LPCA_P2"],
    "RPCA_P1": ["RPCA_P2"],
    "LPCA_P2": [],
    "RPCA_P2": [],
    "LSCA": [],
    "RSCA": [],
    "ACOMM": ["LACA", "RACA"],
    "LACHA": [],
    "RACHA": [],
}

_NO_UPSTREAM: dict[str, str] = {
    "LICA": "inferior",
    "RICA": "inferior",
    "BASILAR": "inferior",
    "LACA": "anterior",
    "RACA": "anterior",
}


def _vessel_info_for_name(name: str) -> VesselInfo:
    side = _SIDE_BY_PREFIX.get(name[:1], "")
    return VesselInfo(
        name=name,
        full_name=name,
        side=side,
        pair=_PAIR_BY_NAME.get(name),
        territory=_TERRITORY_BY_NAME.get(name, ""),
        flow_from=_FLOW_FROM.get(name, ""),
        flow_to=list(_FLOW_TO.get(name, [])),
        no_upstream_start=_NO_UPSTREAM.get(name),
    )


def build_eicab_topology_mapping() -> dict[int, VesselInfo]:
    """Return label-id -> :class:`VesselInfo` for all eICAB vessel labels (1–18)."""
    out: dict[int, VesselInfo] = {}
    for lid in sorted(EICAB_VESSEL_LABEL_IDS):
        name = EICAB_ID_TO_NAME.get(int(lid), f"label_{lid}")
        out[int(lid)] = _vessel_info_for_name(str(name))
    return out


__all__ = ["build_eicab_topology_mapping"]
