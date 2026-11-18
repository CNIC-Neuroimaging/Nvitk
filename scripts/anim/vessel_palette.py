"""Vessel colours shared by the dense animations (left darker, right lighter; a distal split
(A2 / M2) is a lighter shade of its parent)."""

from __future__ import annotations

import numpy as np

from anim_voxels import rgb

#: eICAB label ids (nvitk.pipes.qvtpy.labels EICAB_*).
EICAB_COLORS = {
    1: "#C99A06", 2: "#E5C34A",              # ICAs
    3: "#D9558F",                            # basilar
    4: "#6C3F8F",                            # AComm
    5: "#D63A2F", 6: "#EF867B",              # ACAs (A1)
    7: "#2A72C8", 8: "#7DAEE8",              # MCAs (M1)
    9: "#8E44AD", 10: "#BC8BD6",             # PComms
    11: "#127A70", 12: "#5DB5AA",            # PCA P1
    13: "#2E8C47", 14: "#7FC491",            # PCA P2
    15: "#B4581F", 16: "#DE9A6A",            # SCAs
    17: "#6F7A7B", 18: "#A6AFB0",            # AChAs
    19: "#5B4FC4", 20: "#9A92E0",            # VAs
    21: "#EE7A55", 22: "#F6B39C",            # ACA A2
    23: "#4FA3E6", 24: "#A6CDF3",            # MCA M2
}
EICAB_NAMES = {1: "LICA", 2: "RICA", 3: "BA", 4: "AComm", 5: "LACA", 6: "RACA", 7: "LMCA", 8: "RMCA",
               9: "LPComm", 10: "RPComm", 11: "LPCA P1", 12: "RPCA P1", 13: "LPCA P2", 14: "RPCA P2",
               15: "LSCA", 16: "RSCA", 17: "LAChA", 18: "RAChA", 19: "LVA", 20: "RVA", 21: "LACA A2",
               22: "RACA A2", 23: "LMCA M2", 24: "RMCA M2"}

#: qvtpy label ids (nvitk.pipes.qvtpy.labels QVTPY_*): the eICAB colour of the same vessel; sinuses in slate.
_Q2E = {1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 7, 7: 8, 8: 11, 9: 12, 10: 9, 11: 10, 12: 4, 13: 19, 14: 20, 17: 21, 18: 22,
        19: 23, 20: 24, 21: 13, 22: 14}
QVTPY_COLORS = {**{q: EICAB_COLORS[e] for q, e in _Q2E.items()},
                30: "#8F96A3", 31: "#4B3F72", 32: "#7E6FA8", 33: "#3E6F86", 34: "#86AFC4"}
QVTPY_NAMES = {1: "LICA", 2: "RICA", 3: "BA", 4: "LACA A1", 5: "RACA A1", 6: "LMCA M1", 7: "RMCA M1", 8: "LPCA P1",
               9: "RPCA P1", 10: "LPComm", 11: "RPComm", 12: "AComm", 13: "LVA", 14: "RVA", 17: "LACA A2",
               18: "RACA A2", 19: "LMCA M2", 20: "RMCA M2", 21: "LPCA P2", 22: "RPCA P2", 31: "SSS", 32: "StrS",
               33: "LTS", 34: "RTS"}

GHOST = "#AEB5BF"
REMOVED = "#E8443A"
ADDED = "#2E9E4F"
CUT = "#D6336C"
PATH = "#1B1D23"


def colors_of(labels: np.ndarray, table: dict[int, str] = EICAB_COLORS, default: str = GHOST) -> np.ndarray:
    """RGB (n, 3) float32 per label id."""
    lut = np.tile(rgb(default), (256, 1))
    for k, c in table.items():
        lut[int(k) % 256] = rgb(c)
    return lut[np.asarray(labels, int) % 256].astype("f4")


def sequential(t: np.ndarray, stops=("#F7E7A6", "#F2A541", "#D9482B", "#7A1631")) -> np.ndarray:
    """A warm sequential ramp (low → high) at *t* ∈ [0, 1], (n, 3)."""
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    cs = np.array([rgb(c) for c in stops])
    x = t * (len(cs) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(cs) - 2)
    f = (x - i)[..., None]
    return (cs[i] * (1 - f) + cs[i + 1] * f).astype("f4")


def diverging(t: np.ndarray, stops=("#2C6FBB", "#9DC3E6", "#F4F4F4", "#F2A07B", "#C2352B")) -> np.ndarray:
    """Blue - white - red at *t* ∈ [-1, 1], (n, 3)."""
    return sequential((np.asarray(t, float) + 1.0) / 2.0, stops)


def speed_ramp(t: np.ndarray) -> np.ndarray:
    """Flow speed: deep blue (slow) → cyan → yellow → red (fast), *t* ∈ [0, 1]."""
    return sequential(t, ("#1D3A8A", "#2E86C1", "#3FC1C9", "#F2D43D", "#F08A24", "#C8222C"))


__all__ = ["ADDED", "CUT", "EICAB_COLORS", "EICAB_NAMES", "GHOST", "PATH", "QVTPY_COLORS", "QVTPY_NAMES", "REMOVED",
           "colors_of", "diverging", "sequential", "speed_ramp"]
