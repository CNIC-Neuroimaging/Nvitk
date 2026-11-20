"""Structure colours shared by the PESA-Fat animations (Dixon and CT-PET): left darker, right
lighter; fat in warm yellows, bone marrow in gold; red marks what a step removes, and the QC
report's convention is kept for the two mask kinds (red = the eroded measurement mask, blue = the
volume mask)."""

from __future__ import annotations

import numpy as np

from anim_voxels import rgb

LIVER = "#A8432F"
SPLEEN = "#7E3F8F"
PANCREAS = "#E0A12E"
KIDNEY_L, KIDNEY_R = "#8E2F55", "#C9718F"
PVM_L, PVM_R = "#1F6F8B", "#74B3CC"
QUAD_L, QUAD_R = "#257A4C", "#7CC197"
DELT_L, DELT_R = "#5B4FC4", "#9F96E3"
TRAP = "#3F5F7F"
L3, L4 = "#C08A00", "#EDC65A"
FAT_V, FAT_SC = "#E8892B", "#F4D77A"
IM_LEGS, IM_TRUNK = "#C04A6A", "#6E4AA0"
BONE = "#CFC8B8"
SKIN = "#E9CFBF"
URETER = "#2F80ED"
BLADDER = "#3BA3B8"
REMOVED = "#E8443A"
ADDED = "#2E9E4F"
MEASURE = "#D62828"
VOLUME = "#2B6CB0"
GHOST = "#AEB5BF"


def lut(table: dict[int, str], default: str = GHOST) -> np.ndarray:
    """(256, 3) float32 colour table from ``{label id: hex}``."""
    out = np.tile(rgb(default), (256, 1)).astype("f4")
    for k, c in table.items():
        out[int(k) % 256] = rgb(c)
    return out


def colors_of(labels: np.ndarray, table: dict[int, str], default: str = GHOST) -> np.ndarray:
    return lut(table, default)[np.asarray(labels, int) % 256]


def categorical(ids: np.ndarray, seed: int = 7) -> np.ndarray:
    """A stable pastel colour per arbitrary class id (the raw TotalSegmentator classes), (n, 3)."""
    ids = np.asarray(ids, int)
    rng = np.random.default_rng(seed)
    hues = rng.permutation(256) / 256.0
    h = hues[ids % 256]
    s, v = 0.45, 0.86
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.c_[r, g, b].astype("f4")


def ramp(t: np.ndarray, stops: tuple[str, ...]) -> np.ndarray:
    """Piecewise-linear colour ramp through *stops* at *t* ∈ [0, 1], (n, 3)."""
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    cs = np.array([rgb(c) for c in stops])
    x = t * (len(cs) - 1)
    k = np.minimum(x.astype(int), len(cs) - 2)
    w = (x - k)[..., None]
    return (cs[k] * (1 - w) + cs[k + 1] * w).astype("f4")


#: fat fraction 0 → 50 %: muscle red-brown to fat yellow
FF_STOPS = ("#5A1E2A", "#A8432F", "#E8892B", "#F7E08A", "#FFF8DC")
#: SUV low → high on a white page: pale to dark hot
SUV_STOPS = ("#FBF1DE", "#F5BE7E", "#E0663A", "#A3233A", "#4A0E2E")
