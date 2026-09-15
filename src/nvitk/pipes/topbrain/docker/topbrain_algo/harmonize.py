"""Intensity harmonisation, as stage 0 applied it.

The windows are not here: stage 5 records them in ``models.json`` and the entry point passes
them in. What is here is the shape of the two transforms, which is the part that does not vary
per run:

* **CT** is calibrated, so a fixed HU window is meaningful — clamp and rescale onto ``[0, 1]``.
  Values outside are clamped rather than dropped: a saturated calcification is still a
  calcification, and discarding it would punch a hole in the vessel.
* **MR / TOF** carries arbitrary units, so the window is measured per volume from robust
  percentiles, over **non-zero voxels only** — TOF stores air as exactly 0 across most of the
  field of view, which would otherwise pin the low percentile at 0 and waste the output range.

Only the *clipping* these do survives nnU-Net's per-image z-score; an affine intensity map leaves
a z-score unchanged. That is why the exact window matters and why it travels with the model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

#: Every transform lands here.
TARGET_RANGE: tuple[float, float] = (0.0, 1.0)


def _rescale(data: np.ndarray, lo: float, hi: float,
             target: Sequence[float] = TARGET_RANGE) -> np.ndarray:
    """Affine-map ``[lo, hi]`` onto *target*, clipping outside it.

    A degenerate window yields a constant image rather than dividing by zero: that happens only
    for genuinely empty or constant volumes, and a NaN volume reaching the network is far worse
    than an obviously blank one.
    """
    t_lo, t_hi = float(target[0]), float(target[1])
    if hi <= lo:
        return np.full(data.shape, np.float32(t_lo), dtype=np.float32)
    scaled = (data.astype(np.float32) - np.float32(lo)) / np.float32(hi - lo)
    scaled = np.clip(scaled, np.float32(0.0), np.float32(1.0))
    return scaled * np.float32(t_hi - t_lo) + np.float32(t_lo)


def window_ct(data: np.ndarray, *, window: Sequence[float]) -> np.ndarray:
    """Clip CT to a fixed HU *window* and rescale onto ``[0, 1]``."""
    return _rescale(data, float(window[0]), float(window[1]))


def robust_scale(data: np.ndarray, *, percentiles: Sequence[float],
                 mask_nonzero: bool = True) -> np.ndarray:
    """Rescale MR by measured *percentiles*, over non-zero voxels by default."""
    sample = data[data > 0] if mask_nonzero else data
    if sample.size == 0:
        sample = data.ravel()
    lo, hi = np.percentile(sample.astype(np.float32), list(percentiles))
    return _rescale(data, float(lo), float(hi))


__all__ = ["TARGET_RANGE", "robust_scale", "window_ct"]
