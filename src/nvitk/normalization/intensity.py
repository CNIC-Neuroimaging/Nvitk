"""
Backend-aware intensity normalisation for cross-modality training corpora.

Description
-----------
Segmentation frameworks pick an intensity-normalisation scheme **per input channel**, not per
case. A dataset that mixes CTA and MRA in one channel therefore cannot be normalised correctly
downstream: CT is calibrated (Hounsfield units, air at −1000) while TOF-MRA is uncalibrated
with a hard floor at 0, and a single scheme is wrong for one of them.

These helpers harmonise the two *before* the framework sees them, mapping both onto a common
intensity range so that one channel is legitimately modality-agnostic:

- CT — clip to a fixed diagnostic window in HU, then rescale the window to the target range.
  Fixed rather than percentile-based, because HU are physically meaningful and a percentile
  window would drift with field of view.
- MR — rescale a robust percentile range to the target range. Percentile-based because TOF has
  no standardised scale; the same vessel is a different number on a different scanner.

Array / axis conventions
------------------------
Shape- and axis-agnostic: these are voxelwise maps that never touch geometry. Spacing, affine
and orientation pass through untouched, and an :class:`~nvitk.types.Image` in yields an
``Image`` out via :meth:`~nvitk.types.Image.with_data`.

I/O and arrays: backend ``np`` after ``setup(globals())``; inputs may be NumPy or CuPy arrays
or ``Image``.
"""

from __future__ import annotations

from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.types import Image

setup(globals())

log = Logger()

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Angiographic CT window in Hounsfield units. Keeps contrast-filled lumen and calcified
#: plaque separable while discarding the air and dense-bone extremes.
CTA_WINDOW: tuple[float, float] = (-100.0, 1500.0)

#: Robust percentile range for uncalibrated MR.
MR_PERCENTILES: tuple[float, float] = (0.5, 99.5)

#: Range both modalities are mapped onto.
TARGET_RANGE: tuple[float, float] = (0.0, 1.0)


def _unwrap(image: Any) -> tuple[Any, Image | None]:
    """Split *image* into a backend array and the ``Image`` to rewrap with (or ``None``)."""
    if isinstance(image, Image):
        return as_backend_array(image.data), image
    return as_backend_array(image), None


def _rewrap(data: Any, source: Image | None) -> Any:
    """Rewrap *data* as an ``Image`` when the input was one, preserving geometry."""
    return source.with_data(data) if source is not None else data


def _rescale(data: Any, lo: float, hi: float, target: Sequence[float]) -> Any:
    """Affine-map ``[lo, hi]`` onto *target*, clipping outside it.

    A degenerate window (``hi <= lo``) yields a constant image at the bottom of the target
    range rather than dividing by zero — that happens for genuinely empty or constant volumes,
    and a NaN volume propagating into training is far worse than an obviously blank one.
    """
    t_lo, t_hi = float(target[0]), float(target[1])
    if hi <= lo:
        log.warning("Degenerate intensity window [%g, %g]; emitting a constant image.", lo, hi)
        return np.full(data.shape, np.float32(t_lo), dtype=np.float32)
    scaled = (data.astype(np.float32) - np.float32(lo)) / np.float32(hi - lo)
    scaled = np.clip(scaled, np.float32(0.0), np.float32(1.0))
    return scaled * np.float32(t_hi - t_lo) + np.float32(t_lo)


def window_ct(
    image: Any,
    *,
    window: Sequence[float] = CTA_WINDOW,
    target_range: Sequence[float] = TARGET_RANGE,
) -> Any:
    """Clip CT to a fixed HU *window* and rescale it onto *target_range*.

    Parameters
    ----------
    window
        ``(low_hu, high_hu)``. Values outside are clamped, not discarded — a clamped
        calcification is still a calcification, and dropping it would punch a hole in the mask.

    Returns
    -------
    Image or array
        ``float32``, same type and geometry as the input.

    Examples
    --------
    >>> harmonised = window_ct(cta, window=(-100, 1500))   # doctest: +SKIP
    """
    data, source = _unwrap(image)
    lo, hi = float(window[0]), float(window[1])
    return _rewrap(_rescale(data, lo, hi, target_range), source)


def robust_scale(
    image: Any,
    *,
    percentiles: Sequence[float] = MR_PERCENTILES,
    target_range: Sequence[float] = TARGET_RANGE,
    mask: Any | None = None,
) -> Any:
    """Rescale the *percentiles* range of an uncalibrated volume onto *target_range*.

    Parameters
    ----------
    percentiles
        ``(low, high)`` in percent. Computed over *mask* when given, else the whole volume.
    mask
        Optional boolean region to compute the percentiles over. For TOF, restricting to
        non-zero voxels avoids letting the large air background set the low percentile.

    Returns
    -------
    Image or array
        ``float32``, same type and geometry as the input.
    """
    data, source = _unwrap(image)
    sample = data
    if mask is not None:
        sample = data[as_backend_array(mask).astype(bool)]
        if sample.size == 0:
            log.warning("Empty percentile mask; falling back to the whole volume.")
            sample = data
    # Percentiles are passed as a plain tuple so the call works on both the NumPy and the
    # CuPy backend; the two-element result is pulled to the host to become Python floats.
    bounds = to_numpy(np.percentile(sample, (float(percentiles[0]), float(percentiles[1]))))
    lo, hi = float(bounds[0]), float(bounds[1])
    return _rewrap(_rescale(data, lo, hi, target_range), source)


def harmonize_modality(
    image: Any,
    modality: str,
    *,
    ct_window: Sequence[float] = CTA_WINDOW,
    mr_percentiles: Sequence[float] = MR_PERCENTILES,
    target_range: Sequence[float] = TARGET_RANGE,
    mr_mask_nonzero: bool = True,
) -> Any:
    """Map a CT or MR volume onto a shared intensity range.

    Parameters
    ----------
    modality
        ``"ct"``/``"cta"`` selects the fixed-window branch; ``"mr"``/``"mra"``/``"tof"`` the
        robust-percentile branch. Matched case-insensitively.
    mr_mask_nonzero
        Compute MR percentiles over non-zero voxels only. TOF stores air as exactly 0 over a
        large fraction of the field of view, which otherwise pins the low percentile at 0 and
        wastes most of the output range on background.

    Raises
    ------
    ValueError
        For an unrecognised *modality*. Guessing would silently apply an HU window to
        arbitrary-unit MR data, which looks plausible and is completely wrong.
    """
    key = str(modality).strip().lower()
    if key in ("ct", "cta"):
        return window_ct(image, window=ct_window, target_range=target_range)
    if key in ("mr", "mra", "tof"):
        mask = None
        if mr_mask_nonzero:
            data, _ = _unwrap(image)
            mask = data > 0
        return robust_scale(
            image, percentiles=mr_percentiles, target_range=target_range, mask=mask
        )
    raise ValueError(
        f"Unknown modality {modality!r}; expected one of ct, cta, mr, mra, tof."
    )


__all__ = [
    "CTA_WINDOW",
    "MR_PERCENTILES",
    "TARGET_RANGE",
    "harmonize_modality",
    "robust_scale",
    "window_ct",
]
