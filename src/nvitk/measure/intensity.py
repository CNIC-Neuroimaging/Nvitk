"""Masked intensity statistics (``mean``, ``median``, ``max``, ``p95``, etc.)."""

from __future__ import annotations

from typing import Any, Iterable

from nvitk.core.backend import setup
from nvitk.types import Image

from ._common import bool_mask, ensure_same_shape, resolve_array

setup(globals())


_ALL_STATS = ("mean", "median", "max", "min", "p95", "p5", "std", "sum", "count")


def masked_stats(
    image: Image | Any,
    mask: Image | Any,
    *,
    stats: Iterable[str] = ("mean", "median", "max", "p95", "std", "sum"),
) -> dict[str, float]:
    """
    Compute a set of intensity statistics for *image* restricted to ``mask > 0``.

    Parameters
    ----------
    image
        Intensity image.
    mask
        Label or binary mask image/array with same shape as *image*.
    stats
        Iterable of stat names to compute. Supported:
        ``mean``, ``median``, ``max``, ``min``, ``p95``, ``p5``, ``std``, ``sum``, ``count``.

    Returns
    -------
    dict[str, float]
        Plain Python floats (CuPy arrays are materialized to host at the tail of the pipeline).
    """
    ensure_same_shape(image, mask)
    raw = resolve_array(image)
    m = bool_mask(mask)

    requested = [s.lower() for s in stats]
    unknown = set(requested) - set(_ALL_STATS)
    if unknown:
        raise ValueError(f"Unknown stats requested: {unknown}. Supported: {_ALL_STATS}")

    vals = raw[m]
    if vals.size == 0:
        raise ValueError("Mask is empty; cannot compute intensity statistics.")

    out: dict[str, float] = {}

    def _f(x: Any) -> float:
        # Final per-scalar conversion to Python float.
        from nvitk.core.array import to_numpy

        arr = to_numpy(x)
        return float(arr) if arr.ndim == 0 else float(arr.item())

    if "mean" in requested:
        out["mean"] = _f(np.mean(vals))
    if "median" in requested:
        out["median"] = _f(np.median(vals))
    if "max" in requested:
        out["max"] = _f(np.max(vals))
    if "min" in requested:
        out["min"] = _f(np.min(vals))
    if "p95" in requested:
        out["p95"] = _f(np.percentile(vals, 95))
    if "p5" in requested:
        out["p5"] = _f(np.percentile(vals, 5))
    if "std" in requested:
        out["std"] = _f(np.std(vals))
    if "sum" in requested:
        out["sum"] = _f(np.sum(vals))
    if "count" in requested:
        out["count"] = float(int(vals.size))

    return out


__all__ = ["masked_stats"]
