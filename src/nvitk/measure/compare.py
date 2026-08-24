"""
Intensity correlation utilities (Pearson, Spearman, MAE, RMSE).

Port of the numeric core of the BioImaging ``_pet_correlation.py`` module.
Plotting and report generation are intentionally left out; they can live in a
separate optional submodule if/when needed.

Bug fixes relative to the legacy implementation:

- Adds the missing ``from scipy import stats`` import (the legacy file used
  ``stats.pearsonr`` without ever importing ``stats``).
- Decouples numeric computation from the ``inv_original_lps`` visualization
  scope leak.

All arithmetic stays on the active backend (NumPy or CuPy). The only host
hop is where SciPy's ``scipy.stats`` is used, since CuPy does not provide a
``stats`` submodule and ``pearsonr``/``spearmanr`` require NumPy inputs.
"""

from __future__ import annotations

from typing import Any

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.types import Image

from ._common import resolve_array

setup(globals())


def _ravel(x: Any) -> Any:
    """Cast *x* to the active backend and flatten (no host hop)."""
    return as_backend_array(resolve_array(x)).ravel()


def _float(x: Any) -> float:
    """Final scalar conversion to Python float (host hop)."""
    arr = as_backend_array(x)
    return float(arr) if arr.ndim == 0 else float(arr.item())

def pearson(a: Any, b: Any) -> tuple[float, float]:
    """Return ``(pearson_r, pearson_p)`` for two 1-D arrays."""
    # scipy.stats.pearsonr requires NumPy; materialize here only.
    ah = as_backend_array(resolve_array(a)).ravel()
    bh = as_backend_array(resolve_array(b)).ravel()
    r, p = scipy.stats().pearsonr(ah, bh)
    return float(r), float(p)


def spearman(a: Any, b: Any) -> tuple[float, float]:
    """Return ``(spearman_r, spearman_p)`` for two 1-D arrays."""
    ah = as_backend_array(resolve_array(a)).ravel()
    bh = as_backend_array(resolve_array(b)).ravel()
    r, p = scipy.stats().spearmanr(ah, bh)
    return float(r), float(p)


def rmse(a: Any, b: Any) -> float:
    """Root mean squared error."""
    ah = _ravel(a)
    bh = _ravel(b)
    return _float(np.sqrt(np.mean((ah - bh) ** 2)))


def mae(a: Any, b: Any) -> float:
    """Mean absolute error."""
    ah = _ravel(a)
    bh = _ravel(b)
    return _float(np.mean(np.abs(ah - bh)))


def correlation_stats(a: Any, b: Any) -> dict[str, float]:
    """
    Return Pearson r/p, Spearman r/p, MAE, RMSE, mean % difference, and per-array summaries.

    Arithmetic (diff / means / std) runs on the active backend; ``scipy.stats``
    is the single host hop.
    """
    ah = _ravel(a)
    bh = _ravel(b)
    if ah.shape != bh.shape:
        raise ValueError(f"Shape mismatch: {ah.shape} vs {bh.shape}")

    ah_np = as_backend_array(ah)
    bh_np = as_backend_array(bh)
    _stats = scipy.stats
    pr, pp = _stats.pearsonr(ah_np, bh_np)
    sr, sp = _stats.spearmanr(ah_np, bh_np)

    diff = bh - ah
    eps = 1e-10
    percent_diff = _float(np.mean(np.abs(diff) / (ah + eps)) * 100.0)

    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_r": float(sr),
        "spearman_p": float(sp),
        "mae": _float(np.mean(np.abs(diff))),
        "rmse": _float(np.sqrt(np.mean(diff ** 2))),
        "percent_diff": percent_diff,
        "mean_a": _float(np.mean(ah)),
        "mean_b": _float(np.mean(bh)),
        "std_a": _float(np.std(ah)),
        "std_b": _float(np.std(bh)),
        "n_samples": int(ah.size),
    }


def sample_at_physical_points(
    image: Image | Any,
    affine: Any | None = None,
    physical_points: Any = None,
    *,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
) -> Any:
    """
    Sample *image* at *physical_points* (world coordinates) using the provided *affine*.

    When *image* is an :class:`Image` and *affine* is None, ``image.affine`` is used.
    *physical_points* is an ``(N, 3)`` array-like of world coordinates.

    Returns an ``(N,)`` NumPy array of sampled values (cubic interpolation by default).

    Notes
    -----
    The 4×4 affine inversion runs on the host (trivial cost) using
    ``numpy.linalg.inv``. The heavy voxel sampling (``ndi.map_coordinates``) uses
    the proxy ``ndi``, so it runs on GPU when the active backend is CuPy.
    """
    if physical_points is None:
        raise ValueError("physical_points is required.")
    if affine is None:
        if isinstance(image, Image):
            affine = image.affine
        if affine is None:
            raise ValueError("affine is required (or pass an Image with an affine in metadata).")

    # Host-side prep: invert a 4x4 and build the (N, 3) query array.
    affine_np = as_backend_array(affine)
    if affine_np.shape != (4, 4):
        raise ValueError(f"Affine must be (4, 4), got {affine_np.shape}.")
    inv_affine = np.linalg.inv(affine_np)

    pts = as_backend_array(physical_points).astype(float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"physical_points must be shape (N, 3), got {pts.shape}.")
    n = pts.shape[0]
    pts_hom = as_backend_array([pts.T, np.ones((1, n))]).astype(float)
    voxel_coords_np = (inv_affine @ pts_hom)[:3, :]

    # Move voxel coordinates to the active backend (small N x 3 transfer).
    voxel_coords = as_backend_array(voxel_coords_np)
    data = as_backend_array(resolve_array(image))

    values = ndi.map_coordinates(
        data, voxel_coords, order=order, mode=mode, cval=cval, prefilter=True
    )
    return as_backend_array(values)


__all__ = [
    "pearson",
    "spearman",
    "rmse",
    "mae",
    "correlation_stats",
    "sample_at_physical_points",
]
