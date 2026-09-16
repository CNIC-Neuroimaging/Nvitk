"""Wrappers over ``skimage.filters``, grouped by family.

scikit-image exposes its filters as several dozen loose functions — five edge
operators each with directional variants, seven global threshold rules, three
local ones. Exposing one tool per function would bury the choice that matters
(which *family*) under the one that rarely does (which variant), so each family
is one function here with a ``method`` argument.

Host-only, like every scikit-image wrapper in this package: no ``setup(globals())``,
real NumPy, ``to_numpy`` at the boundary, and callers run them under
``using("numpy")``. Functions that scikit-image only defines in 2D are applied
plane by plane along a chosen axis rather than refused, since a 3D volume is the
normal input here.

Deliberately not wrapped: ``LPIFilter2D``, ``filter_forward``, ``filter_inverse``
and ``wiener`` — they need a caller-supplied impulse response rather than an
image alone; ``gabor_kernel``, ``window`` and ``rank_order``, which build kernels
or orderings rather than filtering; and ``try_all_threshold``, which returns a
Matplotlib figure.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Callable

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.filters.hessian import parse_sigmas
from nvitk.types import Image

#: Ridge scales used when none are given, matching the Hessian filter's.
RIDGE_SIGMAS_DEFAULT: tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0)

#: Families, with the variants each offers. The GUI reads these so the tool's
#: choices and the implementation cannot drift apart.
RIDGE_METHODS: tuple[str, ...] = ("frangi", "sato", "meijering")
EDGE_METHODS: tuple[str, ...] = ("sobel", "scharr", "prewitt", "roberts", "farid")
EDGE_DIRECTIONS: tuple[str, ...] = ("magnitude", "horizontal", "vertical")
BLUR_METHODS: tuple[str, ...] = ("gaussian", "difference_of_gaussians")
GLOBAL_THRESHOLDS: tuple[str, ...] = (
    "otsu", "li", "yen", "isodata", "mean", "minimum", "triangle",
)
LOCAL_THRESHOLDS: tuple[str, ...] = ("local", "niblack", "sauvola")
#: Rank filters that take a plain ``footprint``. Read off the module rather than
#: listed here: the set moves between scikit-image releases, and the ``*_percentile``
#: and ``*_bilateral`` variants need extra arguments this wrapper does not expose.
def _rank_methods() -> tuple[str, ...]:
    """Every ``skimage.filters.rank`` filter this wrapper can drive."""
    try:
        from skimage.filters import rank
    except Exception:  # noqa: BLE001 — the constant is advisory until first use
        return ("mean", "median", "maximum", "minimum")
    skip_suffix = ("_percentile", "_bilateral")
    # windowed_histogram returns a histogram per pixel, so its output has an extra
    # axis and is not an image — the same reason the kernel builders are left out.
    skip_exact = {"windowed_histogram"}
    return tuple(
        name
        for name in sorted(dir(rank))
        if not name.startswith("_")
        and callable(getattr(rank, name, None))
        and not name.endswith(skip_suffix)
        and name not in skip_exact
    )


RANK_METHODS: tuple[str, ...] = _rank_methods()

#: Edge families scikit-image defines only for 2D input. The *directional*
#: variants (``sobel_h`` and friends) are 2D-only for every family, including
#: those whose magnitude form is N-dimensional — see :func:`_is_planar_edge`.
_PLANAR_ONLY: frozenset[str] = frozenset({"roberts", "farid"})


def _is_planar_edge(method: str, direction: str) -> bool:
    """Whether this edge request has to be run plane by plane."""
    return str(method).lower() in _PLANAR_ONLY or str(direction).lower() != "magnitude"


def _unwrap(image: Image | np.ndarray) -> tuple[np.ndarray, Image | None]:
    """``(host array, the Image to rewrap with)`` — ``None`` when given a raw array."""
    if isinstance(image, Image):
        return np.asarray(to_numpy(image.data)), image
    return np.asarray(to_numpy(image)), None


def _rewrap(out: np.ndarray, source: Image | None) -> Image | np.ndarray:
    """Return *out* as an ``Image`` when the input was one, else as the array."""
    return source.with_data(out) if source is not None else out


def _slicewise(func: Callable[[np.ndarray], np.ndarray], arr: np.ndarray, axis: int) -> np.ndarray:
    """Apply a 2D-only *func* to every plane of *arr* along *axis*."""
    if arr.ndim == 2:
        return np.asarray(func(arr), dtype=np.float32)
    ax = int(np.clip(axis, 0, arr.ndim - 1))
    planes = [func(np.take(arr, i, axis=ax)) for i in range(arr.shape[ax])]
    return np.stack(planes, axis=ax).astype(np.float32, copy=False)


def _float(arr: np.ndarray) -> np.ndarray:
    """*arr* as float32 with non-finite values zeroed, which skimage requires."""
    return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ── ridge / vesselness ────────────────────────────────────────────────────────
def ridge_filter(
    image: Image | np.ndarray,
    *,
    method: str = "frangi",
    sigmas: Iterable[float] | Sequence[float] | None = None,
    black_ridges: bool = False,
) -> Image | np.ndarray:
    """Frangi, Sato or Meijering ridge enhancement.

    ``black_ridges`` selects dark structures on a bright background; contrast-
    filled vessels are the bright case, so it defaults off here.
    """
    from skimage import filters as skf

    key = str(method).lower()
    if key not in RIDGE_METHODS:
        raise ValueError(f"Unknown ridge method {method!r}; expected one of {RIDGE_METHODS}.")
    arr, source = _unwrap(image)
    scales = tuple(float(s) for s in (sigmas if sigmas is not None else RIDGE_SIGMAS_DEFAULT))
    out = getattr(skf, key)(_float(arr), sigmas=scales, black_ridges=bool(black_ridges))
    return _rewrap(np.asarray(out, dtype=np.float32), source)


# ── edges ─────────────────────────────────────────────────────────────────────
def edge_filter(
    image: Image | np.ndarray,
    *,
    method: str = "sobel",
    direction: str = "magnitude",
    axis: int = 0,
) -> Image | np.ndarray:
    """Sobel, Scharr, Prewitt, Roberts or Farid edges.

    *direction* picks the gradient magnitude or one of the directional variants;
    Roberts' diagonals are its two directional forms.
    """
    from skimage import filters as skf

    key = str(method).lower()
    if key not in EDGE_METHODS:
        raise ValueError(f"Unknown edge method {method!r}; expected one of {EDGE_METHODS}.")
    want = str(direction).lower()
    if want not in EDGE_DIRECTIONS:
        raise ValueError(f"Unknown direction {direction!r}; expected one of {EDGE_DIRECTIONS}.")

    if key == "roberts":
        names = {"magnitude": "roberts",
                 "horizontal": "roberts_pos_diag",
                 "vertical": "roberts_neg_diag"}
    else:
        suffix = {"magnitude": "", "horizontal": "_h", "vertical": "_v"}[want]
        names = {want: f"{key}{suffix}"}
    func = getattr(skf, names[want] if want in names else names["magnitude"])

    arr, source = _unwrap(image)
    data = _float(arr)
    if _is_planar_edge(key, want) or data.ndim > 3:
        out = _slicewise(func, data, axis)
    else:
        out = np.asarray(func(data), dtype=np.float32)
    return _rewrap(np.asarray(out, dtype=np.float32), source)


def laplace_filter(image: Image | np.ndarray, *, ksize: int = 3) -> Image | np.ndarray:
    """Laplacian of the image — a second-derivative edge/blob response.

    The kernel size is forced odd: an even one has no centre voxel to put the
    operator on, and scikit-image fails on the shape mismatch rather than saying so.
    """
    from skimage import filters as skf

    arr, source = _unwrap(image)
    out = skf.laplace(_float(arr), ksize=max(int(ksize), 3) | 1)
    return _rewrap(np.asarray(out, dtype=np.float32), source)


# ── smoothing and sharpening ──────────────────────────────────────────────────
def blur_filter(
    image: Image | np.ndarray,
    *,
    method: str = "gaussian",
    sigma: float = 1.0,
    sigma_high: float = 2.0,
) -> Image | np.ndarray:
    """Gaussian blur, or a difference of Gaussians band-pass."""
    from skimage import filters as skf

    key = str(method).lower()
    arr, source = _unwrap(image)
    data = _float(arr)
    if key == "difference_of_gaussians":
        low, high = float(sigma), float(sigma_high)
        if high <= low:
            raise ValueError(
                f"The high sigma ({high}) must exceed the low one ({low}) for a "
                "difference of Gaussians."
            )
        out = skf.difference_of_gaussians(data, low, high)
    elif key == "gaussian":
        out = skf.gaussian(data, sigma=float(sigma))
    else:
        raise ValueError(f"Unknown blur method {method!r}; expected one of {BLUR_METHODS}.")
    return _rewrap(np.asarray(out, dtype=np.float32), source)


def median_filter(image: Image | np.ndarray, *, radius: int = 1) -> Image | np.ndarray:
    """Median filter over a cube of the given radius."""
    from skimage import filters as skf

    arr, source = _unwrap(image)
    size = max(int(radius), 1) * 2 + 1
    out = skf.median(arr, footprint=np.ones((size,) * arr.ndim, dtype=bool))
    return _rewrap(np.asarray(out, dtype=arr.dtype), source)


def unsharp_filter(
    image: Image | np.ndarray, *, radius: float = 1.0, amount: float = 1.0
) -> Image | np.ndarray:
    """Unsharp masking: add back a scaled high-frequency residual."""
    from skimage import filters as skf

    arr, source = _unwrap(image)
    out = skf.unsharp_mask(_float(arr), radius=float(radius), amount=float(amount))
    return _rewrap(np.asarray(out, dtype=np.float32), source)


def butterworth_filter(
    image: Image | np.ndarray,
    *,
    cutoff: float = 0.1,
    order: int = 2,
    high_pass: bool = True,
) -> Image | np.ndarray:
    """Butterworth frequency-domain filter; *cutoff* is a fraction of Nyquist."""
    from skimage import filters as skf

    arr, source = _unwrap(image)
    out = skf.butterworth(
        _float(arr),
        cutoff_frequency_ratio=float(np.clip(cutoff, 1e-4, 0.499)),
        order=max(int(order), 1),
        high_pass=bool(high_pass),
    )
    return _rewrap(np.asarray(out, dtype=np.float32), source)


def gabor_filter(
    image: Image | np.ndarray,
    *,
    frequency: float = 0.2,
    theta: float = 0.0,
    axis: int = 0,
) -> Image | np.ndarray:
    """Gabor magnitude response at one frequency and orientation.

    2D only in scikit-image, so a volume is filtered plane by plane. The two
    quadrature responses are combined into a magnitude, which is what makes the
    result independent of where the texture happens to sit in the wave's phase.
    """
    from skimage import filters as skf

    arr, source = _unwrap(image)

    def _one(plane: np.ndarray) -> np.ndarray:
        """Gabor magnitude of a single plane."""
        real, imag = skf.gabor(
            plane, frequency=float(frequency), theta=float(np.deg2rad(theta))
        )
        return np.hypot(real, imag)

    out = _slicewise(_one, _float(arr), axis)
    return _rewrap(np.asarray(out, dtype=np.float32), source)


def rank_filter(
    image: Image | np.ndarray, *, method: str = "mean", radius: int = 2, axis: int = 0
) -> Image | np.ndarray:
    """A local-rank filter from ``skimage.filters.rank``.

    These are defined on integer images over a footprint, so the input is scaled
    into ``uint8`` first and a volume is filtered plane by plane.
    """
    from skimage.filters import rank

    key = str(method).lower()
    if key not in RANK_METHODS:
        raise ValueError(f"Unknown rank method {method!r}; expected one of {RANK_METHODS}.")
    func = getattr(rank, key, None)
    if func is None:
        raise ValueError(f"scikit-image has no rank filter {method!r}.")

    arr, source = _unwrap(image)
    data = _float(arr)
    lo, hi = float(data.min()), float(data.max())
    scaled = (
        np.zeros(data.shape, dtype=np.uint8)
        if hi <= lo
        else ((data - lo) / (hi - lo) * 255.0).astype(np.uint8)
    )
    size = max(int(radius), 1) * 2 + 1
    footprint = np.ones((size, size), dtype=np.uint8)
    out = _slicewise(lambda plane: func(plane, footprint=footprint), scaled, axis)
    return _rewrap(np.asarray(out, dtype=np.float32), source)


# ── thresholds ────────────────────────────────────────────────────────────────
def global_threshold_value(image: Image | np.ndarray, *, method: str = "otsu") -> float:
    """The intensity a global threshold rule picks for *image*."""
    from skimage import filters as skf

    key = str(method).lower()
    if key not in GLOBAL_THRESHOLDS:
        raise ValueError(
            f"Unknown threshold method {method!r}; expected one of {GLOBAL_THRESHOLDS}."
        )
    arr, _source = _unwrap(image)
    return float(getattr(skf, f"threshold_{key}")(_float(arr)))


def global_threshold(image: Image | np.ndarray, *, method: str = "otsu") -> Image | np.ndarray:
    """Binarise with a global rule; returns a ``uint8`` mask."""
    arr, source = _unwrap(image)
    value = global_threshold_value(image, method=method)
    return _rewrap((_float(arr) >= value).astype(np.uint8), source)


def multiotsu_threshold(
    image: Image | np.ndarray, *, classes: int = 3
) -> Image | np.ndarray:
    """Split the intensities into *classes* bands; returns the band index per voxel."""
    from skimage import filters as skf

    arr, source = _unwrap(image)
    n = max(int(classes), 2)
    edges = skf.threshold_multiotsu(_float(arr), classes=n)
    return _rewrap(np.digitize(_float(arr), bins=edges).astype(np.uint8), source)


def local_threshold(
    image: Image | np.ndarray,
    *,
    method: str = "local",
    block_size: int = 35,
    offset: float = 0.0,
    axis: int = 0,
) -> Image | np.ndarray:
    """Binarise against a locally-computed threshold surface.

    ``local`` is a Gaussian-weighted neighbourhood mean; Niblack and Sauvola add
    a local standard-deviation term, which is what makes them hold up on
    unevenly-lit images where one global cut cannot.
    """
    from skimage import filters as skf

    key = str(method).lower()
    if key not in LOCAL_THRESHOLDS:
        raise ValueError(
            f"Unknown local method {method!r}; expected one of {LOCAL_THRESHOLDS}."
        )
    arr, source = _unwrap(image)
    data = _float(arr)
    # Odd window: skimage rejects an even block size outright.
    size = max(int(block_size), 3) | 1

    def _one(plane: np.ndarray) -> np.ndarray:
        """The threshold surface for a single plane."""
        if key == "niblack":
            return skf.threshold_niblack(plane, window_size=size)
        if key == "sauvola":
            return skf.threshold_sauvola(plane, window_size=size)
        return skf.threshold_local(plane, block_size=size, offset=float(offset))

    surface = _slicewise(_one, data, axis)
    mask = data >= (surface + (float(offset) if key != "local" else 0.0))
    return _rewrap(mask.astype(np.uint8), source)


def hysteresis_threshold(
    image: Image | np.ndarray, *, low: float = 0.0, high: float = 0.0
) -> Image | np.ndarray:
    """Keep everything above *low* that connects to something above *high*.

    Both zero means "pick them for me": the Otsu value becomes the high cut and
    half of it the low one, which is the usual starting point for a vessel tree.
    """
    from skimage import filters as skf

    arr, source = _unwrap(image)
    data = _float(arr)
    lo, hi = float(low), float(high)
    if lo == 0.0 and hi == 0.0:
        hi = global_threshold_value(image, method="otsu")
        lo = hi * 0.5
    if hi < lo:
        lo, hi = hi, lo
    out = skf.apply_hysteresis_threshold(data, lo, hi)
    return _rewrap(np.asarray(out, dtype=np.uint8), source)


__all__ = [
    "BLUR_METHODS",
    "EDGE_DIRECTIONS",
    "EDGE_METHODS",
    "GLOBAL_THRESHOLDS",
    "LOCAL_THRESHOLDS",
    "RANK_METHODS",
    "RIDGE_METHODS",
    "RIDGE_SIGMAS_DEFAULT",
    "blur_filter",
    "butterworth_filter",
    "edge_filter",
    "gabor_filter",
    "global_threshold",
    "global_threshold_value",
    "hysteresis_threshold",
    "laplace_filter",
    "local_threshold",
    "median_filter",
    "multiotsu_threshold",
    "parse_sigmas",
    "rank_filter",
    "ridge_filter",
    "unsharp_filter",
]
