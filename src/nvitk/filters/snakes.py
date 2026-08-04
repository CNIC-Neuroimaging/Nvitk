"""Kass snakes (parametric active contours).

Wraps :func:`skimage.segmentation.active_contour`, the discrete
implementation of Kass, Witkin & Terzopoulos (IJCV 1988). An initial
contour (from a binary / label mask) is deformed under internal
smoothness forces and image line/edge energies.

Reference
---------
M. Kass, A. Witkin, D. Terzopoulos,
"Snakes: Active Contour Models",
International Journal of Computer Vision, 1(4):321–331, 1988.
"""

from __future__ import annotations

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.types import Image

SNAKES_ALPHA_DEFAULT: float = 0.01
SNAKES_BETA_DEFAULT: float = 0.1
SNAKES_W_LINE_DEFAULT: float = 0.0
SNAKES_W_EDGE_DEFAULT: float = 1.0
SNAKES_GAMMA_DEFAULT: float = 0.01
SNAKES_MAX_ITER_DEFAULT: int = 2500
SNAKES_CONVERGENCE_DEFAULT: float = 0.1
SNAKES_SIGMA_DEFAULT: float = 1.0
SNAKES_N_POINTS_DEFAULT: int = 400


def _as_numpy_image(image: Image | np.ndarray) -> tuple[np.ndarray, Image | None]:
    """Host NumPy view of *image* plus the original :class:`Image` (or ``None`` for a raw array).

    scikit-image active contours are CPU-only, so we always move to host here.
    """
    if isinstance(image, Image):
        return np.asarray(to_numpy(image.data)), image
    return np.asarray(to_numpy(image)), None


def _mask_array(mask: Image | np.ndarray | None) -> np.ndarray | None:
    """Host NumPy view of an optional mask (``None`` passes through)."""
    if mask is None:
        return None
    if isinstance(mask, Image):
        return np.asarray(to_numpy(mask.data))
    return np.asarray(to_numpy(mask))


def resample_snake(snake: np.ndarray, n_points: int) -> np.ndarray:
    """Arc-length resample a closed contour to *n_points* (row, col)."""
    pts = np.asarray(snake, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"snake must be (K, 2), got {pts.shape}")
    if len(pts) < 3:
        raise ValueError("snake needs at least 3 points")
    n = int(n_points)
    if n < 3:
        raise ValueError("n_points must be >= 3")

    # Close for arc-length, then drop the duplicate endpoint.
    closed = np.vstack([pts, pts[0]])
    seg = np.sqrt(((closed[1:] - closed[:-1]) ** 2).sum(axis=1))
    total = float(seg.sum())
    if total <= 0:
        raise ValueError("snake has zero length")
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2), dtype=np.float64)
    for d in range(2):
        out[:, d] = np.interp(samples, cum, closed[:, d])
    return out


def snake_from_mask(
    mask: np.ndarray,
    *,
    n_points: int = SNAKES_N_POINTS_DEFAULT,
    level: float | None = None,
) -> np.ndarray:
    """Largest iso-contour of a 2-D binary / label mask as a snake (row, col)."""
    from skimage import measure

    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"snake_from_mask expects 2-D mask, got {m.ndim}D")
    binary = m.astype(bool)
    if not binary.any():
        raise ValueError("init mask is empty")

    thr = 0.5 if level is None else float(level)
    contours = measure.find_contours(binary.astype(np.float64), thr)
    if not contours:
        raise ValueError("no contour found in init mask")
    snake = max(contours, key=len)
    return resample_snake(snake, n_points)


def mask_from_snake(snake: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a closed snake into a boolean mask of *shape* (H, W)."""
    from skimage.draw import polygon2mask

    pts = np.asarray(snake, dtype=np.float64)
    return polygon2mask(shape, pts)


def active_contour_2d(
    image: np.ndarray,
    snake: np.ndarray,
    *,
    alpha: float = SNAKES_ALPHA_DEFAULT,
    beta: float = SNAKES_BETA_DEFAULT,
    w_line: float = SNAKES_W_LINE_DEFAULT,
    w_edge: float = SNAKES_W_EDGE_DEFAULT,
    gamma: float = SNAKES_GAMMA_DEFAULT,
    max_num_iter: int = SNAKES_MAX_ITER_DEFAULT,
    convergence: float = SNAKES_CONVERGENCE_DEFAULT,
    boundary_condition: str = "periodic",
    gaussian_sigma: float = SNAKES_SIGMA_DEFAULT,
) -> np.ndarray:
    """Run Kass active contour on a single 2-D image; return refined snake."""
    from skimage.filters import gaussian
    from skimage.segmentation import active_contour

    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        pass  # multichannel OK for skimage
    elif img.ndim != 2:
        raise ValueError(f"active_contour_2d expects 2-D (or MxNx3) image, got {img.ndim}D")

    if float(gaussian_sigma) > 0:
        img = gaussian(img, sigma=float(gaussian_sigma), preserve_range=True)

    return active_contour(
        img,
        np.asarray(snake, dtype=np.float64),
        alpha=float(alpha),
        beta=float(beta),
        w_line=float(w_line),
        w_edge=float(w_edge),
        gamma=float(gamma),
        max_num_iter=int(max_num_iter),
        convergence=float(convergence),
        boundary_condition=str(boundary_condition or "periodic"),
    )


def _snakes_slice(
    image2d: np.ndarray,
    mask2d: np.ndarray,
    *,
    n_points: int,
    kwargs: dict,
) -> np.ndarray:
    """Evolve one 2-D slice: seed a snake from *mask2d*, run active contours, re-rasterize to a mask."""
    snake0 = snake_from_mask(mask2d, n_points=n_points)
    snake1 = active_contour_2d(image2d, snake0, **kwargs)
    return mask_from_snake(snake1, image2d.shape[:2]).astype(np.uint8)


def snakes_filter(
    image: Image | np.ndarray,
    init_mask: Image | np.ndarray,
    *,
    alpha: float = SNAKES_ALPHA_DEFAULT,
    beta: float = SNAKES_BETA_DEFAULT,
    w_line: float = SNAKES_W_LINE_DEFAULT,
    w_edge: float = SNAKES_W_EDGE_DEFAULT,
    gamma: float = SNAKES_GAMMA_DEFAULT,
    max_num_iter: int = SNAKES_MAX_ITER_DEFAULT,
    convergence: float = SNAKES_CONVERGENCE_DEFAULT,
    boundary_condition: str = "periodic",
    gaussian_sigma: float = SNAKES_SIGMA_DEFAULT,
    n_points: int = SNAKES_N_POINTS_DEFAULT,
    axis: int = 0,
) -> Image | np.ndarray:
    """Kass snakes from an initial mask contour; returns a binary mask.

    * 2-D: deform the largest contour of *init_mask* on *image*.
    * 3-D: apply 2-D snakes independently on each slice along *axis*
      where the init mask is nonempty (Kass snakes are 2-D).

    Parameters
    ----------
    image
        Intensity image / volume guiding the external energy.
    init_mask
        Binary or label mask whose contour seeds the snake (required).
    alpha
        Tension (length) weight — higher contracts faster.
    beta
        Rigidity (smoothness) weight — higher → smoother snake.
    w_line
        Attraction to intensity (negative → dark regions).
    w_edge
        Attraction to edges (negative → repel from edges).
    gamma
        Explicit time-step size.
    max_num_iter, convergence
        Optimization stopping criteria.
    boundary_condition
        ``periodic`` (closed), ``free``, ``fixed``, or mixed
        (``free-fixed`` / ``fixed-free``).
    gaussian_sigma
        Pre-smoothing of the image energy (0 disables).
    n_points
        Number of snake control points (resampled from init contour).
    axis
        Slice axis for 3-D volumes (ignored for 2-D).
    """
    arr, wrap = _as_numpy_image(image)
    mask = _mask_array(init_mask)
    if mask is None:
        raise ValueError("init_mask is required for snakes_filter")
    if mask.shape != arr.shape[: mask.ndim]:
        raise ValueError(
            f"init_mask shape {mask.shape} does not match image shape {arr.shape}"
        )

    kw = dict(
        alpha=float(alpha),
        beta=float(beta),
        w_line=float(w_line),
        w_edge=float(w_edge),
        gamma=float(gamma),
        max_num_iter=int(max_num_iter),
        convergence=float(convergence),
        boundary_condition=str(boundary_condition or "periodic"),
        gaussian_sigma=float(gaussian_sigma),
    )
    n_pts = int(n_points)

    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] in (3, 4)):
        out = _snakes_slice(arr, mask, n_points=n_pts, kwargs=kw)
    elif arr.ndim == 3:
        ax = int(axis) % 3
        out = np.zeros(arr.shape, dtype=np.uint8)
        n_slices = arr.shape[ax]
        for i in range(n_slices):
            sl = [slice(None)] * 3
            sl[ax] = i
            m2 = mask[tuple(sl)]
            if not np.any(m2):
                continue
            out[tuple(sl)] = _snakes_slice(arr[tuple(sl)], m2, n_points=n_pts, kwargs=kw)
    else:
        raise ValueError(f"snakes_filter expects 2-D or 3-D data, got {arr.ndim}D")

    if wrap is not None:
        return wrap.with_data(out)
    return out


__all__ = [
    "SNAKES_ALPHA_DEFAULT",
    "SNAKES_BETA_DEFAULT",
    "SNAKES_W_LINE_DEFAULT",
    "SNAKES_W_EDGE_DEFAULT",
    "SNAKES_GAMMA_DEFAULT",
    "SNAKES_MAX_ITER_DEFAULT",
    "SNAKES_CONVERGENCE_DEFAULT",
    "SNAKES_SIGMA_DEFAULT",
    "SNAKES_N_POINTS_DEFAULT",
    "active_contour_2d",
    "mask_from_snake",
    "resample_snake",
    "snake_from_mask",
    "snakes_filter",
]
