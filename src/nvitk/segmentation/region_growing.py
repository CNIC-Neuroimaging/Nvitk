"""6-connected intensity-gated region growing on 3D volumes."""

from __future__ import annotations

from collections import deque
from typing import Callable, Literal

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup

setup(globals())

IntensityPolarity = Literal["hyperintense", "hypointense"]

_NEIGHBOURS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


def _grow_intensity_threshold(
    seed_mean: float,
    intensity_frac: float,
    abs_floor: float | None,
    *,
    polarity: IntensityPolarity,
) -> float:
    """Intensity gate paired with :func:`_intensity_passes_gate`.

    Hyperintense (TOF / CD, default): ``I >= max(mean * frac, abs_floor)``.

    Hypointense (black-blood): ``I <= min(mean * frac, abs_floor)`` when *abs_floor*
    is set as a ceiling cap; otherwise ``I <= mean * frac``. Lower *intensity_frac*
    yields a stricter (darker-only) gate, symmetric to hyperintense semantics.
    """
    frac = float(intensity_frac)
    bound = float(abs_floor) if abs_floor is not None else 0.0
    mean = float(seed_mean)
    if polarity == "hypointense":
        if frac <= 0.0:
            return mean
        ceiling = mean * frac
        if abs_floor is not None:
            ceiling = min(ceiling, bound)
        return ceiling
    return max(mean * frac, bound)


def _intensity_passes_gate(
    value: float,
    threshold: float,
    *,
    polarity: IntensityPolarity,
) -> bool:
    if polarity == "hypointense":
        return value <= threshold
    return value >= threshold


def _bfs_intensity_grow(
    shape: tuple[int, int, int],
    intensity: np.ndarray,
    grow_thresh: float,
    *,
    polarity: IntensityPolarity,
    seed_coords: np.ndarray,
    forbidden: np.ndarray | None,
    can_grow: Callable[[int, int, int], bool],
    claim: Callable[[int, int, int], bool],
) -> int:
    """6-connected BFS with a boolean *visited* volume."""
    nx, ny, nz = shape
    visited = np.zeros((nx, ny, nz), dtype=bool)
    q: deque[tuple[int, int, int]] = deque()
    for i, j, k in seed_coords:
        ii, jj, kk = int(i), int(j), int(k)
        if not visited[ii, jj, kk]:
            visited[ii, jj, kk] = True
            q.append((ii, jj, kk))

    n_added = 0
    while q:
        i, j, k = q.popleft()
        for di, dj, dk in _NEIGHBOURS:
            ni, nj, nk = i + di, j + dj, k + dk
            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                continue
            if visited[ni, nj, nk]:
                continue
            visited[ni, nj, nk] = True
            if forbidden is not None and forbidden[ni, nj, nk]:
                continue
            if not _intensity_passes_gate(
                float(intensity[ni, nj, nk]), grow_thresh, polarity=polarity
            ):
                continue
            if not can_grow(ni, nj, nk):
                continue
            if claim(ni, nj, nk):
                n_added += 1
            q.append((ni, nj, nk))

    return n_added


def region_grow_binary_mask(
    vessel_mask: np.ndarray,
    intensity: np.ndarray,
    *,
    intensity_frac: float,
    abs_floor: float | None = None,
    forbidden: np.ndarray | None = None,
    polarity: IntensityPolarity = "hyperintense",
) -> int:
    """Grow a boolean mask in-place using 6-connectivity and mean-seed intensity gate."""
    mask = np.asarray(vessel_mask, dtype=bool)
    int_np = as_backend_array(intensity).astype(np.float64)
    forb = None if forbidden is None else np.asarray(forbidden, dtype=bool)
    seeds = np.argwhere(mask)
    if seeds.size == 0:
        return 0

    seed_vals = int_np[seeds[:, 0], seeds[:, 1], seeds[:, 2]]
    grow_thresh = _grow_intensity_threshold(
        float(np.mean(seed_vals)),
        intensity_frac,
        abs_floor,
        polarity=polarity,
    )

    def can_grow(ni: int, nj: int, nk: int) -> bool:
        return True

    def claim(ni: int, nj: int, nk: int) -> bool:
        if mask[ni, nj, nk]:
            return False
        mask[ni, nj, nk] = True
        return True

    return _bfs_intensity_grow(
        tuple(int(s) for s in mask.shape[:3]),
        int_np,
        grow_thresh,
        polarity=polarity,
        seed_coords=seeds,
        forbidden=forb,
        can_grow=can_grow,
        claim=claim,
    )


def region_grow_into_label_volume(
    labels: np.ndarray,
    intensity: np.ndarray,
    label_id: int,
    *,
    intensity_frac: float,
    abs_floor: float | None = None,
    forbidden: np.ndarray | None = None,
    polarity: IntensityPolarity = "hyperintense",
) -> int:
    """6-connected region growing into empty voxels (``labels == 0``) for *label_id*."""
    seg_np = as_backend_array(labels)
    int_np = as_backend_array(intensity).astype(np.float64)
    forb = None if forbidden is None else as_backend_array(forbidden).astype(bool, copy=False)
    lid = int(label_id)
    seeds = np.argwhere(seg_np == lid)
    if seeds.size == 0:
        return 0

    seed_vals = int_np[seeds[:, 0], seeds[:, 1], seeds[:, 2]]
    grow_thresh = _grow_intensity_threshold(
        float(np.mean(seed_vals)),
        intensity_frac,
        abs_floor,
        polarity=polarity,
    )

    def can_grow(ni: int, nj: int, nk: int) -> bool:
        return int(seg_np[ni, nj, nk]) == 0

    def claim(ni: int, nj: int, nk: int) -> bool:
        seg_np[ni, nj, nk] = lid
        return True

    return _bfs_intensity_grow(
        tuple(int(s) for s in seg_np.shape[:3]),
        int_np,
        grow_thresh,
        polarity=polarity,
        seed_coords=seeds,
        forbidden=forb,
        can_grow=can_grow,
        claim=claim,
    )


__all__ = [
    "IntensityPolarity",
    "region_grow_binary_mask",
    "region_grow_into_label_volume",
]
