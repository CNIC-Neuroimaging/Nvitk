"""6-connected intensity-gated region growing on 3D volumes."""

from __future__ import annotations

from collections import deque
from typing import Literal

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
    """Intensity gate paired with :func:`_intensity_passes_gate`."""
    frac = float(intensity_frac)
    floor = float(abs_floor) if abs_floor is not None else 0.0
    if polarity == "hypointense":
        if frac <= 0.0:
            return seed_mean
        # Dual of hyperintense ``mean * frac``: admit voxels up to mean / frac.
        ceiling = float(seed_mean) / frac
        if abs_floor is not None:
            ceiling = min(ceiling, floor)
        return max(ceiling, float(seed_mean))
    return max(float(seed_mean) * frac, floor)


def _intensity_passes_gate(
    value: float,
    threshold: float,
    *,
    polarity: IntensityPolarity,
) -> bool:
    if polarity == "hypointense":
        return value <= threshold
    return value >= threshold


def region_grow_binary_mask(
    vessel_mask: np.ndarray,
    intensity: np.ndarray,
    *,
    intensity_frac: float,
    abs_floor: float | None = None,
    forbidden: np.ndarray | None = None,
    polarity: IntensityPolarity = "hyperintense",
) -> int:
    """Grow a boolean mask in-place using 6-connectivity and mean-seed intensity gate.

    *polarity* ``hyperintense`` (default): grow into voxels with
    ``I >= mean(seeds) * intensity_frac``. ``hypointense``: grow into darker voxels with
    ``I <= mean(seeds) / intensity_frac`` (black-blood lumen).
    """
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

    nx, ny, nz = mask.shape
    q: deque[tuple[int, int, int]] = deque(
        (int(i), int(j), int(k)) for i, j, k in seeds
    )
    seen = set(q)
    n_added = 0

    while q:
        i, j, k = q.popleft()
        for di, dj, dk in _NEIGHBOURS:
            ni, nj, nk = i + di, j + dj, k + dk
            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                continue
            if (ni, nj, nk) in seen:
                continue
            seen.add((ni, nj, nk))
            if forb is not None and forb[ni, nj, nk]:
                continue
            if not _intensity_passes_gate(
                float(int_np[ni, nj, nk]), grow_thresh, polarity=polarity
            ):
                continue
            if not mask[ni, nj, nk]:
                mask[ni, nj, nk] = True
                n_added += 1
            q.append((ni, nj, nk))

    return n_added


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
    """6-connected region growing into empty voxels (``labels == 0``) for *label_id*.

    See :func:`region_grow_binary_mask` for *polarity* behaviour.
    """
    seg_np = as_backend_array(labels)
    int_np = as_backend_array(intensity).astype(np.float64)
    forb = None if forbidden is None else as_backend_array(forbidden).astype(bool, copy=False)
    seeds = np.argwhere(seg_np == int(label_id))
    if seeds.size == 0:
        return 0

    seed_vals = int_np[seeds[:, 0], seeds[:, 1], seeds[:, 2]]
    grow_thresh = _grow_intensity_threshold(
        float(np.mean(seed_vals)),
        intensity_frac,
        abs_floor,
        polarity=polarity,
    )

    nx, ny, nz = seg_np.shape
    q: deque[tuple[int, int, int]] = deque(
        (int(i), int(j), int(k)) for i, j, k in seeds
    )
    seen = set(q)
    n_added = 0

    while q:
        i, j, k = q.popleft()
        for di, dj, dk in _NEIGHBOURS:
            ni, nj, nk = i + di, j + dj, k + dk
            if ni < 0 or ni >= nx or nj < 0 or nj >= ny or nk < 0 or nk >= nz:
                continue
            if (ni, nj, nk) in seen:
                continue
            seen.add((ni, nj, nk))
            if seg_np[ni, nj, nk] != 0:
                continue
            if forb is not None and forb[ni, nj, nk]:
                continue
            if not _intensity_passes_gate(
                float(int_np[ni, nj, nk]), grow_thresh, polarity=polarity
            ):
                continue
            seg_np[ni, nj, nk] = int(label_id)
            n_added += 1
            q.append((ni, nj, nk))

    return n_added


__all__ = [
    "IntensityPolarity",
    "region_grow_binary_mask",
    "region_grow_into_label_volume",
]
