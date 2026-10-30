"""Fill the notches in a vessel's cross-sections without moving its outer wall.

On TOF, the blood in a tight loop runs along the loop's outer wall, and the inner wall can come out
darker than the lumen around it: the mask then misses a crescent there. Along the vessel's
centerline, each cross-section of the mask (the 2-D piece around the centerline point, nothing
else of the plane) is compared with its convex outline; the gaps between the two that are at least
*min_depth_mm* deep are the notches, and their voxels are added. The voxel staircase along a
straight wall leaves slivers thinner than that and is not touched, so the outer wall stays where
it was.

Host code (NumPy, SciPy, scikit-image): the callers pass host arrays.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

#: A gap between a cross-section and its convex outline must be this deep (mm) to be a notch.
MIN_DEPTH_MM = 0.35
#: A cross-section whose convex outline exceeds its own area by this factor is skipped (a branch
#: leaving, or the plane grazing a bend), so a fill never bridges two vessels.
MAX_HULL_RATIO = 1.6


def _plane_basis(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane normal to *t*."""
    u = np.cross(t, [1.0, 0.0, 0.0] if abs(t[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return u, np.cross(t, u)


def fill_cross_section_notches(
    mask: np.ndarray,
    path_mm: np.ndarray,
    spacing: Sequence[float],
    *,
    blocked: np.ndarray | None = None,
    min_depth_mm: float = MIN_DEPTH_MM,
    max_hull_ratio: float = MAX_HULL_RATIO,
    step_mm: float = 0.5,
) -> tuple[np.ndarray, dict[str, int]]:
    """The voxels that fill *mask*'s cross-sectional notches along *path_mm* (module docstring).

    *path_mm*: the centerline stretch to treat, ordered points in mm on the voxel axes (voxel index
    times *spacing*). Each cross-section is sampled every *step_mm*, at half the smallest voxel
    size, over 2.5 lumen radii (at least 2 mm). Voxels in *blocked* (e.g. the other labels) and
    *mask* itself are never returned. Returns the added voxels (bool) and counts (sections, sections
    skipped by *max_hull_ratio*, sections filled)."""
    from skimage.morphology import convex_hull_image

    mask = np.asarray(mask, dtype=bool)
    sp = np.asarray(spacing, dtype=float)
    pts = np.asarray(path_mm, dtype=float)
    added = np.zeros(mask.shape, dtype=bool)
    counts = {"sections": 0, "skipped": 0, "filled": 0}
    if len(pts) < 3 or not mask.any():
        return added, counts
    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    radius = ndi.distance_transform_edt(mask, sampling=sp)
    h = float(sp.min()) / 2.0
    shape = np.asarray(mask.shape)
    for s0 in np.arange(0.0, s[-1], step_mm):
        i = int(np.searchsorted(s, s0))
        a = int(np.searchsorted(s, s0 - 1.5))
        b = min(int(np.searchsorted(s, s0 + 1.5)), len(pts) - 1)
        t = pts[b] - pts[max(a, 0)]
        if np.linalg.norm(t) < 1e-6:
            continue
        u, v = _plane_basis(t / np.linalg.norm(t))
        c = np.round(pts[i] / sp).astype(int)
        lo, hi = np.maximum(c - 3, 0), np.minimum(c + 4, shape)
        r = float(radius[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].max())
        g = np.arange(-max(2.5 * r, 2.0), max(2.5 * r, 2.0) + h / 2, h)
        A, B = np.meshgrid(g, g, indexing="ij")
        idx = np.round((pts[i] + A[..., None] * u + B[..., None] * v) / sp).astype(int)
        ok = np.all((idx >= 0) & (idx < shape), axis=-1)
        idx[~ok] = 0
        inside = mask[idx[..., 0], idx[..., 1], idx[..., 2]] & ok
        cc, n = ndi.label(inside, structure=np.ones((3, 3)))
        if n == 0:
            continue
        mid = len(g) // 2
        own = cc[mid, mid]
        if own == 0:  # the centerline point fell just outside: the nearest piece of the section
            _d, (ii, jj) = ndi.distance_transform_edt(cc == 0, return_indices=True)
            own = cc[ii[mid, mid], jj[mid, mid]]
        piece = cc == own
        hull = convex_hull_image(piece)
        counts["sections"] += 1
        if hull.sum() > max_hull_ratio * piece.sum():
            counts["skipped"] += 1
            continue
        gap = hull & ~piece
        gcc, ng = ndi.label(gap, structure=np.ones((3, 3)))
        if ng == 0:
            continue
        depth = ndi.distance_transform_edt(gap) * h
        deep = np.flatnonzero(np.asarray(ndi.maximum(depth, gcc, index=np.arange(1, ng + 1))) >= min_depth_mm) + 1
        notch = np.isin(gcc, deep) & ~inside & ok
        if blocked is not None:
            notch &= ~np.asarray(blocked, dtype=bool)[idx[..., 0], idx[..., 1], idx[..., 2]]
        if notch.any():
            hit = idx[notch]
            added[hit[:, 0], hit[:, 1], hit[:, 2]] = True
            counts["filled"] += 1
    added &= ~mask
    if blocked is not None:
        added &= ~np.asarray(blocked, dtype=bool)
    return added, counts


__all__ = ["MAX_HULL_RATIO", "MIN_DEPTH_MM", "fill_cross_section_notches"]
