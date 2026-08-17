# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/morphology/mst_bridge.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""MST-based bridging of nearby connected components (binary / multilabel).

Builds a complete graph of components with edge weights = nearest-voxel
distance, takes a minimum spanning tree (MST), and draws short tubes along
MST edges shorter than ``max_gap``. Useful for reconnecting fragmented
vessel masks without a global morphological close.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nvitk_vendor.core.array import to_numpy
from nvitk_vendor.core.backend import using
from nvitk_vendor.core.logger import Logger

log = Logger()

DEFAULT_CLOSE_RADIUS: int = 2
DEFAULT_BRIDGE_MAX_GAP: int = 12
DEFAULT_BRIDGE_RADIUS: int = 1


def draw_tube_3d(
    shape: tuple[int, ...],
    p0: np.ndarray | Sequence[float],
    p1: np.ndarray | Sequence[float],
    *,
    radius: int = 1,
) -> np.ndarray:
    """Binary mask of a thick 3-D line segment from *p0* to *p1*."""
    from skimage.draw import line_nd

    a = np.round(np.asarray(p0, dtype=np.float64)).astype(np.int64)
    b = np.round(np.asarray(p1, dtype=np.float64)).astype(np.int64)
    rr = line_nd(tuple(int(x) for x in a), tuple(int(x) for x in b), endpoint=True)
    mask = np.zeros(shape, dtype=bool)
    pts = np.stack(rr, axis=1)
    keep = np.all((pts >= 0) & (pts < np.asarray(shape)), axis=1)
    pts = pts[keep]
    if pts.size == 0:
        return mask
    mask[pts[:, 0], pts[:, 1], pts[:, 2]] = True
    r = max(0, int(radius))
    if r > 0:
        from scipy import ndimage as ndi

        mask = ndi.binary_dilation(mask, iterations=r)
    return mask


def bridge_binary_components_mst(
    mask: np.ndarray | Any,
    *,
    max_gap: int = DEFAULT_BRIDGE_MAX_GAP,
    tube_radius: int = DEFAULT_BRIDGE_RADIUS,
) -> np.ndarray:
    """Connect nearby CCs of a binary mask with short tubes (MST of min distances).

    Parameters
    ----------
    mask
        Binary volume.
    max_gap
        Maximum nearest-voxel distance (voxels) for an MST edge to be drawn.
    tube_radius
        Dilation radius of each bridge polyline.

    Returns
    -------
    np.ndarray
        Boolean mask with bridges OR-ed into the original foreground.
    """
    from scipy import ndimage as ndi
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial import cKDTree

    m = np.asarray(to_numpy(mask), dtype=bool)
    if not m.any():
        return m
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, n_cc = ndi.label(m, structure=structure)
    if int(n_cc) <= 1:
        return m

    coords = [np.argwhere(labeled == i).astype(np.float64) for i in range(1, int(n_cc) + 1)]
    n = len(coords)
    dist_mat = np.full((n, n), np.inf, dtype=np.float64)
    closest: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for i in range(n):
        tree_i = cKDTree(coords[i])
        for j in range(i + 1, n):
            d, idx = tree_i.query(coords[j], k=1)
            d = np.asarray(d, dtype=np.float64)
            idx = np.asarray(idx, dtype=np.int64)
            j_best = int(np.argmin(d))
            d_min = float(d[j_best])
            dist_mat[i, j] = dist_mat[j, i] = d_min
            p_j = coords[j][j_best]
            p_i = coords[i][int(idx[j_best])]
            closest[(i, j)] = (p_i, p_j)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    max_g = float(max(1, int(max_gap)))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist_mat[i, j])
            if np.isfinite(d):
                rows.extend((i, j))
                cols.extend((j, i))
                data.extend((d, d))
    if not data:
        return m

    graph = csr_matrix((data, (rows, cols)), shape=(n, n))
    mst = minimum_spanning_tree(graph)
    coo = mst.tocoo()

    out = m.copy()
    n_bridges = 0
    for i, j, d in zip(coo.row, coo.col, coo.data):
        if i >= j:
            continue
        if float(d) > max_g or float(d) <= 0:
            continue
        key = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
        p_a, p_b = closest[key]
        tube = draw_tube_3d(out.shape, p_a, p_b, radius=int(tube_radius))
        out |= tube
        n_bridges += 1
    if n_bridges:
        log.info(
            f"MST CC bridge: {n_bridges} tube(s), "
            f"max_gap={int(max_gap)}, tube_r={int(tube_radius)}"
        )
    return out


def fill_multilabel_gaps_mst(
    labels: np.ndarray | Any,
    *,
    close_radius: int = DEFAULT_CLOSE_RADIUS,
    bridge_max_gap: int = DEFAULT_BRIDGE_MAX_GAP,
    bridge_radius: int = DEFAULT_BRIDGE_RADIUS,
    label_ids: Sequence[int] | None = None,
) -> np.ndarray:
    """Per-label close + MST CC bridging; never overwrite other label ids.

    Parameters
    ----------
    labels
        Integer label volume (``0`` = background).
    close_radius
        Morphological close footprint radius before bridging (``0`` disables).
    bridge_max_gap, bridge_radius
        Passed to :func:`bridge_binary_components_mst`.
    label_ids
        Labels to process; default = all positive ids present in *labels*.
    """
    from nvitk_vendor.morphology.binary import close

    lab = np.asarray(to_numpy(labels), dtype=np.int32)
    out = lab.copy()
    if label_ids is None:
        ids = sorted(int(v) for v in np.unique(lab) if int(v) != 0)
    else:
        ids = [int(v) for v in label_ids]
    r_close = max(0, int(close_radius))
    filled_total = 0
    with using("numpy"):
        for tid in ids:
            mask = lab == int(tid)
            if not np.any(mask):
                continue
            working = mask.copy()
            if r_close > 0:
                closed = close(
                    working.astype(np.uint8),
                    footprint=r_close,
                    mode="binary",
                    connectivity=3,
                )
                working = to_numpy(closed).astype(bool, copy=False)
            working = bridge_binary_components_mst(
                working,
                max_gap=int(bridge_max_gap),
                tube_radius=int(bridge_radius),
            )
            fill = working & (out == 0)
            n_fill = int(np.count_nonzero(fill))
            if n_fill:
                out[fill] = int(tid)
                filled_total += n_fill
            out[mask] = int(tid)
    log.info(
        f"MST multilabel gap fill: close_r={r_close}, "
        f"bridge_max={int(bridge_max_gap)}, filled_voxels={filled_total}"
    )
    return out


__all__ = [
    "DEFAULT_BRIDGE_MAX_GAP",
    "DEFAULT_BRIDGE_RADIUS",
    "DEFAULT_CLOSE_RADIUS",
    "bridge_binary_components_mst",
    "draw_tube_3d",
    "fill_multilabel_gaps_mst",
]
