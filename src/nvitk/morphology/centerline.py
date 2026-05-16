"""Centerline utilities (skeletonize + polyline ordering).

These helpers are **not** GPU-native: scikit-image's 3D skeletonization is CPU
only. We accept CuPy arrays by converting to NumPy via :func:`nvitk.core.array.to_numpy`.
"""

from __future__ import annotations

from typing import Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.core.exceptions import ValidationError
from nvitk.core.logger import Logger

setup(globals())

log = Logger()


def skeletonize_binary(mask) -> Any:
    """3D/2D skeletonization (CPU skimage; returns backend array of bool)."""
    skeletonize = _require_skeletonize()
    m = to_numpy(mask)
    if m.ndim not in (2, 3):
        raise ValidationError("skeletonize_binary expects a 2D or 3D mask.")
    sk = skeletonize(m.astype(np.uint8, copy=False))
    return as_backend_array(np.asarray(sk, dtype=bool))


def _require_skeletonize():
    try:
        from skimage.morphology import skeletonize  # type: ignore
    except Exception as exc:
        import traceback
        log.exception(traceback.format_exc())
        raise ImportError(
            "Centerline computation requires scikit-image (skeletonize). "
            "Install it with: pip install scikit-image"
        ) from exc
    return skeletonize


def _neighbors26(p: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    x, y, z = p
    out: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                out.append((x + dx, y + dy, z + dz))
    return out


def _centerline_longest_path(coords_xyz: np.ndarray) -> np.ndarray:
    """Order skeleton voxels into an approximate centerline using the graph diameter.

    Input coords are (N,3) voxel indices in (x,y,z) order (as returned by np.argwhere).
    Output is a (M,3) polyline in voxel coordinates (float32).
    """
    if coords_xyz.shape[0] <= 2:
        return coords_xyz.astype(np.float32, copy=False)

    nodes = [tuple(int(v) for v in row) for row in coords_xyz]
    node_set = set(nodes)
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    deg: dict[tuple[int, int, int], int] = {}
    for n in nodes:
        nbrs = [m for m in _neighbors26(n) if m in node_set]
        adj[n] = nbrs
        deg[n] = len(nbrs)

    endpoints = [n for n, d in deg.items() if d <= 1]
    start = endpoints[0] if endpoints else nodes[0]

    def _bfs(
        src: tuple[int, int, int],
    ) -> tuple[
        dict[tuple[int, int, int], int],
        dict[tuple[int, int, int], tuple[int, int, int] | None],
    ]:
        from collections import deque

        dist: dict[tuple[int, int, int], int] = {src: 0}
        parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v in dist:
                    continue
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
        return dist, parent

    d1, _ = _bfs(start)
    a = max(d1, key=d1.get)
    d2, parent = _bfs(a)
    b = max(d2, key=d2.get)

    # Reconstruct path b -> a
    path: list[tuple[int, int, int]] = []
    cur: tuple[int, int, int] | None = b
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur, None)
    path.reverse()
    return np.asarray(path, dtype=np.float32)


def compute_centerlines(
    vessel_mask: np.ndarray,
    *,
    centerline_mask: np.ndarray | None = None,
    labels: Sequence[int] | None = None,
    min_points: int = 20,
) -> dict[int, np.ndarray]:
    """Return per-label ordered centerline points in voxel coordinates.

    Parameters
    ----------
    vessel_mask
        3D integer label mask (NumPy or CuPy).
    centerline_mask
        Optional 3D mask where centerline voxels are >0. If provided, this mask is
        intersected with each `vessel_mask == label` region and ordered.
    labels
        Optional explicit label list. Default: all non-zero labels in `vessel_mask`.
    min_points
        Skip labels whose centerline has fewer than this many voxels.
    """
    arr = as_backend_array(vessel_mask)
    if arr.ndim != 3:
        raise ValidationError("vessel_mask must be 3D for centerline computation.")
    labs = labels or sorted(int(v) for v in np.unique(arr) if int(v) != 0)
    out: dict[int, np.ndarray] = {}

    if centerline_mask is not None:
        cl = as_backend_array(centerline_mask)
        if cl.shape != arr.shape:
            raise ValidationError("centerline_mask must match vessel_mask shape.")
        cl_bin = cl > 0
        for lbl in labs:
            coords = np.argwhere(cl_bin & (arr == int(lbl)))
            if coords.shape[0] < int(min_points):
                continue
            out[int(lbl)] = _centerline_longest_path(coords)
        return out

    skeletonize = _require_skeletonize()
    for lbl in labs:
        roi = (arr == int(lbl))
        if not np.any(roi):
            continue
        sk = skeletonize(to_numpy(roi).astype(np.uint8, copy=False))
        coords = np.argwhere(as_backend_array(sk) > 0)
        if coords.shape[0] < int(min_points):
            continue
        out[int(lbl)] = _centerline_longest_path(coords)
    return out


def centerline_tangents(points_xyz: Any, *, k_half: int = 2) -> Any:
    """Unit tangent vectors along an ordered polyline (voxel coordinates).

    Uses central differences with ``k_half`` samples on each side; endpoints
    use one-sided differences. ``points_xyz`` shape ``(N, 3)``.
    """
    p = as_backend_array(points_xyz).astype(np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValidationError("points_xyz must be (N, 3).")
    n = p.shape[0]
    if n < 2:
        return np.zeros_like(p)

    kh = max(1, int(k_half))
    tang = np.zeros_like(p)
    for i in range(n):
        i0 = max(0, i - kh)
        i1 = min(n - 1, i + kh)
        d = p[i1] - p[i0]
        norm = np.linalg.norm(d)
        tang[i] = d / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return tang.astype(np.float32, copy=False)


__all__ = ["centerline_tangents", "compute_centerlines", "skeletonize_binary"]

