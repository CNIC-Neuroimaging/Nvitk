"""Split LVA/RVA from basilar segmentation via vertebro-basilar skeleton bifurcation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.morphology.centerline import compute_centerlines, skeletonize_binary
from nvitk.pipes.qvtpy.labels import QVTPY_BASILAR, QVTPY_LVA, QVTPY_RVA

setup(globals())


@dataclass(frozen=True)
class VertebralSplitResult:
    """Outcome of basilar → LVA/RVA relabeling."""

    split_applied: bool
    bifurcation_ijk: tuple[int, int, int] | None
    junction_degree: int
    lva_voxels: int
    rva_voxels: int
    basilar_voxels: int
    hemisphere_axis: str
    bifurcation_cut_k: int | None = None
    lva_centerline: np.ndarray | None = None
    rva_centerline: np.ndarray | None = None
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "split_applied": bool(self.split_applied),
            "bifurcation_ijk": (
                [int(self.bifurcation_ijk[0]), int(self.bifurcation_ijk[1]), int(self.bifurcation_ijk[2])]
                if self.bifurcation_ijk is not None
                else None
            ),
            "junction_degree": int(self.junction_degree),
            "lva_voxels": int(self.lva_voxels),
            "rva_voxels": int(self.rva_voxels),
            "basilar_voxels": int(self.basilar_voxels),
            "hemisphere_axis": self.hemisphere_axis,
            "bifurcation_cut_k": self.bifurcation_cut_k,
            "message": self.message,
        }
        if self.lva_centerline is not None and self.lva_centerline.size > 0:
            out["lva_centerline"] = to_numpy(self.lva_centerline).astype(float).tolist()
        if self.rva_centerline is not None and self.rva_centerline.size > 0:
            out["rva_centerline"] = to_numpy(self.rva_centerline).astype(float).tolist()
        return out


def _neighbors26(p: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    i, j, k = p
    out: list[tuple[int, int, int]] = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                out.append((i + di, j + dj, k + dk))
    return out


def _skeleton_branch_nodes(skel: np.ndarray) -> list[tuple[int, int, int]]:
    coords = [tuple(int(v) for v in row) for row in np.argwhere(skel)]
    node_set = set(coords)
    branches: list[tuple[int, int, int]] = []
    for n in coords:
        deg = sum(1 for m in _neighbors26(n) if m in node_set)
        if deg >= 3:
            branches.append(n)
    return branches


def _walk_inferior_skeleton_branch(
    seed: tuple[int, int, int],
    junction: tuple[int, int, int],
    node_set: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    """Skeleton voxels along one inferior branch away from *junction*."""
    jk = int(junction[2])
    path: set[tuple[int, int, int]] = {seed}
    current = seed
    for _ in range(100_000):
        candidates = [
            n
            for n in _neighbors26(current)
            if n in node_set and n not in path and n[2] <= jk
        ]
        if not candidates:
            break
        next_n = min(candidates, key=lambda t: (t[2], abs(t[0] - current[0]) + abs(t[1] - current[1])))
        if next_n[2] > current[2]:
            break
        path.add(next_n)
        current = next_n
    return path


def _flood_fill_from_seeds(
    domain: np.ndarray,
    seeds: set[tuple[int, int, int]],
) -> np.ndarray:
    """6-connected flood fill inside *domain* from *seeds*."""
    out = np.zeros(domain.shape, dtype=bool)
    q: deque[tuple[int, int, int]] = deque()
    for p in seeds:
        i, j, k = p
        if domain[i, j, k] and not out[i, j, k]:
            out[i, j, k] = True
            q.append(p)
    while q:
        i, j, k = q.popleft()
        for ni, nj, nk in _neighbors26((i, j, k)):
            if domain[ni, nj, nk] and not out[ni, nj, nk]:
                out[ni, nj, nk] = True
                q.append((ni, nj, nk))
    return out


def _pair_inferior_branches(
    junction: tuple[int, int, int],
    node_set: set[tuple[int, int, int]],
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]] | None:
    """Return skeleton seed sets for the two inferior VA branches at *junction*."""
    jk = int(junction[2])
    inferior_nbrs = [n for n in _neighbors26(junction) if n in node_set and n[2] < jk]
    if len(inferior_nbrs) < 2:
        return None

    walks = [_walk_inferior_skeleton_branch(n, junction, node_set) for n in inferior_nbrs]
    if len(walks) == 2:
        a, b = walks[0], walks[1]
    else:
        walks.sort(key=lambda w: float(np.mean([p[1] for p in w])))
        a, b = walks[0], walks[-1]

    with using('numpy'):
        mj_a = float(np.mean([p[1] for p in a]))
        mj_b = float(np.mean([p[1] for p in b]))
    if mj_a <= mj_b:
        return a, b
    return b, a


def split_vertebral_from_basilar(
    seg: np.ndarray,
    *,
    min_branch_voxels: int = 5,
    bifurcation_cut_margin: int = 2,
) -> tuple[np.ndarray, VertebralSplitResult]:
    """Relabel inferior basilar into LVA/RVA when a vertebro-basilar Y-junction is found.

    Inferior VA tubes are separated by flood-filling from each inferior skeleton branch
    within ``k < junction_k - bifurcation_cut_margin``, so the merging Y at the
    bifurcation stays basilar and the two VA masks do not bridge.
    """
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    basilar = np.asarray(seg_np == int(QVTPY_BASILAR), dtype=bool)
    n_basilar = int(np.count_nonzero(basilar))
    if n_basilar == 0:
        return seg_np, VertebralSplitResult(
            split_applied=False,
            bifurcation_ijk=None,
            junction_degree=0,
            lva_voxels=0,
            rva_voxels=0,
            basilar_voxels=0,
            hemisphere_axis="y",
            message="empty basilar mask",
        )

    skel = np.asarray(to_numpy(skeletonize_binary(basilar)), dtype=bool)
    branch_nodes = _skeleton_branch_nodes(skel)
    if not branch_nodes:
        return seg_np, VertebralSplitResult(
            split_applied=False,
            bifurcation_ijk=None,
            junction_degree=0,
            lva_voxels=0,
            rva_voxels=0,
            basilar_voxels=n_basilar,
            hemisphere_axis="y",
            message="no skeleton bifurcation",
        )

    junction = min(branch_nodes, key=lambda t: t[2])
    ji, jj, jk = int(junction[0]), int(junction[1]), int(junction[2])
    node_set = set(tuple(int(v) for v in row) for row in np.argwhere(skel))
    nbrs = [n for n in _neighbors26(junction) if n in node_set]
    junction_degree = len(nbrs)
    if len(nbrs) < 2:
        return seg_np, VertebralSplitResult(
            split_applied=False,
            bifurcation_ijk=junction,
            junction_degree=junction_degree,
            lva_voxels=0,
            rva_voxels=0,
            basilar_voxels=n_basilar,
            hemisphere_axis="y",
            message="bifurcation has fewer than two branches",
        )

    branch_pair = _pair_inferior_branches(junction, node_set)
    if branch_pair is None:
        return seg_np, VertebralSplitResult(
            split_applied=False,
            bifurcation_ijk=junction,
            junction_degree=junction_degree,
            lva_voxels=0,
            rva_voxels=0,
            basilar_voxels=n_basilar,
            hemisphere_axis="y",
            message="fewer than two inferior skeleton branches",
        )
    lva_seeds, rva_seeds = branch_pair

    cut_margin = max(0, int(bifurcation_cut_margin))
    cut_k = max(0, jk - cut_margin)
    nz = basilar.shape[2]
    k_axis = np.arange(nz, dtype=np.int32)
    va_domain = basilar & (k_axis[None, None, :] < cut_k)

    if not va_domain.any():
        return seg_np, VertebralSplitResult(
            split_applied=False,
            bifurcation_ijk=junction,
            junction_degree=junction_degree,
            lva_voxels=0,
            rva_voxels=0,
            basilar_voxels=n_basilar,
            hemisphere_axis="y",
            bifurcation_cut_k=int(cut_k),
            message="VA domain empty after bifurcation cut",
        )

    lva_mask = _flood_fill_from_seeds(va_domain, lva_seeds)
    rva_domain = va_domain & ~lva_mask
    rva_mask = _flood_fill_from_seeds(rva_domain, rva_seeds)

    n_lva = int(np.count_nonzero(lva_mask))
    n_rva = int(np.count_nonzero(rva_mask))
    if n_lva < min_branch_voxels or n_rva < min_branch_voxels:
        return seg_np, VertebralSplitResult(
            split_applied=False,
            bifurcation_ijk=junction,
            junction_degree=junction_degree,
            lva_voxels=n_lva,
            rva_voxels=n_rva,
            basilar_voxels=n_basilar,
            hemisphere_axis="y",
            bifurcation_cut_k=int(cut_k),
            message="inferior VA branch too small after skeleton flood-fill",
        )

    superior = basilar & ~(lva_mask | rva_mask)

    out = seg_np.copy()
    out[basilar] = 0
    out[superior] = int(QVTPY_BASILAR)
    out[lva_mask] = int(QVTPY_LVA)
    out[rva_mask] = int(QVTPY_RVA)

    lva_cl = rva_cl = None
    if n_lva > 0:
        lva_cl = compute_centerlines(out, labels=[QVTPY_LVA], min_points=3).get(QVTPY_LVA)
    if n_rva > 0:
        rva_cl = compute_centerlines(out, labels=[QVTPY_RVA], min_points=3).get(QVTPY_RVA)

    return out, VertebralSplitResult(
        split_applied=True,
        bifurcation_ijk=junction,
        junction_degree=junction_degree,
        lva_voxels=n_lva,
        rva_voxels=n_rva,
        basilar_voxels=int(np.count_nonzero(superior)),
        hemisphere_axis="y",
        bifurcation_cut_k=int(cut_k),
        lva_centerline=lva_cl,
        rva_centerline=rva_cl,
        message=None,
    )


__all__ = ["VertebralSplitResult", "split_vertebral_from_basilar"]
