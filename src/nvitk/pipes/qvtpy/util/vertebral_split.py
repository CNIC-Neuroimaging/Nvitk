"""Split LVA/RVA from grown basilar via vertebro-basilar centerline bifurcation.

After stage-4 basilar region growing, the basilar mask often includes the two
vertebral arteries that merge into it (before the superior PCA territory).

Anatomy (array axes ``i=X``, ``j=Y``, ``k=Z``):

- Each VA runs along **+Z** on its own L/R side of the **X** axis.
- The two arms merge at an inferior Y-junction into the basilar.
- The basilar continues along **+Z** after the confluence.

This module:

1. Builds a detailed centerline of the basilar CC (trunk + bifurcation arms).
2. Detects an inferior Y-junction (VA confluence). If none → VAs absent.
3. Labels the two inferior arms as LVA/RVA by **mean X** (hemisphere-aware via
   ICA when available); the remainder after the merge stays basilar.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.core.logger import Logger
from nvitk.morphology.centerline import (
    compute_centerline_branches,
    compute_centerlines,
    skeletonize_binary,
)
from nvitk.pipes.qvtpy.labels import (
    QVTPY_BASILAR,
    QVTPY_LICA,
    QVTPY_LVA,
    QVTPY_RICA,
    QVTPY_RVA,
)

setup(globals())

log = Logger()

# Array axis treated as L↔R (matches distal expand / typical RAS i=X).
_LR_AXIS = 0
_Z_AXIS = 2


@dataclass(frozen=True)
class VertebralSplitResult:
    """Outcome of basilar → LVA/RVA relabeling."""

    split_applied: bool
    bifurcation_ijk: tuple[int, int, int] | None
    junction_degree: int
    n_centerline_branches: int
    lva_voxels: int
    rva_voxels: int
    basilar_voxels: int
    hemisphere_axis: str
    bifurcation_cut_k: int | None = None
    confidence: float = 0.0
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
            "n_centerline_branches": int(self.n_centerline_branches),
            "lva_voxels": int(self.lva_voxels),
            "rva_voxels": int(self.rva_voxels),
            "basilar_voxels": int(self.basilar_voxels),
            "hemisphere_axis": self.hemisphere_axis,
            "bifurcation_cut_k": self.bifurcation_cut_k,
            "vertebral_split_confidence": float(self.confidence),
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


def _walk_inferior_skeleton_branch(
    seed: tuple[int, int, int],
    junction: tuple[int, int, int],
    node_set: set[tuple[int, int, int]],
) -> set[tuple[int, int, int]]:
    """Skeleton voxels along one inferior branch (descending −Z away from merge)."""
    jk = int(junction[_Z_AXIS])
    path: set[tuple[int, int, int]] = {seed}
    current = seed
    for _ in range(100_000):
        candidates = [
            n
            for n in _neighbors26(current)
            if n in node_set and n not in path and n[_Z_AXIS] <= jk
        ]
        if not candidates:
            break
        next_n = min(
            candidates,
            key=lambda t: (
                t[_Z_AXIS],
                abs(t[0] - current[0]) + abs(t[1] - current[1]),
            ),
        )
        if next_n[_Z_AXIS] > current[_Z_AXIS]:
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
        if (
            0 <= i < domain.shape[0]
            and 0 <= j < domain.shape[1]
            and 0 <= k < domain.shape[2]
            and domain[i, j, k]
            and not out[i, j, k]
        ):
            out[i, j, k] = True
            q.append(p)
    while q:
        i, j, k = q.popleft()
        for ni, nj, nk in (
            (i - 1, j, k),
            (i + 1, j, k),
            (i, j - 1, k),
            (i, j + 1, k),
            (i, j, k - 1),
            (i, j, k + 1),
        ):
            if (
                0 <= ni < domain.shape[0]
                and 0 <= nj < domain.shape[1]
                and 0 <= nk < domain.shape[2]
                and domain[ni, nj, nk]
                and not out[ni, nj, nk]
            ):
                out[ni, nj, nk] = True
                q.append((ni, nj, nk))
    return out


def _mean_axis(points: set[tuple[int, int, int]], axis: int) -> float:
    with using("numpy"):
        return float(np.mean([p[axis] for p in points]))


def _min_axis(points: set[tuple[int, int, int]], axis: int) -> float:
    return float(min(p[axis] for p in points))


def _arm_components_from_cluster(
    cluster: set[tuple[int, int, int]],
    node_set: set[tuple[int, int, int]],
) -> list[set[tuple[int, int, int]]]:
    """Connected components of ``node_set \\ cluster`` that touch the cluster."""
    outside = node_set - cluster
    if not outside:
        return []
    exits = {m for c in cluster for m in _neighbors26(c) if m in outside}
    if not exits:
        return []
    visited: set[tuple[int, int, int]] = set()
    arms: list[set[tuple[int, int, int]]] = []
    for seed in sorted(exits):
        if seed in visited:
            continue
        comp: set[tuple[int, int, int]] = set()
        q: deque[tuple[int, int, int]] = deque([seed])
        visited.add(seed)
        while q:
            u = q.popleft()
            comp.add(u)
            for v in _neighbors26(u):
                if v in outside and v not in visited:
                    visited.add(v)
                    q.append(v)
        if comp & exits:
            arms.append(comp)
    return arms


def _pair_va_arms_from_cluster(
    cluster: set[tuple[int, int, int]],
    node_set: set[tuple[int, int, int]],
    *,
    min_arm_points: int = 3,
    min_x_separation: float = 1.5,
) -> tuple[
    tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]] | None,
    dict[str, Any],
]:
    """Pick the two inferior VA arms at a Y-junction cluster.

    Arms are the connected components leaving the junction blob. The highest-mean-Z
    arm is treated as the basilar continuation (+Z); the two lowest-mean-Z arms are
    the VAs (ascending along +Z into the merge on distinct X sides).
    """
    arms = _arm_components_from_cluster(cluster, node_set)
    long_arms = [a for a in arms if len(a) >= int(min_arm_points)]
    diag: dict[str, Any] = {
        "n_arms": len(arms),
        "n_long_arms": len(long_arms),
        "arm_mean_xyz": [
            (
                round(_mean_axis(a, 0), 2),
                round(_mean_axis(a, 1), 2),
                round(_mean_axis(a, 2), 2),
                len(a),
            )
            for a in long_arms
        ],
    }
    if len(long_arms) < 3:
        diag["reject"] = "need ≥3 long arms (2 VA + basilar)"
        return None, diag

    # Lowest two mean-Z → VAs; highest → basilar continuation along +Z.
    ranked = sorted(long_arms, key=lambda a: (_mean_axis(a, _Z_AXIS), -len(a)))
    va_a, va_b = ranked[0], ranked[1]
    bas = ranked[-1]
    va_z = 0.5 * (_mean_axis(va_a, _Z_AXIS) + _mean_axis(va_b, _Z_AXIS))
    bas_z = _mean_axis(bas, _Z_AXIS)
    dx = abs(_mean_axis(va_a, _LR_AXIS) - _mean_axis(va_b, _LR_AXIS))
    diag.update(
        {
            "va_mean_z": round(va_z, 2),
            "basilar_mean_z": round(bas_z, 2),
            "va_x_separation": round(dx, 2),
        }
    )
    if va_z >= bas_z - 0.25:
        diag["reject"] = "lowest arms are not inferior to basilar (+Z) arm"
        return None, diag
    if dx < float(min_x_separation):
        diag["reject"] = f"VA arms not separated on X (dx={dx:.2f})"
        return None, diag
    jk = _mean_axis(cluster, _Z_AXIS)
    if _min_axis(va_a, _Z_AXIS) > jk + 1 and _min_axis(va_b, _Z_AXIS) > jk + 1:
        diag["reject"] = "candidate VA arms do not extend below confluence Z"
        return None, diag
    return (va_a, va_b), diag


def _pair_va_from_inferior_endpoints(
    node_set: set[tuple[int, int, int]],
    deg: dict[tuple[int, int, int], int],
    *,
    min_x_separation: float = 1.5,
    min_walk: int = 3,
) -> tuple[
    tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]] | None,
    tuple[int, int, int] | None,
    dict[str, Any],
]:
    """Fallback: two lowest-Z skeleton endpoints separated on X as VA tips.

    Walks each tip toward higher Z until a junction; the first shared / nearest
    junction is the confluence. Useful when the merge is a soft blob without a
    clean 3-arm cluster decomposition.
    """
    endpoints = [n for n, d in deg.items() if int(d) <= 1]
    diag: dict[str, Any] = {"n_endpoints": len(endpoints)}
    if len(endpoints) < 2:
        diag["reject"] = "fewer than 2 skeleton endpoints"
        return None, None, diag
    # Candidate VA tips: lowest-Z endpoints.
    ranked = sorted(endpoints, key=lambda n: (n[_Z_AXIS], n[_LR_AXIS]))
    # Try pairs among the most inferior endpoints.
    candidates = ranked[: min(6, len(ranked))]
    best = None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            a, b = candidates[i], candidates[j]
            dx = abs(a[_LR_AXIS] - b[_LR_AXIS])
            if dx < min_x_separation:
                continue
            # Walk each tip up (+Z preference) collecting path until degree ≥3.
            def _walk_up(start: tuple[int, int, int]) -> set[tuple[int, int, int]]:
                path: set[tuple[int, int, int]] = {start}
                prev = None
                cur = start
                for _ in range(100_000):
                    nbrs = [m for m in _neighbors26(cur) if m in node_set and m != prev]
                    if not nbrs:
                        break
                    # Prefer ascending / staying near tip Z while exploring.
                    nxt = max(
                        nbrs,
                        key=lambda t: (t[_Z_AXIS], -abs(t[_LR_AXIS] - start[_LR_AXIS])),
                    )
                    if nxt in path:
                        break
                    path.add(nxt)
                    prev, cur = cur, nxt
                    if deg.get(cur, 0) >= 3:
                        break
                return path

            wa, wb = _walk_up(a), _walk_up(b)
            if len(wa) < min_walk or len(wb) < min_walk:
                continue
            # Confluence ≈ highest-Z voxel common to both paths, else midpoint of tips' max Z.
            common = wa & wb
            if common:
                bif = max(common, key=lambda n: n[_Z_AXIS])
            else:
                # Nearest pair across path ends.
                end_a = max(wa, key=lambda n: n[_Z_AXIS])
                end_b = max(wb, key=lambda n: n[_Z_AXIS])
                bif = end_a if end_a[_Z_AXIS] <= end_b[_Z_AXIS] else end_b
            score = dx - 0.1 * abs(a[_Z_AXIS] - b[_Z_AXIS])
            if best is None or score > best[0]:
                best = (score, wa, wb, bif, a, b)
    if best is None:
        diag["reject"] = "no X-separated inferior endpoint pair"
        return None, None, diag
    _score, wa, wb, bif, tip_a, tip_b = best
    diag.update(
        {
            "tips": [list(tip_a), list(tip_b)],
            "confluence": list(bif),
            "walk_lens": [len(wa), len(wb)],
        }
    )
    return (wa, wb), bif, diag


def _pair_inferior_branches_legacy(
    junction: tuple[int, int, int],
    node_set: set[tuple[int, int, int]],
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]] | None:
    """Fallback: two immediate neighbors with z ≤ junction_z."""
    jk = int(junction[_Z_AXIS])
    inferior_nbrs = [
        n for n in _neighbors26(junction) if n in node_set and n[_Z_AXIS] <= jk
    ]
    if len(inferior_nbrs) < 2:
        return None
    walks = [_walk_inferior_skeleton_branch(n, junction, node_set) for n in inferior_nbrs]
    walks = [w for w in walks if len(w) >= 2]
    if len(walks) < 2:
        return None
    if len(walks) == 2:
        a, b = walks[0], walks[1]
    else:
        walks.sort(key=lambda w: _mean_axis(w, _LR_AXIS))
        a, b = walks[0], walks[-1]
    if abs(_mean_axis(a, _LR_AXIS) - _mean_axis(b, _LR_AXIS)) < 1.0:
        return None
    return a, b


def _left_has_higher_x(seg_np: np.ndarray) -> bool | None:
    """Whether patient-left has higher array-X (RAS-like). None if ICA missing."""
    left = as_backend_array(np.argwhere(seg_np == int(QVTPY_LICA))).astype(np.float64)
    right = as_backend_array(np.argwhere(seg_np == int(QVTPY_RICA))).astype(np.float64)
    if left.size == 0 or right.size == 0:
        return None
    return float(left[:, _LR_AXIS].mean()) >= float(right[:, _LR_AXIS].mean())


def _order_branches_left_right(
    a: set[tuple[int, int, int]],
    b: set[tuple[int, int, int]],
    seg_np: np.ndarray,
) -> tuple[set[tuple[int, int, int]], set[tuple[int, int, int]], str, float, float]:
    """Order (a, b) into (LVA_seeds, RVA_seeds) by mean X position.

    L/R is decided on the X axis: each VA sits on its own L/R X side before
    ascending along +Z into the basilar confluence. ICA centroids resolve which
    X extreme is patient-left; fallback assumes RAS (left = higher X).
    """
    mx_a = _mean_axis(a, _LR_AXIS)
    mx_b = _mean_axis(b, _LR_AXIS)
    left_higher = _left_has_higher_x(seg_np)
    if left_higher is None:
        left_higher = True
        mode = "x_ras"
    else:
        mode = "x_ica"
    if left_higher:
        lva, rva = (a, b) if mx_a >= mx_b else (b, a)
    else:
        lva, rva = (a, b) if mx_a <= mx_b else (b, a)
    return lva, rva, mode, mx_a, mx_b


def _prefer_superior_basilar(
    basilar: np.ndarray,
    prefer_points: np.ndarray | None,
) -> np.ndarray | None:
    """Bias the basilar trunk toward superior (post-merge) so VA arms are side branches."""
    if prefer_points is not None:
        pts = as_backend_array(prefer_points).astype(np.float64).reshape(-1, 3)
        if pts.shape[0] >= 2:
            return pts
    coords = np.argwhere(basilar)
    if coords.size == 0:
        return None
    kmax = int(coords[:, 2].max())
    top = coords[coords[:, 2] >= max(0, kmax - 3)]
    return top.astype(np.float64) if top.size else None


def split_vertebral_from_basilar(
    seg: np.ndarray,
    *,
    prefer_basilar_centerline: np.ndarray | None = None,
    min_branch_voxels: int = 15,
    min_branch_points: int = 3,
    bifurcation_cut_margin: int = 0,
) -> tuple[np.ndarray, VertebralSplitResult]:
    """Relabel inferior basilar into LVA/RVA when a vertebro-basilar bifurcation exists.

    VAs ascend along **+Z** on distinct L/R **X** sides, merge at an inferior
    Y-junction, and the basilar continues along +Z. The split is placed
    **exactly on the bifurcation cluster**: junction voxels stay basilar; the two
    inferior arms are flood-filled into LVA/RVA (X half-spaces). Optional
    *bifurcation_cut_margin* (>0) adds a legacy soft ``z < junction_z - margin``
    gate. Without a qualifying bifurcation, the basilar mask is left unchanged.
    """
    seg_np = as_backend_array(seg).astype(np.int32, copy=False)
    basilar = as_backend_array(seg_np == int(QVTPY_BASILAR)).astype(bool)
    n_basilar = int(np.count_nonzero(basilar))
    empty = VertebralSplitResult(
        split_applied=False,
        bifurcation_ijk=None,
        junction_degree=0,
        n_centerline_branches=0,
        lva_voxels=0,
        rva_voxels=0,
        basilar_voxels=n_basilar,
        hemisphere_axis="x",
    )
    log.step(
        f"vertebral split: start (basilar_voxels={n_basilar}, "
        f"anatomy=+Z ascent on L/R X → merge → basilar +Z)"
    )
    if n_basilar == 0:
        log.info("vertebral split: empty basilar mask — VAs absent")
        return seg_np, replace(empty, basilar_voxels=0, message="empty basilar mask")

    # Detailed centerline: trunk (superior-biased) + all bifurcation side arms.
    prefer = _prefer_superior_basilar(basilar, prefer_basilar_centerline)
    basilar_lab = basilar.astype(np.int32) * int(QVTPY_BASILAR)
    branch_map = compute_centerline_branches(
        basilar_lab,
        labels=[int(QVTPY_BASILAR)],
        min_points=int(min_branch_points),
        min_branch_points=int(min_branch_points),
        prefer_points_by_label=(
            {int(QVTPY_BASILAR): prefer} if prefer is not None else None
        ),
    )
    paths = branch_map.get(int(QVTPY_BASILAR)) or []
    n_branches = len(paths)
    log.info(
        f"vertebral split: basilar centerline branches={n_branches} "
        f"(prefer={'stage3/superior' if prefer is not None else 'none'})"
    )

    skel = as_backend_array(skeletonize_binary(basilar)).astype(bool)
    n_skel = int(np.count_nonzero(skel))
    skel_coords = np.argwhere(skel)
    from nvitk.morphology.polyline_graph import junction_clusters, skeleton_graph

    nodes, _adj, deg = skeleton_graph(skel_coords.astype(np.float64))
    node_set = set(nodes)
    clusters = junction_clusters(nodes, deg, min_degree=3)
    log.info(
        f"vertebral split: skeleton voxels={n_skel}, "
        f"junction clusters={len(clusters)} "
        f"(centerline_branches={n_branches})"
    )
    junction: tuple[int, int, int] | None = None
    junction_cluster: set[tuple[int, int, int]] = set()
    branch_pair: tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]] | None = None
    junction_degree = 0

    if not clusters:
        log.info(
            "vertebral split: no degree≥3 junction clusters; "
            "trying inferior-endpoint fallback"
        )
        ep_pair, ep_bif, ep_diag = _pair_va_from_inferior_endpoints(node_set, deg)
        if ep_pair is None or ep_bif is None:
            log.info(
                f"vertebral split: VAs absent ({ep_diag.get('reject') or 'no clusters'})"
            )
            return seg_np, replace(
                empty,
                n_centerline_branches=n_branches,
                message="no skeleton bifurcation (VAs absent)",
            )
        branch_pair = ep_pair
        junction = ep_bif
        junction_cluster = {ep_bif}
        # Expand to the full junction cluster containing the bifurcation if any.
        for cluster_list in clusters:
            if ep_bif in cluster_list:
                junction_cluster = set(cluster_list)
                break
        junction_degree = int(deg.get(ep_bif, 0))
        log.info(
            f"vertebral split: accepted via inferior-endpoint fallback "
            f"(tips={ep_diag.get('tips')} confluence={list(ep_bif)})"
        )
    else:
        # Prefer inferior clusters first (VA confluence is below the basilar stem).
        def _cluster_z(cluster_list: list[tuple[int, int, int]]) -> float:
            return float(np.mean([n[_Z_AXIS] for n in cluster_list]))

        n_reject_logged = 0
        for cluster_list in sorted(clusters, key=_cluster_z):
            cluster = set(cluster_list)
            rep = max(cluster_list, key=lambda n: (deg.get(n, 0), n))
            pair, diag = _pair_va_arms_from_cluster(
                cluster, node_set, min_arm_points=max(2, int(min_branch_points) - 1)
            )
            if pair is None:
                legacy = _pair_inferior_branches_legacy(rep, node_set)
                if legacy is None:
                    if n_reject_logged < 5:
                        log.info(
                            f"vertebral split: cluster@{rep} rejected "
                            f"({diag.get('reject')}; arms={diag.get('arm_mean_xyz')})"
                        )
                        n_reject_logged += 1
                    continue
                branch_pair = legacy
                log.info(
                    f"vertebral split: cluster@{rep} accepted via legacy "
                    "inferior-neighbor walk"
                )
            else:
                branch_pair = pair
                log.info(
                    f"vertebral split: cluster@{rep} accepted as VA confluence "
                    f"(va_z={diag.get('va_mean_z')} bas_z={diag.get('basilar_mean_z')} "
                    f"dx={diag.get('va_x_separation')} arms={diag.get('arm_mean_xyz')})"
                )
            junction = rep
            junction_cluster = cluster
            junction_degree = int(
                deg.get(rep, len([n for n in _neighbors26(rep) if n in node_set]))
            )
            break

        if junction is None or branch_pair is None:
            ep_pair, ep_bif, ep_diag = _pair_va_from_inferior_endpoints(node_set, deg)
            if ep_pair is not None and ep_bif is not None:
                branch_pair = ep_pair
                junction = ep_bif
                junction_cluster = {ep_bif}
                for cluster_list in clusters:
                    if ep_bif in cluster_list:
                        junction_cluster = set(cluster_list)
                        break
                junction_degree = int(deg.get(ep_bif, 0))
                log.info(
                    f"vertebral split: accepted via inferior-endpoint fallback "
                    f"(tips={ep_diag.get('tips')} confluence={ep_diag.get('confluence')} "
                    f"walks={ep_diag.get('walk_lens')})"
                )
            else:
                log.info(
                    "vertebral split: no Y-junction with two inferior X-separated VA arms "
                    f"— VAs absent ({ep_diag.get('reject')})"
                )
                return seg_np, replace(
                    empty,
                    n_centerline_branches=n_branches,
                    message="no inferior VA confluence bifurcation (VAs absent)",
                )

    assert junction is not None and branch_pair is not None
    ji, jj, jk = int(junction[0]), int(junction[1]), int(junction[2])
    log.step(
        f"vertebral split: confluence at ijk=({ji},{jj},{jk}) "
        f"degree={junction_degree} cluster_voxels={len(junction_cluster)} "
        f"(VAs ascend +Z on L/R X → merge → basilar +Z)"
    )

    # Seeds are arm voxels outside the junction; drop any accidental junction hits.
    raw_a = {p for p in branch_pair[0] if p not in junction_cluster}
    raw_b = {p for p in branch_pair[1] if p not in junction_cluster}
    if len(raw_a) < 2 or len(raw_b) < 2:
        raw_a, raw_b = branch_pair[0], branch_pair[1]
    lva_seeds, rva_seeds, hemisphere_axis, mx_a, mx_b = _order_branches_left_right(
        raw_a, raw_b, seg_np
    )
    lva_x = _mean_axis(lva_seeds, _LR_AXIS)
    rva_x = _mean_axis(rva_seeds, _LR_AXIS)
    lva_z = _mean_axis(lva_seeds, _Z_AXIS)
    rva_z = _mean_axis(rva_seeds, _Z_AXIS)
    log.info(
        f"vertebral split: L/R by X-axis ({hemisphere_axis}): "
        f"arm0_mean_x={mx_a:.2f} arm1_mean_x={mx_b:.2f} → "
        f"LVA mean_x={lva_x:.2f} mean_z={lva_z:.2f} "
        f"(skel={len(lva_seeds)}), "
        f"RVA mean_x={rva_x:.2f} mean_z={rva_z:.2f} "
        f"(skel={len(rva_seeds)})"
    )

    # Exact bifurcation cut:
    # - junction-cluster voxels stay basilar (the merge point itself)
    # - everything with z > junction_z stays basilar (post-merge stem along +Z)
    # - VA flood is only at z <= junction_z, excluding the junction, on X half-spaces
    cut_margin = max(0, int(bifurcation_cut_margin))
    cut_k = int(jk) - cut_margin  # default margin=0 → cut exactly at bifurcation Z
    junction_mask = np.zeros(basilar.shape, dtype=bool)
    if junction_cluster:
        jc = np.asarray(list(junction_cluster), dtype=np.int64)
        junction_mask[jc[:, 0], jc[:, 1], jc[:, 2]] = True
    k_axis = np.arange(basilar.shape[_Z_AXIS], dtype=np.int32)
    va_domain = basilar & ~junction_mask & (k_axis[None, None, :] <= cut_k)
    n_va_domain = int(np.count_nonzero(va_domain))
    # Split the VA domain on X so L/R arms cannot flood into each other when
    # they briefly touch near the confluence.
    mid_x = 0.5 * (lva_x + rva_x)
    x_coords = np.arange(basilar.shape[_LR_AXIS], dtype=np.float64)
    if _LR_AXIS == 0:
        x_grid = x_coords[:, None, None]
    else:
        x_grid = x_coords[None, :, None]
    if lva_x >= rva_x:
        lva_domain = va_domain & (x_grid >= mid_x - 0.5)
        rva_domain = va_domain & (x_grid <= mid_x + 0.5)
    else:
        lva_domain = va_domain & (x_grid <= mid_x + 0.5)
        rva_domain = va_domain & (x_grid >= mid_x - 0.5)
    log.info(
        f"vertebral split: cut at bifurcation "
        f"(ijk=({ji},{jj},{jk}), cluster={len(junction_cluster)}, "
        f"z<={cut_k}, margin={cut_margin}) "
        f"VA_domain={n_va_domain} X-midline={mid_x:.1f} "
        f"(LVA_x={lva_x:.1f}, RVA_x={rva_x:.1f})"
    )
    if not va_domain.any():
        log.info("vertebral split: VA domain empty after bifurcation cut — skip")
        return seg_np, replace(
            empty,
            bifurcation_ijk=junction,
            junction_degree=junction_degree,
            n_centerline_branches=n_branches,
            hemisphere_axis=hemisphere_axis,
            bifurcation_cut_k=int(cut_k),
            message="VA domain empty after bifurcation cut",
        )

    lva_mask = _flood_fill_from_seeds(lva_domain, lva_seeds)
    rva_mask = _flood_fill_from_seeds(rva_domain & ~lva_mask, rva_seeds)
    # Never relabel the bifurcation itself.
    lva_mask &= ~junction_mask
    rva_mask &= ~junction_mask
    n_lva = int(np.count_nonzero(lva_mask))
    n_rva = int(np.count_nonzero(rva_mask))
    if n_lva < min_branch_voxels or n_rva < min_branch_voxels:
        log.info(
            f"vertebral split: VA flood too small "
            f"(LVA={n_lva}, RVA={n_rva}, min={min_branch_voxels}) — skip"
        )
        return seg_np, replace(
            empty,
            bifurcation_ijk=junction,
            junction_degree=junction_degree,
            n_centerline_branches=n_branches,
            lva_voxels=n_lva,
            rva_voxels=n_rva,
            hemisphere_axis=hemisphere_axis,
            bifurcation_cut_k=int(cut_k),
            message="inferior VA branch too small after bifurcation flood-fill",
        )

    superior = basilar & ~(lva_mask | rva_mask)
    n_bas_keep = int(np.count_nonzero(superior))
    out = seg_np.copy()
    out[basilar] = 0
    out[superior] = int(QVTPY_BASILAR)
    out[lva_mask] = int(QVTPY_LVA)
    out[rva_mask] = int(QVTPY_RVA)

    lva_cl = compute_centerlines(out, labels=[QVTPY_LVA], min_points=3).get(QVTPY_LVA)
    rva_cl = compute_centerlines(out, labels=[QVTPY_RVA], min_points=3).get(QVTPY_RVA)
    balance = (
        float(min(n_lva, n_rva)) / float(max(n_lva, n_rva)) if max(n_lva, n_rva) > 0 else 0.0
    )
    confidence = balance if hemisphere_axis.startswith("x_ica") else 0.5 * balance

    log.step(
        f"vertebral split: applied — "
        f"LVA={n_lva} (mean_x={lva_x:.1f}) "
        f"RVA={n_rva} (mean_x={rva_x:.1f}) "
        f"basilar={n_bas_keep} "
        f"confluence=({ji},{jj},{jk}) hemi={hemisphere_axis} "
        f"confidence={confidence:.3f}"
    )

    return out, VertebralSplitResult(
        split_applied=True,
        bifurcation_ijk=(ji, jj, jk),
        junction_degree=junction_degree,
        n_centerline_branches=n_branches,
        lva_voxels=n_lva,
        rva_voxels=n_rva,
        basilar_voxels=n_bas_keep,
        hemisphere_axis=hemisphere_axis,
        bifurcation_cut_k=int(cut_k),
        confidence=confidence,
        lva_centerline=lva_cl,
        rva_centerline=rva_cl,
        message=None,
    )


__all__ = ["VertebralSplitResult", "split_vertebral_from_basilar"]
