"""Split LVA/RVA from grown basilar via vertebro-basilar **centerline** bifurcation.

After stage-4 basilar region growing, the basilar mask may include the two
vertebral arteries that merge into it (before the superior PCA territory).

Anatomy (array axes ``i=X``, ``j=Y``, ``k=Z``):

- Each VA runs along **+Z** on its own L/R side of the **X** axis.
- The two arms merge at an inferior Y-junction into the basilar.
- The basilar continues along **+Z** after the confluence.

This module detects that Y **only on the centerline branch graph**
(``compute_centerline_branches``: trunk + side branches that attach to it).
If the centerline is a single polyline (no bifurcation), VAs are treated as
absent and the basilar mask is left unchanged — no skeleton-tip / venous-style
junction fallbacks that can longitudinally bisect a tube.
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
# Minimum X separation (voxels) between the two VA centerline arms.
_MIN_VA_X_SEPARATION = 1.5
# Confluence must leave enough superior stem and sit below this Z percentile
# of the trunk (avoids cutting at the superior tip of a single tube).
_MAX_CONFLUENCE_Z_PERCENTILE = 70.0
_MIN_BASILAR_KEEP_FRAC = 0.15


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


def _pts_to_tuples(pts: np.ndarray) -> list[tuple[int, int, int]]:
    arr = to_numpy(pts).astype(np.int64).reshape(-1, 3)
    return [tuple(int(v) for v in row) for row in arr]


def _pts_to_set(pts: np.ndarray) -> set[tuple[int, int, int]]:
    return set(_pts_to_tuples(pts))


def _nearest_index(
    trunk: list[tuple[int, int, int]],
    query: tuple[int, int, int],
) -> int:
    best_i = 0
    best_d = None
    qx, qy, qz = query
    for i, (x, y, z) in enumerate(trunk):
        d = (x - qx) ** 2 + (y - qy) ** 2 + (z - qz) ** 2
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    return int(best_i)


def _arm_set_from_polyline(
    pts: list[tuple[int, int, int]],
    *,
    drop: tuple[int, int, int] | None = None,
) -> set[tuple[int, int, int]]:
    out = set(pts)
    if drop is not None:
        out.discard(drop)
    return out


def _classify_va_bifurcation_at_attach(
    trunk: list[tuple[int, int, int]],
    attach_idx: int,
    side_arm: set[tuple[int, int, int]],
    *,
    min_arm_points: int,
    min_x_separation: float,
) -> tuple[
    tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]] | None,
    dict[str, Any],
]:
    """At trunk attachment *attach_idx*, pick two inferior X-separated VA arms.

    The three arms leaving the confluence are:

    - trunk segment before the attach index
    - trunk segment after the attach index
    - the side branch (excluding the junction voxel)

    Exactly one should be the superior basilar continuation (+Z); the other two
    are the VAs (lower mean Z, separated on X).
    """
    junction = trunk[attach_idx]
    left = _arm_set_from_polyline(trunk[: attach_idx + 1], drop=junction)
    right = _arm_set_from_polyline(trunk[attach_idx:], drop=junction)
    side = set(side_arm)
    side.discard(junction)
    arms = [a for a in (left, right, side) if len(a) >= int(min_arm_points)]
    diag: dict[str, Any] = {
        "attach_idx": int(attach_idx),
        "junction": list(junction),
        "arm_sizes": [len(left), len(right), len(side)],
        "arm_mean_xyz": [
            (
                round(_mean_axis(a, 0), 2),
                round(_mean_axis(a, 1), 2),
                round(_mean_axis(a, 2), 2),
                len(a),
            )
            for a in arms
        ],
    }
    if len(arms) < 3:
        diag["reject"] = "need 3 arms at centerline bifurcation (2 VA + basilar)"
        return None, diag

    # Highest mean-Z → basilar continuation; lowest two → VAs.
    ranked = sorted(arms, key=lambda a: (_mean_axis(a, _Z_AXIS), -len(a)))
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
    # Both VA arms should reach at/below the confluence Z.
    jk = float(junction[_Z_AXIS])
    if min(p[_Z_AXIS] for p in va_a) > jk + 1 and min(p[_Z_AXIS] for p in va_b) > jk + 1:
        diag["reject"] = "candidate VA arms do not extend below confluence Z"
        return None, diag
    return (va_a, va_b), diag


def _bifurcation_from_centerline_branches(
    paths: list[np.ndarray],
    *,
    min_arm_points: int = 3,
    min_x_separation: float = _MIN_VA_X_SEPARATION,
) -> tuple[
    tuple[int, int, int] | None,
    tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]] | None,
    dict[str, Any],
]:
    """Find VA confluence from trunk + side-branch centerline polylines only.

    ``compute_centerline_branches`` returns trunk first, then each side branch
    oriented **junction → endpoint**. No skeleton tip / venous junction heuristics.
    """
    diag: dict[str, Any] = {"n_paths": len(paths)}
    if len(paths) < 2:
        diag["reject"] = "no centerline bifurcation (single polyline)"
        return None, None, diag

    trunk = _pts_to_tuples(paths[0])
    if len(trunk) < 2 * int(min_arm_points):
        diag["reject"] = "trunk too short for VA bifurcation"
        return None, None, diag

    trunk_z = [p[_Z_AXIS] for p in trunk]
    z_hi = float(np.percentile(trunk_z, _MAX_CONFLUENCE_Z_PERCENTILE))
    diag["trunk_len"] = len(trunk)
    diag["max_confluence_z"] = round(z_hi, 2)

    best: tuple[
        float,
        tuple[int, int, int],
        tuple[set[tuple[int, int, int]], set[tuple[int, int, int]]],
        dict[str, Any],
    ] | None = None
    rejects: list[str] = []

    for bi, side_path in enumerate(paths[1:]):
        side_pts = _pts_to_tuples(side_path)
        if len(side_pts) < int(min_arm_points):
            continue
        # Oriented junction → endpoint by construction.
        attach_pt = side_pts[0]
        attach_idx = _nearest_index(trunk, attach_pt)
        # Snap attach to the nearest trunk voxel (handles 1-voxel offset).
        attach = trunk[attach_idx]
        if attach[_Z_AXIS] > z_hi:
            rejects.append(
                f"branch{bi + 1}@{list(attach)}: confluence too superior "
                f"(z={attach[_Z_AXIS]} > p{_MAX_CONFLUENCE_Z_PERCENTILE:g}={z_hi:.1f})"
            )
            continue
        # Need room on both sides of the trunk for a Y (not an endpoint spur).
        if attach_idx < int(min_arm_points) - 1 or attach_idx > len(trunk) - int(min_arm_points):
            rejects.append(
                f"branch{bi + 1}@{list(attach)}: attach near trunk endpoint "
                f"(idx={attach_idx}/{len(trunk)})"
            )
            continue
        pair, sub = _classify_va_bifurcation_at_attach(
            trunk,
            attach_idx,
            _pts_to_set(side_path),
            min_arm_points=min_arm_points,
            min_x_separation=min_x_separation,
        )
        if pair is None:
            rejects.append(
                f"branch{bi + 1}@{list(attach)}: {sub.get('reject')}"
            )
            continue
        # Prefer more inferior confluence with larger X separation.
        score = float(sub["va_x_separation"]) - 0.05 * float(attach[_Z_AXIS])
        if best is None or score > best[0]:
            best = (score, attach, pair, sub)

    if best is None:
        diag["reject"] = "no qualifying centerline VA bifurcation"
        diag["branch_rejects"] = rejects[:8]
        return None, None, diag

    _score, junction, pair, sub = best
    diag.update(
        {
            "accepted": sub,
            "branch_rejects": rejects[:8],
        }
    )
    return junction, pair, diag


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
    """Order (a, b) into (LVA_seeds, RVA_seeds) by mean X position."""
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
    """Relabel inferior basilar into LVA/RVA when a centerline Y-bifurcation exists.

    Detection uses only ``compute_centerline_branches`` (trunk + attaching side
    branches). A single-polyline centerline means VAs are absent — no tip-based
    fallback that can longitudinally bisect the basilar tube. The cut is at the
    bifurcation: junction voxel stays basilar; inferior arms flood into LVA/RVA
    on X half-spaces.
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

    junction, branch_pair, bif_diag = _bifurcation_from_centerline_branches(
        paths,
        min_arm_points=max(2, int(min_branch_points) - 1),
        min_x_separation=_MIN_VA_X_SEPARATION,
    )
    if junction is None or branch_pair is None:
        reason = bif_diag.get("reject") or "no centerline VA bifurcation"
        extra = bif_diag.get("branch_rejects") or []
        if extra:
            log.info(
                f"vertebral split: VAs absent ({reason}); "
                f"branch checks={extra[:3]}"
            )
        else:
            log.info(f"vertebral split: VAs absent ({reason})")
        return seg_np, replace(
            empty,
            n_centerline_branches=n_branches,
            message=str(reason),
        )

    ji, jj, jk = int(junction[0]), int(junction[1]), int(junction[2])
    accepted = bif_diag.get("accepted") or {}
    log.step(
        f"vertebral split: centerline confluence at ijk=({ji},{jj},{jk}) "
        f"branches={n_branches} "
        f"va_z={accepted.get('va_mean_z')} bas_z={accepted.get('basilar_mean_z')} "
        f"dx={accepted.get('va_x_separation')} "
        f"(VAs ascend +Z on L/R X → merge → basilar +Z)"
    )

    junction_cluster = {junction}
    # Seeds are arm voxels outside the junction.
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
    dx_lr = abs(lva_x - rva_x)
    log.info(
        f"vertebral split: L/R by X-axis ({hemisphere_axis}): "
        f"arm0_mean_x={mx_a:.2f} arm1_mean_x={mx_b:.2f} → "
        f"LVA mean_x={lva_x:.2f} mean_z={lva_z:.2f} "
        f"(skel={len(lva_seeds)}), "
        f"RVA mean_x={rva_x:.2f} mean_z={rva_z:.2f} "
        f"(skel={len(rva_seeds)})"
    )
    if dx_lr < _MIN_VA_X_SEPARATION:
        log.info(
            f"vertebral split: L/R arms not separated on X "
            f"(dx={dx_lr:.2f} < {_MIN_VA_X_SEPARATION}) — skip"
        )
        return seg_np, replace(
            empty,
            bifurcation_ijk=junction,
            n_centerline_branches=n_branches,
            hemisphere_axis=hemisphere_axis,
            message="L/R VA arms not separated on X",
        )

    cut_margin = max(0, int(bifurcation_cut_margin))
    cut_k = int(jk) - cut_margin
    junction_mask = np.zeros(basilar.shape, dtype=bool)
    junction_mask[ji, jj, jk] = True
    k_axis = np.arange(basilar.shape[_Z_AXIS], dtype=np.int32)
    va_domain = basilar & ~junction_mask & (k_axis[None, None, :] <= cut_k)
    n_va_domain = int(np.count_nonzero(va_domain))
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
        f"vertebral split: cut at centerline bifurcation "
        f"(ijk=({ji},{jj},{jk}), z<={cut_k}, margin={cut_margin}) "
        f"VA_domain={n_va_domain} X-midline={mid_x:.1f} "
        f"(LVA_x={lva_x:.1f}, RVA_x={rva_x:.1f})"
    )
    if not va_domain.any():
        log.info("vertebral split: VA domain empty after bifurcation cut — skip")
        return seg_np, replace(
            empty,
            bifurcation_ijk=junction,
            n_centerline_branches=n_branches,
            hemisphere_axis=hemisphere_axis,
            bifurcation_cut_k=int(cut_k),
            message="VA domain empty after bifurcation cut",
        )

    lva_mask = _flood_fill_from_seeds(lva_domain, lva_seeds)
    rva_mask = _flood_fill_from_seeds(rva_domain & ~lva_mask, rva_seeds)
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
            n_centerline_branches=n_branches,
            lva_voxels=n_lva,
            rva_voxels=n_rva,
            hemisphere_axis=hemisphere_axis,
            bifurcation_cut_k=int(cut_k),
            message="inferior VA branch too small after bifurcation flood-fill",
        )

    superior = basilar & ~(lva_mask | rva_mask)
    n_bas_keep = int(np.count_nonzero(superior))
    min_bas_keep = max(
        int(min_branch_voxels),
        int(round(_MIN_BASILAR_KEEP_FRAC * float(n_basilar))),
    )
    if n_bas_keep < min_bas_keep:
        log.info(
            f"vertebral split: basilar remnant too small after cut "
            f"({n_bas_keep} < {min_bas_keep}) — skip "
            "(likely false confluence on a single tube)"
        )
        return seg_np, replace(
            empty,
            bifurcation_ijk=junction,
            n_centerline_branches=n_branches,
            lva_voxels=n_lva,
            rva_voxels=n_rva,
            basilar_voxels=n_bas_keep,
            hemisphere_axis=hemisphere_axis,
            bifurcation_cut_k=int(cut_k),
            message="basilar remnant too small (false confluence guard)",
        )

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
        junction_degree=3,  # centerline Y (trunk split + side branch)
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
