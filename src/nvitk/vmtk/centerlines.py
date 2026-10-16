"""VMTK centerlines: seed-to-seed extraction, automatic networks, branches and geometry.

All coordinates are in the frame of the input surface (millimetres for every caller in nvitk).
Centerlines carry VMTK's ``MaximumInscribedSphereRadius`` point array (the Voronoi-sphere radius,
i.e. the distance from the centerline to the wall).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ._core import (
    cell_array,
    closest_surface_point,
    line_point_ids,
    point_array,
    points_of,
    polylines_of,
    run_script,
)

RADIUS_ARRAY = "MaximumInscribedSphereRadius"


def centerlines_between_points(
    surface: Any,
    source: Sequence[float],
    targets: Sequence[Sequence[float]] | Sequence[float],
    *,
    resampling_step: float | None = None,
    append_endpoints: bool = True,
    geometry: bool = True,
) -> Any:
    """VMTK centerlines from one *source* point to each of *targets* (one line per target).

    The seeds are snapped to the nearest surface vertex and passed to ``vmtkCenterlines`` with the
    ``pointlist`` selector, then ``vmtkCenterlineGeometry`` (smoothing off) adds ``Curvature``,
    ``Torsion``, ``Tortuosity`` and ``Length`` — exactly cow_morpho's call, which the eICAB
    morphometrics profile was tuned with.
    """
    src = closest_surface_point(surface, source)
    tgt_list = np.asarray(targets, dtype=float).reshape(-1, 3)
    tgts = np.vstack([closest_surface_point(surface, t) for t in tgt_list])
    members: dict[str, Any] = {
        "Surface": surface,
        "SeedSelectorName": "pointlist",
        "SourcePoints": [float(v) for v in src],
        "TargetPoints": [float(v) for v in tgts.ravel()],
        "AppendEndPoints": 1 if append_endpoints else 0,
    }
    if resampling_step is not None and resampling_step > 0:
        members.update(Resampling=1, ResamplingStepLength=float(resampling_step))
    cl = run_script("vmtkCenterlines", **members).Centerlines
    if cl is None or cl.GetNumberOfPoints() == 0:
        raise RuntimeError("vmtkCenterlines returned an empty result.")
    if not geometry:
        return cl
    geom = centerline_geometry(cl)
    return geom if geom.GetNumberOfPoints() > 0 else cl


def centerline_geometry(
    centerlines: Any,
    *,
    smoothing: bool = False,
    iterations: int = 100,
    factor: float = 0.1,
) -> Any:
    """``vmtkCenterlineGeometry``: per-point ``Curvature`` / ``Torsion`` / Frenet frame, per-line
    ``Length`` / ``Tortuosity`` (cell data). Smoothing is off by default (cow parity)."""
    return run_script(
        "vmtkCenterlineGeometry",
        Centerlines=centerlines,
        LineSmoothing=1 if smoothing else 0,
        OutputSmoothedLines=1 if smoothing else 0,
        NumberOfSmoothingIterations=int(iterations) if smoothing else 0,
        SmoothingFactor=float(factor) if smoothing else 0.0,
        FrenetTangentArrayName="FrenetTangent",
        FrenetNormalArrayName="FrenetNormal",
        FrenetBinormalArrayName="FrenetBinormal",
    ).Centerlines


def centerline_attributes(centerlines: Any) -> Any:
    """``vmtkCenterlineAttributes``: ``Abscissas`` (arc length) and parallel-transport ``ParallelTransportNormals``."""
    return run_script("vmtkCenterlineAttributes", Centerlines=centerlines).Centerlines


def smooth_centerlines(centerlines: Any, *, iterations: int = 100, factor: float = 0.1) -> Any:
    """``vmtkCenterlineSmoothing`` (Laplacian, endpoints fixed)."""
    return run_script(
        "vmtkCenterlineSmoothing",
        Centerlines=centerlines,
        NumberOfSmoothingIterations=int(iterations),
        SmoothingFactor=float(factor),
    ).Centerlines


def resample_centerlines(centerlines: Any, step: float) -> Any:
    """``vmtkCenterlineResampling`` at a fixed arc-length *step*."""
    return run_script("vmtkCenterlineResampling", Centerlines=centerlines, Length=float(step)).Centerlines


# ──────────────────────────────────────────────────────────────────────────────
# Networks and branches
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class VmtkBranch:
    """One branch of a split centerline network (``vmtkBranchExtractor`` group)."""

    group_id: int
    points: np.ndarray
    radius: np.ndarray
    #: True for the bifurcation pieces VMTK blanks out between branches.
    blanking: bool = False
    centerline_ids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


@dataclass
class VmtkNetwork:
    """A VMTK centerline network: the raw centerlines plus their branch split."""

    centerlines: Any
    split: Any
    branches: list[VmtkBranch]
    source: np.ndarray
    targets: np.ndarray

    @property
    def bifurcations(self) -> list[VmtkBranch]:
        return [b for b in self.branches if b.blanking]

    @property
    def vessel_branches(self) -> list[VmtkBranch]:
        return [b for b in self.branches if not b.blanking]

    def joined_branches(self, reach: float | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """``(points, radius)`` of each vessel branch, every end that meets a bifurcation piece
        extended to that bifurcation's centre.

        VMTK blanks a stretch of each centerline at a bifurcation, so its branches stop short of
        one another; joined, the branches of one bifurcation share an end point. An end meets a
        piece when it lies within *reach* mm of it (default: the branch's median radius, ≥ 1 mm).
        """
        # Every centerline crosses a bifurcation through its own blanked cell: match against all of
        # them, each standing for its group's centre.
        cells = self._blanked_cells()
        centres = {gid: (np.vstack([c for g, c, _ in cells if g == gid]).mean(axis=0),
                         float(np.median(np.concatenate([r for g, _, r in cells if g == gid]))))
                   for gid in {g for g, _, _ in cells}}
        out = []
        for b in self.vessel_branches:
            pts = np.asarray(b.points, dtype=float)
            rad = np.asarray(b.radius, dtype=float)
            if len(pts) < 2:
                continue
            tol = float(reach) if reach is not None else max(1.0, float(np.nanmedian(rad)))
            for end in (0, -1):
                if not cells:
                    break
                dist = [float(np.min(np.linalg.norm(c - pts[end], axis=1))) for _, c, _ in cells]
                k = int(np.argmin(dist))
                if dist[k] > tol:
                    continue
                centre, r = centres[cells[k][0]]
                if end == 0:
                    pts, rad = np.vstack([centre, pts]), np.r_[r, rad]
                else:
                    pts, rad = np.vstack([pts, centre]), np.r_[rad, r]
            out.append((pts, rad))
        return out

    def _blanked_cells(self) -> list[tuple[int, np.ndarray, np.ndarray]]:
        """``(group_id, points, radius)`` of every blanked cell of the split centerlines."""
        groups = cell_array(self.split, "GroupIds", default=0).astype(int)
        blanked = cell_array(self.split, "Blanking", default=0).astype(bool)
        lines = polylines_of(self.split, (RADIUS_ARRAY,))
        return [(int(g), line["points"], line[RADIUS_ARRAY])
                for g, is_blank, line in zip(groups, blanked, lines) if is_blank and len(line["points"])]

    @property
    def bifurcation_centres(self) -> np.ndarray:
        """One point per bifurcation: the centre of its blanked cells."""
        cells = self._blanked_cells()
        pts = [np.vstack([c for g, c, _ in cells if g == gid]).mean(axis=0) for gid in sorted({g for g, _, _ in cells})]
        return np.asarray(pts, dtype=float) if pts else np.empty((0, 3))


def split_branches(centerlines: Any) -> tuple[Any, list[VmtkBranch]]:
    """``vmtkBranchExtractor``: group overlapping centerlines into branches + bifurcation pieces.

    Returns the split polydata (cell arrays ``GroupIds``, ``CenterlineIds``, ``TractIds``,
    ``Blanking``) and one :class:`VmtkBranch` per group, represented by its longest cell.
    """
    split = run_script(
        "vmtkBranchExtractor",
        Centerlines=centerlines,
        RadiusArrayName=RADIUS_ARRAY,
        GroupIdsArrayName="GroupIds",
        CenterlineIdsArrayName="CenterlineIds",
        TractIdsArrayName="TractIds",
        BlankingArrayName="Blanking",
    ).Centerlines
    pts = points_of(split)
    radius = point_array(split, RADIUS_ARRAY)
    groups = cell_array(split, "GroupIds", default=0).astype(int)
    blanking = cell_array(split, "Blanking", default=0).astype(int)
    cl_ids = cell_array(split, "CenterlineIds", default=0).astype(int)
    cells = line_point_ids(split)
    by_group: dict[int, list[int]] = {}
    for c, _ids in enumerate(cells):
        by_group.setdefault(int(groups[c]), []).append(c)
    branches = []
    for gid, members in sorted(by_group.items()):
        best = max(members, key=lambda c: len(cells[c]))
        ids = cells[best]
        branches.append(
            VmtkBranch(
                group_id=int(gid),
                points=pts[ids],
                radius=radius[ids],
                blanking=bool(blanking[best]),
                centerline_ids=tuple(sorted({int(cl_ids[c]) for c in members})),
            )
        )
    return split, branches


def mask_endpoints(
    mask: np.ndarray,
    spacing: Sequence[float],
    *,
    prune_spur_mm: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Skeleton endpoints of a binary mask, in ``index × spacing`` millimetres, and the inscribed
    radius (EDT maximum within 2 mm) near each — the seeds for :func:`centerline_network`."""
    from scipy import ndimage as ndi

    from nvitk.measure.morpho.skeleton import prune_short_terminal_spurs, skeleton_graph, skeletonize_mask

    spacing = np.asarray(spacing, dtype=float)
    m = np.asarray(mask).astype(bool)
    skel = skeletonize_mask(m).astype(bool)
    if prune_spur_mm > 0:
        skel = prune_short_terminal_spurs(skel, float(prune_spur_mm), spacing=spacing).astype(bool)
    pts, _neighbors, degree = skeleton_graph(skel)
    if len(pts) == 0:
        return np.empty((0, 3)), np.empty(0)
    ends = pts[np.asarray(degree) == 1]
    if len(ends) == 0:  # a closed loop: pick the two most distant skeleton voxels
        d = np.linalg.norm((pts[:, None, :] - pts[None, :, :]) * spacing, axis=2)
        i, j = np.unravel_index(int(np.argmax(d)), d.shape)
        ends = pts[[i, j]]
    edt = ndi.distance_transform_edt(m, sampling=spacing)
    sk_mm = pts.astype(float) * spacing
    sk_r = edt[tuple(pts.T)]
    radii = []
    for e in ends.astype(float) * spacing:
        near = np.linalg.norm(sk_mm - e, axis=1) <= 2.0
        radii.append(float(sk_r[near].max()) if near.any() else float(edt[tuple(np.round(e / spacing).astype(int))]))
    return ends.astype(float) * spacing, np.asarray(radii, dtype=float)


def centerline_network(
    surface: Any,
    endpoints: np.ndarray,
    *,
    source_index: int | None = None,
    endpoint_radii: np.ndarray | None = None,
    resampling_step: float | None = None,
) -> VmtkNetwork:
    """Centerlines from one endpoint to all the others, split into branches.

    *endpoints* (``(N, 3)``, surface frame) are typically :func:`mask_endpoints`. The source is
    *source_index*, else the endpoint with the largest *endpoint_radii* (the widest end is the
    inlet), else the first one. (``vmtkcenterlinesnetwork`` is not used: it opens the surface at a
    random cell, which adds a spurious endpoint, and fails on numpy 2 for single-branch trees.)
    """
    ends = np.asarray(endpoints, dtype=float).reshape(-1, 3)
    if len(ends) < 2:
        raise ValueError("A centerline network needs at least two endpoints.")
    if source_index is None:
        source_index = int(np.argmax(endpoint_radii)) if endpoint_radii is not None else 0
    src = ends[int(source_index)]
    tgts = np.delete(ends, int(source_index), axis=0)
    cl = centerlines_between_points(surface, src, tgts, resampling_step=resampling_step)
    if len(tgts) >= 2:
        split, branches = split_branches(cl)
    else:
        split = cl
        branches = [
            VmtkBranch(group_id=0, points=points_of(cl), radius=point_array(cl, RADIUS_ARRAY), centerline_ids=(0,))
        ]
    return VmtkNetwork(centerlines=cl, split=split, branches=branches, source=src, targets=tgts)


# ──────────────────────────────────────────────────────────────────────────────
# Full networks: every branch, loop and fragment
# ──────────────────────────────────────────────────────────────────────────────

#: How much of a vessel tree a VMTK network covers (see :func:`full_centerline_network`).
NETWORK_COVERAGES: tuple[str, ...] = ("full", "tree")


@dataclass
class FullCenterlineNetwork:
    """Every centerline of a vessel mask: all branches, the loops and each fragment."""

    branches: list[np.ndarray] = field(default_factory=list)
    radii: list[np.ndarray] = field(default_factory=list)
    #: Where each branch comes from: ``"tree"`` (the VMTK run from a component's widest end),
    #: ``"loop"`` (a VMTK run from another end, adding what the first one skipped), ``"gap"``
    #: (VMTK between the two ends of a still-uncovered skeleton branch) or ``"skeleton"`` (that
    #: skeleton branch re-centred on the lumen, where VMTK could not trace it).
    sources: list[str] = field(default_factory=list)
    #: Bifurcations and the points where added pieces join the network.
    junctions: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    #: Continuous centerlines (each VMTK line and every added piece), e.g. for rasterizing.
    paths: list[np.ndarray] = field(default_factory=list)
    n_components: int = 0
    n_vmtk_runs: int = 0


def _arc_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def _true_runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """``(start, end)`` index pairs (inclusive) of the runs of True in *flag*."""
    runs, start = [], None
    for i, v in enumerate(np.append(np.asarray(flag, dtype=bool), False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    return runs


def _uncovered_pieces(
    points: np.ndarray, radius: np.ndarray, tree: Any, voxel: float
) -> list[tuple[np.ndarray, np.ndarray, list[np.ndarray]]]:
    """The stretches of a centerline the accepted network (*tree*, a KD-tree of its points) does
    not run along yet, each joined to the network at its ends.

    A point is covered within max(1.5 voxels, half its radius). Each uncovered stretch is extended
    to the network point nearest its covered neighbours, so the piece meets the network; those
    meeting points are returned with it. Stretches shorter than max(3 voxels, 1 mm) are dropped —
    two centerlines through the same junction diverge that much — and so are stretches that stay
    within one radius (≥ 2 voxels) of the network on 80 % of their length: inside the same lumen,
    a second centerline drifting off the first is not a new branch.
    """
    pts = np.asarray(points, dtype=float)
    rad = np.asarray(radius, dtype=float)
    if tree is None:
        return [(pts, rad, [])] if _arc_length(pts) > 0 else []
    d, _ = tree.query(pts)
    tol = np.maximum(1.5 * voxel, 0.5 * np.where(np.isfinite(rad), rad, 0.0))
    pieces = []
    for a, b in _true_runs(d > tol):
        seg, seg_r, joins = pts[a:b + 1], rad[a:b + 1], []
        if a > 0:
            _, j = tree.query(pts[a - 1])
            seg, seg_r = np.vstack([tree.data[j], seg]), np.r_[rad[a - 1], seg_r]
            joins.append(tree.data[j])
        if b < len(pts) - 1:
            _, j = tree.query(pts[b + 1])
            seg, seg_r = np.vstack([seg, tree.data[j]]), np.r_[seg_r, rad[b + 1]]
            joins.append(tree.data[j])
        if _arc_length(seg) < max(3.0 * voxel, 1.0):
            continue
        inner, inner_r = pts[a:b + 1], rad[a:b + 1]
        d_in, _ = tree.query(inner)
        same_lumen = d_in <= np.maximum(2.0 * voxel, np.where(np.isfinite(inner_r), inner_r, 0.0))
        if float(np.mean(same_lumen)) >= 0.8:
            continue
        pieces.append((seg, seg_r, joins))
    return pieces


def _surface_regions(surface: Any, min_points: int = 20) -> list[Any]:
    """The connected pieces of *surface* (largest first), each as its own polydata."""
    import vtk

    probe = vtk.vtkPolyDataConnectivityFilter()
    probe.SetInputData(surface)
    probe.SetExtractionModeToAllRegions()
    probe.Update()
    sizes = [int(probe.GetRegionSizes().GetValue(i)) for i in range(probe.GetNumberOfExtractedRegions())]
    if len(sizes) <= 1:
        return [surface]
    regions = []
    for rid in np.argsort(sizes)[::-1]:
        if sizes[rid] < int(min_points):
            continue
        pick = vtk.vtkPolyDataConnectivityFilter()
        pick.SetInputData(surface)
        pick.SetExtractionModeToSpecifiedRegions()
        pick.AddSpecifiedRegion(int(rid))
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(pick.GetOutputPort())
        clean.Update()
        piece = vtk.vtkPolyData()
        piece.DeepCopy(clean.GetOutput())
        regions.append(piece)
    return regions


def full_centerline_network(
    surface: Any,
    mask: np.ndarray,
    spacing: Sequence[float],
    *,
    voxel_to_surface: np.ndarray | None = None,
    resampling_step: float | None = None,
    prune_spur_mm: float = 2.0,
    max_sources: int = 4,
    max_gap_runs: int = 40,
) -> FullCenterlineNetwork:
    """VMTK centerlines of **every** branch of a vessel mask, loops and fragments included.

    One source → all end points (:func:`centerline_network`) traces a shortest-path tree. It
    skips one segment of every loop (the circle of Willis is a ring) and reaches nothing outside
    the source's connected piece. Here, per connected piece of *surface* (the mask's skeleton
    voxels go to the piece whose wall is nearest — voxels touching only at an edge or corner are
    one mask component but two marching-cubes pieces):

    1. VMTK from the widest end point to all others, on that component's piece of *surface*;
    2. the skeleton branches the result does not run along (within max(1.5 voxels, 0.6 × radius)
       on 75 % of their length) are "uncovered"; while any are left, VMTK runs again from the
       unused end point nearest to them (up to *max_sources* runs), keeping only the new stretches;
    3. each branch still uncovered gets VMTK between its own two skeleton ends (up to
       *max_gap_runs* in total), accepted when it follows the skeleton branch (length ratio
       0.5–2, median distance ≤ max(2 voxels, the branch's median radius)); otherwise the skeleton
       branch is re-centred on the lumen (:func:`nvitk.morphology.centering.center_polyline`).

    *mask* is in voxel indices; *voxel_to_surface* (4 × 4, default ``diag(spacing)``) maps them
    into the surface frame (:func:`nvitk.morphology.centering.mask_surface` uses that default).
    Skeleton spurs shorter than *prune_spur_mm* are ignored. Every branch keeps the inscribed radius (VMTK's
    ``MaximumInscribedSphereRadius``, or the distance to the surface for re-centred ones).
    """
    from scipy import ndimage as ndi
    from scipy.spatial import cKDTree

    from nvitk.core.array import to_numpy
    from nvitk.core.backend import using
    from nvitk.measure.morpho.skeleton import prune_short_terminal_spurs, skeletonize_mask
    from nvitk.morphology.centering import center_polyline, inscribed_radius
    from nvitk.morphology.polyline_graph import branch_polylines_from_skeleton, skeleton_graph

    sp = np.asarray(spacing, dtype=float)[:3]
    aff = np.asarray(voxel_to_surface, dtype=float) if voxel_to_surface is not None else np.diag([*sp, 1.0])

    def to_surface(ijk: np.ndarray) -> np.ndarray:
        return np.asarray(ijk, dtype=float).reshape(-1, 3) @ aff[:3, :3].T + aff[:3, 3]

    voxel = float(np.mean(np.linalg.norm(aff[:3, :3], axis=0)))
    step = float(resampling_step) if resampling_step and resampling_step > 0 else 0.5 * voxel
    m = np.asarray(mask).astype(bool)
    out = FullCenterlineNetwork()
    if not m.any():
        return out

    edt = ndi.distance_transform_edt(m, sampling=sp)
    skel = skeletonize_mask(m).astype(bool)
    if prune_spur_mm > 0:
        skel = prune_short_terminal_spurs(skel, float(prune_spur_mm), spacing=sp).astype(bool)
    all_coords = np.argwhere(skel)
    if len(all_coords) < 2:
        return out
    regions = _surface_regions(surface)
    if len(regions) == 1:
        owner = np.zeros(len(all_coords), dtype=int)
    else:
        wall = np.column_stack([cKDTree(points_of(r)).query(to_surface(all_coords))[0] for r in regions])
        owner = np.argmin(wall, axis=1)
    hi = np.asarray(m.shape) - 1
    accepted: list[np.ndarray] = []
    junctions: list[np.ndarray] = []
    gap_budget = int(max_gap_runs)

    def add(points: np.ndarray, radius: np.ndarray, source: str) -> None:
        out.branches.append(np.asarray(points, dtype=float))
        out.radii.append(np.asarray(radius, dtype=float))
        out.sources.append(source)
        accepted.append(np.asarray(points, dtype=float))

    def accepted_tree() -> Any:
        return cKDTree(np.vstack(accepted)) if accepted else None

    for k, region in enumerate(regions):
        coords = all_coords[owner == k]
        if len(coords) < 2:
            continue
        out.n_components += 1
        with using("cpu"):
            chains_vox = [np.asarray(to_numpy(c), dtype=float)
                          for c in branch_polylines_from_skeleton(coords.astype(np.float32), min_points=2)]
            _nodes, _adj, deg = skeleton_graph(coords.astype(np.float32))
        chains = []
        for ch in chains_vox:
            idx = np.clip(np.round(ch).astype(int), 0, hi)
            pts = to_surface(ch)
            if _arc_length(pts) >= max(3.0 * voxel, 1.0):
                chains.append((pts, edt[tuple(idx.T)]))
        if not chains:
            continue

        def uncovered(tree: Any) -> list[tuple[np.ndarray, np.ndarray]]:
            if tree is None:
                return list(chains)
            todo = []
            for pts, r in chains:
                d, _ = tree.query(pts)
                if float(np.mean(d <= np.maximum(1.5 * voxel, 0.6 * r))) < 0.75:
                    todo.append((pts, r))
            return todo

        # 1–2. VMTK from end points until every skeleton branch is covered.
        ends_vox = np.asarray([n for n, d in deg.items() if d <= 1], dtype=float).reshape(-1, 3)
        tree = accepted_tree()
        if len(ends_vox) >= 2:
            ends = to_surface(ends_vox)
            sk_tree = cKDTree(coords * sp)
            sk_r = edt[tuple(coords.T)]
            end_r = np.asarray([sk_r[sk_tree.query_ball_point(e * sp, 2.0)].max() for e in ends_vox])
            used: list[int] = []
            first = True
            for _run in range(max(1, int(max_sources))):
                todo = uncovered(tree) if not first else chains
                if not todo:
                    break
                if first:
                    src = int(np.argmax(end_r))
                else:
                    free = [i for i in range(len(ends)) if i not in used]
                    if not free:
                        break
                    near = cKDTree(np.vstack([p for p, _ in todo]))
                    src = min(free, key=lambda i: float(near.query(ends[i])[0]))
                used.append(src)
                try:
                    net = centerline_network(region, ends, source_index=src, endpoint_radii=end_r,
                                             resampling_step=step)
                except Exception:  # noqa: BLE001 — a failed run leaves the branches to the next steps
                    continue
                out.n_vmtk_runs += 1
                if first:
                    for pts, rad in net.joined_branches():
                        add(pts, rad, "tree")
                    junctions.extend(net.bifurcation_centres)
                    out.paths.extend(line["points"] for line in polylines_of(net.centerlines))
                    first = False
                else:
                    # Only stretches joining the network at both ends: a loop segment. An open-ended
                    # stretch is where this run's source tip parts from the first run's line.
                    for line in polylines_of(net.centerlines, (RADIUS_ARRAY,)):
                        for pts, rad, joins in _uncovered_pieces(line["points"], line[RADIUS_ARRAY], tree, voxel):
                            if len(joins) == 2:
                                add(pts, rad, "loop")
                                out.paths.append(pts)
                                junctions.extend(joins)
                                tree = accepted_tree()
                tree = accepted_tree()

        # 3. Whatever is left: VMTK between the branch's own ends, else the re-centred skeleton.
        for pts, r in uncovered(tree):
            piece, source = None, "skeleton"
            if gap_budget > 0 and _arc_length(pts) > 0 and float(np.linalg.norm(pts[0] - pts[-1])) > voxel:
                gap_budget -= 1
                try:
                    cl = centerlines_between_points(region, pts[0], [pts[-1]], resampling_step=step,
                                                    append_endpoints=False, geometry=False)
                    out.n_vmtk_runs += 1
                    q, qr = points_of(cl), point_array(cl, RADIUS_ARRAY)
                    ratio = _arc_length(q) / max(_arc_length(pts), 1e-9)
                    d, _ = cKDTree(pts).query(q)
                    if 0.5 <= ratio <= 2.0 and float(np.median(d)) <= max(2.0 * voxel, float(np.median(r))):
                        piece, source = (q, qr), "gap"
                except Exception:  # noqa: BLE001
                    piece = None
            if piece is None:
                centred = center_polyline(pts, region, iterations=3, smooth_mm=1.5 * voxel, sample_step_mm=step)
                piece = (centred, inscribed_radius(region, centred))
            for p2, r2, joins in _uncovered_pieces(piece[0], piece[1], tree, voxel):
                add(p2, r2, source)
                out.paths.append(p2)
                junctions.extend(joins)
            tree = accepted_tree()

    if junctions:
        jun = np.asarray(junctions, dtype=float).reshape(-1, 3)
        keep: list[np.ndarray] = []
        for p in jun:
            if all(float(np.linalg.norm(p - k)) > 1.5 * voxel for k in keep):
                keep.append(p)
        out.junctions = np.asarray(keep)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Surface ↔ centerline measurements
# ──────────────────────────────────────────────────────────────────────────────


def centerline_sections(surface: Any, centerlines: Any) -> tuple[Any, Any]:
    """``vmtkCenterlineSections``: per-centerline-point lumen section ``CenterlineSectionArea``,
    ``…MinSize`` / ``…MaxSize`` (min/max Feret), ``…Shape`` (min/max) and ``…Closed``.

    Returns ``(centerlines_with_arrays, sections_polydata)``.
    """
    s = run_script("vmtkCenterlineSections", Surface=surface, Centerlines=centerlines)
    return s.Centerlines, s.CenterlineSections


def distance_to_centerlines(surface: Any, centerlines: Any, *, use_radius: bool = False) -> np.ndarray:
    """``vmtkDistanceToCenterlines``: per-vertex distance from the wall to the centerline (or, with
    *use_radius*, normalised by the local inscribed radius)."""
    out = run_script(
        "vmtkDistanceToCenterlines",
        Surface=surface,
        Centerlines=centerlines,
        UseRadiusInformation=1 if use_radius else 0,
        RadiusArrayName=RADIUS_ARRAY,
        DistanceToCenterlinesArrayName="DistanceToCenterlines",
    ).Surface
    return point_array(out, "DistanceToCenterlines")


def branch_clip(surface: Any, split_centerlines: Any) -> np.ndarray:
    """``vmtkBranchClipper`` (non-interactive): the branch ``GroupIds`` of every surface vertex.

    The clipper returns the surface cut into branch pieces; each original vertex takes the group
    of the nearest clipped vertex.
    """
    from scipy.spatial import cKDTree

    clipped = run_script(
        "vmtkBranchClipper",
        Surface=surface,
        Centerlines=split_centerlines,
        GroupIdsArrayName="GroupIds",
        RadiusArrayName=RADIUS_ARRAY,
        BlankingArrayName="Blanking",
        Interactive=0,
    ).Surface
    groups = point_array(clipped, "GroupIds", default=-1)
    tree = cKDTree(points_of(clipped))
    _d, idx = tree.query(points_of(surface))
    return groups[idx].astype(int)


def bifurcation_geometry(split_centerlines: Any) -> list[dict[str, Any]]:
    """``vmtkBifurcationReferenceSystems`` + ``vmtkBifurcationVectors``: per-bifurcation origin,
    plane normal, and for each daughter its in-plane / out-of-plane angle (degrees)."""
    refs = run_script(
        "vmtkBifurcationReferenceSystems",
        Centerlines=split_centerlines,
        RadiusArrayName=RADIUS_ARRAY,
        BlankingArrayName="Blanking",
        GroupIdsArrayName="GroupIds",
    ).ReferenceSystems
    vecs = run_script(
        "vmtkBifurcationVectors",
        Centerlines=split_centerlines,
        ReferenceSystems=refs,
        RadiusArrayName=RADIUS_ARRAY,
        GroupIdsArrayName="GroupIds",
        CenterlineIdsArrayName="CenterlineIds",
        TractIdsArrayName="TractIds",
        BlankingArrayName="Blanking",
        NormalizeBifurcationVectors=1,
    ).BifurcationVectors
    origins = points_of(refs)
    normals = point_array_vectors(refs, "Normal")
    ref_groups = point_array(refs, "GroupIds", default=-1).astype(int)
    bif_group = point_array(vecs, "BifurcationGroupIds", default=-1).astype(int)
    member_group = point_array(vecs, "GroupIds", default=-1).astype(int)
    in_plane = np.degrees(point_array(vecs, "InPlaneBifurcationVectorAngles"))
    out_plane = np.degrees(point_array(vecs, "OutOfPlaneBifurcationVectorAngles"))
    orientation = point_array(vecs, "BifurcationVectorsOrientation", default=-1).astype(int)
    out = []
    for k, gid in enumerate(ref_groups):
        rows = np.flatnonzero(bif_group == gid)
        out.append({
            "group_id": int(gid),
            "origin": origins[k],
            "normal": normals[k] if len(normals) > k else np.full(3, np.nan),
            "branches": [
                {
                    "group_id": int(member_group[r]),
                    "role": "parent" if orientation[r] == 0 else "daughter",
                    "in_plane_angle_deg": float(in_plane[r]),
                    "out_of_plane_angle_deg": float(out_plane[r]),
                }
                for r in rows
            ],
        })
    return out


def point_array_vectors(poly: Any, name: str) -> np.ndarray:
    """A multi-component point array as ``(N, k)`` (empty when absent)."""
    from vtk.util import numpy_support

    arr = None if poly is None else poly.GetPointData().GetArray(name)
    if arr is None:
        return np.empty((0, 3))
    return numpy_support.vtk_to_numpy(arr).astype(float)


__all__ = [
    "NETWORK_COVERAGES",
    "RADIUS_ARRAY",
    "FullCenterlineNetwork",
    "VmtkBranch",
    "VmtkNetwork",
    "bifurcation_geometry",
    "branch_clip",
    "centerline_attributes",
    "centerline_geometry",
    "centerline_network",
    "centerline_sections",
    "centerlines_between_points",
    "distance_to_centerlines",
    "full_centerline_network",
    "mask_endpoints",
    "resample_centerlines",
    "smooth_centerlines",
    "split_branches",
]
