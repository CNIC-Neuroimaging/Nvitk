# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/vmtk/centerlines.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
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

    from nvitk_vendor.measure.morpho.skeleton import prune_short_terminal_spurs, skeleton_graph, skeletonize_mask

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
    "RADIUS_ARRAY",
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
    "mask_endpoints",
    "resample_centerlines",
    "smooth_centerlines",
    "split_branches",
]
