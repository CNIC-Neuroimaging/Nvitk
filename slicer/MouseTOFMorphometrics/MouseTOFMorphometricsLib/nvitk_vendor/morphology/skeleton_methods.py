# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM nvitk — DO NOT EDIT.
# Source: src/nvitk/morphology/skeleton_methods.py
# Regenerate: python MouseTOFMorphometricsLib/vendor_sync.py
# The only change from upstream is the root package rename nvitk -> nvitk_vendor.
# ─────────────────────────────────────────────────────────────────────────
"""Skeletonization methods: VMTK, native (scikit-image + re-centring), ray-cast (experimental).

Every skeleton / centerline tool in nvitk_vendor picks one of :data:`SKELETON_METHODS`:

``vmtk``
    VMTK centerlines on the mask's surface (Voronoi-based, sub-voxel, with the maximum inscribed
    sphere radius), between the voxel-skeleton end points (:mod:`nvitk_vendor.vmtk.centerlines`). The GUI
    default.
``native``
    scikit-image's 3D thinning (the historical behaviour); branch polylines are re-centred on the
    lumen when a network is requested (:mod:`nvitk_vendor.morphology.centering`). The library default, so
    the validated 4D-flow qvtpy stages keep their outputs.
``raycast``
    EXPIGEO ray probing of the mask surface (:mod:`nvitk_vendor.morphology.raycast_skeleton`).
    Experimental: offered everywhere, used by default nowhere.

:func:`centerline_network` returns sub-voxel branch polylines (voxel coordinates) with radii;
:func:`method_skeleton` rasterizes them into a 1-voxel skeleton mask so every voxel-graph tool
(junctions, polyline extraction, trees) works unchanged on any method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

SKELETON_METHODS: tuple[str, ...] = ("vmtk", "native", "raycast")
#: GUI choice labels (``normalize_skeleton_method`` accepts either form).
SKELETON_METHOD_LABELS: dict[str, str] = {
    "vmtk": "vmtk",
    "native": "native (skimage)",
    "raycast": "raycast (experimental)",
}
SKELETON_METHOD_CHOICES: tuple[str, ...] = tuple(SKELETON_METHOD_LABELS.values())
DEFAULT_GUI_METHOD = "vmtk"
DEFAULT_LIBRARY_METHOD = "native"


def normalize_skeleton_method(method: str | None, *, default: str = DEFAULT_LIBRARY_METHOD) -> str:
    """``"vmtk"`` / ``"native"`` / ``"raycast"`` from a method name or GUI label."""
    token = str(method or "").strip().lower()
    if not token:
        return default
    token = token.split()[0]
    if token in ("skimage", "voxel"):
        token = "native"
    if token not in SKELETON_METHODS:
        raise ValueError(f"Unknown skeleton method {method!r}; expected one of {SKELETON_METHODS}.")
    return token


@dataclass
class CenterlineNetwork:
    """Branch polylines of one binary mask, in voxel coordinates, with inscribed radii (mm)."""

    method: str
    branches: list[np.ndarray] = field(default_factory=list)
    radii: list[np.ndarray] = field(default_factory=list)
    #: Junction points (voxel coordinates).
    junctions: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    #: Continuous root→terminal paths (voxel coordinates) when the method has them (``vmtk``: its
    #: source→target centerlines); the branches are pieces of these.
    paths: list[np.ndarray] = field(default_factory=list)

    @property
    def lengths_vox(self) -> list[float]:
        return [float(np.linalg.norm(np.diff(b, axis=0), axis=1).sum()) if len(b) > 1 else 0.0 for b in self.branches]


def _mask_and_spacing(mask: np.ndarray, spacing: Sequence[float] | None) -> tuple[np.ndarray, np.ndarray]:
    m = np.asarray(mask).astype(bool)
    if m.ndim != 3:
        raise ValueError("Skeletonization methods expect a 3D mask.")
    sp = np.asarray(spacing if spacing is not None else (1.0, 1.0, 1.0), dtype=float)[:3]
    return m, sp


def _vmtk_network(mask: np.ndarray, spacing: np.ndarray) -> CenterlineNetwork:
    from nvitk_vendor.morphology.centering import mask_surface
    from nvitk_vendor.vmtk.centerlines import centerline_network as vmtk_network
    from nvitk_vendor.vmtk.centerlines import mask_endpoints

    from nvitk_vendor.vmtk._core import polylines_of

    ends, radii = mask_endpoints(mask, spacing)
    if len(ends) < 2:
        return CenterlineNetwork("vmtk")
    surface = mask_surface(mask, spacing)
    net = vmtk_network(surface, ends, endpoint_radii=radii, resampling_step=0.5 * float(spacing.min()))
    joined = net.joined_branches()
    branches = [pts / spacing for pts, _ in joined]
    rads = [rad for _, rad in joined]
    junctions = net.bifurcation_centres / spacing
    paths = [line["points"] / spacing for line in polylines_of(net.centerlines) if len(line["points"]) >= 2]
    return CenterlineNetwork("vmtk", branches, rads, junctions, paths)


def _native_network(mask: np.ndarray, spacing: np.ndarray, *, center: bool) -> CenterlineNetwork:
    from nvitk_vendor.morphology.centerline import skeletonize_binary
    from nvitk_vendor.morphology.polyline_graph import branch_polylines_from_skeleton, junction_nodes_from_skeleton
    from nvitk_vendor.core.array import to_numpy

    sk = to_numpy(skeletonize_binary(mask)).astype(bool)
    coords = np.argwhere(sk).astype(np.float32)
    if len(coords) < 2:
        return CenterlineNetwork("native")
    polys = [np.asarray(to_numpy(p), dtype=float) for p in branch_polylines_from_skeleton(coords, min_points=2)]
    try:
        junctions = np.asarray(to_numpy(junction_nodes_from_skeleton(coords)), dtype=float).reshape(-1, 3)
    except Exception:  # noqa: BLE001
        junctions = np.empty((0, 3))
    if center and polys:
        from nvitk_vendor.morphology.centering import center_voxel_paths

        polys, rads = center_voxel_paths(polys, mask, spacing)
    else:
        from scipy import ndimage as ndi

        edt = ndi.distance_transform_edt(mask, sampling=spacing)
        rads = [edt[tuple(np.clip(np.round(p).astype(int), 0, np.array(mask.shape) - 1).T)] for p in polys]
    return CenterlineNetwork("native", polys, rads, junctions)


def _raycast_network(mask: np.ndarray, spacing: np.ndarray) -> CenterlineNetwork:
    from nvitk_vendor.morphology.raycast_skeleton import raycast_skeleton_from_mask

    skel, _surface = raycast_skeleton_from_mask(mask, spacing)
    branches, rads = [], []
    for chain in skel.branches():
        if len(chain) >= 2:
            branches.append(skel.vertices[chain] / spacing)
            rads.append(skel.radius[chain])
    junctions = skel.vertices[skel.degree >= 3] / spacing
    return CenterlineNetwork("raycast", branches, rads, junctions)


def centerline_network(
    mask: np.ndarray,
    *,
    method: str = DEFAULT_LIBRARY_METHOD,
    spacing: Sequence[float] | None = None,
    center_native: bool = True,
) -> CenterlineNetwork:
    """Branch polylines (voxel coordinates) of a binary 3D mask with the given *method*."""
    m, sp = _mask_and_spacing(mask, spacing)
    meth = normalize_skeleton_method(method)
    if not m.any():
        return CenterlineNetwork(meth)
    if meth == "vmtk":
        return _vmtk_network(m, sp)
    if meth == "raycast":
        return _raycast_network(m, sp)
    return _native_network(m, sp, center=center_native)


def rasterize_polylines(shape: Sequence[int], polylines: Sequence[np.ndarray], *, step: float = 0.25) -> np.ndarray:
    """Boolean volume with every polyline (voxel coordinates) drawn as a 26-connected voxel chain."""
    out = np.zeros(tuple(int(s) for s in shape), dtype=bool)
    hi = np.asarray(out.shape) - 1
    for poly in polylines:
        p = np.asarray(poly, dtype=float).reshape(-1, 3)
        if len(p) == 0:
            continue
        if len(p) == 1:
            dense = p
        else:
            seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            ss = np.arange(0.0, s[-1] + step, step)
            dense = np.column_stack([np.interp(ss, s, p[:, k]) for k in range(3)])
        idx = np.clip(np.round(dense).astype(int), 0, hi)
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def method_skeleton(
    mask: np.ndarray,
    *,
    method: str = DEFAULT_LIBRARY_METHOD,
    spacing: Sequence[float] | None = None,
) -> np.ndarray:
    """1-voxel skeleton mask of a binary 3D mask with *method*.

    ``native`` is scikit-image's thinning as-is; ``vmtk`` / ``raycast`` rasterize their centerline
    network (VMTK: its continuous source→target centerlines), clipped to the mask and thinned again
    so junctions stay one voxel wide.
    """
    from skimage.morphology import skeletonize

    m, sp = _mask_and_spacing(mask, spacing)
    meth = normalize_skeleton_method(method)
    if meth == "native" or not m.any():
        return skeletonize(m.astype(np.uint8)).astype(bool)
    net = centerline_network(m, method=meth, spacing=sp)
    if not net.branches:
        return skeletonize(m.astype(np.uint8)).astype(bool)
    drawn = rasterize_polylines(m.shape, net.paths or net.branches)
    return skeletonize((drawn & m).astype(np.uint8)).astype(bool) if (drawn & m).any() else drawn


__all__ = [
    "DEFAULT_GUI_METHOD",
    "DEFAULT_LIBRARY_METHOD",
    "SKELETON_METHODS",
    "SKELETON_METHOD_CHOICES",
    "SKELETON_METHOD_LABELS",
    "CenterlineNetwork",
    "centerline_network",
    "method_skeleton",
    "normalize_skeleton_method",
    "rasterize_polylines",
]
