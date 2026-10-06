"""Moving surfaces: affine transforms, centring, principal-axis and ICP alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

import numpy as np

from nvitk.meshlab.convert import points_of
from nvitk.types import Mesh, PointCloud

_S = TypeVar("_S", Mesh, PointCloud)


def _with_points(obj: _S, points: np.ndarray, *, linear: np.ndarray | None = None) -> _S:
    """*obj* moved to *points*; normals in point data are rotated with *linear*."""
    if isinstance(obj, Mesh):
        out = obj.with_vertices(points)
        if linear is not None and np.linalg.det(linear) < 0:
            out.faces = out.faces[:, ::-1].copy()  # a mirror reverses the winding
    else:
        out = obj.copy()
        out.points = np.asarray(points, dtype=float)
    normals = out.point_data.get("normals")
    if normals is not None and linear is not None:
        n = normals @ np.linalg.inv(linear)  # normals transform by the inverse transpose
        norm = np.linalg.norm(n, axis=1, keepdims=True)
        out.point_data["normals"] = np.divide(n, norm, out=np.zeros_like(n), where=norm > 0)
    return out


def apply_affine(obj: _S, affine: np.ndarray) -> _S:
    """Map every vertex / point through a 4x4 affine."""
    aff = np.asarray(affine, dtype=float)
    if aff.shape != (4, 4):
        raise ValueError("affine must be 4x4.")
    pts = points_of(obj)
    moved = pts @ aff[:3, :3].T + aff[:3, 3]
    return _with_points(obj, moved, linear=aff[:3, :3])


def rotation_matrix(angles_deg: Sequence[float]) -> np.ndarray:
    """3x3 rotation about x, then y, then z (degrees, right-handed)."""
    ax, ay, az = (np.radians(float(a)) for a in angles_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ ry @ rx


def compose_affine(
    *,
    translate: Sequence[float] = (0.0, 0.0, 0.0),
    rotate_deg: Sequence[float] = (0.0, 0.0, 0.0),
    scale: Sequence[float] | float = 1.0,
    centre: Sequence[float] | None = None,
) -> np.ndarray:
    """4x4 affine: scale and rotate about *centre* (origin by default), then translate."""
    s = np.broadcast_to(np.asarray(scale, dtype=float), (3,))
    lin = rotation_matrix(rotate_deg) @ np.diag(s)
    c = np.zeros(3) if centre is None else np.asarray(centre, dtype=float)
    aff = np.eye(4)
    aff[:3, :3] = lin
    aff[:3, 3] = c - lin @ c + np.asarray(translate, dtype=float)
    return aff


def transform_surface(
    obj: _S,
    *,
    translate: Sequence[float] = (0.0, 0.0, 0.0),
    rotate_deg: Sequence[float] = (0.0, 0.0, 0.0),
    scale: Sequence[float] | float = 1.0,
    about_centroid: bool = True,
) -> _S:
    """Scale / rotate (about the centroid by default) and translate."""
    centre = points_of(obj).mean(axis=0) if about_centroid and len(points_of(obj)) else None
    return apply_affine(obj, compose_affine(translate=translate, rotate_deg=rotate_deg, scale=scale, centre=centre))


def centre_at_origin(obj: _S) -> _S:
    """Translate so the centroid sits at the origin."""
    c = points_of(obj).mean(axis=0)
    return apply_affine(obj, compose_affine(translate=-c))


def align_principal_axes(obj: _S) -> tuple[_S, np.ndarray]:
    """Rotate the principal axes onto x, y, z (largest spread on x), centred.

    Returns ``(aligned, affine)``. Axis signs are chosen so the result is a proper
    rotation and the third moment along each axis is positive (stable under
    re-meshing).
    """
    pts = points_of(obj)
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    proj = (pts - centre) @ vt.T
    signs = np.sign((proj ** 3).sum(axis=0))
    signs[signs == 0] = 1.0
    rot = vt * signs[:, None]
    if np.linalg.det(rot) < 0:
        rot[2] *= -1
    aff = np.eye(4)
    aff[:3, :3] = rot
    aff[:3, 3] = -rot @ centre
    return apply_affine(obj, aff), aff


@dataclass(frozen=True)
class IcpResult:
    """ICP outcome: the 4x4 moving→fixed affine and the fit."""

    affine: np.ndarray
    rms: float
    iterations: int
    converged: bool


def _best_fit(src: np.ndarray, dst: np.ndarray, *, scaling: bool) -> np.ndarray:
    """Least-squares rigid (or similarity) transform src → dst (Umeyama)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - mu_s, dst - mu_d
    cov = b.T @ a / len(src)
    u, d, vt = np.linalg.svd(cov)
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s[2, 2] = -1
    rot = u @ s @ vt
    scale = (np.trace(np.diag(d) @ s) / (a ** 2).sum(axis=1).mean()) if scaling else 1.0
    aff = np.eye(4)
    aff[:3, :3] = scale * rot
    aff[:3, 3] = mu_d - scale * rot @ mu_s
    return aff


def icp(
    moving: Mesh | PointCloud | np.ndarray,
    fixed: Mesh | PointCloud | np.ndarray,
    *,
    iterations: int = 50,
    tolerance: float = 1e-6,
    scaling: bool = False,
    max_points: int = 20000,
    initial: np.ndarray | None = None,
    reject_quantile: float = 0.95,
    seed: int = 0,
) -> IcpResult:
    """Iterative closest point: the affine aligning *moving* onto *fixed*.

    Point-to-point with a k-d tree, rigid (or similarity with *scaling*), on at
    most *max_points* random points of each set; pairs farther than the
    *reject_quantile* of the distances are ignored each round (outlier
    rejection). Start from *initial* (e.g. a principal-axes guess).
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    src_all = points_of(moving)
    dst = points_of(fixed)
    if len(src_all) > max_points:
        src_all = src_all[rng.choice(len(src_all), max_points, replace=False)]
    if len(dst) > max_points * 2:
        dst = dst[rng.choice(len(dst), max_points * 2, replace=False)]
    tree = cKDTree(dst)
    total = np.eye(4) if initial is None else np.asarray(initial, dtype=float).copy()
    src = src_all @ total[:3, :3].T + total[:3, 3]
    prev = np.inf
    rms = np.inf
    converged = False
    it = 0
    for it in range(1, int(iterations) + 1):
        dist, idx = tree.query(src)
        cut = np.quantile(dist, reject_quantile) if 0 < reject_quantile < 1 else np.inf
        keep = dist <= cut
        step = _best_fit(src[keep], dst[idx[keep]], scaling=scaling)
        src = src @ step[:3, :3].T + step[:3, 3]
        total = step @ total
        rms = float(np.sqrt((dist[keep] ** 2).mean()))
        if abs(prev - rms) < tolerance:
            converged = True
            break
        prev = rms
    dist, _ = tree.query(src)
    return IcpResult(affine=total, rms=float(np.sqrt((dist ** 2).mean())), iterations=it, converged=converged)


def align_icp(moving: _S, fixed: Mesh | PointCloud, **kwargs: object) -> tuple[_S, IcpResult]:
    """Align *moving* to *fixed* with :func:`icp`; returns the moved copy and the fit."""
    res = icp(moving, fixed, **kwargs)  # type: ignore[arg-type]
    return apply_affine(moving, res.affine), res


__all__ = [
    "IcpResult",
    "align_icp",
    "align_principal_axes",
    "apply_affine",
    "centre_at_origin",
    "compose_affine",
    "icp",
    "rotation_matrix",
    "transform_surface",
]
