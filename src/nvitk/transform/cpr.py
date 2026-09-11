"""Curved planar reformation: resample a volume along a centerline so a vessel lies flat.

A vessel curves, so no single plane through a volume shows one along its length.
Straightened CPR (Kanitsar et al., *IEEE Vis* 2002) resamples the volume in a frame
that travels with the centerline: one row per station, spaced uniformly in arc
length, each row a ray cut across the vessel. The result is an image whose vertical
axis is millimetres along the vessel, so a lesion's length and the calibre either
side of it can be read straight off it.

The frame is the part that decides whether the result is usable. The textbook
Frenet-Serret frame is defined by the curvature vector, so its normal flips through
an inflection and spins where curvature approaches zero — the reformation twists and
tears. :func:`rotation_minimizing_frames` instead carries one frame along the curve
by double reflection (Wang et al., *ACM TOG* 2008), which is stable, linear-time and
needs no second derivative.

Like :mod:`nvitk.transform.oblique`, whose sampler this generalises, the module uses
``nvitk.core.backend.setup`` so ``np`` and ``ndi`` follow the active backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nvitk.core.array import to_numpy
from nvitk.core.backend import setup, using
from nvitk.core import as_backend_array

setup(globals())

#: Below this the two ends of a segment are treated as the same point.
_EPS = 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Frames
# ──────────────────────────────────────────────────────────────────────────────
def _unit(vectors: Any) -> Any:
    """Row-wise unit vectors on the active backend; a zero row stays zero, not NaN."""
    arr = as_backend_array(vectors).astype(float, copy=False)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    safe = np.where(norms > _EPS, norms, 1.0)
    return np.where(norms > _EPS, arr / safe, 0.0)


def _seed_normal(tangent: Any) -> Any:
    """Any unit vector perpendicular to *tangent*, for the first station only.

    Only the *starting* orientation is arbitrary — every later frame is carried
    from this one, so the choice sets the reformation's rotation offset and
    nothing else. The theta control exists precisely to rotate it.
    """
    t = _unit(as_backend_array(tangent).reshape(3))
    axis = np.eye(3)[int(np.argmin(np.abs(t)))]
    return _unit(np.cross(t, axis))


def rotation_minimizing_frames(
    points: Any,
    tangents: Any,
    *,
    seed_normal: Any = None,
) -> tuple[Any, Any]:
    """Carry an orthonormal frame along a curve with minimal twist.

    The double-reflection method of Wang et al. (*ACM TOG* 2008). The first
    reflection is in the plane bisecting the two *sample points* — not the two
    tangents; reflecting across the tangent difference instead lets the frame
    invert wherever the curve passes through an inflection, which is the failure
    the whole method exists to avoid.

    Parameters
    ----------
    points
        ``(N, 3)`` positions along the curve, in the same isotropic space as
        *tangents* (millimetres, not voxels).
    tangents
        ``(N, 3)`` unit tangents.
    seed_normal
        Orientation of the first frame. Defaults to an arbitrary perpendicular;
        pass one to keep a reformation stable across re-renders.

    Returns
    -------
    (u, v)
        Two ``(N, 3)`` arrays completing a right-handed frame with *tangents*.
    """
    # The walk is sequential and works on single 3-vectors: on CuPy every dot
    # product would be its own kernel launch, so this is one of the regions the
    # backend policy carves out for the host. Convert in, convert out.
    with using("cpu"):
        x = to_numpy(points).astype(float, copy=False).reshape(-1, 3)
        t = to_numpy(_unit(tangents)).astype(float, copy=False).reshape(-1, 3)
        n = int(min(x.shape[0], t.shape[0]))
        if n == 0:
            empty = np.zeros((0, 3), dtype=float)
            return as_backend_array(empty), as_backend_array(empty)
        x, t = x[:n], t[:n]

        u = np.zeros((n, 3), dtype=float)
        seed = to_numpy(_seed_normal(t[0])) if seed_normal is None else to_numpy(
            seed_normal
        ).astype(float, copy=False).reshape(3)
        u[0] = seed / max(float(np.linalg.norm(seed)), _EPS)
        # Guard a seed that is not perpendicular: project it onto the first plane.
        u[0] = u[0] - float(np.dot(u[0], t[0])) * t[0]
        norm0 = float(np.linalg.norm(u[0]))
        u[0] = u[0] / norm0 if norm0 > _EPS else to_numpy(_seed_normal(t[0]))

        for i in range(n - 1):
            # Reflection 1: across the plane bisecting points i and i+1.
            v1 = x[i + 1] - x[i]
            c1 = float(np.dot(v1, v1))
            if c1 <= _EPS:
                u[i + 1] = u[i]
                continue
            u_l = u[i] - (2.0 / c1) * float(np.dot(v1, u[i])) * v1
            t_l = t[i] - (2.0 / c1) * float(np.dot(v1, t[i])) * v1
            # Reflection 2: brings the transported tangent onto the real one.
            v2 = t[i + 1] - t_l
            c2 = float(np.dot(v2, v2))
            u[i + 1] = u_l if c2 <= _EPS else u_l - (2.0 / c2) * float(np.dot(v2, u_l)) * v2

        norms = np.linalg.norm(u, axis=1, keepdims=True)
        u = u / np.where(norms > _EPS, norms, 1.0)
        # Re-orthogonalise: the reflections keep u perpendicular to t analytically,
        # but a long curve accumulates enough float drift to matter.
        u = u - (u * t).sum(axis=1, keepdims=True) * t
        norms = np.linalg.norm(u, axis=1, keepdims=True)
        u = u / np.where(norms > _EPS, norms, 1.0)
        v = np.cross(t, u)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        v = v / np.where(norms > _EPS, norms, 1.0)

    return as_backend_array(u), as_backend_array(v)


def frame_twist(tangents: Any, u: Any) -> float:
    """Total rotation of *u* about the tangent, in radians — 0 for a perfect RMF.

    The measure the rotation-minimizing frame exists to keep small; used to compare
    it against a naive frame in tests.
    """
    # Sequential over pairs of stations, like the frame walk it measures.
    with using("cpu"):
        t = to_numpy(_unit(tangents)).astype(float, copy=False).reshape(-1, 3)
        uu = to_numpy(_unit(u)).astype(float, copy=False).reshape(-1, 3)
        total = 0.0
        for i in range(int(t.shape[0]) - 1):
            # Project the previous normal into the next plane and measure the
            # residual angle: that is the twist the step introduced.
            projected = uu[i] - float(np.dot(uu[i], t[i + 1])) * t[i + 1]
            norm = float(np.linalg.norm(projected))
            if norm <= _EPS:
                continue
            projected = projected / norm
            cos = float(np.clip(np.dot(projected, uu[i + 1]), -1.0, 1.0))
            total += abs(float(np.arccos(cos)))
    return total


# ──────────────────────────────────────────────────────────────────────────────
# Centerline sampling
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CenterlineSamples:
    """A centerline resampled at uniform arc length, with its travelling frame."""

    points_vox: Any
    points_mm: Any
    tangents: Any
    u: Any
    v: Any
    arc_length_mm: Any
    spacing: tuple[float, float, float]

    @property
    def n_stations(self) -> int:
        """How many stations the centerline was resampled to."""
        return int(to_numpy(self.points_vox).shape[0])

    @property
    def length_mm(self) -> float:
        """Total vessel length along the centerline."""
        s = to_numpy(self.arc_length_mm)
        return float(s[-1]) if s.size else 0.0


def _spacing_tuple(spacing: Any) -> tuple[float, float, float]:
    """Coerce *spacing* to a 3-tuple of millimetres per voxel."""
    values = [float(v) for v in to_numpy(spacing).reshape(-1)]
    while len(values) < 3:
        values.append(1.0)
    return (values[0], values[1], values[2])


def arc_lengths(points_mm: Any) -> Any:
    """Cumulative arc length along a polyline, starting at 0."""
    pts = as_backend_array(points_mm).astype(float, copy=False).reshape(-1, 3)
    if pts.shape[0] < 2:
        return np.zeros((pts.shape[0],), dtype=float)
    steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([np.zeros(1, dtype=float), np.cumsum(steps)])


#: How much finer than the requested station spacing the smoothing spline is read
#: back at, and the ceiling on that. Eight samples per station makes the chord
#: error of the linear resampling that follows negligible against voxel size.
_SPLINE_OVERSAMPLE = 8
_MAX_SPLINE_POINTS = 200_000


def resample_uniform(points_mm: Any, *, step_mm: float) -> Any:
    """Resample a polyline to uniform *step_mm* spacing by arc length."""
    pts = as_backend_array(points_mm).astype(float, copy=False).reshape(-1, 3)
    if pts.shape[0] < 2:
        return pts
    s = arc_lengths(pts)
    total = float(to_numpy(s[-1]))
    if total <= _EPS:
        return pts
    n = max(int(total / max(float(step_mm), _EPS)) + 2, 2)
    target = np.linspace(0.0, total, n)
    return np.stack([np.interp(target, s, pts[:, k]) for k in range(3)], axis=1)


def _smooth_points(points_mm: Any, *, smooth: float, n_out: int | None = None) -> Any:
    """Fit a smoothing spline through a polyline, falling back to the input.

    *n_out* is how many points the fitted curve is read back at. It has to be well
    above the number of stations the caller will finally resample to: the spline is
    evaluated into a polyline and then interpolated linearly, so reading it back at
    the input's own resolution puts the voxel staircase straight back into a curve
    the fit had just taken it out of.
    """
    # CuPy has no splprep, so the fit runs on the host and comes back to the
    # active backend.
    with using("cpu"):
        pts = to_numpy(points_mm).astype(float, copy=False).reshape(-1, 3)
        n_in = int(pts.shape[0])
        if n_in < 4 or smooth <= 0:
            return as_backend_array(pts)
        try:
            from scipy.interpolate import splev, splprep

            # s scales with the point count: a fixed s over-smooths a short
            # segment and leaves a long one as staircased as it started.
            tck, _u = splprep(pts.T, s=float(smooth) * n_in, k=min(3, n_in - 1))
            n = max(int(n_out or n_in), n_in)
            out = np.stack(splev(np.linspace(0.0, 1.0, n), tck), axis=1)
        except Exception:
            out = pts
    return as_backend_array(out)


def resample_centerline(
    points_vox: Any,
    spacing: Any,
    *,
    step_mm: float = 0.5,
    smooth: float = 0.35,
    seed_normal: Any = None,
) -> CenterlineSamples:
    """Smooth a centerline, resample it at uniform arc length, and frame it.

    Everything happens in millimetres. Tangents taken on voxel coordinates of an
    anisotropic volume point the wrong way — on 3 x 1 x 1 mm voxels a diagonal
    vessel's tangent is off by the spacing ratio, and the reformation is cut at a
    tilt — so the polyline is converted to millimetres first and only mapped back
    to voxels for sampling.
    """
    sp = _spacing_tuple(spacing)
    vox = as_backend_array(points_vox).astype(float, copy=False).reshape(-1, 3)
    if vox.shape[0] < 2:
        raise ValueError("A centerline needs at least two points.")

    scale = as_backend_array(sp).astype(float, copy=False)
    raw_mm = vox * scale
    # Read the fitted curve back far finer than the station spacing so the uniform
    # resampling that follows interpolates *along* the spline instead of chording
    # across it. Without this the curve is only ever as fine as the skeleton was,
    # and asking for 0.2 mm stations on 0.5 mm voxels buys nothing.
    span = float(to_numpy(arc_lengths(raw_mm)[-1]))
    dense = int(
        min(
            max(span / max(float(step_mm), _EPS) * _SPLINE_OVERSAMPLE, vox.shape[0]),
            _MAX_SPLINE_POINTS,
        )
    )
    mm = _smooth_points(raw_mm, smooth=smooth, n_out=dense)
    mm = resample_uniform(mm, step_mm=step_mm)

    tangents = _tangents(mm)
    u, v = rotation_minimizing_frames(mm, tangents, seed_normal=seed_normal)
    return CenterlineSamples(
        points_vox=mm / scale,
        points_mm=mm,
        tangents=tangents,
        u=u,
        v=v,
        arc_length_mm=arc_lengths(mm),
        spacing=sp,
    )


def _tangents(points_mm: Any, *, k_half: int = 2) -> Any:
    """Unit tangents by central differences, one-sided at the ends."""
    pts = as_backend_array(points_mm).astype(float, copy=False).reshape(-1, 3)
    n = int(pts.shape[0])
    if n < 2:
        return np.zeros((n, 3), dtype=float)
    idx = np.arange(n)
    lo = np.clip(idx - int(k_half), 0, n - 1)
    hi = np.clip(idx + int(k_half), 0, n - 1)
    out = _unit(pts[hi] - pts[lo])
    # A degenerate run leaves a zero tangent, which would collapse the frame.
    # Carrying the previous one forward is inherently sequential, so it happens
    # on the host — and only when there is actually a gap to fill.
    norms = np.linalg.norm(out, axis=1)
    if bool(to_numpy(np.any(norms <= _EPS))):
        with using("cpu"):
            host = to_numpy(out)
            for i in range(n):
                if float(np.linalg.norm(host[i])) <= _EPS:
                    host[i] = host[i - 1] if i else np.array([0.0, 0.0, 1.0])
        out = as_backend_array(host)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Reformation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CprResult:
    """One straightened reformation of a volume along a centerline."""

    image: Any
    arc_length_mm: Any
    ray_mm: Any
    points_vox: Any
    u: Any
    v: Any
    angle_deg: float
    fold_warning: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_stations, n_ray)`` of the reformatted image."""
        arr = to_numpy(self.image)
        return (int(arr.shape[0]), int(arr.shape[1]))


def cut_direction(samples: CenterlineSamples, angle_deg: float) -> Any:
    """The in-plane direction the reformation cuts along, at *angle_deg*."""
    theta = float(angle_deg) * 3.141592653589793 / 180.0
    u = as_backend_array(samples.u).astype(float, copy=False)
    v = as_backend_array(samples.v).astype(float, copy=False)
    return float(np.cos(theta)) * u + float(np.sin(theta)) * v


def cpr_coords(samples: CenterlineSamples, *, angle_deg: float, ray_mm: float, n_ray: int) -> Any:
    """``(3, n_stations, n_ray)`` voxel coordinates for one reformation.

    Built as a single grid so the whole reformation is one ``map_coordinates``
    call rather than one per station.
    """
    direction = cut_direction(samples, angle_deg)
    centers_mm = as_backend_array(samples.points_mm).astype(float, copy=False)
    scale = as_backend_array(samples.spacing).astype(float, copy=False)

    offsets = np.linspace(-float(ray_mm), float(ray_mm), int(n_ray))
    # (stations, ray, 3) in mm, then back to voxels for sampling.
    pts_mm = centers_mm[:, None, :] + offsets[None, :, None] * direction[:, None, :]
    pts_vox = pts_mm / scale
    return np.stack([pts_vox[..., 0], pts_vox[..., 1], pts_vox[..., 2]], axis=0)


def polar_coords(
    samples: CenterlineSamples, *, ray_mm: float, n_angle: int, n_radius: int
) -> Any:
    """``(3, stations, angles, radii)`` voxel coordinates of the perpendicular discs.

    One grid for every station at once, so a whole vessel's cross-sections are a
    single ``map_coordinates`` call. Radii start one step out from the centerline,
    which is the ray's origin and carries no area.
    """
    centers_mm = as_backend_array(samples.points_mm).astype(float, copy=False)
    u = as_backend_array(samples.u).astype(float, copy=False)
    v = as_backend_array(samples.v).astype(float, copy=False)
    scale = as_backend_array(samples.spacing).astype(float, copy=False)

    radii = np.linspace(
        float(ray_mm) / int(n_radius), float(ray_mm), int(n_radius)
    )
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angle), endpoint=False)
    dirs = (
        np.cos(angles)[None, :, None] * u[:, None, :]
        + np.sin(angles)[None, :, None] * v[:, None, :]
    )
    pts_mm = centers_mm[:, None, None, :] + radii[None, None, :, None] * dirs[:, :, None, :]
    pts_vox = pts_mm / scale
    return np.stack([pts_vox[..., 0], pts_vox[..., 1], pts_vox[..., 2]], axis=0)


def fold_warning(samples: CenterlineSamples, *, ray_mm: float) -> str:
    """Warn when the rays cross, which folds the reformation onto itself.

    Straightening compresses the inside of a bend and stretches the outside. Once
    the radius of curvature drops below the ray half-length the rays on the inside
    intersect, and the image there is not a picture of anything.
    """
    pts = as_backend_array(samples.points_mm).astype(float, copy=False)
    if pts.shape[0] < 3:
        return ""
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab, bc, ac = b - a, c - b, c - a
    cross = np.linalg.norm(np.cross(ab, bc), axis=1)
    denom = (
        np.linalg.norm(ab, axis=1)
        * np.linalg.norm(bc, axis=1)
        * np.linalg.norm(ac, axis=1)
    )
    curvature = np.where(denom > _EPS, 2.0 * cross / np.where(denom > _EPS, denom, 1.0), 0.0)
    peak = float(to_numpy(np.max(curvature))) if curvature.size else 0.0
    if peak <= _EPS:
        return ""
    radius = 1.0 / peak
    if radius >= float(ray_mm):
        return ""
    return (
        f"Tightest bend has a {radius:.1f} mm radius, inside the {float(ray_mm):.1f} mm "
        "ray — the reformation folds over itself there. Reduce the ray length."
    )


def cpr_sample(
    volume: Any,
    samples: CenterlineSamples,
    *,
    angle_deg: float = 0.0,
    ray_mm: float = 12.0,
    n_ray: int = 129,
    order: int = 1,
    mode: str = "constant",
    cval: float = 0.0,
) -> CprResult:
    """Reformat *volume* along *samples* into a straightened image.

    Use ``order=1`` for intensities and ``order=0`` for masks — interpolating label
    ids produces values that were never in the segmentation.
    """
    coords = cpr_coords(samples, angle_deg=angle_deg, ray_mm=ray_mm, n_ray=n_ray)
    image = ndi.map_coordinates(
        as_backend_array(volume),
        as_backend_array(coords),
        order=int(order),
        mode=mode,
        cval=float(cval),
    )
    return CprResult(
        image=image,
        arc_length_mm=samples.arc_length_mm,
        ray_mm=np.linspace(-float(ray_mm), float(ray_mm), int(n_ray)),
        points_vox=samples.points_vox,
        u=samples.u,
        v=samples.v,
        angle_deg=float(angle_deg),
        fold_warning=fold_warning(samples, ray_mm=ray_mm),
    )


def cpr_sample_mask(
    mask: Any,
    samples: CenterlineSamples,
    *,
    angle_deg: float = 0.0,
    ray_mm: float = 12.0,
    n_ray: int = 129,
) -> Any:
    """Reformat a label mask with nearest-neighbour sampling; returns the array."""
    return cpr_sample(
        mask,
        samples,
        angle_deg=angle_deg,
        ray_mm=ray_mm,
        n_ray=n_ray,
        order=0,
    ).image


def cross_section_at(
    volume: Any,
    samples: CenterlineSamples,
    station: int,
    *,
    ray_mm: float = 12.0,
    res: int = 129,
    order: int = 1,
):
    """The true perpendicular plane at one station, for the detail view."""
    from nvitk.transform.oblique import oblique_slice

    idx = int(max(0, min(int(station), samples.n_stations - 1)))
    scale = as_backend_array(samples.spacing).astype(float, copy=False)
    # oblique_slice works in voxels, so the frame and the radius convert with it.
    u_vox = as_backend_array(samples.u).astype(float, copy=False)[idx] / scale
    v_vox = as_backend_array(samples.v).astype(float, copy=False)[idx] / scale
    center = as_backend_array(samples.points_vox).astype(float, copy=False)[idx]
    radius_vox = float(ray_mm) / min(samples.spacing)
    return oblique_slice(
        volume,
        center_xyz=center,
        u_xyz=u_vox,
        v_xyz=v_vox,
        radius_vox=radius_vox,
        res=int(res),
        order=int(order),
    )


__all__ = [
    "CenterlineSamples",
    "CprResult",
    "arc_lengths",
    "cpr_coords",
    "polar_coords",
    "cpr_sample",
    "cpr_sample_mask",
    "cross_section_at",
    "cut_direction",
    "fold_warning",
    "frame_twist",
    "resample_centerline",
    "resample_uniform",
    "rotation_minimizing_frames",
]
