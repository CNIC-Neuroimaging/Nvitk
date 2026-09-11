"""Per-vessel curved planar reformation: layers in, flattened vessels out.

Wraps the geometry in :mod:`nvitk.transform.cpr` with everything a vessel needs —
finding a centerline when none was supplied, splitting a multilabel mask into the
vessels the caller asked for, resampling the lumen and wall alongside the image,
and measuring the calibre at every station.

No Qt and no Napari here; the panel in :mod:`nvitk.gui.viz.vessel_cpr_panel` is a
view onto these results, and the same functions serve a script or a QC figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from nvitk.core import as_backend_array
from nvitk.core.array import to_numpy
from nvitk.core.backend import setup, using
from nvitk.transform.cpr import (
    CenterlineSamples,
    CprResult,
    cpr_sample,
    cpr_sample_mask,
    cross_section_at,
    polar_coords,
    resample_centerline,
)

setup(globals())

#: Centerlines shorter than this are noise rather than vessels.
MIN_CENTERLINE_POINTS = 8

#: Default reach when joining a vessel's centerline pieces to each other. A
#: segmentation of one artery routinely arrives in several components — a signal
#: dropout at a bend, a clip artefact, a centerline traced per sub-segment — and
#: without joining them only the largest piece is reformatted.
DEFAULT_JOIN_GAP_VOX = 8

#: How far outside the lumen a supplied centerline voxel may sit and still count.
#: Nearest-neighbour resampling onto another grid moves a one-voxel-wide curve by
#: up to a voxel, which is enough to put most of it outside a thin vessel.
CENTERLINE_TOLERANCE_VOX = 2

#: Default distance either side of the centerline that a reformation covers.
DEFAULT_RAY_MM = 10.0

#: Default samples across a reformation row. Odd, so one lands on the centerline.
DEFAULT_N_RAY = 161


@dataclass(frozen=True)
class VesselCpr:
    """One vessel, flattened, with its overlays and calibre."""

    label: int
    name: str
    cpr: CprResult
    lumen: Any
    wall: Any | None
    radius_mm: Any
    samples: CenterlineSamples
    diameter: Any = None

    @property
    def length_mm(self) -> float:
        """Vessel length along the centerline."""
        return self.samples.length_mm

    @property
    def n_stations(self) -> int:
        """Rows in the reformatted image."""
        return self.samples.n_stations

    def diameter_mm(self) -> Any:
        """Area-equivalent lumen diameter at each station.

        Falls back to counting lumen samples across each reformation row when no
        area measurement was made. That fallback is quantised to the ray's sample
        step and is measured along one cut direction only, so it reads as a
        staircase on a vessel whose true calibre varies smoothly — prefer the
        measured value.
        """
        if self.diameter is not None:
            return as_backend_array(self.diameter)
        lumen = as_backend_array(self.lumen)
        ray = to_numpy(self.cpr.ray_mm)
        step = float(ray[1] - ray[0]) if ray.size > 1 else 1.0
        return (lumen > 0).sum(axis=1).astype(float) * step


def vessel_name(label: int) -> str:
    """A readable name for *label*, using the qvtpy vocabulary when it knows one."""
    try:
        from nvitk.pipes.qvtpy.labels import qvtpy_vessel_name

        name = str(qvtpy_vessel_name(int(label)) or "").strip()
        # An unmapped id comes back as the sentinel "QVTPY_UNKNOWN_<id>", which is
        # an internal token, not a vessel name — fall through to a plain label.
        if name and "unknown" not in name.lower():
            return name
    except Exception:
        pass
    return f"Label {int(label)}"


def labels_in(mask: Any, *, labels: Sequence[int] | None = None) -> list[int]:
    """Non-zero label ids in *mask*, or the requested subset that is actually present.

    A binary mask has one label — ``1`` — so a caller that passes a plain
    segmentation gets a single vessel without having to say so.
    """
    present = sorted(
        int(v) for v in to_numpy(np.unique(as_backend_array(mask))) if int(v) != 0
    )
    if labels is None:
        return present
    wanted = [int(v) for v in labels]
    return [v for v in wanted if v in present]


def derive_wall(lumen_mask: Any, *, label: int, thickness_vox: int = 1) -> Any:
    """A wall ring around *label*, for when no wall segmentation was supplied.

    ``dilate(lumen) - lumen``: a band the given number of voxels thick hugging the
    outside of the lumen. It is a stand-in for a real wall segmentation, not a
    measurement of one — it says where the wall *would* be, at the thickness asked
    for, not where it is.
    """
    lumen = as_backend_array(lumen_mask) == int(label)
    if not bool(to_numpy(lumen.any())):
        return np.zeros(lumen.shape, dtype=np.int32)
    # brute_force is required by CuPy for iterations > 1 and is accepted by SciPy
    # with the same result — it changes the algorithm, not the output.
    grown = ndi.binary_dilation(
        lumen, iterations=max(int(thickness_vox), 1), brute_force=True
    )
    return (grown & ~lumen).astype(np.int32)


class CenterlineUnavailable(RuntimeError):
    """No usable centerline for one label, with the reason spelled out.

    A vessel that cannot be flattened is a normal outcome — the label is a speck,
    the supplied centerline misses it — and the useful thing to hand back is which
    label and why, not a bare ``None`` that leaves the caller guessing.
    """

    def __init__(self, label: int, reason: str) -> None:
        """Record *label* and the human-readable *reason* it produced no centerline."""
        super().__init__(f"label {int(label)}: {reason}")
        self.label = int(label)
        self.reason = str(reason)


def _join_components(region: Any, *, max_gap: int) -> Any:
    """Thread a mask's separate pieces together with one-voxel bridges.

    Host-side, alongside the skeletonisation that follows it. ``tube_radius=0``
    keeps each bridge one voxel wide: a fatter one reads as a junction cluster to
    the graph walk and fragments the very path it was meant to join.
    """
    from nvitk.morphology.mst_bridge import bridge_binary_components_mst

    joined = to_numpy(bridge_binary_components_mst(region > 0, max_gap=int(max_gap), tube_radius=0))
    return (joined > 0).astype(np.int32)


def _skeleton_points(binary: Any, label: int, *, centerline_mask: Any = None) -> Any:
    """Ordered centerline voxels for a 0/1 *binary* volume, or raise ``CenterlineUnavailable``.

    Host-side: skeletonisation is skimage. Callers are already inside ``using("cpu")``.
    """
    from nvitk.morphology.centerline import compute_centerlines

    try:
        found = compute_centerlines(
            binary,
            centerline_mask=centerline_mask,
            labels=[1],
            min_points=MIN_CENTERLINE_POINTS,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim as the reason
        raise CenterlineUnavailable(label, f"skeletonisation failed ({exc})") from exc
    points = found.get(1)
    if points is None:
        raise CenterlineUnavailable(
            label,
            f"the skeleton has fewer than {MIN_CENTERLINE_POINTS} voxels — "
            "the vessel is too short or too thin to trace",
        )
    n = int(to_numpy(points).shape[0])
    if n < 2:
        raise CenterlineUnavailable(label, f"the centerline is only {n} point(s) long")
    return points


def upsample_plan(shape, spacing, *, factor: float) -> tuple[tuple, tuple]:
    """Target ``(shape, spacing)`` for upsampling a grid by *factor*.

    The spacing is derived from the shapes rather than by dividing by *factor*,
    because a zoom aligns the first and last voxel *centres*: the physical extent
    is preserved, so the new spacing is the old one scaled by the ratio of the
    two grids' spans, and rounding the shape would otherwise leave it slightly off.
    """
    f = float(factor)
    shape = tuple(int(n) for n in shape)
    spacing = tuple(float(v) for v in spacing)
    if f <= 1.0:
        return shape, spacing
    new_shape = tuple(max(int(round(n * f)), 1) for n in shape)
    new_spacing = tuple(
        s * (max(n - 1, 1) / max(m - 1, 1)) for s, n, m in zip(spacing, shape, new_shape)
    )
    return new_shape, new_spacing


def upsample_volume(volume: Any, new_shape, *, order: int) -> Any:
    """Resample *volume* onto *new_shape*.

    ``order=1`` for intensities and ``order=0`` for labels — interpolating label
    ids invents values that were never in the segmentation.
    """
    arr = as_backend_array(volume)
    if tuple(arr.shape) == tuple(new_shape):
        return arr
    zoom = tuple(float(m) / float(n) for n, m in zip(arr.shape, new_shape))
    return ndi.zoom(arr, zoom, order=int(order), mode="nearest")


def centerline_for_label(
    lumen_mask: Any,
    label: int,
    *,
    spacing: Any,
    centerline_mask: Any = None,
    step_mm: float = 0.5,
    smooth: float = 0.35,
    seed_normal: Any = None,
    join_gap_vox: int = DEFAULT_JOIN_GAP_VOX,
) -> CenterlineSamples:
    """The centerline of one vessel, resampled and framed.

    Uses a supplied centerline where there is one, and otherwise skeletonises the
    label and walks the skeleton's longest path — the graph diameter, which for a
    single vessel is the vessel. A supplied centerline that does not land inside
    the label (a different grid, a one-voxel drift) falls back to skeletonising
    rather than losing the vessel.

    One label's pieces are threaded together first, out to *join_gap_vox*. The
    longest path runs through a single connected component, so a vessel that
    arrives in two pieces would otherwise be reformatted from whichever piece is
    longer and the rest of it silently dropped.

    Raises
    ------
    CenterlineUnavailable
        When no centerline can be traced, carrying the label and the reason.
    """
    # Skeletonisation is skimage, which is host-only — convert at the boundary.
    with using("cpu"):
        binary = (to_numpy(lumen_mask) == int(label)).astype(np.int32)
        n_vox = int(binary.sum())
        if n_vox == 0:
            raise CenterlineUnavailable(label, "the label is empty on this grid")
        if n_vox < MIN_CENTERLINE_POINTS:
            raise CenterlineUnavailable(
                label,
                f"only {n_vox} voxel(s) — a centerline needs at least "
                f"{MIN_CENTERLINE_POINTS}",
            )

        supplied = None
        if centerline_mask is not None:
            cl = to_numpy(centerline_mask)
            own = cl == int(label)
            if int(own.sum()) >= MIN_CENTERLINE_POINTS:
                # The centerline layer carries this vessel's own id, so take it
                # whole. Intersecting it with the lumen first is what throws most
                # of a supplied curve away: a centerline resampled onto this grid,
                # or drawn a little long, sits partly outside the mask.
                supplied = own
            else:
                # Unlabelled (or differently labelled) centerlines: keep what lands
                # in the lumen or within a voxel or two of it, rather than only
                # what lands exactly inside.
                near = ndi.binary_dilation(
                    binary > 0, iterations=CENTERLINE_TOLERANCE_VOX, brute_force=True
                )
                supplied = (cl > 0) & to_numpy(near)
            if int(supplied.sum()) < MIN_CENTERLINE_POINTS:
                # The supplied curve misses this vessel; skeletonise it instead.
                supplied = None

        try:
            # Trace through the supplied curve's own extent when there is one.
            # compute_centerlines intersects the centerline with the region it is
            # given, so passing the lumen here would undo the tolerance above.
            region = binary if supplied is None else supplied.astype(np.int32)
            if int(join_gap_vox) > 0:
                region = _join_components(region, max_gap=int(join_gap_vox))
                if supplied is not None:
                    # The join has to be visible on both, or the threads that
                    # connect the pieces are intersected straight back out.
                    supplied = region > 0
            points = _skeleton_points(region, label, centerline_mask=supplied)
        except CenterlineUnavailable as exc:
            if centerline_mask is not None and supplied is None:
                raise CenterlineUnavailable(
                    label,
                    f"{exc.reason}; the supplied centerline layer has no voxels "
                    "inside this label either",
                ) from exc
            raise

    try:
        return resample_centerline(
            points,
            spacing,
            step_mm=step_mm,
            smooth=smooth,
            seed_normal=seed_normal,
        )
    except Exception as exc:  # noqa: BLE001 — surfaced verbatim as the reason
        raise CenterlineUnavailable(label, f"resampling failed ({exc})") from exc


def centerlines_for_labels(
    lumen_mask: Any,
    *,
    spacing: Any,
    labels: Sequence[int] | None = None,
    centerline_mask: Any = None,
    step_mm: float = 0.5,
    smooth: float = 0.35,
    reasons: dict[int, str] | None = None,
    join_gap_vox: int = DEFAULT_JOIN_GAP_VOX,
) -> dict[int, CenterlineSamples]:
    """A framed centerline per requested label.

    Labels that yield none are skipped; pass a dict as *reasons* to collect why,
    keyed by label, so the caller can say which vessel failed and for what.
    """
    out: dict[int, CenterlineSamples] = {}
    wanted = [int(v) for v in (labels or [])]
    usable = labels_in(lumen_mask, labels=labels)
    if reasons is not None:
        for label in wanted:
            if label not in usable:
                reasons[label] = "the label is not present in the mask"
    for label in usable:
        try:
            out[int(label)] = centerline_for_label(
                lumen_mask,
                label,
                spacing=spacing,
                centerline_mask=centerline_mask,
                step_mm=step_mm,
                smooth=smooth,
                join_gap_vox=join_gap_vox,
            )
        except CenterlineUnavailable as exc:
            if reasons is not None:
                reasons[int(label)] = exc.reason
    return out


def build_vessel_cpr(
    intensity: Any,
    lumen_mask: Any,
    *,
    label: int,
    samples: CenterlineSamples,
    wall_mask: Any = None,
    wall_thickness_vox: int = 1,
    angle_deg: float = 0.0,
    ray_mm: float = DEFAULT_RAY_MM,
    n_ray: int = DEFAULT_N_RAY,
) -> VesselCpr:
    """Flatten one vessel, with its lumen, wall and per-station calibre.

    The image is sampled linearly and both masks nearest-neighbour, so the overlays
    stay the ids they were segmented as.
    """
    lumen_full = as_backend_array(lumen_mask)
    binary = (lumen_full == int(label)).astype(np.int32)

    cpr = cpr_sample(
        intensity, samples, angle_deg=angle_deg, ray_mm=ray_mm, n_ray=n_ray, order=1
    )
    lumen = cpr_sample_mask(binary, samples, angle_deg=angle_deg, ray_mm=ray_mm, n_ray=n_ray)

    if wall_mask is None:
        wall_vol = derive_wall(lumen_full, label=int(label), thickness_vox=wall_thickness_vox)
    else:
        wall_vol = (as_backend_array(wall_mask) > 0).astype(np.int32)
    wall = cpr_sample_mask(wall_vol, samples, angle_deg=angle_deg, ray_mm=ray_mm, n_ray=n_ray)

    return VesselCpr(
        label=int(label),
        name=vessel_name(label),
        cpr=cpr,
        lumen=lumen,
        wall=wall,
        radius_mm=station_radius_mm(binary, samples),
        samples=samples,
        diameter=station_area_diameter_mm(binary, samples, ray_mm=ray_mm),
    )


#: Rays cast around each station, and samples along each ray, for the area
#: measurement. 72 rays is one every 5 degrees; 128 radial samples over a 10 mm
#: ray puts the boundary search step well under a tenth of a millimetre.
DEFAULT_N_ANGLE = 72
DEFAULT_N_RADIUS = 128
_AREA_EPS = 1e-9


def station_area_diameter_mm(
    binary_mask: Any,
    samples: CenterlineSamples,
    *,
    ray_mm: float = DEFAULT_RAY_MM,
    n_angle: int = DEFAULT_N_ANGLE,
    n_radius: int = DEFAULT_N_RADIUS,
) -> Any:
    """Area-equivalent lumen diameter at each station, to sub-voxel precision.

    Rays are cast outward from the centerline on the perpendicular plane and each
    one is cut where the mask first falls below half. Cutting at the *first* gap
    means a neighbouring vessel that happens to lie inside the ray is not counted,
    and interpolating the crossing puts the boundary between samples instead of
    snapping it to one — which is what makes this smooth where both the distance
    transform and a row-count are quantised to the grid. The enclosed area is
    reported as the diameter of the circle of equal area.
    """
    binary = (as_backend_array(binary_mask) > 0).astype(float)
    coords = polar_coords(samples, ray_mm=ray_mm, n_angle=n_angle, n_radius=n_radius)
    occ = ndi.map_coordinates(binary, coords, order=1, mode="constant", cval=0.0)

    dr = float(ray_mm) / int(n_radius)
    inside = (occ >= 0.5).astype(np.int8)
    # cumprod stays 1 only while every sample so far has been inside, so the sum
    # is the index of the first gap — the lumen boundary along that ray.
    k = np.cumprod(inside, axis=2).sum(axis=2)

    kc = np.clip(k, 1, int(n_radius) - 1)
    lo = np.take_along_axis(occ, (kc - 1)[..., None], axis=2)[..., 0]
    hi = np.take_along_axis(occ, kc[..., None], axis=2)[..., 0]
    drop = lo - hi
    safe = np.abs(drop) > _AREA_EPS
    frac = np.clip(np.where(safe, (lo - 0.5) / np.where(safe, drop, 1.0), 0.0), 0.0, 1.0)

    # radii[j] is (j + 1) * dr, so the last sample still inside — index k - 1 —
    # sits at k * dr, and the crossing is that plus the interpolated fraction of
    # the next step. Anchoring on (k - 1) instead costs a fixed dr of radius,
    # which is a constant underestimate of the calibre at every station.
    radius = np.where(k > 0, (k.astype(float) + frac) * dr, 0.0)
    area = 0.5 * (radius**2).sum(axis=1) * (2.0 * np.pi / float(n_angle))
    return 2.0 * np.sqrt(area / np.pi)


def station_radius_mm(binary_mask: Any, samples: CenterlineSamples) -> Any:
    """Inscribed radius in mm at each station, from a spacing-aware distance transform."""
    mask = as_backend_array(binary_mask) > 0
    if not bool(to_numpy(mask.any())):
        return np.zeros((samples.n_stations,), dtype=float)
    dist = ndi.distance_transform_edt(mask, sampling=samples.spacing)
    pts = as_backend_array(samples.points_vox).astype(float, copy=False)
    # Trilinear, not nearest. Stations are spaced far finer than a voxel, so
    # rounding to the containing voxel returns the same handful of distances over
    # and over and a constant-calibre vessel comes out as a staircase.
    coords = np.stack(
        [np.clip(pts[:, k], 0.0, float(dist.shape[k]) - 1.0) for k in range(3)], axis=0
    )
    return ndi.map_coordinates(dist, coords, order=1, mode="nearest").astype(float)


def vessel_cross_section(
    intensity: Any,
    vessel: VesselCpr,
    station: int,
    *,
    ray_mm: float = DEFAULT_RAY_MM,
    res: int = 129,
    lumen_mask: Any = None,
) -> tuple[Any, Any | None]:
    """``(intensity, lumen)`` on the true perpendicular plane at one station."""
    image = cross_section_at(intensity, vessel.samples, station, ray_mm=ray_mm, res=res, order=1)
    if lumen_mask is None:
        return image, None
    binary = (as_backend_array(lumen_mask) == vessel.label).astype(np.int32)
    mask = cross_section_at(binary, vessel.samples, station, ray_mm=ray_mm, res=res, order=0)
    return image, mask


def station_world_points(vessel: VesselCpr) -> Any:
    """Centerline stations as voxel coordinates, for a Napari overlay layer."""
    # Napari layers hold host arrays.
    return to_numpy(vessel.samples.points_vox)


def plane_corners(vessel: VesselCpr, station: int, *, ray_mm: float = DEFAULT_RAY_MM) -> Any:
    """The four corners of the cross-section plane at *station*, in voxel coords.

    Drawn in 3D so the flat view and the volume agree about where the cut is.
    """
    # Feeds a Napari Shapes layer, so it is built on the host.
    with using("cpu"):
        idx = int(max(0, min(int(station), vessel.n_stations - 1)))
        scale = to_numpy(vessel.samples.spacing).astype(float)
        center = to_numpy(vessel.samples.points_vox)[idx]
        u = to_numpy(vessel.samples.u)[idx] / scale
        v = to_numpy(vessel.samples.v)[idx] / scale
        r = float(ray_mm)
        return np.stack(
            [
                center + r * (u + v),
                center + r * (u - v),
                center + r * (-u - v),
                center + r * (-u + v),
            ]
        ).astype(np.float32)


__all__ = [
    "DEFAULT_N_RAY",
    "DEFAULT_RAY_MM",
    "CENTERLINE_TOLERANCE_VOX",
    "DEFAULT_JOIN_GAP_VOX",
    "MIN_CENTERLINE_POINTS",
    "upsample_plan",
    "upsample_volume",
    "VesselCpr",
    "build_vessel_cpr",
    "CenterlineUnavailable",
    "centerline_for_label",
    "centerlines_for_labels",
    "derive_wall",
    "labels_in",
    "plane_corners",
    "station_area_diameter_mm",
    "station_radius_mm",
    "station_world_points",
    "vessel_cross_section",
    "vessel_name",
]
