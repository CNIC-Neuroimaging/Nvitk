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
    resample_centerline,
)

setup(globals())

#: Centerlines shorter than this are noise rather than vessels.
MIN_CENTERLINE_POINTS = 8

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

    @property
    def length_mm(self) -> float:
        """Vessel length along the centerline."""
        return self.samples.length_mm

    @property
    def n_stations(self) -> int:
        """Rows in the reformatted image."""
        return self.samples.n_stations

    def diameter_mm(self) -> Any:
        """Lumen width at each station, measured off the reformation itself.

        Counts lumen samples across each row rather than doubling the distance
        transform, so it reflects what the image actually shows — including a
        stenosis the centerline passes straight through.
        """
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


def centerline_for_label(
    lumen_mask: Any,
    label: int,
    *,
    spacing: Any,
    centerline_mask: Any = None,
    step_mm: float = 0.5,
    smooth: float = 0.35,
    seed_normal: Any = None,
) -> CenterlineSamples:
    """The centerline of one vessel, resampled and framed.

    Uses a supplied centerline where there is one, and otherwise skeletonises the
    label and walks the skeleton's longest path — the graph diameter, which for a
    single vessel is the vessel. A supplied centerline that does not land inside
    the label (a different grid, a one-voxel drift) falls back to skeletonising
    rather than losing the vessel.

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
            supplied = (to_numpy(centerline_mask) > 0) & (binary > 0)
            if int(supplied.sum()) < MIN_CENTERLINE_POINTS:
                # The supplied curve misses this vessel; skeletonise it instead.
                supplied = None

        try:
            points = _skeleton_points(binary, label, centerline_mask=supplied)
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
    )


def station_radius_mm(binary_mask: Any, samples: CenterlineSamples) -> Any:
    """Inscribed radius in mm at each station, from a spacing-aware distance transform."""
    mask = as_backend_array(binary_mask) > 0
    if not bool(to_numpy(mask.any())):
        return np.zeros((samples.n_stations,), dtype=float)
    dist = ndi.distance_transform_edt(mask, sampling=samples.spacing)
    pts = np.rint(as_backend_array(samples.points_vox)).astype(int)
    for axis in range(3):
        pts[:, axis] = np.clip(pts[:, axis], 0, int(dist.shape[axis]) - 1)
    return dist[pts[:, 0], pts[:, 1], pts[:, 2]].astype(float)


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
    "MIN_CENTERLINE_POINTS",
    "VesselCpr",
    "build_vessel_cpr",
    "CenterlineUnavailable",
    "centerline_for_label",
    "centerlines_for_labels",
    "derive_wall",
    "labels_in",
    "plane_corners",
    "station_radius_mm",
    "station_world_points",
    "vessel_cross_section",
    "vessel_name",
]
