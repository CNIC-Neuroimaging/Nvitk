"""
Dynamic (3D+t / 2D+t) DICOM series — geometry splitting and temporal stacking.

Description
-----------
Two failure modes of time-resolved acquisitions that the volume-oriented
``dicom2nifti`` path cannot handle, solved before (or after) it runs:

1. **Mixed geometry in one series.** Some series carry images of several sizes or
   orientations — two surview projections, reformats with different heights,
   secondary captures. Stacking them fails (``could not broadcast (158,512) into
   (122,512)``) or, worse, succeeds with a nonsense slice step.
   :func:`split_by_geometry` cuts the series into sub-series whose images share
   ``Rows``/``Columns``/``ImageOrientationPatient``/``PixelSpacing``.

2. **Repeated slice positions.** A dynamic acquisition images the same position
   (or the same stack of positions) many times: bolus tracking monitors one slice
   over time, a perfusion or cine series repeats a whole stack. A position step of
   zero gives ``dicom2nifti`` a NaN slice direction (``NON_CUBICAL_IMAGE``) and an
   un-decomposable affine. :func:`detect_temporal_layout` recognises the repetition
   and :func:`build_temporal_image` writes a proper ``X x Y x Z x T`` NIfTI with
   the frame times in the header and the sidecar.

Array / axis conventions
------------------------
The volume layout and affine reproduce ``dicom2nifti.common.create_affine`` and
its ``(columns, rows, slices)`` transpose exactly, so a 4D series written here sits
in the same RAS world frame as the 3D series ``dicom2nifti`` writes from the same
study. Axes are ``XYZT``; the time axis is last, in seconds.

Time ordering
-------------
Frames are ordered by the first DICOM attribute that actually varies across
them, in this order of trust: ``TemporalPositionIdentifier``, ``TriggerTime``,
``NominalPercentageOfCardiacPhase``, acquisition/content date-time, and finally
``InstanceNumber``. Frame times (seconds from the first frame) come from the
first *time-valued* attribute that varies; when none does, the time axis is
an index with a step of 1 and the sidecar says so.

I/O: pydicom datasets in, nibabel images out; NumPy only (host-side conversion).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from nvitk.core.logger import Logger

try:
    import nibabel as nib
except Exception:  # pragma: no cover — guarded by _require_deps upstream
    nib = None

log = Logger()

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

#: Two slice positions closer than this (mm, along the normal) are the same slice.
POSITION_TOL_MM = 1e-2

#: Direction cosines are compared after rounding to this many decimals.
_IOP_DECIMALS = 3

#: Pixel spacings are compared after rounding to this many decimals (mm).
_SPACING_DECIMALS = 4


# ---------------------------------------------------------------------------
# Small DICOM readers
# ---------------------------------------------------------------------------


def _floats(value: Any, n: int | None = None) -> tuple[float, ...] | None:
    """A DICOM multi-value as floats, or ``None`` when absent or malformed."""
    if value is None:
        return None
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if n is not None and len(out) < n:
        return None
    if not all(np.isfinite(out)):
        return None
    return out


def _float(value: Any) -> float | None:
    """A DICOM scalar as a finite float, or ``None``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _seconds_of_day(time_value: Any) -> float | None:
    """``HHMMSS.ffffff`` DICOM TM → seconds since midnight."""
    text = str(time_value or "").strip()
    if not text:
        return None
    try:
        hh = int(text[0:2])
        mm = int(text[2:4]) if len(text) >= 4 else 0
        ss = float(text[4:]) if len(text) > 4 else 0.0
    except ValueError:
        return None
    return hh * 3600.0 + mm * 60.0 + ss


def _datetime_seconds(ds: Any, dt_tag: str, date_tag: str, time_tag: str) -> float | None:
    """Absolute seconds from a DT attribute, else from a DA + TM pair."""
    dt_text = str(ds.get(dt_tag, "") or "").strip() if dt_tag else ""
    if len(dt_text) >= 14:
        try:
            base = datetime.strptime(dt_text[:14], "%Y%m%d%H%M%S")
            frac = float("0" + dt_text[14:].split("+")[0].split("-")[0]) if len(dt_text) > 14 else 0.0
            return base.timestamp() + frac
        except ValueError:
            pass
    tod = _seconds_of_day(ds.get(time_tag, None))
    if tod is None:
        return None
    date_text = str(ds.get(date_tag, "") or "").strip()
    if len(date_text) >= 8:
        try:
            return datetime.strptime(date_text[:8], "%Y%m%d").timestamp() + tod
        except ValueError:
            pass
    return tod


def slice_normal(ds: Any) -> np.ndarray | None:
    """Unit slice normal ``row × col`` from ``ImageOrientationPatient``."""
    iop = _floats(ds.get("ImageOrientationPatient", None), 6)
    if iop is None:
        return None
    normal = np.cross(np.asarray(iop[:3]), np.asarray(iop[3:6]))
    norm = float(np.linalg.norm(normal))
    return normal / norm if norm > 1e-8 else None


def slice_position(ds: Any, normal: np.ndarray) -> float | None:
    """Position of *ds* along *normal* (mm), from ``ImagePositionPatient``."""
    ipp = _floats(ds.get("ImagePositionPatient", None), 3)
    if ipp is None:
        return None
    return float(np.dot(np.asarray(ipp[:3]), normal))


def series_label(ds_or_md: Any) -> str:
    """``series 404 'IMR, 83%'`` — how log messages name a series."""
    get = ds_or_md.get if hasattr(ds_or_md, "get") else (lambda k, d=None: getattr(ds_or_md, k, d))
    number = str(get("SeriesNumber", "") or "").strip()
    desc = str(get("SeriesDescription", "") or "").strip()
    label = f"series {number}" if number else "series"
    return f"{label} '{desc}'" if desc else label


# ---------------------------------------------------------------------------
# Geometry splitting
# ---------------------------------------------------------------------------


def geometry_signature(ds: Any) -> tuple:
    """What two images must share to be stacked into one volume."""
    try:
        rows = int(ds.get("Rows", 0) or 0)
        cols = int(ds.get("Columns", 0) or 0)
    except (TypeError, ValueError):
        rows = cols = 0
    iop = _floats(ds.get("ImageOrientationPatient", None), 6)
    spacing = _floats(ds.get("PixelSpacing", None), 2)
    return (
        rows,
        cols,
        None if iop is None else tuple(round(v, _IOP_DECIMALS) + 0.0 for v in iop[:6]),
        None if spacing is None else tuple(round(v, _SPACING_DECIMALS) for v in spacing[:2]),
    )


def _plane_name(iop: tuple[float, ...] | None) -> str:
    """``AX`` / ``COR`` / ``SAG`` / ``OBL`` for a direction-cosine pair."""
    if iop is None:
        return "NOGEOM"
    normal = np.abs(np.cross(np.asarray(iop[:3]), np.asarray(iop[3:6])))
    axis = int(np.argmax(normal))
    if float(normal[axis]) < 0.9:
        return "OBL"
    return ("SAG", "COR", "AX")[axis]


def split_by_geometry(ds_list: Sequence[Any]) -> list[tuple[str, list[Any]]]:
    """
    Group *ds_list* by :func:`geometry_signature`.

    Returns
    -------
    list of (suffix, datasets)
        One entry per geometry, largest group first. The suffix is empty when the
        series is homogeneous; otherwise it names what differs — the plane
        (``AX``/``COR``/``SAG``/``OBL``) when orientations differ, the matrix size
        (``158x512``) when sizes differ — plus an index when that is still
        ambiguous, so every sub-series gets a distinct, readable file name.
    """
    groups: dict[tuple, list[Any]] = {}
    for ds in ds_list:
        groups.setdefault(geometry_signature(ds), []).append(ds)
    if len(groups) <= 1:
        return [("", list(ds_list))]

    keys = sorted(groups, key=lambda k: -len(groups[k]))
    orientations = {k[2] for k in keys}
    sizes = {(k[0], k[1]) for k in keys}
    spacings = {k[3] for k in keys}
    labels: list[str] = []
    for key in keys:
        parts: list[str] = []
        if len(orientations) > 1:
            parts.append(_plane_name(key[2]))
        if len(sizes) > 1:
            parts.append(f"{key[0]}x{key[1]}")
        if len(spacings) > 1 and key[3] is not None:
            parts.append(f"{key[3][0]:g}mm")
        labels.append("_".join(parts) or "GEOM")
    # Disambiguate labels that still collide (e.g. two oblique planes).
    seen: dict[str, int] = {}
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
    counters: dict[str, int] = {}
    out: list[tuple[str, list[Any]]] = []
    for key, label in zip(keys, labels):
        if seen[label] > 1:
            counters[label] = counters.get(label, 0) + 1
            label = f"{label}_{counters[label]}"
        out.append((label, groups[key]))
    return out


# ---------------------------------------------------------------------------
# Temporal layout
# ---------------------------------------------------------------------------


#: Ordering attributes, most trusted first: (name, reader, is_time_in_seconds).
def _order_readers() -> list[tuple[str, Any, bool]]:
    return [
        ("TemporalPositionIdentifier", lambda d: _float(d.get("TemporalPositionIdentifier", None)), False),
        ("TriggerTime", lambda d: (lambda v: None if v is None else v / 1000.0)(_float(d.get("TriggerTime", None))), True),
        ("NominalPercentageOfCardiacPhase",
         lambda d: _float(d.get("NominalPercentageOfCardiacPhase", None)), False),
        ("AcquisitionDateTime",
         lambda d: _datetime_seconds(d, "AcquisitionDateTime", "AcquisitionDate", "AcquisitionTime"), True),
        ("ContentTime", lambda d: _datetime_seconds(d, "", "ContentDate", "ContentTime"), True),
        ("InstanceNumber", lambda d: _float(d.get("InstanceNumber", None)), False),
    ]


@dataclass
class TemporalLayout:
    """A dynamic series arranged as ``frames[t][z]``."""

    #: Slice positions along the normal, in stacking order (mm).
    positions: list[float]
    #: ``n_t`` lists of ``n_z`` datasets, each list sorted like *positions*.
    frames: list[list[Any]]
    #: Seconds from the first frame (mean over each frame's slices); index when unknown.
    times_s: list[float]
    #: Attribute the frames were ordered by.
    order_source: str
    #: Attribute the times came from (``"index"`` when no time attribute varied).
    time_source: str
    #: Instances left out because a position had more repeats than the others.
    dropped: int = 0
    #: Per-frame cardiac phase (%), when the series records one.
    cardiac_phases: list[float] | None = field(default=None)

    @property
    def n_t(self) -> int:
        """Number of time points."""
        return len(self.frames)

    @property
    def n_z(self) -> int:
        """Number of slice positions per time point."""
        return len(self.positions)

    @property
    def t_res(self) -> float:
        """Median frame interval in seconds (1.0 when time is an index)."""
        if len(self.times_s) < 2:
            return 1.0
        diffs = np.diff(np.asarray(self.times_s, dtype=float))
        diffs = diffs[diffs > 0]
        return float(np.median(diffs)) if diffs.size else 1.0


def _cluster_positions(values: Sequence[float]) -> list[float]:
    """Distinct slice positions (within :data:`POSITION_TOL_MM`), ascending."""
    out: list[float] = []
    for v in sorted(values):
        if not out or abs(v - out[-1]) > POSITION_TOL_MM:
            out.append(v)
    return out


def _ordering(ds_items: Sequence[Any]) -> tuple[str, Any, bool]:
    """The first ordering attribute that is present on every item and varies."""
    for name, reader, is_time in _order_readers():
        vals = [reader(d) for d in ds_items]
        if any(v is None for v in vals):
            continue
        if len({round(float(v), 6) for v in vals}) > 1:
            return name, reader, is_time
    return "file order", None, False


def detect_temporal_layout(ds_list: Sequence[Any]) -> TemporalLayout | None:
    """
    Recognise a dynamic series: one geometry, slice positions repeated over time.

    Returns ``None`` for an ordinary volume (every position seen once) or when
    the geometry needed to decide is missing. A series whose positions repeat
    unevenly keeps, per position, as many frames as the *least* repeated position
    has — the common time base — and reports how many instances it dropped.
    """
    if len(ds_list) < 2:
        return None
    if len({geometry_signature(ds) for ds in ds_list}) != 1:
        return None
    normal = slice_normal(ds_list[0])
    if normal is None:
        return None
    pos_of: list[tuple[float, Any]] = []
    for ds in ds_list:
        p = slice_position(ds, normal)
        if p is None:
            return None
        pos_of.append((p, ds))
    positions = _cluster_positions([p for p, _ in pos_of])
    if len(positions) == len(ds_list):
        return None  # every position once: an ordinary volume

    # ---- 1. Bucket instances by slice position ---------------------------------
    buckets: list[list[Any]] = [[] for _ in positions]
    for p, ds in pos_of:
        idx = int(np.argmin([abs(p - q) for q in positions]))
        buckets[idx].append(ds)
    counts = [len(b) for b in buckets]
    n_t = min(counts)
    if n_t < 2:
        # Some positions are seen once: overlapping stacks, not a time series.
        return None

    # ---- 2. Order each position's instances in time ----------------------------
    order_name, reader, _ = _ordering([ds for _, ds in pos_of])
    for b in buckets:
        if reader is not None:
            b.sort(key=lambda d: float(reader(d)))
        else:
            b.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0) or 0))
    dropped = sum(c - n_t for c in counts)

    # ---- 3. Stack in dicom2nifti's slice order --------------------------------
    order = _dicom2nifti_slice_order(buckets)
    frames = [[buckets[z][t] for z in order] for t in range(n_t)]
    stack_positions = [positions[z] for z in order]

    # ---- 4. Frame times --------------------------------------------------------
    times, time_source = _frame_times(frames)
    phases = None
    phase_vals = [_float(f[0].get("NominalPercentageOfCardiacPhase", None)) for f in frames]
    if all(v is not None for v in phase_vals):
        phases = [float(v) for v in phase_vals]
    return TemporalLayout(
        positions=stack_positions,
        frames=frames,
        times_s=times,
        order_source=order_name,
        time_source=time_source,
        dropped=dropped,
        cardiac_phases=phases,
    )


def _dicom2nifti_slice_order(buckets: Sequence[Sequence[Any]]) -> list[int]:
    """Bucket order matching ``dicom2nifti.common.sort_dicoms`` (most-varying axis, ascending)."""
    reps = [_floats(b[0].get("ImagePositionPatient", None), 3) for b in buckets]
    arr = np.asarray(reps, dtype=float)
    if arr.shape[0] <= 1:
        return list(range(arr.shape[0]))
    spread = arr.max(axis=0) - arr.min(axis=0)
    # Ties go to the earlier axis, as in sort_dicoms' x-then-y-then-z checks.
    axis = int(np.argmax(spread))
    return [int(i) for i in np.argsort(arr[:, axis], kind="stable")]


def _frame_times(frames: Sequence[Sequence[Any]]) -> tuple[list[float], str]:
    """Per-frame seconds from the first frame, and the attribute they came from."""
    for name, reader, is_time in _order_readers():
        if not is_time:
            continue
        per_frame: list[float] = []
        for frame in frames:
            vals = [reader(d) for d in frame]
            if any(v is None for v in vals):
                per_frame = []
                break
            per_frame.append(float(np.mean(vals)))
        if len(per_frame) == len(frames) and len({round(v, 6) for v in per_frame}) > 1:
            t0 = per_frame[0]
            return [v - t0 for v in per_frame], name
    return [float(i) for i in range(len(frames))], "index"


# ---------------------------------------------------------------------------
# Volume building
# ---------------------------------------------------------------------------


def rescaled_slice(ds: Any) -> np.ndarray:
    """One slice's pixels with ``RescaleSlope``/``RescaleIntercept`` applied.

    Integer slope and intercept (every CT: 1 and -1024) stay in integer
    arithmetic, as ``dicom2nifti`` does — a dynamic CT is many volumes, and
    float32 would double what int16 holds exactly.
    """
    arr = ds.pixel_array
    slope = _float(ds.get("RescaleSlope", None))
    intercept = _float(ds.get("RescaleIntercept", None))
    slope = 1.0 if slope is None else slope
    intercept = 0.0 if intercept is None else intercept
    if slope == 1.0 and intercept == 0.0:
        return arr
    if float(slope).is_integer() and float(intercept).is_integer():
        return arr.astype(np.int32) * int(slope) + int(intercept)
    return arr.astype(np.float32) * np.float32(slope) + np.float32(intercept)


def storage_dtype(arrays: Sequence[np.ndarray]) -> Any:
    """Smallest dtype holding every value of *arrays* exactly (float32 for floats)."""
    if any(a.dtype.kind == "f" for a in arrays):
        return np.float32
    lo = min(int(a.min()) for a in arrays)
    hi = max(int(a.max()) for a in arrays)
    for dtype in (np.int16, np.int32):
        info = np.iinfo(dtype)
        if info.min <= lo and hi <= info.max:
            return dtype
    return np.int64


def dicom2nifti_affine(sorted_slices: Sequence[Any]) -> np.ndarray:
    """
    RAS voxel-to-world affine for ``(columns, rows, slices)`` voxels.

    The same formula as ``dicom2nifti.common.create_affine``, except that a single
    slice — or a step of zero — uses ``SpacingBetweenSlices`` / ``SliceThickness``
    along the normal instead of raising ``NOT_A_VOLUME``.
    """
    first = sorted_slices[0]
    iop = _floats(first.get("ImageOrientationPatient", None), 6)
    ipp = _floats(first.get("ImagePositionPatient", None), 3)
    ps = _floats(first.get("PixelSpacing", None), 2)
    if iop is None or ipp is None:
        raise ValueError("ImageOrientationPatient / ImagePositionPatient are required.")
    if ps is None:
        ps = (1.0, 1.0)
    orient1 = np.asarray(iop[:3])
    orient2 = np.asarray(iop[3:6])
    delta_r, delta_c = float(ps[0]), float(ps[1])
    pos = np.asarray(ipp[:3])
    step = np.zeros(3)
    if len(sorted_slices) > 1:
        last = np.asarray(_floats(sorted_slices[-1].get("ImagePositionPatient", None), 3))
        step = (pos - last) / (1 - len(sorted_slices))
    if float(np.linalg.norm(step)) == 0.0:
        thick = _float(first.get("SpacingBetweenSlices", None)) or _float(first.get("SliceThickness", None)) or 1.0
        step = -np.cross(orient1, orient2) * abs(float(thick))
    return np.array(
        [
            [-orient1[0] * delta_c, -orient2[0] * delta_r, -step[0], -pos[0]],
            [-orient1[1] * delta_c, -orient2[1] * delta_r, -step[1], -pos[1]],
            [orient1[2] * delta_c, orient2[2] * delta_r, step[2], pos[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def build_temporal_image(layout: TemporalLayout) -> tuple[Any, dict[str, Any]]:
    """
    Write *layout* as a NIfTI image: ``(columns, rows, n_z, n_t)``.

    Returns
    -------
    (image, extra_metadata)
        A ``nibabel.Nifti1Image`` with spatial units mm and temporal units s
        (``pixdim[4]`` = the median frame interval), and the metadata the caller
        should merge into the series' sidecar.
    """
    if nib is None:
        raise RuntimeError("nibabel is required to build a temporal NIfTI.")
    # ---- 1. Pixels, frame by frame, already in HU / scanner units ------------
    first = [rescaled_slice(ds) for ds in layout.frames[0]]
    rows, cols = first[0].shape[:2]
    volume = np.empty((cols, rows, layout.n_z, layout.n_t), dtype=storage_dtype(first))
    for t, frame in enumerate(layout.frames):
        slices = first if t == 0 else [rescaled_slice(ds) for ds in frame]
        if t:
            # A later frame can leave the first frame's range (contrast arriving):
            # widen rather than wrap.
            wanted = np.promote_types(volume.dtype, storage_dtype(slices))
            if wanted != volume.dtype:
                volume = volume.astype(wanted)
        for z, arr in enumerate(slices):
            # (rows, cols) → (cols, rows): dicom2nifti's transpose(2, 1, 0).
            volume[:, :, z, t] = arr.T

    # ---- 2. Geometry ----------------------------------------------------------
    affine = dicom2nifti_affine(layout.frames[0])
    image = nib.Nifti1Image(volume, affine)
    image.header.set_xyzt_units("mm", "sec")
    zooms = list(image.header.get_zooms())
    zooms[3] = float(layout.t_res)
    image.header.set_zooms(tuple(zooms))
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)

    extra: dict[str, Any] = {
        "axes": "XYZT",
        "dynamic": True,
        "n_timepoints": int(layout.n_t),
        "n_slices_per_timepoint": int(layout.n_z),
        "t_res": float(layout.t_res),
        "temporal_resolution": float(layout.t_res),
        "frame_times_s": [round(float(t), 6) for t in layout.times_s],
        "temporal_order_source": layout.order_source,
        "temporal_time_source": layout.time_source,
    }
    if layout.cardiac_phases is not None:
        extra["cardiac_phases_percent"] = layout.cardiac_phases
    if layout.dropped:
        extra["temporal_dropped_instances"] = int(layout.dropped)
    return image, extra


__all__ = [
    "POSITION_TOL_MM",
    "TemporalLayout",
    "build_temporal_image",
    "detect_temporal_layout",
    "dicom2nifti_affine",
    "geometry_signature",
    "rescaled_slice",
    "storage_dtype",
    "series_label",
    "slice_normal",
    "slice_position",
    "split_by_geometry",
]
