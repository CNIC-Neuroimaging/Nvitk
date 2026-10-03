"""
3D+t (time-series) operations on :class:`~nvitk.types.Image`.

Description
-----------
The everyday reductions of a dynamic volume — one frame, a projection over time,
the time–intensity curve of a region — and the inverse, stacking 3D volumes into
a 3D+t one. Used by the GUI's time tools and usable from any pipeline (dynamic
CT perfusion, bolus tracking, cine, phase stacks from ``dcm2nii --stack-phases``).

Array / axis conventions
------------------------
The time axis is found from ``image.axes`` (``T``, else ``C``), so ``XYZT`` files
and the GUI's time-first ``TXYZ`` layers are both handled; without axis labels a
4D array is taken as NIfTI ``XYZT``. Frame times come from
``metadata["frame_times_s"]`` when the converter recorded them, else from
``t_res`` (seconds per frame), else the frame index.

Outputs keep the spatial metadata (affine, spacing) and drop the temporal keys.
I/O and arrays: backend ``np`` after ``setup(globals())`` — CuPy arrays stay on
the GPU through every reduction.
"""

from __future__ import annotations

from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.types import Image

setup(globals())

#: Reductions :func:`temporal_projection` offers. ``ttp`` is the time to peak
#: (seconds, or frames when no timing is known); ``auc`` the area under the
#: curve by the trapezoidal rule over the frame times.
TEMPORAL_METHODS: tuple[str, ...] = ("max", "mean", "min", "std", "sum", "median", "ttp", "auc")

#: Metadata keys that describe the time axis and must not survive a reduction.
_TEMPORAL_KEYS: tuple[str, ...] = (
    "t_res", "temporal_resolution", "frame_times_s", "n_timepoints", "dynamic",
    "cardiac_phases_percent", "frame_offsets_percent_RR", "phase_stack",
    "temporal_order_source", "temporal_time_source", "t_units", "time_leading",
    "source_axes", "spectral_energies_kev", "spectral_stack", "stack_sources",
)


# ---------------------------------------------------------------------------
# Axis / timing helpers
# ---------------------------------------------------------------------------


def time_axis(image: Image) -> int:
    """Array axis holding time (``T``, else ``C``; ``XYZT`` order when unlabelled).

    Raises
    ------
    ValueError
        For an image that is not 4D.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected a 3D+t (4D) image; got ndim={image.ndim}.")
    axes = str(image.axes or "").upper()
    if len(axes) == 4:
        for ch in ("T", "C"):
            if ch in axes:
                return axes.index(ch)
    return 3


def frame_times(image: Image) -> list[float]:
    """Position of each frame on the fourth axis: seconds, keV, or the frame index.

    A monoenergetic spectral stack (``dcm2nii --stack-energies``) is a "series" whose
    frames are energies: its axis values are ``spectral_energies_kev``, so every tool
    built on frame times — the curve plot above all — works in keV unchanged. See
    :func:`frame_axis_unit` for the unit.
    """
    md = image.metadata or {}
    n = int(image.shape[time_axis(image)])
    energies = md.get("spectral_energies_kev")
    if isinstance(energies, (list, tuple)) and len(energies) == n:
        try:
            return [float(e) for e in energies]
        except (TypeError, ValueError):
            pass
    times = md.get("frame_times_s")
    if isinstance(times, (list, tuple)) and len(times) == n:
        try:
            return [float(t) for t in times]
        except (TypeError, ValueError):
            pass
    t_res = md.get("t_res", md.get("temporal_resolution"))
    try:
        step = float(t_res)
        if step > 0:
            return [i * step for i in range(n)]
    except (TypeError, ValueError):
        pass
    return [float(i) for i in range(n)]


def frame_axis_unit(image: Image) -> str:
    """Unit of :func:`frame_times`: ``"keV"``, ``"s"``, or ``"frame"``."""
    md = image.metadata or {}
    n = int(image.shape[time_axis(image)])
    energies = md.get("spectral_energies_kev")
    if isinstance(energies, (list, tuple)) and len(energies) == n:
        return "keV"
    if isinstance(md.get("frame_times_s"), (list, tuple)):
        return "s"
    try:
        if float(md.get("t_res", md.get("temporal_resolution"))) > 0 and md.get("t_units", "s") == "s":
            return "s"
    except (TypeError, ValueError):
        pass
    return str(md.get("t_units") or "frame")


def _spatial_image(image: Image, data: Any, *, name: str) -> Image:
    """Wrap a 3D result of *image* with its spatial metadata and axes."""
    t_ax = time_axis(image)
    md = {k: v for k, v in dict(image.metadata or {}).items() if k not in _TEMPORAL_KEYS}
    axes = str(image.axes or "").upper()
    spatial_axes = "".join(ch for i, ch in enumerate(axes) if i != t_ax) if len(axes) == 4 else "XYZ"
    md["axes"] = spatial_axes
    md["shape"] = tuple(int(v) for v in data.shape)
    return Image(data=data, metadata=md, axes=spatial_axes, name=name, orientation=image.orientation)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def extract_frame(image: Image, index: int) -> Image:
    """The 3D volume at time point *index* (negative counts from the end)."""
    t_ax = time_axis(image)
    n = int(image.shape[t_ax])
    idx = int(index) + n if int(index) < 0 else int(index)
    if not 0 <= idx < n:
        raise ValueError(f"Frame {index} is out of range for {n} time points.")
    key = [slice(None)] * 4
    key[t_ax] = idx
    data = as_backend_array(image.data)[tuple(key)]
    out = _spatial_image(image, data, name=f"{image.name or 'image'}_t{idx}")
    out.metadata["frame_index"] = idx
    out.metadata["frame_time_s"] = frame_times(image)[idx]
    return out


def temporal_projection(image: Image, method: str = "max") -> Image:
    """
    Collapse the time axis of *image* with *method*.

    Parameters
    ----------
    method
        One of :data:`TEMPORAL_METHODS`. ``ttp`` returns, per voxel, the time of
        its maximum (seconds when frame times are known); ``auc`` integrates the
        curve over the frame times (intensity·s).

    Returns
    -------
    Image
        Float32 3D image on the spatial grid of *image*.
    """
    key = str(method).strip().lower()
    if key not in TEMPORAL_METHODS:
        raise ValueError(f"Unknown temporal method {method!r}; use one of {TEMPORAL_METHODS}.")
    t_ax = time_axis(image)
    data = as_backend_array(image.data)
    if key == "max":
        out = np.max(data, axis=t_ax)
    elif key == "min":
        out = np.min(data, axis=t_ax)
    elif key == "mean":
        out = np.mean(data, axis=t_ax, dtype=np.float32)
    elif key == "std":
        out = np.std(data, axis=t_ax, dtype=np.float32)
    elif key == "sum":
        out = np.sum(data, axis=t_ax, dtype=np.float32)
    elif key == "median":
        out = np.median(data, axis=t_ax)
    elif key == "ttp":
        times = as_backend_array(frame_times(image)).astype(np.float32)
        out = times[np.argmax(data, axis=t_ax)]
    else:  # auc — trapezoid over the frame times
        times = as_backend_array(frame_times(image)).astype(np.float32)
        moved = np.moveaxis(data, t_ax, -1).astype(np.float32, copy=False)
        dt = np.diff(times)
        out = np.sum(0.5 * (moved[..., 1:] + moved[..., :-1]) * dt, axis=-1)
    out = out.astype(np.float32, copy=False)
    res = _spatial_image(image, out, name=f"{image.name or 'image'}_t{key}")
    res.metadata["temporal_projection"] = key
    return res


def time_intensity_curve(
    image: Image,
    *,
    mask: Any = None,
    voxel: Sequence[int] | None = None,
    statistic: str = "mean",
) -> tuple[list[float], list[float]]:
    """
    ``(times, values)`` — the time–intensity curve of a region or a voxel.

    Parameters
    ----------
    mask
        Boolean / label array on the spatial grid (``> 0`` is the region).
    voxel
        Spatial index ``(i, j, k)`` in the image's spatial axis order, when no
        *mask* is given.
    statistic
        ``mean`` (default), ``median``, ``max`` or ``sum`` over the region.

    Raises
    ------
    ValueError
        When neither a non-empty *mask* nor a *voxel* is given.
    """
    t_ax = time_axis(image)
    data = np.moveaxis(as_backend_array(image.data), t_ax, -1)
    times = frame_times(image)
    if mask is not None:
        region = as_backend_array(mask) > 0
        if tuple(region.shape) != tuple(data.shape[:3]):
            raise ValueError(f"Mask shape {tuple(region.shape)} != spatial shape {tuple(data.shape[:3])}.")
        if not bool(to_numpy(region.any())):
            raise ValueError("The mask is empty.")
        samples = data[region]  # (n_voxels, n_t)
        stat = str(statistic).lower()
        if stat == "median":
            values = np.median(samples, axis=0)
        elif stat == "max":
            values = np.max(samples, axis=0)
        elif stat == "sum":
            values = np.sum(samples, axis=0, dtype=np.float64)
        else:
            values = np.mean(samples, axis=0, dtype=np.float64)
    elif voxel is not None:
        i, j, k = (int(v) for v in voxel)
        values = data[i, j, k, :]
    else:
        raise ValueError("Give a mask or a voxel for the time–intensity curve.")
    return times, [float(v) for v in to_numpy(values)]


# ---------------------------------------------------------------------------
# Stacking
# ---------------------------------------------------------------------------


def stack_frames(
    images: Sequence[Image],
    *,
    frame_times_s: Sequence[float] | None = None,
    t_res: float | None = None,
    name: str | None = None,
) -> Image:
    """
    Stack 3D images on one grid into an ``XYZT`` image (time last).

    The first image's metadata (affine, spacing) is kept; all images must share
    its shape. *frame_times_s* (one per image) or a constant *t_res* set the
    timing; without either the time axis is a frame index.
    """
    if len(images) < 2:
        raise ValueError("Stacking needs at least two volumes.")
    first = images[0]
    shape = tuple(first.shape)
    for img in images[1:]:
        if tuple(img.shape) != shape:
            raise ValueError(f"All volumes must share one grid: {tuple(img.shape)} != {shape}.")
    arrays = [as_backend_array(img.data) for img in images]
    dtype = np.result_type(*[a.dtype for a in arrays])
    data = np.stack([a.astype(dtype, copy=False) for a in arrays], axis=-1)
    md = {k: v for k, v in dict(first.metadata or {}).items() if k not in _TEMPORAL_KEYS}
    md["axes"] = "XYZT"
    md["dynamic"] = True
    md["n_timepoints"] = len(images)
    if frame_times_s is not None:
        times = [float(t) for t in frame_times_s]
        if len(times) != len(images):
            raise ValueError("frame_times_s needs one time per volume.")
        md["frame_times_s"] = times
        diffs = [b - a for a, b in zip(times, times[1:]) if b > a]
        md["t_res"] = float(sorted(diffs)[len(diffs) // 2]) if diffs else 1.0
    else:
        md["t_res"] = float(t_res) if t_res else 1.0
    md["temporal_resolution"] = md["t_res"]
    md["shape"] = tuple(int(v) for v in data.shape)
    return Image(data=data, metadata=md, axes="XYZT", name=name or f"{first.name or 'stack'}_4d",
                 orientation=first.orientation)


__all__ = [
    "TEMPORAL_METHODS",
    "extract_frame",
    "frame_axis_unit",
    "frame_times",
    "stack_frames",
    "temporal_projection",
    "time_axis",
    "time_intensity_curve",
]
