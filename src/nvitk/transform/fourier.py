"""
K-space — forward/inverse FFT of images and volumes, and k-space filtering.

Description
-----------
The spatial Fourier transform of an :class:`~nvitk.types.Image`, in the
convention MRI uses for k-space, plus the inverse and the usual k-space
manipulations (component display, radial low/high/band filters). A 3D+t image is
transformed over its **spatial** axes only, one transform per time point.

Two transform modes:

- ``"3d"`` — over every spatial axis (a 3D acquisition's k-space);
- ``"2d"`` — over the two in-plane axes, slice by slice (a 2D multi-slice
  acquisition's k-space). The slice axis defaults to the spatial axis with the
  coarsest spacing, which is the through-plane axis of any 2D stack.

Convention
----------
``centered=True`` (the default) uses the symmetric, phase-correct form

    k = fftshift(fftn(ifftshift(x)))        x = fftshift(ifftn(ifftshift(k)))

so DC sits in the middle of the k-space grid *and* the image origin is taken as
the grid centre — the phase of a centred object is then flat rather than a
checkerboard. ``norm="ortho"`` keeps energy (Parseval) and makes the inverse the
adjoint. The inverse reads every setting back from the k-space metadata, so a
round trip needs no arguments.

Array / axis conventions
------------------------
Axis roles come from ``image.axes`` (``X``/``Y``/``Z`` spatial, ``T``/``C``
non-spatial), defaulting to the NIfTI order ``XYZ[T]``. The k-space image keeps
the source affine, so it is drawn *pixel-for-pixel* over the image it came from;
the physical frequency step of each axis, ``1 / (N * dx)`` in cycles/mm, is stored
in ``metadata["kspace"]["frequency_spacing"]``.

Backends
--------
Host arrays use ``scipy.fft`` (pocketfft, multi-threaded with the shared worker
budget of :mod:`nvitk.core.parallel`); CuPy arrays use ``cupyx.scipy.fft`` on the
GPU. Single precision stays single precision (float32 → complex64).
"""

from __future__ import annotations

from typing import Any, Sequence

from nvitk.core.array import as_backend_array
from nvitk.core.backend import get_current_backend, setup
from nvitk.core.logger import Logger
from nvitk.types import Image

setup(globals())

log = Logger()

#: Metadata key holding everything the inverse needs.
KSPACE_KEY = "kspace"

#: Display components :func:`kspace_component` can extract.
KSPACE_COMPONENTS: tuple[str, ...] = ("log_magnitude", "magnitude", "phase", "real", "imag")

#: Radial filter shapes for :func:`kspace_filter_mask`.
FILTER_KINDS: tuple[str, ...] = ("lowpass", "highpass", "bandpass", "bandstop")
FILTER_WINDOWS: tuple[str, ...] = ("hann", "gaussian", "butterworth", "ideal")


# ---------------------------------------------------------------------------
# Axis roles
# ---------------------------------------------------------------------------


def _axes_string(image: Image) -> str:
    """*image*'s axis labels, defaulting to NIfTI ``XYZ[T…]`` order."""
    from nvitk.io._common import default_nifti_axes

    axes = str(image.axes or "").upper()
    return axes if len(axes) == image.ndim else default_nifti_axes(image.ndim).upper()


def _spatial_spacing(image: Image, spatial: Sequence[int], axes: str) -> list[float]:
    """mm spacing of each spatial array axis (1.0 where unknown)."""
    md = image.metadata or {}
    key = {"X": "x_res", "Y": "y_res", "Z": "z_res"}
    out: list[float] = []
    sp = image.spacing or ()
    for ax in spatial:
        val = md.get(key.get(axes[ax], ""))
        if val is None:
            # metadata["spacing"] is in X, Y, Z order for spatial axes.
            rank = "XYZ".find(axes[ax])
            val = sp[rank] if 0 <= rank < len(sp) else None
        try:
            fval = float(val) if val is not None else 1.0
        except (TypeError, ValueError):
            fval = 1.0
        out.append(fval if fval > 0 else 1.0)
    return out


def kspace_axes(
    image: Image,
    *,
    mode: str = "3d",
    slice_axis: int | None = None,
) -> tuple[int, ...]:
    """
    Array axes the transform runs over.

    Parameters
    ----------
    image
        Source image; ``image.axes`` decides which axes are spatial.
    mode
        ``"3d"`` — every spatial axis; ``"2d"`` — the in-plane pair only.
    slice_axis
        For ``"2d"``: the through-plane array axis. ``None`` picks the spatial
        axis with the coarsest spacing (ties → the ``Z`` axis).

    Raises
    ------
    ValueError
        When the image has fewer than two spatial axes, or *slice_axis* is not one.
    """
    axes = _axes_string(image)
    spatial = [i for i, ch in enumerate(axes) if ch in "XYZ"]
    if len(spatial) < 2:
        raise ValueError(f"K-space needs at least two spatial axes; image axes are {axes!r}.")
    key = str(mode).strip().lower()
    if key == "3d" or len(spatial) == 2:
        return tuple(spatial)
    if key != "2d":
        raise ValueError(f"Unknown k-space mode {mode!r}; use '3d' or '2d'.")
    if slice_axis is None:
        spacing = _spatial_spacing(image, spatial, axes)
        coarsest = max(spacing)
        candidates = [ax for ax, s in zip(spatial, spacing) if s >= coarsest - 1e-6]
        slice_axis = next((ax for ax in candidates if axes[ax] == "Z"), candidates[-1])
    if int(slice_axis) not in spatial:
        raise ValueError(f"slice_axis={slice_axis} is not a spatial axis of {axes!r}.")
    return tuple(ax for ax in spatial if ax != int(slice_axis))


# ---------------------------------------------------------------------------
# FFT backend
# ---------------------------------------------------------------------------


def _fft_ops() -> tuple[Any, dict[str, Any]]:
    """``(fft module, extra kwargs)`` for the active backend."""
    if get_current_backend() == "cupy":
        # cupyx.scipy.fft: no ``workers`` — the GPU is already parallel.
        return scipy.fft, {}
    from nvitk.core.parallel import get_worker_count

    return scipy.fft, {"workers": get_worker_count()}


def _as_float_or_complex(arr: Any) -> Any:
    """Integers → float32 (FFT of a 16-bit CT in float64 doubles memory for nothing)."""
    if np.iscomplexobj(arr):
        return arr
    if arr.dtype.kind in "iub":
        return arr.astype(np.float32)
    return arr


def _forward(arr: Any, axes: tuple[int, ...], centered: bool, norm: str) -> Any:
    """FFT over *axes* with the module's centring convention."""
    fft, kw = _fft_ops()
    data = fft.ifftshift(arr, axes=axes) if centered else arr
    out = fft.fftn(data, axes=axes, norm=norm, **kw)
    return fft.fftshift(out, axes=axes) if centered else out


def _inverse(arr: Any, axes: tuple[int, ...], centered: bool, norm: str) -> Any:
    """Inverse of :func:`_forward`."""
    fft, kw = _fft_ops()
    data = fft.ifftshift(arr, axes=axes) if centered else arr
    out = fft.ifftn(data, axes=axes, norm=norm, **kw)
    return fft.fftshift(out, axes=axes) if centered else out


# ---------------------------------------------------------------------------
# Forward / inverse
# ---------------------------------------------------------------------------


def kspace(
    image: Image,
    *,
    mode: str = "3d",
    slice_axis: int | None = None,
    centered: bool = True,
    norm: str = "ortho",
) -> Image:
    """
    K-space of *image*: its spatial FFT, as a complex :class:`~nvitk.types.Image`.

    Parameters
    ----------
    image
        2D, 3D or 3D+t image (any real or complex dtype).
    mode, slice_axis
        See :func:`kspace_axes`.
    centered
        DC in the middle of the grid, symmetric (phase-correct) convention.
    norm
        ``"ortho"`` (default, energy-preserving), ``"backward"`` or ``"forward"``.

    Returns
    -------
    Image
        Complex data on the source grid (same affine, axes and name + ``_kspace``);
        ``metadata["kspace"]`` records the axes, convention, source dtype and the
        per-axis frequency spacing in cycles/mm.
    """
    axes_str = _axes_string(image)
    axes = kspace_axes(image, mode=mode, slice_axis=slice_axis)
    data = _as_float_or_complex(as_backend_array(image.data))
    kdata = _forward(data, axes, bool(centered), str(norm))

    spacing = _spatial_spacing(image, axes, axes_str)
    freq = [1.0 / (float(image.shape[ax]) * float(dx)) for ax, dx in zip(axes, spacing)]
    md = dict(image.metadata or {})
    md[KSPACE_KEY] = {
        "axes": [int(a) for a in axes],
        "axis_labels": "".join(axes_str[a] for a in axes),
        "mode": str(mode).lower(),
        "centered": bool(centered),
        "norm": str(norm),
        "source_dtype": str(image.dtype),
        "source_is_complex": bool(np.iscomplexobj(image.data)),
        "spacing_mm": [float(s) for s in spacing],
        "frequency_spacing": [float(f) for f in freq],
        "frequency_units": "cycles/mm",
    }
    md["shape"] = tuple(int(s) for s in kdata.shape)
    log.info(
        "K-space (%s, axes %s, %s): %s → %s",
        md[KSPACE_KEY]["mode"], md[KSPACE_KEY]["axis_labels"],
        "centred" if centered else "uncentred", tuple(image.shape), kdata.dtype,
    )
    return Image(
        data=kdata,
        metadata=md,
        axes=image.axes,
        name=f"{image.name or 'image'}_kspace",
        orientation=image.orientation,
    )


def is_kspace(image: Image) -> bool:
    """True when *image* carries the k-space metadata :func:`kspace` writes."""
    return isinstance((image.metadata or {}).get(KSPACE_KEY), dict)


def inverse_kspace(kimage: Image, *, real: bool | None = None) -> Image:
    """
    Back to image space, using the settings stored by :func:`kspace`.

    Parameters
    ----------
    kimage
        Output of :func:`kspace` (possibly modified — masked, filtered, edited).
    real
        Return the real part. ``None`` → real unless the original was complex.
        The real part, not the magnitude: a filtered image legitimately has
        negative values (ringing, a high-pass), and ``abs`` would fold them up.

    Raises
    ------
    ValueError
        When *kimage* has no k-space metadata.
    """
    info = (kimage.metadata or {}).get(KSPACE_KEY)
    if not isinstance(info, dict):
        raise ValueError("Not a k-space image: metadata['kspace'] is missing.")
    axes = tuple(int(a) for a in info["axes"])
    data = as_backend_array(kimage.data)
    out = _inverse(data, axes, bool(info.get("centered", True)), str(info.get("norm", "ortho")))
    keep_real = (not bool(info.get("source_is_complex", False))) if real is None else bool(real)
    if keep_real:
        out = out.real
        if str(info.get("source_dtype", "")).startswith(("float32", "int", "uint")):
            out = out.astype(np.float32, copy=False)
    md = dict(kimage.metadata or {})
    md.pop(KSPACE_KEY, None)
    md["shape"] = tuple(int(s) for s in out.shape)
    name = str(kimage.name or "image")
    name = name[: -len("_kspace")] if name.endswith("_kspace") else name
    return Image(data=out, metadata=md, axes=kimage.axes, name=f"{name}_ifft",
                 orientation=kimage.orientation)


# ---------------------------------------------------------------------------
# Display components
# ---------------------------------------------------------------------------


def kspace_component(kimage: Image, component: str = "log_magnitude") -> Image:
    """
    One real-valued view of complex k-space, as float32.

    ``log_magnitude`` is ``log1p(|k|)`` — the only one of these that shows the
    periphery of k-space at all, since |k| spans many decades between DC and the
    edge. ``phase`` is in radians, ``(-π, π]``.
    """
    key = str(component).strip().lower()
    data = as_backend_array(kimage.data)
    if key == "magnitude":
        out = np.abs(data)
    elif key == "log_magnitude":
        out = np.log1p(np.abs(data))
    elif key == "phase":
        out = np.angle(data)
    elif key == "real":
        out = np.real(data)
    elif key in ("imag", "imaginary"):
        out = np.imag(data)
    else:
        raise ValueError(f"Unknown k-space component {component!r}; use one of {KSPACE_COMPONENTS}.")
    md = dict(kimage.metadata or {})
    md["kspace_component"] = key
    md["shape"] = tuple(int(s) for s in out.shape)
    return Image(data=out.astype(np.float32, copy=False), metadata=md, axes=kimage.axes,
                 name=f"{kimage.name or 'kspace'}_{key}", orientation=kimage.orientation)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def kspace_radius(kimage: Image) -> Any:
    """
    Normalised radial frequency of every k-space sample, broadcastable to the data.

    ``0`` at DC, ``1`` at the Nyquist frequency of each transformed axis (so an
    anisotropic grid gets an elliptical, not a circular, pass band in index space
    — circular in *normalised* frequency, which is what resolution means).
    """
    info = (kimage.metadata or {}).get(KSPACE_KEY)
    if not isinstance(info, dict):
        raise ValueError("Not a k-space image: metadata['kspace'] is missing.")
    axes = tuple(int(a) for a in info["axes"])
    centered = bool(info.get("centered", True))
    fft, _kw = _fft_ops()
    radius2 = None
    for ax in axes:
        n = int(kimage.shape[ax])
        # Cycles per sample in [-0.5, 0.5); ×2 → fraction of Nyquist.
        f = fft.fftfreq(n) * 2.0
        if centered:
            f = fft.fftshift(f)
        shape = [1] * kimage.ndim
        shape[ax] = n
        term = (as_backend_array(f).reshape(shape)) ** 2
        radius2 = term if radius2 is None else radius2 + term
    return np.sqrt(radius2)


def kspace_filter_mask(
    kimage: Image,
    *,
    kind: str = "lowpass",
    cutoff: float = 0.5,
    cutoff_high: float | None = None,
    window: str = "hann",
    order: int = 2,
) -> Any:
    """
    Radial k-space weighting in ``[0, 1]``.

    Parameters
    ----------
    kind
        ``lowpass``, ``highpass``, ``bandpass`` or ``bandstop``.
    cutoff
        Pass-band edge as a fraction of Nyquist (``0 < cutoff <= sqrt(ndim)``).
        For band filters, the lower edge.
    cutoff_high
        Upper band edge (band filters only).
    window
        ``hann`` (raised-cosine roll-off over ±10 % of the cutoff), ``gaussian``
        (σ = cutoff), ``butterworth`` (of *order*), or ``ideal`` (hard edge —
        rings, which is sometimes the point of showing it).
    """
    r = kspace_radius(kimage)

    def _low(edge: float) -> Any:
        edge = max(float(edge), 1e-6)
        w = str(window).strip().lower()
        if w == "ideal":
            return (r <= edge).astype(np.float32)
        if w == "gaussian":
            return np.exp(-0.5 * (r / edge) ** 2).astype(np.float32)
        if w == "butterworth":
            return (1.0 / (1.0 + (r / edge) ** (2 * max(int(order), 1)))).astype(np.float32)
        if w == "hann":
            width = 0.1 * edge
            t = np.clip((r - (edge - width)) / (2.0 * width), 0.0, 1.0)
            return (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
        raise ValueError(f"Unknown window {window!r}; use one of {FILTER_WINDOWS}.")

    key = str(kind).strip().lower()
    if key == "lowpass":
        return _low(cutoff)
    if key == "highpass":
        return (1.0 - _low(cutoff)).astype(np.float32)
    if key in ("bandpass", "bandstop"):
        if cutoff_high is None or float(cutoff_high) <= float(cutoff):
            raise ValueError("Band filters need cutoff_high > cutoff.")
        band = (_low(cutoff_high) - _low(cutoff)).clip(0.0, 1.0).astype(np.float32)
        return band if key == "bandpass" else (1.0 - band).astype(np.float32)
    raise ValueError(f"Unknown filter kind {kind!r}; use one of {FILTER_KINDS}.")


def apply_kspace_mask(kimage: Image, mask: Any) -> Image:
    """*kimage* multiplied by a real weighting (filter mask, painted ROI, …)."""
    weights = as_backend_array(mask)
    out = as_backend_array(kimage.data) * weights.astype(np.float32, copy=False)
    return kimage.with_data(out)


def kspace_filter(
    image: Image,
    *,
    kind: str = "lowpass",
    cutoff: float = 0.5,
    cutoff_high: float | None = None,
    window: str = "hann",
    order: int = 2,
    mode: str = "3d",
    slice_axis: int | None = None,
) -> Image:
    """
    Filter *image* in k-space: FFT → radial weighting → inverse FFT.

    Returns a real image on the source grid (complex sources stay complex). See
    :func:`kspace_filter_mask` for the filter parameters and :func:`kspace_axes`
    for *mode* / *slice_axis*.
    """
    kim = kspace(image, mode=mode, slice_axis=slice_axis)
    mask = kspace_filter_mask(kim, kind=kind, cutoff=cutoff, cutoff_high=cutoff_high,
                              window=window, order=order)
    out = inverse_kspace(apply_kspace_mask(kim, mask))
    out.name = f"{image.name or 'image'}_{str(kind).lower()}"
    return out


__all__ = [
    "FILTER_KINDS",
    "FILTER_WINDOWS",
    "KSPACE_COMPONENTS",
    "KSPACE_KEY",
    "apply_kspace_mask",
    "inverse_kspace",
    "is_kspace",
    "kspace",
    "kspace_axes",
    "kspace_component",
    "kspace_filter",
    "kspace_filter_mask",
    "kspace_radius",
]
