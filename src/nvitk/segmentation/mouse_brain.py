"""Mouse brain extraction / parcellation via ANTsPyNet."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import (
    ants_result_to_array,
    require_ants,
    require_antspynet,
    to_ants_image,
)
from nvitk.types import Image

log = Logger()

MouseBrainMode = Literal["extraction", "parcellation"]
MOUSE_EXTRACTION_MODALITIES: tuple[str, ...] = ("t2", "ex5coronal", "ex5sagittal")
MouseModality = Literal["t2", "ex5coronal", "ex5sagittal"]
MouseParcellation = Literal["nick", "tct", "jay"]


def _fmt_tuple(values: Any, *, nd: int = 4) -> str:
    """Format a numeric tuple for logging with *nd* significant digits."""
    try:
        seq = tuple(float(x) for x in values)
    except Exception:
        return repr(values)
    return "(" + ", ".join(f"{v:.{nd}g}" for v in seq) + ")"


def _intensity_stats(arr: np.ndarray) -> str:
    """One-line min/max/mean/std/p01/p99 summary of finite voxels (for QC logging)."""
    a = np.asarray(arr, dtype=np.float64)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return "empty/non-finite"
    return (
        f"min={finite.min():.4g} max={finite.max():.4g} "
        f"mean={finite.mean():.4g} std={finite.std():.4g} "
        f"p01={np.percentile(finite, 1):.4g} p99={np.percentile(finite, 99):.4g}"
    )


def _bbox_and_counts(mask: np.ndarray) -> str:
    """Log summary: foreground voxel count, percentage, and tight bounding box."""
    fg = np.asarray(mask) > 0
    n = int(fg.sum())
    if n == 0:
        return "foreground=0"
    coords = np.argwhere(fg)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return (
        f"foreground={n} ({100.0 * n / fg.size:.2f}% of voxels) "
        f"bbox_lo={tuple(int(x) for x in lo)} bbox_hi={tuple(int(x) for x in hi)}"
    )


def _label_histogram(seg: np.ndarray, *, max_labels: int = 20) -> str:
    """Log summary: per-label voxel counts (largest first), truncated to *max_labels*."""
    vals, counts = np.unique(np.asarray(seg), return_counts=True)
    # Drop background 0 from the headline if present.
    pairs = sorted(zip(vals.tolist(), counts.tolist()), key=lambda x: (-x[1], x[0]))
    parts = [f"{int(v)}:{int(c)}" for v, c in pairs[:max_labels]]
    extra = "" if len(pairs) <= max_labels else f" …(+{len(pairs) - max_labels} more)"
    return f"n_labels={len(pairs)} hist=[{', '.join(parts)}]{extra}"


def _log_ants_geometry(tag: str, ants_img: Any, *, data: np.ndarray | None = None) -> None:
    """Log an ANTs image's geometry (shape, spacing, origin, direction) under *tag* for QC."""
    shape = tuple(int(s) for s in ants_img.shape)
    sp = tuple(float(x) for x in ants_img.spacing)
    origin = tuple(float(x) for x in ants_img.origin)
    try:
        direction = np.asarray(ants_img.direction, dtype=float)
        dir_s = np.array2string(direction, precision=3, suppress_small=True)
    except Exception:
        dir_s = "?"
    fov = tuple(shape[i] * sp[i] for i in range(min(len(shape), len(sp))))
    log.info(
        f"mouse brain [{tag}]: shape={shape} spacing={_fmt_tuple(sp)} "
        f"fov_mm≈{_fmt_tuple(fov)} origin={_fmt_tuple(origin)}"
    )
    log.info(f"mouse brain [{tag}]: direction=\n{dir_s}")
    arr = data if data is not None else np.asarray(ants_img.numpy())
    log.info(f"mouse brain [{tag}]: intensity {_intensity_stats(arr)}")


def _spacing_looks_implausible(ants_img, shape: tuple[int, ...]) -> bool:
    """True when spacing looks like unit voxels instead of mouse-mm geometry."""
    sp = tuple(float(x) for x in ants_img.spacing)
    if len(sp) < 3:
        return False
    mn = min(sp)
    return mn >= 0.5 and max(shape[:3]) >= 64


def _warn_if_spacing_implausible(ants_img, shape: tuple[int, ...]) -> None:
    """Mouse MRI is typically ~0.05–0.2 mm; unit spacing breaks template COM align."""
    sp = tuple(float(x) for x in ants_img.spacing)
    if len(sp) < 3:
        return
    mn, mx = min(sp), max(sp)
    if _spacing_looks_implausible(ants_img, shape):
        fov = tuple(shape[i] * sp[i] for i in range(3))
        log.warning(
            f"Mouse brain: voxel spacing={_fmt_tuple(sp)} → fov_mm≈{_fmt_tuple(fov)} "
            f"is far too large for a mouse brain (template FOV ≈ 20 mm). "
            "ANTsPyNet COM alignment will fail and produce a tiny/random blob."
        )
    elif mx / max(mn, 1e-8) > 50:
        log.warning(
            f"Mouse brain: extreme anisotropy spacing={_fmt_tuple(sp)} (shape={shape}). "
            "Results may be poor; consider isotropic resampling first."
        )


def _fix_spacing_to_mouse_fov(
    ants_img: Any,
    *,
    target_fov_mm: float = 20.0,
) -> Any:
    """
    Uniformly rescale spacing/origin so max FOV matches a mouse-sized FOV.

    Used when the NIfTI header has unit spacing (1,1,1) but the volume is a
    mouse brain. The ANTsPyNet T2 template FOV is ~20 mm.
    """
    ants = require_ants()
    shape = tuple(int(s) for s in ants_img.shape)
    sp = tuple(float(x) for x in ants_img.spacing)
    origin = tuple(float(x) for x in ants_img.origin)
    fov = tuple(shape[i] * sp[i] for i in range(min(3, len(shape), len(sp))))
    max_fov = max(fov) if fov else 0.0
    if max_fov <= 0:
        return ants_img
    factor = float(target_fov_mm) / float(max_fov)
    new_sp = tuple(s * factor for s in sp)
    new_origin = tuple(o * factor for o in origin)
    log.info(
        f"mouse brain: fixing spacing { _fmt_tuple(sp) } → {_fmt_tuple(new_sp)} "
        f"(scale×{factor:.4g}) so max FOV {max_fov:.4g} mm → {target_fov_mm:.4g} mm"
    )
    # Clone to avoid mutating caller's image metadata unexpectedly.
    fixed = ants.image_clone(ants_img)
    fixed.set_spacing(list(new_sp))
    fixed.set_origin(list(new_origin))
    new_fov = tuple(shape[i] * new_sp[i] for i in range(3))
    log.info(
        f"mouse brain: after spacing fix fov_mm≈{_fmt_tuple(new_fov)} "
        f"origin={_fmt_tuple(new_origin)}"
    )
    return fixed


def mouse_brain_segmentation(
    image: Image | np.ndarray,
    *,
    mode: MouseBrainMode = "extraction",
    modality: str = "t2",
    which_parcellation: str = "nick",
    mask: Image | np.ndarray | None = None,
    do_n4: bool = True,
    binarize: bool = True,
    threshold: float = 0.5,
    return_isotropic_output: bool = False,
    which_axis: int = 2,
    fix_spacing: bool = True,
    target_fov_mm: float = 20.0,
    verbose: bool = False,
) -> np.ndarray:
    """Segment a mouse brain MRI with ANTsPyNet deep models.

    Parameters
    ----------
    image
        Input intensity volume. **T2-weighted mouse MRI** for ``t2`` / nick / tct;
        histology stacks for ``ex5*``. Correct voxel spacing (mm) is required.
    mode
        ``extraction`` → brain mask (probability, optionally binarized).
        ``parcellation`` → regional labels (nick / tct / jay).
    modality
        Extraction only: ``t2``, ``ex5coronal``, or ``ex5sagittal``.
        There is **no T1 mouse extraction model** in ANTsPyNet.
    which_parcellation
        ``nick``, ``tct``, or ``jay``.
    mask
        Optional brain mask for parcellation; estimated via T2 extraction if omitted.
    do_n4
        Run ANTs N4 bias correction first (recommended; matches ANTsX examples).
    binarize
        For extraction, threshold the probability map (default True).
    threshold
        Probability cutoff when *binarize* is True.
    return_isotropic_output
        Forwarded to ANTsPyNet (resample output to isotropic min-spacing).
    which_axis
        Slice axis for ``ex5*`` modalities.
    fix_spacing
        If True (default), when spacing looks like unit voxels, rescale so the
        max FOV matches *target_fov_mm* (~mouse template size).
    target_fov_mm
        Target max FOV in mm used by *fix_spacing* (default 20).
    verbose
        Forwarded to ANTsPyNet (prints internal preprocess / predict steps).
        N4 always runs quietly to avoid flooding the log.
    """
    mode_l = str(mode).strip().lower()
    mod = str(modality).strip().lower()
    parc = str(which_parcellation).strip().lower()

    log.info("=" * 60)
    log.info(
        f"mouse brain: start mode={mode_l!r} modality={mod!r} "
        f"parcellation={parc!r} do_n4={bool(do_n4)} binarize={bool(binarize)} "
        f"threshold={float(threshold)} isotropic_out={bool(return_isotropic_output)} "
        f"fix_spacing={bool(fix_spacing)} target_fov_mm={float(target_fov_mm)} "
        f"which_axis={int(which_axis)} verbose={bool(verbose)}"
    )

    antspynet = require_antspynet()
    ants = require_ants()
    log.info("mouse brain: ANTsPy / ANTsPyNet imports OK")

    raw = to_numpy(getattr(image, "data", image))
    meta = getattr(image, "metadata", None) or {}
    log.info(
        f"mouse brain: input array dtype={raw.dtype} shape={tuple(raw.shape)} "
        f"name={getattr(image, 'name', None)!r} "
        f"has_affine={'affine' in meta or getattr(image, 'affine', None) is not None} "
        f"meta_spacing={meta.get('spacing')!r}"
    )

    ants_img = to_ants_image(image)
    shape = tuple(int(s) for s in raw.shape)
    _log_ants_geometry("input→ants", ants_img, data=raw)
    _warn_if_spacing_implausible(ants_img, shape)

    if fix_spacing and _spacing_looks_implausible(ants_img, shape):
        ants_img = _fix_spacing_to_mouse_fov(
            ants_img, target_fov_mm=float(target_fov_mm)
        )
        _log_ants_geometry("after spacing fix", ants_img)
    elif fix_spacing:
        log.info("mouse brain: spacing looks plausible — no FOV rescale")
    else:
        log.info("mouse brain: fix_spacing=False — leaving header spacing unchanged")

    if do_n4:
        import contextlib
        import io

        log.info("mouse brain: running N4 bias-field correction (shrink=2, spline=20 mm)…")
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            ants_img = ants.n4_bias_field_correction(
                ants_img,
                rescale_intensities=True,
                shrink_factor=2,
                convergence={"iters": [50, 50, 50, 50], "tol": 0.0},
                spline_param=20,
                verbose=False,
            )
        _log_ants_geometry("after N4", ants_img)
    else:
        log.info("mouse brain: skipping N4 (do_n4=False)")

    if mode_l in ("extraction", "extract", "brain_extraction", "mask"):
        if mod == "t1":
            raise ValueError(
                "ANTsPyNet mouse_brain_extraction has no T1 model. "
                f"Use modality in {MOUSE_EXTRACTION_MODALITIES} "
                "(typically 't2' for mouse MRI)."
            )
        if mod not in MOUSE_EXTRACTION_MODALITIES:
            raise ValueError(
                f"Unrecognized mouse extraction modality {modality!r}; "
                f"expected one of {MOUSE_EXTRACTION_MODALITIES}."
            )
        log.info(
            f"mouse brain: calling antspynet.mouse_brain_extraction(modality={mod!r})…"
        )
        out = antspynet.mouse_brain_extraction(
            ants_img,
            modality=mod,
            return_isotropic_output=bool(return_isotropic_output),
            which_axis=int(which_axis),
            verbose=bool(verbose),
        )
        arr = ants_result_to_array(out)
        log.info(
            f"mouse brain: extraction probability map shape={arr.shape} dtype={arr.dtype} "
            f"{_intensity_stats(arr)}"
        )
        soft_fg = int(np.count_nonzero(arr > 0.1))
        soft_hi = int(np.count_nonzero(arr >= float(threshold)))
        log.info(
            f"mouse brain: soft mask voxels>0.1={soft_fg} "
            f"voxels>={threshold}={soft_hi}"
        )
        if binarize:
            binary = (arr >= float(threshold)).astype(np.uint8)
            log.info(
                f"mouse brain: binarized @ {threshold} → {_bbox_and_counts(binary)}"
            )
            if int(np.count_nonzero(binary)) < 100:
                log.warning(
                    "Extracted mask is nearly empty. Check voxel spacing (must be in mm), "
                    "orientation, and that the image is T2 mouse MRI."
                )
            log.info("mouse brain: extraction done")
            log.info("=" * 60)
            return binary
        log.info("mouse brain: returning probability map (binarize=False)")
        log.info("=" * 60)
        return arr

    if mode_l in ("parcellation", "parcel", "labels"):
        ants_mask = None
        if mask is not None:
            ants_mask = to_ants_image(mask)
            mask_arr = np.asarray(ants_mask.numpy())
            log.info(
                f"mouse brain: user mask before thresh shape={mask_arr.shape} "
                f"{_intensity_stats(mask_arr)} {_bbox_and_counts(mask_arr)}"
            )
            ants_mask = ants.threshold_image(ants_mask, 0.5, 1e12, 1, 0)
            mask_bin = np.asarray(ants_mask.numpy())
            log.info(
                f"mouse brain: user mask after thresh≥0.5 → {_bbox_and_counts(mask_bin)}"
            )
        else:
            log.info(
                "mouse brain: no user mask — ANTsPyNet will auto-extract (modality=t2)"
            )

        log.info(
            f"mouse brain: calling antspynet.mouse_brain_parcellation("
            f"which_parcellation={parc!r})…"
        )
        out = antspynet.mouse_brain_parcellation(
            ants_img,
            mask=ants_mask,
            return_isotropic_output=bool(return_isotropic_output),
            which_parcellation=parc,
            verbose=bool(verbose),
        )
        if isinstance(out, dict):
            keys = list(out.keys())
            log.info(f"mouse brain: parcellation returned dict keys={keys}")
        arr = ants_result_to_array(out)
        log.info(
            f"mouse brain: segmentation shape={arr.shape} dtype={arr.dtype} "
            f"{_bbox_and_counts(arr)}"
        )
        log.info(f"mouse brain: {_label_histogram(arr)}")
        log.info("mouse brain: parcellation done")
        log.info("=" * 60)
        return arr

    raise ValueError(
        f"Unknown mouse brain mode {mode!r}; use 'extraction' or 'parcellation'."
    )


__all__ = [
    "MOUSE_EXTRACTION_MODALITIES",
    "MouseBrainMode",
    "MouseModality",
    "mouse_brain_segmentation",
]
