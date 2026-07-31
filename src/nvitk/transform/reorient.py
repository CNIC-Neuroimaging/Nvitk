"""Reorient volumes by axis permute / flips / target codes (optionally vs a reference)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as _host_np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import ValidationError
from nvitk.io._common import orientation_codes_from_affine
from nvitk.transform.swap_axes import permute_axes
from nvitk.types import Image

# Preclinical TurboRARE-style volumes in this lab store anatomy as
# (axis0=L/R, axis1=S/I, axis2=A/P) even when the NIfTI header claims
# (L/R, A/P, S/I). The ANTsPy mouse gallery example stores (L/R, A/P, S/I)
# with LAS polarity (R→L, P→A, I→S).
MOUSE_ANATOMY_PERMUTE: tuple[int, int, int] = (0, 2, 1)
MOUSE_TARGET_ORIENTATION: str = "LAS"

_AXCODE_TO_RAS = {
    "R": _host_np.array([1.0, 0.0, 0.0]),
    "L": _host_np.array([-1.0, 0.0, 0.0]),
    "A": _host_np.array([0.0, 1.0, 0.0]),
    "P": _host_np.array([0.0, -1.0, 0.0]),
    "S": _host_np.array([0.0, 0.0, 1.0]),
    "I": _host_np.array([0.0, 0.0, -1.0]),
}


def _parse_permute(order: str | Sequence[int] | None) -> tuple[int, ...] | None:
    if order is None:
        return None
    if isinstance(order, str):
        text = order.strip()
        if not text:
            return None
        parts = [p.strip() for p in text.replace(" ", ",").split(",") if p.strip()]
        if len(parts) == 1 and len(parts[0]) == 3 and parts[0].isdigit():
            parts = list(parts[0])
        ord_t = tuple(int(p) for p in parts)
    else:
        ord_t = tuple(int(i) for i in order)
    if len(ord_t) != 3 or sorted(ord_t) != [0, 1, 2]:
        raise ValidationError(
            f"permute_order must be a permutation of 0,1,2; got {order!r}"
        )
    return ord_t


def _validate_axcodes(codes: str) -> str:
    target = str(codes).strip().upper()
    if len(target) != 3:
        raise ValidationError(f"orientation must have length 3; got {codes!r}")
    if any(c not in _AXCODE_TO_RAS for c in target):
        raise ValidationError(f"Invalid orientation codes: {codes!r}")
    pairs = (("L", "R"), ("A", "P"), ("S", "I"))
    axes_used = set()
    for c in target:
        for a, b in pairs:
            if c in (a, b):
                if a in axes_used or b in axes_used:
                    raise ValidationError(f"orientation repeats an axis pair: {codes!r}")
                axes_used.add(a)
                axes_used.add(b)
                break
    return target


def _spacing_from_image(image: Image) -> tuple[float, float, float]:
    sp = image.spacing
    if sp is not None and len(sp) >= 3:
        return (float(sp[0]), float(sp[1]), float(sp[2]))
    aff = image.affine
    if aff is not None:
        A = _host_np.asarray(aff, dtype=float)
        return tuple(float(_host_np.linalg.norm(A[:3, i])) for i in range(3))  # type: ignore[return-value]
    return (1.0, 1.0, 1.0)


def canonical_affine_for_axcodes(
    spacing: Sequence[float],
    axcodes: str,
    *,
    shape: Sequence[int] | None = None,
    origin: Sequence[float] | None = None,
) -> _host_np.ndarray:
    """Build a 4×4 voxel→RAS-world affine for data already stored as *axcodes*."""
    codes = _validate_axcodes(axcodes)
    sx, sy, sz = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    direction = _host_np.column_stack(
        [_AXCODE_TO_RAS[codes[0]], _AXCODE_TO_RAS[codes[1]], _AXCODE_TO_RAS[codes[2]]]
    )
    scales = _host_np.diag([sx, sy, sz])
    R = direction @ scales
    aff = _host_np.eye(4, dtype=float)
    aff[:3, :3] = R
    if origin is not None:
        aff[:3, 3] = _host_np.asarray(origin, dtype=float)[:3]
    elif shape is not None:
        center = (_host_np.asarray(shape[:3], dtype=float) - 1.0) * 0.5
        aff[:3, 3] = -(R @ center)
    return aff


def _apply_flips(image: Image, flip_axes: Sequence[bool] | None) -> Image:
    if not flip_axes:
        return image
    flags = [bool(x) for x in flip_axes[:3]]
    while len(flags) < 3:
        flags.append(False)
    if not any(flags):
        return image
    data = to_numpy(image.data, copy=True)
    aff = image.affine
    A = _host_np.asarray(aff, dtype=float).copy() if aff is not None else None
    for axis, do_flip in enumerate(flags):
        if not do_flip:
            continue
        data = _host_np.flip(data, axis=axis)
        if A is not None and A.shape == (4, 4):
            n = data.shape[axis]
            # x' = (n-1) - x  =>  column_axis *= -1; origin += (n-1)*old_column
            col = A[:3, axis].copy()
            A[:3, 3] = A[:3, 3] + (n - 1) * col
            A[:3, axis] = -col
    md = dict(image.metadata or {})
    if A is not None:
        md["affine"] = A
        codes = orientation_codes_from_affine(A)
        if codes:
            md["orientation"] = codes
        for i, key in enumerate(("x_res", "y_res", "z_res")):
            md[key] = float(_host_np.linalg.norm(A[:3, i]))
        md["spacing"] = (md["x_res"], md["y_res"], md["z_res"])
    ori = md.get("orientation", image.orientation)
    return Image(data=data, metadata=md, axes=image.axes, name=image.name, orientation=ori)


def _assign_canonical_orientation(image: Image, axcodes: str) -> Image:
    """Replace direction/origin so *axcodes* describe the current array layout."""
    codes = _validate_axcodes(axcodes)
    data = image.data
    shape = tuple(int(s) for s in getattr(data, "shape", ())[:3])
    spacing = _spacing_from_image(image)
    aff = canonical_affine_for_axcodes(spacing, codes, shape=shape)
    md = dict(image.metadata or {})
    md["affine"] = aff
    md["orientation"] = codes
    for i, key in enumerate(("x_res", "y_res", "z_res")):
        md[key] = float(spacing[i])
    md["spacing"] = spacing
    return Image(data=data, metadata=md, axes=image.axes, name=image.name, orientation=codes)


def reorient_volume(
    image: Image | Any,
    *,
    mode: str = "manual",
    reference: Image | Any | None = None,
    target_orientation: str | None = None,
    permute_order: str | Sequence[int] | None = None,
    flip_axes: Sequence[bool] | None = None,
    reset_affine: bool = False,
) -> Image:
    """
    Reorient a 3D volume via optional axis permute/flips and target axis codes.

    Parameters
    ----------
    image
        Input :class:`~nvitk.types.Image` (or array coerced via metadata-less Image).
    mode
        ``manual`` — permute / flip / target codes from arguments.
        ``reference`` — match *reference* orientation codes (after optional permute/flips).
        ``mouse`` — preclinical preset: permute ``(0,2,1)`` then canonical ``LAS``
        (AP moves from Z→Y; SI ends on Z), matching the ANTsPy mouse gallery layout.
    reference
        Reference image whose orientation codes are the target when ``mode='reference'``.
    target_orientation
        Three-letter codes (e.g. ``LAS``, ``RAS``). Used in manual mode, and as
        fallback when reference has no readable affine.
    permute_order
        Spatial axis permutation (e.g. ``\"0,2,1\"`` or ``(0, 2, 1)``).
    flip_axes
        Length-3 booleans for flipping array axes 0/1/2 after permute.
    reset_affine
        If True, after permute/flips assign a fresh canonical affine for
        *target_orientation* (use when the header axes do not match anatomy).
        Forced True for ``mode='mouse'``.
    """
    if not isinstance(image, Image):
        image = Image(data=image)
    if image.ndim < 3:
        raise ValidationError("reorient_volume requires at least 3 dimensions.")

    mode_l = str(mode or "manual").strip().lower()
    if mode_l in ("mouse", "mouse_preset", "ants_mouse"):
        permute = MOUSE_ANATOMY_PERMUTE
        target = MOUSE_TARGET_ORIENTATION
        reset_affine = True
    elif mode_l in ("reference", "ref"):
        permute = _parse_permute(permute_order)
        if reference is None:
            raise ValidationError("mode='reference' requires a reference image.")
        ref_img = reference if isinstance(reference, Image) else Image(data=reference)
        ref_codes = None
        if ref_img.affine is not None:
            ref_codes = orientation_codes_from_affine(ref_img.affine)
        if not ref_codes and ref_img.orientation:
            ref_codes = str(ref_img.orientation).upper()
        if not ref_codes and target_orientation:
            ref_codes = _validate_axcodes(target_orientation)
        if not ref_codes:
            raise ValidationError(
                "reference image has no readable orientation; pass target_orientation."
            )
        target = _validate_axcodes(ref_codes)
    elif mode_l in ("manual", "custom"):
        permute = _parse_permute(permute_order)
        target = _validate_axcodes(target_orientation) if target_orientation else None
    else:
        raise ValidationError(
            f"Unknown reorient mode {mode!r}; use 'manual', 'reference', or 'mouse'."
        )

    out = image
    if permute is not None and permute != (0, 1, 2):
        out = permute_axes(out, permute)

    out = _apply_flips(out, flip_axes)

    if reset_affine:
        if target is None:
            raise ValidationError("reset_affine requires target_orientation.")
        return _assign_canonical_orientation(out, target)

    if target is None:
        return out

    if out.affine is None:
        return _assign_canonical_orientation(out, target)

    current = orientation_codes_from_affine(out.affine)
    if current == target:
        out.orientation = target
        out.metadata["orientation"] = target
        return out
    return out.orient_to(target)


def mouse_reorient_volume(image: Image | Any) -> Image:
    """Apply the lab mouse / ANTsPy-gallery axis convention (see :data:`MOUSE_TARGET_ORIENTATION`)."""
    return reorient_volume(image, mode="mouse")


def mouse_reorient_nifti(nifti_image: Any) -> Any:
    """
    Apply :func:`mouse_reorient_volume` to a nibabel spatial image and return a NIfTI.
    """
    import nibabel as nib

    data = to_numpy(nifti_image.dataobj)
    aff = to_numpy(nifti_image.affine)
    img = Image(data=data, metadata={"affine": aff})
    out = mouse_reorient_volume(img)
    new_aff = to_numpy(out.affine)
    nii = nib.Nifti1Image(to_numpy(out.data), new_aff, header=nifti_image.header.copy())
    nii.set_sform(new_aff, code=1)
    nii.set_qform(new_aff, code=1)
    return nii


__all__ = [
    "MOUSE_ANATOMY_PERMUTE",
    "MOUSE_TARGET_ORIENTATION",
    "canonical_affine_for_axcodes",
    "reorient_volume",
    "mouse_reorient_volume",
    "mouse_reorient_nifti",
]
