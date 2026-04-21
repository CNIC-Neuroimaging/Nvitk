"""Binary (and gray) morphological operators: dilate, erode, open, close, fill_holes."""

from __future__ import annotations

from typing import Any, Sequence

from nvitk.core.backend import get_current_backend, setup
from nvitk.types import Image

from ._common import (
    _as_array,
    _coerce_to_current_backend,
    _resolve_structure,
    _wrap_like,
)

setup(globals())


def dilate(
    img: Image | Any,
    footprint: int | Any | None = None,
    *,
    iterations: int = 1,
    mode: str = "binary",
    isotropic: bool = False,
    spacing: Sequence[float] | None = None,
    connectivity: int = 2,
) -> Image | Any:
    """Morphological dilation.

    Parameters
    ----------
    img:
        Input mask (:class:`Image` or array).
    footprint:
        Structuring element. ``None`` → unit ball; ``int`` → iterated unit ball
        of that radius; array → used as-is (cast to the active backend).
    iterations:
        Passed to ``ndi.binary_dilation`` (integer ≥ 1).
    mode:
        ``"binary"`` or ``"gray"``.
    isotropic, spacing:
        When ``footprint`` is ``int`` and ``isotropic=True``, resample the
        ball to isotropic voxels using ``spacing``.
    connectivity:
        Base connectivity for the unit structure (1 = face, 2 = edge, ...).
    """
    if mode not in ("binary", "gray"):
        raise ValueError("mode must be 'binary' or 'gray'.")
    arr = _coerce_to_current_backend(_as_array(img))
    structure = _resolve_structure(
        arr.ndim,
        footprint,
        connectivity=connectivity,
        isotropic=isotropic,
        spacing=spacing,
    )
    op = ndi.binary_dilation if mode == "binary" else ndi.grey_dilation
    kwargs: dict[str, Any] = {"iterations": int(iterations)}
    if get_current_backend() == "cupy" and mode == "binary":
        kwargs["brute_force"] = True
    if mode == "gray":
        kwargs.pop("iterations", None)
    out = op(arr, structure=structure, **kwargs)
    return _wrap_like(img, out.astype(np.uint8))


def erode(
    img: Image | Any,
    footprint: int | Any | None = None,
    *,
    iterations: int = 1,
    mode: str = "binary",
    isotropic: bool = False,
    spacing: Sequence[float] | None = None,
    connectivity: int = 1,
) -> Image | Any:
    """Morphological erosion (counterpart to :func:`dilate`)."""
    if mode not in ("binary", "gray"):
        raise ValueError("mode must be 'binary' or 'gray'.")
    arr = _coerce_to_current_backend(_as_array(img))
    structure = _resolve_structure(
        arr.ndim,
        footprint,
        connectivity=connectivity,
        isotropic=isotropic,
        spacing=spacing,
    )
    op = ndi.binary_erosion if mode == "binary" else ndi.grey_erosion
    kwargs: dict[str, Any] = {"iterations": int(iterations)}
    if get_current_backend() == "cupy" and mode == "binary":
        kwargs["brute_force"] = True
    if mode == "gray":
        kwargs.pop("iterations", None)
    out = op(arr, structure=structure, **kwargs)
    return _wrap_like(img, out.astype(np.uint8))


def open(  # noqa: A001 - shadow builtin intentionally
    img: Image | Any,
    footprint: int | Any | None = None,
    *,
    iterations: int = 1,
    mode: str = "binary",
    isotropic: bool = False,
    spacing: Sequence[float] | None = None,
    connectivity: int = 1,
) -> Image | Any:
    """Morphological opening = erode followed by dilate."""
    eroded = erode(
        img,
        footprint,
        iterations=iterations,
        mode=mode,
        isotropic=isotropic,
        spacing=spacing,
        connectivity=connectivity,
    )
    return dilate(
        eroded,
        footprint,
        iterations=iterations,
        mode=mode,
        isotropic=isotropic,
        spacing=spacing,
        connectivity=connectivity,
    )


def close(
    img: Image | Any,
    footprint: int | Any | None = None,
    *,
    iterations: int = 1,
    mode: str = "binary",
    isotropic: bool = False,
    spacing: Sequence[float] | None = None,
    connectivity: int = 1,
) -> Image | Any:
    """Morphological closing = dilate followed by erode."""
    dilated = dilate(
        img,
        footprint,
        iterations=iterations,
        mode=mode,
        isotropic=isotropic,
        spacing=spacing,
        connectivity=connectivity,
    )
    return erode(
        dilated,
        footprint,
        iterations=iterations,
        mode=mode,
        isotropic=isotropic,
        spacing=spacing,
        connectivity=connectivity,
    )


def fill_holes(
    img: Image | Any,
    *,
    axis: int | None = None,
    structure: Any | None = None,
    mode: str = "binary",
) -> Image | Any:
    """Fill holes in a binary mask.

    Parameters
    ----------
    img:
        Input binary mask.
    axis:
        If provided, apply slicewise along *axis* (useful for axial fills that
        should not close inter-slice voids). If ``None``, fill in 3D.
    structure, mode:
        See ``scipy.ndimage.binary_fill_holes``.
    """
    if mode not in ("binary", "gray"):
        raise ValueError("mode must be 'binary' or 'gray'.")

    arr = _coerce_to_current_backend(_as_array(img))
    op = ndi.binary_fill_holes if mode == "binary" else ndi.grey_closing

    if axis is None:
        filled = op(arr, structure=structure)
        return _wrap_like(img, filled.astype(np.uint8))

    if axis not in range(-arr.ndim, arr.ndim):
        raise ValueError(f"axis={axis} is out of range for ndim={arr.ndim}.")
    ax = axis % arr.ndim
    out = np.zeros_like(arr, dtype=arr.dtype)
    moved = np.moveaxis(arr, ax, 0)
    out_moved = np.moveaxis(out, ax, 0)
    for i in range(moved.shape[0]):
        out_moved[i] = op(moved[i], structure=structure)
    out = np.moveaxis(out_moved, 0, ax)
    return _wrap_like(img, out.astype(np.uint8))


__all__ = ["dilate", "erode", "open", "close", "fill_holes"]
