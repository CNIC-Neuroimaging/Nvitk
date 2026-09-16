"""Internal helpers shared by the measure primitives."""

from __future__ import annotations

from typing import Any

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.types import Image

setup(globals())


def resolve_array(img: Image | Any) -> Any:
    """Return the voxel array for *img*, or *img* itself when already an array."""
    return img.data if isinstance(img, Image) else img


def resolve_spacing(img: Image | Any, spacing: tuple[float, ...] | None) -> tuple[float, ...]:
    """
    Return a spacing tuple for *img*.

    Resolution order:
    1. explicit *spacing* argument;
    2. ``img.spacing`` when *img* is an :class:`Image`;
    3. :class:`ValueError`.
    """
    if spacing is not None:
        return tuple(float(s) for s in spacing)
    if isinstance(img, Image) and img.spacing is not None:
        return tuple(float(s) for s in img.spacing)
    raise ValueError("Spacing is required (pass explicitly or via an Image with metadata spacing).")


def bool_mask(mask: Image | Any) -> Any:
    """Boolean-cast *mask* onto the active backend (NumPy/CuPy).

    Routed through :func:`as_backend_array` even when the input already casts:
    a Napari layer holds host arrays, so returning them untouched under the CuPy
    backend hands NumPy operands to a CuPy ufunc, which rejects them.
    """
    return as_backend_array(resolve_array(mask)).astype(bool)


def backend_array(img: Image | Any) -> Any:
    """The voxel array for *img*, on the active backend.

    Napari hands out host arrays whatever the compute backend is, so anything
    that will be combined with a backend array — indexed by a mask, fed to a
    ufunc — has to come through here first, or CuPy rejects the pairing.
    """
    return as_backend_array(resolve_array(img))


def label_mask(mask: Image | Any, label: int | None = None) -> Any:
    """*mask* as booleans: every non-zero voxel, or only those equal to *label*.

    The ``label`` form is what a multi-label segmentation needs. Collapsing one
    to "non-zero" erases exactly the distinction a comparison against another
    label map is meant to measure, and scores a prediction that assigns every
    voxel to the wrong structure as a perfect match.
    """
    if label is None:
        return bool_mask(mask)
    return as_backend_array(resolve_array(mask)) == int(label)


def ensure_same_shape(a: Image | Any, b: Image | Any) -> None:
    """Raise ``ValueError`` unless *a* and *b* (Image or array) have the same shape."""
    sa = resolve_array(a).shape
    sb = resolve_array(b).shape
    if tuple(sa) != tuple(sb):
        raise ValueError(f"Shape mismatch: {sa} vs {sb}")
