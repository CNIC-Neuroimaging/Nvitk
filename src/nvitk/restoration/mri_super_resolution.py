"""MRI super-resolution via ANTsPyNet."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import ants_result_to_array, require_antspynet, to_ants_image
from nvitk.types import Image

log = Logger()

# Integer factors only — ANTsPyNet builds model ids as ``1x1x2``, not ``1.0x1.0x2.0``.
MRI_SR_EXPANSION_FACTORS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 2),
    (1, 1, 3),
    (1, 1, 4),
    (1, 1, 6),
    (2, 2, 2),
    (2, 2, 4),
)
MRI_SR_FEATURES: tuple[str, ...] = ("vgg", "grader")
# grader has no pretrained weights for 1x1x6 in current antspynet.
_MRI_SR_UNAVAILABLE: frozenset[tuple[tuple[int, int, int], str]] = frozenset({
    ((1, 1, 6), "grader"),
})


def _normalize_expansion_factor(
    expansion_factor: Sequence[float | int],
) -> tuple[int, int, int]:
    """Validate and coerce a per-axis super-resolution expansion factor to a 3-int tuple."""
    if len(expansion_factor) != 3:
        raise ValueError(
            "expansion_factor must have three values (ix, iy, iz); "
            f"got {list(expansion_factor)!r}"
        )
    exp: list[int] = []
    for x in expansion_factor:
        fx = float(x)
        ix = int(round(fx))
        if abs(fx - ix) > 1e-6:
            raise ValueError(
                f"expansion_factor values must be integers (got {x!r}); "
                f"valid options: {', '.join('x'.join(map(str, e)) for e in MRI_SR_EXPANSION_FACTORS)}"
            )
        exp.append(ix)
    key = (exp[0], exp[1], exp[2])
    if key not in MRI_SR_EXPANSION_FACTORS:
        raise ValueError(
            f"Unsupported expansion_factor {key}; "
            f"valid options: {', '.join('x'.join(map(str, e)) for e in MRI_SR_EXPANSION_FACTORS)}"
        )
    return key


def mri_super_resolution(
    image: Image | np.ndarray,
    *,
    expansion_factor: Sequence[float | int] | list[float] | tuple[float, ...] = (1, 1, 2),
    feature: str = "vgg",
    verbose: bool = False,
) -> np.ndarray:
    """ANTsPyNet deep back-projection MRI super-resolution.

    Parameters
    ----------
    image
        Input MRI volume.
    expansion_factor
        Per-axis integer upsampling factors. Supported: ``1,1,2``, ``1,1,3``,
        ``1,1,4``, ``1,1,6``, ``2,2,2``, ``2,2,4``. Must be ints (floats like
        ``1.0`` are coerced only when exactly integral).
    feature
        Feature backbone: ``vgg`` or ``grader`` (``1,1,6`` + ``grader`` has no weights).
    """
    antspynet = require_antspynet()
    feat = str(feature).strip().lower()
    if feat not in MRI_SR_FEATURES:
        raise ValueError(f"feature must be one of {MRI_SR_FEATURES}, got {feature!r}")
    exp = _normalize_expansion_factor(expansion_factor)
    if (exp, feat) in _MRI_SR_UNAVAILABLE:
        raise ValueError(
            f"No pretrained weights for expansion_factor={exp} with feature={feat!r}; "
            "try feature='vgg'."
        )
    ants_img = to_ants_image(image)
    shape = tuple(to_numpy(getattr(image, "data", image)).shape)
    # Pass ints so antspynet builds ``…_1x1x2_…`` model ids (not ``1.0x1.0x2.0``).
    exp_list = [int(x) for x in exp]
    log.info(
        f"mri_super_resolution: shape={shape}, expansion_factor={exp_list}, feature={feat!r}"
    )
    out = antspynet.mri_super_resolution(
        ants_img,
        expansion_factor=exp_list,
        feature=feat,
        verbose=bool(verbose),
    )
    return ants_result_to_array(out)


__all__ = [
    "MRI_SR_EXPANSION_FACTORS",
    "MRI_SR_FEATURES",
    "mri_super_resolution",
]
