"""N4 bias-field correction via ANTsPy (``ants.n4_bias_field_correction``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import from_ants_image, require_ants, to_ants_image
from nvitk.types import Image

log = Logger()


def n4_bias_field_correction(
    image: Image | np.ndarray,
    mask: Image | np.ndarray | None = None,
    *,
    shrink_factor: int = 4,
    spline_param: float | int | None = None,
    rescale_intensities: bool = False,
    convergence_iters: tuple[int, ...] | list[int] | None = None,
    convergence_tol: float = 1e-7,
    return_bias_field: bool = False,
    verbose: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Correct intensity inhomogeneity with ANTs N4.

    Parameters
    ----------
    image
        Input intensity volume (typically MRI).
    mask
        Optional soft-tissue / brain mask restricting the bias estimate.
    shrink_factor
        Downsampling factor for the bias estimate (ANTs default 4).
    spline_param
        B-spline fitting distance (voxels); ``None`` uses ANTs default.
    rescale_intensities
        Rescale intensities into ``[0, 1]`` before correction.
    return_bias_field
        When true, also return the estimated bias field volume.

    Returns
    -------
    corrected
        Bias-corrected intensity array (NumPy).
    (corrected, bias_field)
        When ``return_bias_field`` is true.
    """
    ants = require_ants()
    ants_img = to_ants_image(image)
    ants_mask = to_ants_image(mask) if mask is not None else None
    iters = list(convergence_iters) if convergence_iters is not None else [50, 50, 50, 50]
    kw: dict[str, Any] = {
        "image": ants_img,
        "mask": ants_mask,
        "shrink_factor": int(shrink_factor),
        "rescale_intensities": bool(rescale_intensities),
        "convergence": {"iters": iters, "tol": float(convergence_tol)},
        "return_bias_field": bool(return_bias_field),
        "verbose": bool(verbose),
    }
    if spline_param is not None and float(spline_param) > 0:
        kw["spline_param"] = float(spline_param)

    log.info(
        f"N4 bias correction: shape={tuple(to_numpy(getattr(image, 'data', image)).shape)}, "
        f"shrink_factor={int(shrink_factor)}, mask={'yes' if ants_mask is not None else 'no'}"
    )
    # ANTsPy fills both corrected + bias internally but returns only one based on
    # *return_bias_field*; always fetch the corrected image first.
    corrected_ants = ants.n4_bias_field_correction(**{**kw, "return_bias_field": False})
    corrected = from_ants_image(corrected_ants)
    if not return_bias_field:
        return corrected
    bias_ants = ants.n4_bias_field_correction(**{**kw, "return_bias_field": True})
    return corrected, from_ants_image(bias_ants)


__all__ = ["n4_bias_field_correction"]
