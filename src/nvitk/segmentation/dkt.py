"""Desikan-Killiany-Tourville cortical parcellation via ANTsPyNet."""

from __future__ import annotations

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import ants_result_to_array, require_antspynet, to_ants_image
from nvitk.types import Image

log = Logger()


def desikan_killiany_tourville_labeling(
    image: Image | np.ndarray,
    *,
    do_preprocessing: bool = True,
    do_lobar_parcellation: bool = False,
    do_denoising: bool = True,
    version: int = 0,
    verbose: bool = False,
) -> np.ndarray:
    """ANTsPyNet Desikan-Killiany-Tourville (DKT) labeling on T1w MRI.

    Returns the segmentation label map (probability maps are discarded).
    """
    antspynet = require_antspynet()
    ants_t1 = to_ants_image(image)
    shape = tuple(to_numpy(getattr(image, "data", image)).shape)
    log.info(
        f"DKT labeling: shape={shape}, preprocessing={bool(do_preprocessing)}, "
        f"lobar={bool(do_lobar_parcellation)}, denoising={bool(do_denoising)}, "
        f"version={int(version)}"
    )
    out = antspynet.desikan_killiany_tourville_labeling(
        ants_t1,
        do_preprocessing=bool(do_preprocessing),
        return_probability_images=False,
        do_lobar_parcellation=bool(do_lobar_parcellation),
        do_denoising=bool(do_denoising),
        version=int(version),
        verbose=bool(verbose),
    )
    return ants_result_to_array(out)


__all__ = ["desikan_killiany_tourville_labeling"]
