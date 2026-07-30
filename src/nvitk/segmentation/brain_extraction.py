"""Multi-modal human brain extraction via ANTsPyNet."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import ants_result_to_array, require_antspynet, to_ants_image
from nvitk.types import Image

log = Logger()

# Common ANTsPyNet ``brain_extraction`` modality strings (see antspynet docs).
BRAIN_EXTRACTION_MODALITIES: tuple[str, ...] = (
    "t1",
    "t1nobrainer",
    "t1combined",
    "t1threetissue",
    "t1hemi",
    "t1lobes",
    "flair",
    "t2",
    "t2star",
    "bold",
    "fa",
    "mra",
    "t1t2infant",
    "t1infant",
    "t2infant",
)


def brain_extraction(
    image: Image | np.ndarray | Sequence[Image | np.ndarray],
    *,
    modality: str = "t1",
    verbose: bool = False,
) -> np.ndarray:
    """ANTsPyNet multi-modal brain extraction.

    Parameters
    ----------
    image
        One intensity volume, or a sequence of volumes for multi-modal
        modalities such as ``t1t2infant``.
    modality
        ANTsPyNet modality string (``t1``, ``t2``, ``flair``, ``mra``, …).
    """
    antspynet = require_antspynet()
    mod = str(modality).strip().lower()
    if isinstance(image, (list, tuple)):
        ants_in: Any = [to_ants_image(im) for im in image]
        shapes = [tuple(to_numpy(getattr(im, "data", im)).shape) for im in image]
        log.info(f"brain_extraction: multimodal n={len(ants_in)} shapes={shapes}, modality={mod}")
    else:
        ants_in = to_ants_image(image)
        shape = tuple(to_numpy(getattr(image, "data", image)).shape)
        log.info(f"brain_extraction: shape={shape}, modality={mod}")
    out = antspynet.brain_extraction(ants_in, modality=mod, verbose=bool(verbose))
    return ants_result_to_array(out)


__all__ = ["BRAIN_EXTRACTION_MODALITIES", "brain_extraction"]
