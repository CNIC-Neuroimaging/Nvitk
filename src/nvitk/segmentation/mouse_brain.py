"""Mouse brain extraction / parcellation via ANTsPyNet."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import from_ants_image, require_antspynet, to_ants_image
from nvitk.types import Image

log = Logger()

MouseBrainMode = Literal["extraction", "parcellation"]
MouseModality = Literal["t2", "t1"]


def mouse_brain_segmentation(
    image: Image | np.ndarray,
    *,
    mode: MouseBrainMode = "extraction",
    modality: MouseModality = "t2",
    which_parcellation: str = "nick",
    mask: Image | np.ndarray | None = None,
    return_isotropic_output: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Segment a mouse brain MRI with ANTsPyNet deep models.

    Parameters
    ----------
    image
        Input intensity volume (typically ex-/in-vivo T2w mouse MRI).
    mode
        ``extraction`` → binary / probabilistic brain mask via
        :func:`antspynet.mouse_brain_extraction`.
        ``parcellation`` → regional labels via
        :func:`antspynet.mouse_brain_parcellation`.
    modality
        Imaging contrast for extraction (``t2`` or ``t1``).
    which_parcellation
        Parcellation scheme name (ANTsPyNet default ``nick``).
    mask
        Optional brain mask for parcellation (recommended after extraction).
    """
    antspynet = require_antspynet()
    ants_img = to_ants_image(image)
    shape = tuple(to_numpy(getattr(image, "data", image)).shape)
    mode_l = str(mode).strip().lower()
    if mode_l in ("extraction", "extract", "brain_extraction", "mask"):
        log.info(f"mouse brain extraction: shape={shape}, modality={modality}")
        out = antspynet.mouse_brain_extraction(
            ants_img,
            modality=str(modality).lower(),
            return_isotropic_output=bool(return_isotropic_output),
            verbose=bool(verbose),
        )
    elif mode_l in ("parcellation", "parcel", "labels"):
        ants_mask = to_ants_image(mask) if mask is not None else None
        log.info(
            f"mouse brain parcellation: shape={shape}, "
            f"which_parcellation={which_parcellation!r}, mask={'yes' if ants_mask is not None else 'no'}"
        )
        out = antspynet.mouse_brain_parcellation(
            ants_img,
            mask=ants_mask,
            return_isotropic_output=bool(return_isotropic_output),
            which_parcellation=str(which_parcellation),
            verbose=bool(verbose),
        )
    else:
        raise ValueError(
            f"Unknown mouse brain mode {mode!r}; use 'extraction' or 'parcellation'."
        )

    if isinstance(out, dict):
        # Some antspynet builds return probability maps + segmentation.
        for key in ("segmentation_image", "segmentation", "mask", "probability_image"):
            if key in out and out[key] is not None:
                return from_ants_image(out[key])
        first = next(iter(out.values()))
        return from_ants_image(first)
    return from_ants_image(out)


__all__ = ["mouse_brain_segmentation", "MouseBrainMode", "MouseModality"]
