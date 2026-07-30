"""MRA-TOF vessel segmentation via ANTsPyNet."""

from __future__ import annotations

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import ants_result_to_array, require_ants, require_antspynet, to_ants_image
from nvitk.types import Image

log = Logger()


def _as_binary_brain_mask(mask: Image | np.ndarray, *, threshold: float = 0.5):
    """Return an ANTs mask with values exactly ``0`` / ``1``.

    ANTsPyNet indexes with ``mask == 1``; soft masks, label ids ≠ 1, or
    ``255`` foreground therefore crash with a zero-size ``min()``.
    """
    ants = require_ants()
    ants_mask = to_ants_image(mask)
    # Match antspynet auto path: threshold_image(mask, 0.5, 1.1, 1, 0)
    binary = ants.threshold_image(ants_mask, float(threshold), 1e12, 1, 0)
    n_fg = int(np.count_nonzero(np.asarray(binary.numpy()) == 1))
    if n_fg == 0:
        raise ValueError(
            "Brain mask has no foreground voxels after binarization "
            f"(threshold={threshold}). Pass a binary brain mask, or omit the "
            "mask to let ANTsPyNet estimate one from the MRA."
        )
    return binary


def mra_vessel_segmentation(
    image: Image | np.ndarray,
    mask: Image | np.ndarray | None = None,
    *,
    prediction_batch_size: int = 16,
    patch_stride_length: int = 32,
    verbose: bool = False,
) -> np.ndarray:
    """ANTsPyNet MRA-TOF vessel segmentation (probability image).

    Parameters
    ----------
    image
        MRA / TOF intensity volume.
    mask
        Optional brain mask (any positive foreground is accepted; binarized to
        ``0/1``). If omitted, ANTsPyNet estimates an MRA brain mask.
    prediction_batch_size
        Batch size for patch prediction (GPU memory trade-off).
    patch_stride_length
        Stride (voxels) for overlapping patch prediction.
    """
    antspynet = require_antspynet()
    ants_mra = to_ants_image(image)
    ants_mask = _as_binary_brain_mask(mask) if mask is not None else None
    shape = tuple(to_numpy(getattr(image, "data", image)).shape)
    log.info(
        f"mra_vessel_segmentation: shape={shape}, mask={'yes' if ants_mask is not None else 'auto'}, "
        f"batch={int(prediction_batch_size)}, stride={int(patch_stride_length)}"
    )
    out = antspynet.brain_mra_vessel_segmentation(
        ants_mra,
        mask=ants_mask,
        prediction_batch_size=int(prediction_batch_size),
        patch_stride_length=int(patch_stride_length),
        verbose=bool(verbose),
    )
    return ants_result_to_array(out)


__all__ = ["mra_vessel_segmentation"]
