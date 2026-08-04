"""MRA-TOF vessel segmentation via ANTsPyNet (memory-safe patch streaming)."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io.ants_bridge import ants_result_to_array, require_ants, require_antspynet, to_ants_image
from nvitk.types import Image

log = Logger()

# ANTsPyNet pretrained MRA vessel U-Net input size (fixed by weights).
_PATCH_SIZE: tuple[int, int, int] = (160, 160, 160)
_DEFAULT_BATCH: int = 2
_DEFAULT_STRIDE: int = 32


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


def _stride_tuple(stride: int | tuple[int, ...] | list[int]) -> tuple[int, int, int]:
    """Normalize a patch stride (scalar or length-3) to a 3-tuple of ints ≥ 1."""
    if isinstance(stride, (tuple, list)):
        if len(stride) != 3:
            raise ValueError("patch_stride_length must be an int or length-3 tuple.")
        out = tuple(int(max(1, s)) for s in stride)
        return out[0], out[1], out[2]
    s = int(max(1, stride))
    return (s, s, s)


def _patch_starts(
    image_shape: tuple[int, ...],
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    """Same regular grid as ``ants.extract_image_patches(..., max_number_of_patches='all')``."""
    starts: list[tuple[int, int, int]] = []
    for i in range(0, image_shape[0] - patch_size[0] + 1, stride[0]):
        for j in range(0, image_shape[1] - patch_size[1] + 1, stride[1]):
            for k in range(0, image_shape[2] - patch_size[2] + 1, stride[2]):
                starts.append((i, j, k))
    return starts


def _estimate_antspynet_peak_gb(n_patches: int, batch_size: int) -> float:
    """Peak RAM of stock ANTsPyNet (all patches + all predictions + one batch), float64."""
    voxels = float(np.prod(_PATCH_SIZE))
    arrays = 3.0 * n_patches + float(max(1, batch_size))
    return arrays * voxels * 8.0 / (1024.0**3)


def _configure_tf_for_inference() -> None:
    """Reduce TensorFlow CPU / allocator pressure before model creation."""
    try:
        import tensorflow as tf

        try:
            tf.config.threading.set_intra_op_parallelism_threads(4)
            tf.config.threading.set_inter_op_parallelism_threads(2)
        except Exception:
            pass
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    except Exception:
        pass


def _brain_mra_vessel_segmentation_streaming(
    mra: Any,
    mask: Any | None,
    *,
    prediction_batch_size: int,
    patch_stride_length: tuple[int, int, int],
    verbose: bool,
) -> Any:
    """ANTsPyNet MRA vessel segmentation without materializing all patches in RAM.

    Matches :func:`antspynet.brain_mra_vessel_segmentation` (template registration +
    160³ U-Net + patch average), but predicts in batches and accumulates into a
    float32 sum/count volume so peak memory stays O(template + batch) instead of
    O(n_patches × 160³).
    """
    ants = require_ants()
    require_antspynet()

    from antspynet.architectures import create_unet_model_3d
    from antspynet.utilities import get_antsxnet_data, get_pretrained_network
    from antspynet.utilities import brain_extraction as ants_brain_extraction

    if mask is None:
        mask = ants_brain_extraction(mra, modality="mra", verbose=verbose)
        mask = ants.threshold_image(mask, 0.5, 1.1, 1, 0)

    template = ants.image_read(get_antsxnet_data("mraTemplate"))
    template_brain_mask = ants.image_read(get_antsxnet_data("mraTemplateBrainMask"))

    mra_preprocessed = ants.image_clone(mra)
    fg = mask == 1
    mra_preprocessed[fg] = (
        (mra_preprocessed[fg] - mra_preprocessed[fg].min())
        / (mra_preprocessed[fg].max() - mra_preprocessed[fg].min())
    )
    log.info("mra_vessel: registering MRA to template (SyNQuick[a]) …")
    reg = ants.registration(
        template * template_brain_mask,
        mra_preprocessed * mask,
        type_of_transform="antsRegistrationSyNQuick[a]",
        verbose=verbose,
    )
    mra_preprocessed = ants.image_clone(reg["warpedmovout"])

    patch_size = _PATCH_SIZE
    if np.any(np.asarray(mra_preprocessed.shape) < np.asarray(patch_size)):
        raise ValueError("Images must be > 160 voxels per dimension after template warp.")

    template_mra_prior = ants.image_read(get_antsxnet_data("mraTemplateVesselPrior"))
    template_mra_prior = (
        (template_mra_prior - template_mra_prior.min())
        / (template_mra_prior.max() - template_mra_prior.min())
    )

    starts = _patch_starts(tuple(mra_preprocessed.shape), patch_size, patch_stride_length)
    n_patches = len(starts)
    if n_patches == 0:
        raise ValueError("No patches to predict (check template shape / stride).")

    stock_gb = _estimate_antspynet_peak_gb(n_patches, prediction_batch_size)
    log.info(
        f"mra_vessel: template={tuple(mra_preprocessed.shape)}, "
        f"patches={n_patches}, batch={prediction_batch_size}, "
        f"stride={patch_stride_length} "
        f"(stock ANTsPyNet would peak ~{stock_gb:.1f} GiB; using streaming)"
    )

    _configure_tf_for_inference()
    channel_size = 2
    model = create_unet_model_3d(
        (*patch_size, channel_size),
        number_of_outputs=1,
        mode="sigmoid",
        number_of_filters=(32, 64, 128, 256, 512),
        convolution_kernel_size=(3, 3, 3),
        deconvolution_kernel_size=(2, 2, 2),
        dropout_rate=0.0,
        weight_decay=0,
    )
    model.load_weights(get_pretrained_network("mraVesselWeights_160"))

    mra_arr = np.asarray(mra_preprocessed.numpy(), dtype=np.float32)
    prior_arr = np.asarray(template_mra_prior.numpy(), dtype=np.float32)
    # Match reconstruct_image_from_patches (stride != 1): average overlapping patches.
    sum_arr = np.zeros(mra_arr.shape, dtype=np.float32)
    count_arr = np.zeros(mra_arr.shape, dtype=np.float32)

    batch_n = max(1, int(prediction_batch_size))
    n_batches = (n_patches + batch_n - 1) // batch_n
    ps0, ps1, ps2 = patch_size

    for b in range(n_batches):
        i0 = b * batch_n
        i1 = min(i0 + batch_n, n_patches)
        cur = i1 - i0
        batch_x = np.zeros((cur, ps0, ps1, ps2, channel_size), dtype=np.float32)
        batch_starts = starts[i0:i1]
        for bi, (i, j, k) in enumerate(batch_starts):
            batch_x[bi, :, :, :, 0] = mra_arr[i : i + ps0, j : j + ps1, k : k + ps2]
            batch_x[bi, :, :, :, 1] = prior_arr[i : i + ps0, j : j + ps1, k : k + ps2]

        if verbose or (b == 0 or (b + 1) % max(1, n_batches // 10) == 0 or b + 1 == n_batches):
            log.info(f"mra_vessel: predicting batch {b + 1}/{n_batches} ({cur} patches)")

        pred = model.predict(batch_x, verbose=0 if not verbose else 1)
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 5:
            pred = pred[..., 0]

        for bi, (i, j, k) in enumerate(batch_starts):
            sum_arr[i : i + ps0, j : j + ps1, k : k + ps2] += pred[bi]
            count_arr[i : i + ps0, j : j + ps1, k : k + ps2] += 1.0

        del batch_x, pred
        gc.collect()

    count_arr[count_arr == 0] = 1.0
    prob_arr = sum_arr / count_arr
    del sum_arr, count_arr, mra_arr, prior_arr
    gc.collect()

    probability_image_warped = ants.from_numpy(
        prob_arr,
        origin=template_brain_mask.origin,
        spacing=template_brain_mask.spacing,
        direction=template_brain_mask.direction,
    )
    probability_image = ants.apply_transforms(
        mra,
        probability_image_warped,
        transformlist=reg["invtransforms"],
        whichtoinvert=[True],
        verbose=verbose,
    )

    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    del model
    gc.collect()

    return probability_image


def mra_vessel_segmentation(
    image: Image | np.ndarray,
    mask: Image | np.ndarray | None = None,
    *,
    prediction_batch_size: int = _DEFAULT_BATCH,
    patch_stride_length: int = _DEFAULT_STRIDE,
    verbose: bool = False,
) -> np.ndarray:
    """ANTsPyNet MRA-TOF vessel segmentation (probability image).

    Uses a **streaming** patch predictor (same model/weights as ANTsPyNet) so
    large TOFs do not allocate all ``160³`` patches at once — the stock
    ``antspynet.brain_mra_vessel_segmentation`` path can need tens of GiB at
    small strides (e.g. stride 16 ≈ 25–50 GiB) and get OOM-killed.

    Parameters
    ----------
    image
        MRA / TOF intensity volume.
    mask
        Optional brain mask (any positive foreground is accepted; binarized to
        ``0/1``). If omitted, ANTsPyNet estimates an MRA brain mask.
    prediction_batch_size
        Patches per ``model.predict`` call (keep small on CPU / limited RAM).
    patch_stride_length
        Stride (voxels) for overlapping patch prediction. Smaller → denser
        overlap and more compute; does **not** inflate RAM in the streaming path.
    """
    require_antspynet()
    ants_mra = to_ants_image(image)
    ants_mask = _as_binary_brain_mask(mask) if mask is not None else None
    shape = tuple(to_numpy(getattr(image, "data", image)).shape)

    batch = max(1, int(prediction_batch_size))
    stride = _stride_tuple(patch_stride_length)
    n_est = len(_patch_starts((260, 311, 260), _PATCH_SIZE, stride))
    stock_gb = _estimate_antspynet_peak_gb(n_est, batch)
    if stock_gb > 12.0:
        log.warning(
            f"mra_vessel: stock ANTsPyNet would peak ~{stock_gb:.0f} GiB for "
            f"stride={stride} (n≈{n_est} patches); using streaming inference instead."
        )

    log.info(
        f"mra_vessel_segmentation: shape={shape}, mask={'yes' if ants_mask is not None else 'auto'}, "
        f"batch={batch}, stride={stride}"
    )
    out = _brain_mra_vessel_segmentation_streaming(
        ants_mra,
        ants_mask,
        prediction_batch_size=batch,
        patch_stride_length=stride,
        verbose=bool(verbose),
    )
    return ants_result_to_array(out)


__all__ = ["mra_vessel_segmentation"]
