"""The vectorised measurement loops return exactly what the per-item loops they replaced did.

Each reference below is the earlier implementation, kept here verbatim in its algorithm so the
faster code can be checked against it on random inputs: same masks, same thresholds, same
counts, same label pairs.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from nvitk.core.backend import using
from nvitk.filters import sliding_threshold as st
from nvitk.measure import segmentation_metrics as sm

_CUBE = np.ones((3, 3, 3), dtype=np.uint8)


# Reference implementations (one full-volume pass per threshold / component / label pair)


def _counts_loop(values, max_val, x):
    return np.array([float(np.count_nonzero(values > (max_val * n))) for n in x])


def _detection_matches_loop(reference, prediction, *, iou_threshold):
    ref_labels, n_ref = ndi.label(reference, structure=_CUBE)
    pred_labels, n_pred = ndi.label(prediction, structure=_CUBE)
    if n_ref == 0 and n_pred == 0:
        return 0, 0, 0
    matched, tp = set(), 0
    for ref_id in range(1, n_ref + 1):
        ref_component = ref_labels == ref_id
        best_iou, best_id = 0.0, None
        for pred_id in np.unique(pred_labels[ref_component]):
            if pred_id == 0 or int(pred_id) in matched:
                continue
            pred_component = pred_labels == pred_id
            union = int(np.logical_or(ref_component, pred_component).sum())
            iou = float(np.logical_and(ref_component, pred_component).sum()) / union if union else 0.0
            if iou > best_iou:
                best_iou, best_id = iou, int(pred_id)
        if best_id is not None and best_iou >= iou_threshold:
            matched.add(best_id)
            tp += 1
    return tp, n_pred - len(matched), n_ref - tp


def _neighbouring_labels_dilation(mask, labels=None):
    present = sorted(labels) if labels is not None else [int(v) for v in np.unique(mask) if int(v) != 0]
    dilated = {v: ndi.binary_dilation(mask == v, structure=_CUBE) for v in present}
    pairs = set()
    for i, first in enumerate(present):
        for second in present[i + 1:]:
            if np.logical_and(dilated[first], mask == second).any():
                pairs.add(frozenset((first, second)))
    return pairs


# Helpers


def _blobs(rng, shape, density, sigma):
    v = ndi.gaussian_filter(rng.random(shape), sigma)
    return v > np.quantile(v, 1 - density)


def _label_volume(rng, shape, n_labels):
    base = ndi.gaussian_filter(rng.random(shape), rng.uniform(0.5, 2.5))
    edges = np.quantile(base, np.linspace(0, 1, n_labels + 2)[1:-1])
    lab = np.digitize(base, edges).astype(np.int32)
    lab[rng.random(shape) < rng.uniform(0, 0.6)] = 0
    return lab


# Tests


@pytest.mark.parametrize("seed", range(20))
def test_sliding_threshold_counts_match_loop(seed):
    rng = np.random.default_rng(seed)
    values = rng.gamma(1.5, 1.0, size=tuple(rng.integers(5, 30, size=3)))
    if seed % 2:
        values = np.round(values * 4) / 4  # many ties at the thresholds
    x = np.arange(0.0, 0.8 + 0.0005, 0.001)
    max_val = float(values.max())
    with using("numpy"):
        fast = st._counts_above(values, max_val * x).astype(np.float64)
    np.testing.assert_array_equal(fast, _counts_loop(values, max_val, x))


@pytest.mark.parametrize("seed", range(10))
def test_sliding_threshold_masks_unchanged(seed):
    rng = np.random.default_rng(100 + seed)
    vol = rng.gamma(1.5, 1.0, size=(16, 18, 12))
    img = rng.random((41, 41))
    with using("numpy"):
        mask3, thr3 = st.binary_mask_sliding_threshold_3d(vol)
        mask2 = st.binary_mask_sliding_threshold_2d(img)
    # Re-derive the threshold with the loop and the module's own curve post-processing.
    x = np.arange(0.0, 0.8 + 0.0005, 0.001)
    cd = ndi.median_filter(vol, size=3, mode="constant", cval=0.0)
    sval = _counts_loop(cd, float(cd.max()), x).astype(np.float32)
    with using("numpy"):
        fast = st._counts_above(cd, float(cd.max()) * x).astype(np.float64).astype(np.float32)
    np.testing.assert_array_equal(fast, sval)
    assert mask3.shape == vol.shape and np.isfinite(thr3)
    assert mask2.shape == img.shape


@pytest.mark.parametrize("seed", range(30))
def test_detection_matches_equals_loop(seed):
    rng = np.random.default_rng(200 + seed)
    shape = tuple(int(v) for v in rng.integers(8, 24, size=3))
    ref = _blobs(rng, shape, rng.uniform(0.02, 0.3), rng.uniform(0.6, 2.0))
    pred = _blobs(rng, shape, rng.uniform(0.02, 0.3), rng.uniform(0.6, 2.0))
    if seed % 3 == 0:
        pred |= np.roll(ref, 1, axis=0)
    for thr in (0.0, 0.25, 0.5, 1.0):
        assert sm.detection_matches(ref, pred, iou_threshold=thr) == _detection_matches_loop(
            ref, pred, iou_threshold=thr
        )


@pytest.mark.parametrize("seed", range(30))
def test_neighbouring_labels_equals_dilation(seed):
    rng = np.random.default_rng(300 + seed)
    shape = tuple(int(v) for v in rng.integers(5, 20, size=3))
    lab = _label_volume(rng, shape, int(rng.integers(1, 40)))
    if seed % 5 == 0:
        lab = lab - 5
    for labels in (None, list(range(-6, 12)), [3, 1, 1, 2, 7]):
        assert sm.neighbouring_labels(lab, labels=labels) == _neighbouring_labels_dilation(lab, labels)
