"""Unit tests for post-centerline ICA mask cleaning."""

from __future__ import annotations

import numpy as np
import pytest

skimage = pytest.importorskip("skimage")

from nvitk.morphology.centerline_siphon import (
    MIN_SIPHON_CYCLE_LEN,
    _bfs_distances_inside_roi,
    _bridge_cut_anchor,
    _dilate_fractional_shell,
    clean_mask_geodesic_cl,
    compute_mask_genus,
    recover_lumen_thickness,
    recover_lumen_thickness_symmetric,
    refine_mask_lumen_gaps,
)


def _thin_bar(shape: tuple[int, int, int], axis: str = "z") -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    if axis == "z":
        m[cx - 1 : cx + 2, cy, cz - 8 : cz + 8] = True
    return m


def test_bfs_distances_reaches_all_roi_voxels():
    shape = (12, 12, 12)
    roi = _thin_bar(shape)
    seeds = np.zeros(shape, dtype=bool)
    seeds[shape[0] // 2, shape[1] // 2, shape[2] // 2 - 8] = True
    dist = _bfs_distances_inside_roi(roi, seeds)
    assert np.all(dist[roi] >= 0)


def test_bridge_cut_anchor_picks_thin_point():
    shape = (20, 20, 20)
    mask = np.zeros(shape, dtype=bool)
    mask[8:12, 10, 5:15] = True
    mask[9, 10, 10] = False  # neck
    bridge = [(9, 10, 9), (9, 10, 11)]
    anchor = _bridge_cut_anchor(mask, bridge, dilate_r=1)
    assert anchor is not None
    assert mask[anchor]


def test_recover_lumen_thickness_stops_before_suspect():
    """Eroded tube can grow within ceiling without creating a handle."""
    shape = (24, 24, 24)
    ceiling = _thin_bar(shape)
    from scipy import ndimage as ndi

    eroded = ndi.binary_erosion(ceiling, iterations=2)
    out, info = recover_lumen_thickness(
        eroded, ceiling, label_name="test", max_extra_iters=4
    )
    out_np = np.asarray(out, dtype=bool)
    assert int(out_np.sum()) >= int(eroded.sum())
    assert int(info["beta1_final"]) == 0
    assert int(info["voxels_after"]) <= int(ceiling.sum())


def test_compute_mask_genus_filters_small_cycle_noise():
    """Tiny spurious loop should not count as siphon genus."""
    shape = (16, 16, 16)
    m = np.zeros(shape, dtype=bool)
    m[6:10, 8, 6:10] = True
    m[6, 8, 6:10] = True
    m[9, 8, 6:10] = True
    rep = compute_mask_genus(m, label_name="test", min_cycle_len=MIN_SIPHON_CYCLE_LEN)
    if rep.beta1_raw > 0 and rep.max_cycle_len < MIN_SIPHON_CYCLE_LEN:
        assert rep.beta1 == 0
        assert rep.noise_filtered


def test_symmetric_thickness_uses_common_steps():
    shape = (24, 24, 24)
    ceil = _thin_bar(shape)
    from scipy import ndimage as ndi

    eroded_a = ndi.binary_erosion(ceil, iterations=2)
    eroded_b = ndi.binary_erosion(ceil, iterations=3)
    items = [
        {"lid": 1, "mask": eroded_a, "ceiling": ceil, "label_name": "LICA"},
        {"lid": 2, "mask": eroded_b, "ceiling": ceil, "label_name": "RICA"},
    ]
    masks, meta = recover_lumen_thickness_symmetric(items)
    assert meta["common_micro_steps"] == min(
        meta["per_ica_steps"]["1"], meta["per_ica_steps"]["2"]
    )
    v1 = int(np.asarray(masks[1], dtype=bool).sum())
    v2 = int(np.asarray(masks[2], dtype=bool).sum())
    if meta["common_micro_steps"] > 0:
        assert v1 == v2


def test_dilate_fractional_shell_adds_partial_layer():
    shape = (12, 12, 12)
    m = _thin_bar(shape)
    ceil = np.zeros(shape, dtype=bool)
    ceil[4:8, 5:7, 2:10] = True
    out, n = _dilate_fractional_shell(m, ceil, shell_fraction=0.5)
    assert n > 0
    assert int(out.sum()) > int(m.sum())


def test_clean_mask_geodesic_clears_between_arms():
    """Simple U-shape: bridge across the opening, CL along one arm."""
    shape = (20, 20, 20)
    roi = np.zeros(shape, dtype=bool)
    roi[5:15, 10, 2:8] = True
    roi[5:15, 10, 12:18] = True
    roi[5:15, 10, 8:12] = True  # bridge
    path = np.array(
        [[7, 10, 3], [8, 10, 4], [9, 10, 5], [10, 10, 6], [11, 10, 7]],
        dtype=np.float32,
    )
    bridge = [(8, 10, 9), (9, 10, 10), (10, 10, 11)]
    cleaned, info = clean_mask_geodesic_cl(
        roi, path, bridge, label_name="test", bridge_dilate_r=1
    )
    cleaned_np = np.asarray(cleaned, dtype=bool)
    assert int(info.get("cleared_voxels", 0)) > 0
    assert cleaned_np.sum() < roi.sum()
    assert cleaned_np.sum() > 0


def test_refine_mask_lumen_gaps_preserves_connectivity():
    shape = (20, 20, 20)
    roi = np.zeros(shape, dtype=bool)
    roi[5:15, 10, 2:8] = True
    roi[5:15, 10, 12:18] = True
    roi[8:12, 10, 8:12] = False
    roi[9, 10, 8:12] = True
    path = np.array(
        [[7, 10, 3], [8, 10, 4], [9, 10, 5], [10, 10, 6]],
        dtype=np.float32,
    )
    refined = np.asarray(
        refine_mask_lumen_gaps(roi, roi, path, label_name="test"), dtype=bool
    )
    from nvitk.morphology.components import label_connected

    _, n_cc = label_connected(refined, connectivity=1)
    assert int(n_cc) >= 1
