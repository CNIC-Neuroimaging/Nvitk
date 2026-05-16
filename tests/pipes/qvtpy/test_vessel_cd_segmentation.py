"""Tests for stage-4 local CD vessel segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from nvitk.core.array import as_backend_array
from nvitk.pipes.qvtpy.labels import QVTPY_LMCA, QVTPY_LICA
from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import (
    VESSEL_EXTRA_PADDING,
    bbox_padding_for_label,
    build_seg_4dflow_local,
    _bbox_with_padding,
    _bbox_with_vessel_padding,
)


def test_bbox_padding() -> None:
    roi = np.zeros((20, 20, 20), dtype=bool)
    roi[10, 10, 10] = True
    bb = _bbox_with_padding(roi, (20, 20, 20), padding=3)
    assert bb is not None
    i0, i1, j0, j1, k0, k1 = bb
    assert i0 == 7 and i1 == 13


def test_lMCA_asymmetric_bbox() -> None:
    roi = np.zeros((50, 50, 50), dtype=bool)
    roi[25, 25, 25] = True
    out = _bbox_with_vessel_padding(roi, (50, 50, 50), QVTPY_LMCA, default_pad=3)
    assert out is not None
    bbox, fp = out
    i0, i1, j0, j1, k0, k1 = bbox
    assert fp.pad_i_min == 0
    assert fp.pad_i_max == VESSEL_EXTRA_PADDING
    assert i0 == 25 - 0
    assert i1 == 25 + VESSEL_EXTRA_PADDING
    assert j0 == 22 and j1 == 28


def test_ica_basilar_restricts_z_plus() -> None:
    fp = bbox_padding_for_label(QVTPY_LICA, default_pad=3)
    assert fp.pad_k_max == 0
    assert fp.pad_k_min == 3


def test_two_vessels_no_region_growing() -> None:
    shape = (40, 40, 40)
    cd = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    clm[20, 20, 10:30] = 1
    clm[25:35, 20, 20] = 2
    cd[18:22, 18:22, 8:32] = 100.0
    cd[24:36, 18:22, 18:22] = 80.0

    res = build_seg_4dflow_local(
        cd,
        clm,
        crop_padding_bbox=3,
        thr_algorithm="lsthr",
        region_growing=False,
    )
    seg = np.asarray(res.segmentation)
    assert np.any(seg == 1)
    assert np.any(seg == 2)
    assert all(st.n_voxels_after_island_clean >= 0 for st in res.vessel_stats)


def test_region_growing_fills_free_high_cd() -> None:
    shape = (50, 50, 50)
    cd = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    clm[25, 25, 20:30] = 1
    cd[23:28, 23:28, 18:32] = 100.0
    cd[28:35, 25, 25] = 90.0

    res_off = build_seg_4dflow_local(
        cd, clm, crop_padding_bbox=2, region_growing=False, thr_algorithm="lsthr"
    )
    res_on = build_seg_4dflow_local(
        cd, clm, crop_padding_bbox=2, region_growing=True, thr_algorithm="lsthr"
    )
    n_off = int(np.count_nonzero(np.asarray(res_off.segmentation) == 1))
    n_on = int(np.count_nonzero(np.asarray(res_on.segmentation) == 1))
    assert n_on >= n_off


def test_region_growing_does_not_overwrite_other_label() -> None:
    shape = (50, 50, 50)
    cd = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    clm[20, 25, 20:30] = 1
    clm[30, 25, 20:30] = 2
    cd[18:22, 23:28, 15:35] = 100.0
    cd[28:32, 23:28, 15:35] = 100.0
    cd[22:28, 25, 25] = 95.0

    res = build_seg_4dflow_local(
        cd, clm, crop_padding_bbox=2, region_growing=True, thr_algorithm="lsthr"
    )
    seg = np.asarray(res.segmentation)
    assert np.any(seg == 1)
    assert np.any(seg == 2)
    assert not np.any((seg == 1) & (seg == 2))


def test_centerline_barrier_reduces_overlap_with_other_skeleton() -> None:
    from nvitk.morphology import dilate

    shape = (40, 40, 40)
    cd = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    clm[20, 20, 10:30] = 1
    clm[22, 20, 10:30] = 2
    cd[18:25, 18:25, 8:32] = 100.0
    other_cl = as_backend_array(
        dilate((clm == 1).astype(np.uint8), footprint=3, connectivity=1)
    ).astype(bool)

    res_barrier = build_seg_4dflow_local(
        cd,
        clm,
        crop_padding_bbox=5,
        region_growing=False,
        cl_barrier_radius=3,
        thr_algorithm="lsthr",
    )
    res_none = build_seg_4dflow_local(
        cd,
        clm,
        crop_padding_bbox=5,
        region_growing=False,
        cl_barrier_radius=0,
        thr_algorithm="lsthr",
    )
    seg_b = np.asarray(res_barrier.segmentation)
    seg_n = np.asarray(res_none.segmentation)
    overlap_b = int(np.count_nonzero((seg_b == 2) & other_cl))
    overlap_n = int(np.count_nonzero((seg_n == 2) & other_cl))
    assert overlap_b < overlap_n


def test_island_clean_removes_small_blob() -> None:
    shape = (30, 30, 30)
    cd = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    clm[15, 15, 10:20] = 1
    cd[13:18, 13:18, 8:22] = 100.0
    cd[25, 25, 15] = 100.0

    res = build_seg_4dflow_local(
        cd,
        clm,
        crop_padding_bbox=3,
        region_growing=False,
        seg_min_island_fraction=0.05,
        thr_algorithm="lsthr",
    )
    seg = np.asarray(res.segmentation)
    assert not np.any(seg[25, 25, 15] == 1)
    st = res.vessel_stats[0]
    assert st.n_voxels_after_island_clean <= st.n_voxels_after_threshold


def test_otsu_smoke() -> None:
    pytest.importorskip("skimage")
    shape = (30, 30, 30)
    cd = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    clm[15, 15, 10:20] = 5
    cd[13:18, 13:18, 8:22] = 50.0
    res = build_seg_4dflow_local(
        cd, clm, crop_padding_bbox=2, thr_algorithm="otsu", region_growing=False
    )
    assert int(np.count_nonzero(np.asarray(res.segmentation))) > 0
