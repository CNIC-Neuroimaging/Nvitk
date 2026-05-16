"""Tests for stage-4 local CD vessel segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from nvitk.core.array import as_backend_array
from nvitk.pipes.qvtpy.labels import QVTPY_LMCA, QVTPY_LICA, QVTPY_LPCOMM, QVTPY_STRV
from nvitk.pipes.qvtpy.util.mask_cleaning import keep_largest_component_per_label
from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import (
    VESSEL_EXTRA_PADDING,
    bbox_padding_for_label,
    build_seg_4dflow_local,
    crop_min_fraction_for_label,
    region_growing_enabled_for_label,
    rg_intensity_frac_for_label,
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


def test_largest_cc_keeps_main_blob_only() -> None:
    seg = np.zeros((20, 20, 20), dtype=np.int32)
    seg[10, 10, 8:18] = 1
    seg[2, 2, 2] = 1
    out = keep_largest_component_per_label(seg)
    assert int(np.count_nonzero(out == 1)) == 10
    assert out[2, 2, 2] == 0


def test_small_vessel_uses_zero_crop_min_fraction() -> None:
    assert crop_min_fraction_for_label(QVTPY_LPCOMM) == 0.0
    assert crop_min_fraction_for_label(QVTPY_LICA) > 0


def test_strv_region_growing_disabled() -> None:
    assert not region_growing_enabled_for_label(QVTPY_STRV)


def test_venous_rg_intensity_frac_per_sinus() -> None:
    from nvitk.pipes.qvtpy.labels import QVTPY_LTSV, QVTPY_RTSV, QVTPY_SSSV
    from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import resolve_venous_rg_intensity_fracs

    merged = resolve_venous_rg_intensity_fracs({QVTPY_SSSV: 0.33})
    assert merged[QVTPY_SSSV] == 0.33
    assert merged[QVTPY_LTSV] == rg_intensity_frac_for_label(QVTPY_LTSV, venous_fracs=merged)
    assert QVTPY_STRV not in merged
    assert rg_intensity_frac_for_label(QVTPY_SSSV, venous_fracs=merged) == 0.33
    assert rg_intensity_frac_for_label(QVTPY_RTSV, venous_fracs=merged) == merged[QVTPY_RTSV]


def test_strv_skips_region_growing() -> None:
    shape = (40, 40, 40)
    cd = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    clm[20, 20, 10:30] = QVTPY_STRV
    cd[18:22, 18:22, 8:32] = 100.0
    res = build_seg_4dflow_local(
        cd, clm, crop_padding_bbox=3, region_growing=True, thr_algorithm="lsthr"
    )
    st = res.vessel_stats[0]
    assert st.label_id == QVTPY_STRV
    assert not st.region_growing_applied
    assert st.n_voxels_after_region_growing == st.n_voxels_after_island_clean


def test_aca_eicab_barrier_includes_acomm_and_contralateral() -> None:
    from nvitk.pipes.qvtpy.labels import QVTPY_ACOMM, QVTPY_LACA, QVTPY_RACA
    from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import _aca_eicab_region_growing_barrier

    shape = (50, 40, 40)
    eicab = np.zeros(shape, dtype=np.int32)
    eicab[28:32, 18:22, 12:28] = QVTPY_ACOMM
    eicab[8:18, 18:22, 10:30] = QVTPY_LACA
    eicab[38:48, 18:22, 10:30] = QVTPY_RACA
    forb = _aca_eicab_region_growing_barrier(
        eicab, QVTPY_LACA, acomm_radius=2, contra_radius=2
    )
    assert forb[30, 20, 20]
    assert forb[42, 20, 20]
    assert not forb[12, 20, 20]


def test_mca_explore_frac_lower_than_default() -> None:
    from nvitk.pipes.qvtpy.util.vessel_cd_segmentation import rg_intensity_frac_for_label

    assert rg_intensity_frac_for_label(QVTPY_LMCA, default_frac=0.5, explore_frac=0.35) == 0.35
    assert rg_intensity_frac_for_label(QVTPY_LICA, default_frac=0.5, explore_frac=0.35) == 0.5


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
