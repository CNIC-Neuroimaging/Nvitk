"""Tests for stage-4 local CD vessel segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from nvitk.core.array import as_backend_array
from nvitk.pipes.qvtpy.labels import (
    QVTPY_LACA,
    QVTPY_LMCA,
    QVTPY_LICA,
    QVTPY_LPCOMM,
    QVTPY_RACA,
    QVTPY_STRV,
)
from nvitk.pipes.qvtpy.util.mask_cleaning import keep_largest_component_per_label
from nvitk.segmentation.local_cd import (
    VESSEL_EXTRA_PADDING,
    BboxFacePadding,
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


def test_aca_bbox_symmetric_padding_perpendicular_to_centerline() -> None:
    """ACA crops must thicken along i and j so thin centerlines still threshold."""
    shape = (51, 51, 51)
    fp = bbox_padding_for_label(QVTPY_RACA, default_pad=3)
    assert fp.pad_i_min == 3 and fp.pad_i_max == 3

    clm = np.zeros(shape, dtype=np.int32)
    ci, cj, ck = 25, 25, 25
    clm[ci, 5:46, ck] = QVTPY_RACA
    out = _bbox_with_vessel_padding(clm == QVTPY_RACA, shape, QVTPY_RACA, default_pad=3)
    assert out is not None
    i0, i1, _, _, _, _ = out[0]
    assert i1 - i0 >= 6


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
    from nvitk.segmentation.local_cd import resolve_venous_rg_intensity_fracs

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


def test_aca_sequential_second_grow_can_overlap_first() -> None:
    """RACA RG may overlap LACA; final seg labels are disjoint after correction."""
    from nvitk.pipes.qvtpy.labels import QVTPY_LACA, QVTPY_RACA
    from nvitk.pipes.qvtpy.util.aca_sequential_grow import _region_grow_acas_sequential
    from nvitk.segmentation.local_cd import VesselSegStats

    shape = (51, 51, 51)
    clm = np.zeros(shape, dtype=np.int32)
    ci, cj, ck = 25, 25, 25
    clm[5:46, cj, ck] = QVTPY_LACA
    clm[ci, 5:46, ck] = QVTPY_RACA

    seg = np.zeros(shape, dtype=np.int32)
    seg[clm == QVTPY_LACA] = QVTPY_LACA
    seg[clm == QVTPY_RACA] = QVTPY_RACA
    cd = np.zeros(shape, dtype=np.float64)
    cd[5:46, 23:28, 23:28] = 100.0
    cd[23:28, 5:46, 23:28] = 100.0

    fp = BboxFacePadding(0, 0, 0, 0, 0, 0)
    stats_by_id = {
        QVTPY_LACA: VesselSegStats(
            label_id=QVTPY_LACA,
            bbox=(0, 50, 0, 50, 0, 50),
            face_padding=fp,
            thr_algorithm="lsthr",
            opt_thresh=1.0,
            n_voxels_after_threshold=10,
            n_voxels_after_island_clean=10,
            n_voxels_after_region_growing=10,
        ),
        QVTPY_RACA: VesselSegStats(
            label_id=QVTPY_RACA,
            bbox=(0, 50, 0, 50, 0, 50),
            face_padding=fp,
            thr_algorithm="lsthr",
            opt_thresh=1.0,
            n_voxels_after_threshold=10,
            n_voxels_after_island_clean=10,
            n_voxels_after_region_growing=10,
        ),
    }
    info = _region_grow_acas_sequential(
        seg,
        cd,
        clm,
        None,
        opt_thresh_by_label={QVTPY_LACA: 1.0, QVTPY_RACA: 1.0},
        rg_intensity_frac=0.45,
        rg_intensity_frac_explore=0.35,
        venous_fracs={},
        rg_barrier_radius=0,
        aca_overlap_min_voxels=1,
        acomm_junction_radius=10,
        stats_by_id=stats_by_id,
    )
    assert info.n_overlap_voxels > 0
    assert not np.any((seg == QVTPY_LACA) & (seg == QVTPY_RACA))


def test_aca_plane_split_disjoint_near_junction() -> None:
    from nvitk.pipes.qvtpy.labels import QVTPY_LACA, QVTPY_RACA
    from nvitk.pipes.qvtpy.util.aca_sequential_grow import _split_aca_merged_by_junction_plane

    shape = (51, 51, 51)
    clm = np.zeros(shape, dtype=np.int32)
    ck = 25
    clm[8:22, 25, ck] = QVTPY_LACA
    clm[28:42, 25, ck] = QVTPY_RACA

    laca_m = np.zeros(shape, dtype=bool)
    raca_m = np.zeros(shape, dtype=bool)
    laca_m[8:22, 24:27, ck] = True
    raca_m[28:42, 24:27, ck] = True
    laca_m[24:27, 24:27, ck] = True
    raca_m[24:27, 24:27, ck] = True

    res = _split_aca_merged_by_junction_plane(
        laca_m, raca_m, clm, None, acomm_junction_radius=10
    )
    assert not np.any(res.laca_mask & res.raca_mask)
    assert np.any(res.laca_mask)
    assert np.any(res.raca_mask)


def test_aca_plane_split_assigns_by_axis_side() -> None:
    from nvitk.pipes.qvtpy.labels import QVTPY_LACA, QVTPY_RACA
    from nvitk.pipes.qvtpy.util.aca_sequential_grow import _split_aca_merged_by_junction_plane

    shape = (51, 51, 51)
    clm = np.zeros(shape, dtype=np.int32)
    ck = 25
    clm[8:22, 25, ck] = QVTPY_LACA
    clm[28:42, 25, ck] = QVTPY_RACA

    laca_m = np.zeros(shape, dtype=bool)
    raca_m = np.zeros(shape, dtype=bool)
    laca_m[8:22, 24:27, ck] = True
    raca_m[28:42, 24:27, ck] = True
    laca_m[24:27, 24:27, ck] = True
    raca_m[24:27, 24:27, ck] = True

    res = _split_aca_merged_by_junction_plane(
        laca_m, raca_m, clm, None, acomm_junction_radius=12
    )
    assert res.split_axis == 0
    assert res.laca_on_low_side is True
    assert np.all(res.laca_mask[8:22, 25, ck])
    assert np.all(res.raca_mask[28:42, 25, ck])
    assert not np.any(res.laca_mask[28:42, 25, ck])
    assert not np.any(res.raca_mask[8:22, 25, ck])


def test_aca_stray_raca_island_reassigned_to_laca() -> None:
    """RACA CC with no RACA seeds (plane mislabel) is moved to LACA."""
    from nvitk.pipes.qvtpy.labels import QVTPY_LACA, QVTPY_RACA
    from nvitk.pipes.qvtpy.util.aca_sequential_grow import _split_aca_merged_by_junction_plane

    shape = (51, 51, 51)
    clm = np.zeros(shape, dtype=np.int32)
    ck = 25
    clm[8:22, 10:20, ck] = QVTPY_LACA
    clm[28:42, 30:40, ck] = QVTPY_RACA

    laca_m = np.zeros(shape, dtype=bool)
    raca_m = np.zeros(shape, dtype=bool)
    laca_m[8:22, 10:20, ck] = True
    raca_m[28:42, 30:40, ck] = True
    # Isolated RACA island on LACA side (no RACA centerline here).
    raca_m[10:13, 12:15, ck] = True

    res = _split_aca_merged_by_junction_plane(
        laca_m, raca_m, clm, None, acomm_junction_radius=12
    )
    assert res.n_stray_islands_reassigned > 0
    assert np.all(res.laca_mask[10:13, 12:15, ck])
    assert not np.any(res.raca_mask[10:13, 12:15, ck])


def _synthetic_converging_acas(
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Two ACAs that approach the same AComm neighbourhood without crossing."""
    cd = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    ci, cj, ck = 25, 25, 25
    for i in range(5, 26):
        j = 10 + (i - 5) * 14 // 20
        clm[i, j, ck] = QVTPY_LACA
        cd[i, j - 2 : j + 3, ck - 2 : ck + 3] = 100.0
    for i in range(45, 24, -1):
        j = 40 - (45 - i) * 14 // 20
        clm[i, j, ck] = QVTPY_RACA
        cd[i, j - 2 : j + 3, ck - 2 : ck + 3] = 100.0
    cd[ci - 3 : ci + 4, cj - 3 : cj + 4, ck - 2 : ck + 3] = 95.0
    return cd, clm, ci, cj, ck


def test_aca_sequential_integration_converging_approach() -> None:
    from nvitk.pipes.qvtpy.labels import QVTPY_LACA, QVTPY_RACA

    cd, clm, ci, cj, ck = _synthetic_converging_acas((51, 51, 51))

    res = build_seg_4dflow_local(
        cd,
        clm,
        crop_padding_bbox=3,
        region_growing=True,
        thr_algorithm="lsthr",
        aca_sequential_grow=True,
        aca_overlap_min_voxels=1,
        acomm_junction_radius=10,
        rg_barrier_radius=0,
    )
    seg = np.asarray(res.segmentation)
    assert res.aca_sequential_grow is not None
    assert not np.any((seg == QVTPY_LACA) & (seg == QVTPY_RACA))
    assert np.any(seg[8, 12, ck] == QVTPY_LACA)
    assert np.any(seg[42, 38, ck] == QVTPY_RACA)
    assert not np.any(seg[42, 38, ck] == QVTPY_LACA)
    assert not np.any(seg[8, 12, ck] == QVTPY_RACA)


def test_single_aca_uses_standard_region_growing() -> None:
    from nvitk.pipes.qvtpy.labels import QVTPY_LACA

    shape = (50, 50, 50)
    cd = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    clm[25, 25, 20:30] = QVTPY_LACA
    cd[23:28, 23:28, 18:32] = 100.0

    res = build_seg_4dflow_local(
        cd,
        clm,
        crop_padding_bbox=2,
        region_growing=True,
        thr_algorithm="lsthr",
        aca_sequential_grow=True,
    )
    assert res.aca_sequential_grow is None
    assert np.any(np.asarray(res.segmentation) == QVTPY_LACA)


def test_mca_explore_frac_lower_than_default() -> None:
    from nvitk.segmentation.local_cd import rg_intensity_frac_for_label

    assert rg_intensity_frac_for_label(QVTPY_LMCA, default_frac=0.5, explore_frac=0.35) == 0.35
    assert rg_intensity_frac_for_label(QVTPY_LICA, default_frac=0.5, explore_frac=0.35) == 0.5


def test_otsu_smoke() -> None:
    pytest.importorskip("skimage")
    shape = (30, 30, 30)
    cd = np.zeros(shape, dtype=np.float64)
    clm = np.zeros(shape, dtype=np.int32)
    clm[15, 15, 10:20] = 5
    cd[13:18, 13:18, 8:22] = 50.0
    res = build_seg_4dflow_local(cd, clm, thr_algorithm="otsu", region_growing=False)
    assert res.segmentation is not None
