"""Tests for eICAB-mask hypointense threshold BB segmentation."""

from __future__ import annotations

import numpy as np
import pytest

from nvitk.pipes.pesa_brain.black_blood.util.bb_vessel_segmentation import build_seg_bb
from nvitk.pipes.pesa_brain.black_blood.util.vwi_preprocess import preprocess_vwi_bb


def _synthetic_tube(
    shape: tuple[int, int, int] = (40, 40, 40),
    *,
    label_id: int = 1,
    lumen_val: float = 20.0,
    wall_val: float = 120.0,
    bg_val: float = 180.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bright background, dark lumen along z-axis center."""
    wvi = np.full(shape, bg_val, dtype=np.float64)
    eicab = np.zeros(shape, dtype=np.int32)
    cx, cy = shape[0] // 2, shape[1] // 2
    for z in range(8, shape[2] - 8):
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                wvi[cx + dx, cy + dy, z] = lumen_val
                if abs(dx) <= 1 and abs(dy) <= 1:
                    eicab[cx + dx, cy + dy, z] = label_id
                else:
                    wvi[cx + dx, cy + dy, z] = wall_val
                    eicab[cx + dx, cy + dy, z] = label_id
    return wvi, eicab


def test_build_seg_bb_finds_dark_lumen_in_dilated_roi() -> None:
    wvi, eicab = _synthetic_tube()
    result = build_seg_bb(
        wvi,
        eicab,
        eicab_dilate=2,
        thr_algorithm="lsthr",
        min_component_frac=0.0,
    )
    assert np.any(result.seg == 1)
    lumen_voxels = int(np.count_nonzero((result.seg == 1) & (wvi < 80)))
    assert lumen_voxels > 0


def test_thr_algorithms_accepted() -> None:
    wvi, eicab = _synthetic_tube()
    algos: list[str] = ["lsthr", "lthr"]
    try:
        import skimage.filters  # noqa: F401
    except ImportError:
        pass
    else:
        algos.append("otsu")
    for algo in algos:
        result = build_seg_bb(
            wvi, eicab, eicab_dilate=2, thr_algorithm=algo, min_component_frac=0.0  # type: ignore[arg-type]
        )
        assert result.seg.shape == wvi.shape


def test_vwi_preprocess_median() -> None:
    wvi, _ = _synthetic_tube()
    out = preprocess_vwi_bb(wvi, "median", median_size=3)
    assert out.shape == wvi.shape
    assert not np.allclose(out, wvi)
