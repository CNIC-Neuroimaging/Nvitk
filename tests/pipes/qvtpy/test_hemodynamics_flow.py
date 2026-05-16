"""Tests for hemodynamic indices on flow series."""

from __future__ import annotations

import numpy as np

from nvitk.measure.hemodynamics import mean_velocity_mm_s, pulsatility_index, resistivity_index


def test_pi_ri_on_pulsatile_flow() -> None:
    flow = np.array([1.0, 3.0, 2.0, 4.0, 1.5], dtype=np.float64).reshape(1, -1)
    pi = float(pulsatility_index(flow)[0])
    ri = float(resistivity_index(flow)[0])
    assert pi > 0.0
    assert 0.0 < ri <= 1.0


def test_mean_velocity() -> None:
    v = np.array([10.0, -5.0, 15.0], dtype=np.float64)
    assert mean_velocity_mm_s(v) == 20.0 / 3.0
