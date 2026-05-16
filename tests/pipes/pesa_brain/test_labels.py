"""Tests for black_blood label relabeling."""

from __future__ import annotations

import numpy as np

from nvitk.pipes.pesa_brain.black_blood.labels import (
    BB_LPCA,
    BB_RPCA,
    EICAB_LPCA_P1,
    EICAB_LPCA_P2,
    EICAB_LSCA,
    EICAB_RPCA_P1,
    EICAB_RPCA_P2,
    relabel_eicab_to_bb,
)


def test_relabel_eicab_to_bb_pca_merge_and_sca_drop() -> None:
    vol = np.zeros((5, 5, 5), dtype=np.int32)
    vol[1, 1, 1] = EICAB_LPCA_P1
    vol[2, 2, 2] = EICAB_LPCA_P2
    vol[3, 3, 3] = EICAB_RPCA_P1
    vol[1, 2, 1] = EICAB_RPCA_P2
    vol[0, 0, 0] = EICAB_LSCA

    out = relabel_eicab_to_bb(vol)
    assert int(out[1, 1, 1]) == BB_LPCA
    assert int(out[2, 2, 2]) == BB_LPCA
    assert int(out[3, 3, 3]) == BB_RPCA
    assert int(out[1, 2, 1]) == BB_RPCA
    assert int(out[0, 0, 0]) == 0
