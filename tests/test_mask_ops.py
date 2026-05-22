"""Tests for binary mask logical operators."""

import numpy as np

from nvitk.segmentation.mask_ops import (
    mask_complement,
    mask_intersection,
    mask_subtract,
    mask_union,
    mask_xor,
)


def test_mask_union_intersection():
    a = np.zeros((4, 4, 4), dtype=np.uint8)
    b = np.zeros_like(a)
    a[1:3, 1:3, 1:3] = 1
    b[2:4, 2:4, 2:4] = 1
    u = mask_union(a, b)
    i = mask_intersection(a, b)
    assert int(u.sum()) > int(a.sum())
    assert int(i.sum()) < int(a.sum())


def test_mask_subtract_and_xor():
    a = np.zeros((5, 5, 5), dtype=np.uint8)
    b = np.zeros_like(a)
    a[:, :, :3] = 1
    b[:, :, 2:] = 1
    s = mask_subtract(a, b)
    x = mask_xor(a, b)
    assert int(s.sum()) < int(a.sum())
    assert int(x.sum()) > 0


def test_mask_complement_within():
    m = np.ones((3, 3, 3), dtype=np.uint8)
    w = np.zeros_like(m)
    w[1, 1, 1] = 1
    c = mask_complement(m, w)
    assert int(c.sum()) == 0
    c2 = mask_complement(m)
    assert int(c2.sum()) == 0
