"""Tests for skeleton-graph polyline extraction."""

import numpy as np

from nvitk.morphology.polyline_graph import (
    branch_polylines_from_skeleton,
    detect_junctions_from_centerline,
    extract_polylines_from_centerline,
    junction_nodes_from_skeleton,
)


def test_junction_split_y_shape():
    # Y skeleton: junction at (5,5,5)
    pts = []
    for z in range(5, 11):
        pts.append([5, 5, z])
    for x in range(6, 10):
        pts.append([x, 5, 5])
    coords = np.asarray(pts, dtype=np.float32)
    branches = branch_polylines_from_skeleton(coords, min_points=3)
    assert len(branches) >= 2
    junctions = junction_nodes_from_skeleton(coords, min_degree=3)
    assert junctions.shape[0] >= 1


def test_extract_from_centerline_mask():
    vol = np.zeros((20, 20, 20), dtype=np.uint8)
    vol[10, 10, 5:15] = 1
    vol[10, 8:12, 10] = 1
    branches = extract_polylines_from_centerline(
        vol, mode="junction_split", min_points=3, per_connected_component=True
    )
    assert len(branches) >= 2
    junc = detect_junctions_from_centerline(vol, min_degree=3)
    assert junc.shape[0] >= 1
