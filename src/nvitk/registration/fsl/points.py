"""Map voxel indices between images using a FLIRT ``*.mat`` and NIfTI affines."""

from __future__ import annotations

from pathlib import Path

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup

setup(globals())


def load_flirt_matrix(mat_path: str | Path) -> np.ndarray:
    """Load a 4×4 FLIRT transform (input mm → reference mm)."""
    raw = np.loadtxt(str(mat_path), dtype=np.float64)
    mat = as_backend_array(raw).astype(np.float64, copy=False).reshape((4, 4))
    return mat


def transform_ijk_points_flirt(
    points: np.ndarray,
    moving_affine: np.ndarray,
    fixed_affine: np.ndarray,
    flirt_mat: np.ndarray,
) -> np.ndarray:
    """Map ``(i,j,k)`` points from moving image voxels to fixed image voxels.

    Uses moving/fixed NIfTI affines and the FLIRT matrix from
    ``flirt -in <moving> -ref <fixed>`` (same as ``tof_to_vwi_bb.mat``).
    """
    pts = as_backend_array(points).astype(np.float64, copy=False)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be Nx3, got {pts.shape}")
    moving_affine = as_backend_array(moving_affine).astype(np.float64, copy=False).reshape((4, 4))
    fixed_affine = as_backend_array(fixed_affine).astype(np.float64, copy=False).reshape((4, 4))
    flirt_mat = as_backend_array(flirt_mat).astype(np.float64, copy=False).reshape((4, 4))

    n = pts.shape[0]
    ijk_h = np.ones((n, 4), dtype=np.float64)
    ijk_h[:, :3] = pts
    moving_mm = (moving_affine @ ijk_h.T).T
    ref_mm = (flirt_mat @ moving_mm.T).T
    fixed_inv = np.linalg.inv(fixed_affine)
    ref_ijk = (fixed_inv @ ref_mm.T).T[:, :3]
    return ref_ijk
