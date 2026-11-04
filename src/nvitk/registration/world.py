"""
World-space forms of linear registration results — FLIRT and ANTs as 4x4 matrices.

Description
-----------
FLIRT and ANTs both reslice their output onto the fixed image's grid, which crops anything the
fixed field of view does not cover: a whole-brain T1 registered to a 90 mm TOF slab loses the
vertex. A linear transform has a grid-free form, a 4x4 matrix from the moving image's world (mm)
to the fixed image's world, which lets the caller resample onto any grid it chooses.

Conventions
-----------
World is the NIfTI world of the files the registration read: RAS mm, ``img.affine``.

* **FLIRT** ``.mat`` files map FSL *scaled-voxel* coordinates of the moving image to those of the
  reference. Scaled-voxel coordinates are voxel indices times voxel size, with the first axis
  flipped when the image's affine has a positive determinant (FSL's "neurological" storage).
  So ``world_map = A_ref @ inv(F_ref) @ mat @ F_mov @ inv(A_mov)``.
* **ANTs** linear transforms (``*GenericAffine.mat``) map *fixed* physical points to *moving*
  ones, in ITK's LPS. ``world_map`` is the inverse of that map, conjugated into RAS.

Only linear transforms have a world matrix; a deformable ANTs result raises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from nvitk.core.array import to_numpy
from nvitk.core.backend import setup, using

setup(globals())

#: LPS <-> RAS (its own inverse).
_LPS_RAS = (-1.0, -1.0, 1.0, 1.0)


def fsl_scaled_voxel_matrix(affine: Any, shape: Sequence[int], zooms: Sequence[float]) -> Any:
    """
    Voxel index -> FSL scaled-voxel coordinates for an image with *affine*, *shape*, *zooms*.

    FSL works in mm along the stored voxel axes, flipping the first one when the voxel-to-world
    affine has a positive determinant, so its internal space is always radiological.
    """
    with using("cpu"):
        aff = to_numpy(affine).astype(float)[:4, :4]
        scale = np.diag([float(zooms[0]), float(zooms[1]), float(zooms[2]), 1.0])
        if np.linalg.det(aff[:3, :3]) > 0:
            flip = np.eye(4)
            flip[0, 0] = -1.0
            flip[0, 3] = float(shape[0]) - 1.0
            return scale @ flip
        return scale


def _nifti_geometry(path: str | Path) -> tuple[Any, tuple[int, ...], tuple[float, ...]]:
    """``(affine, shape, zooms)`` of a NIfTI header (the data is not read)."""
    import nibabel as nib

    img = nib.load(str(path))
    return img.affine, tuple(int(s) for s in img.shape[:3]), tuple(float(z) for z in img.header.get_zooms()[:3])


def flirt_matrix_to_world(mat: str | Path | Any, moving: str | Path, reference: str | Path) -> Any:
    """
    A FLIRT matrix as a moving-world -> reference-world 4x4 (RAS mm).

    Parameters
    ----------
    mat : path or (4, 4) array
        The ``.mat`` FLIRT wrote (or its contents).
    moving, reference : path
        The NIfTI files FLIRT registered: the matrix is only meaningful against their headers.
    """
    with using("cpu"):
        if isinstance(mat, (str, Path)):
            m = np.loadtxt(str(mat)).astype(float)
        else:
            m = to_numpy(mat).astype(float)
        a_mov, s_mov, z_mov = _nifti_geometry(moving)
        a_ref, s_ref, z_ref = _nifti_geometry(reference)
        f_mov = fsl_scaled_voxel_matrix(a_mov, s_mov, z_mov)
        f_ref = fsl_scaled_voxel_matrix(a_ref, s_ref, z_ref)
        return a_ref @ np.linalg.inv(f_ref) @ m @ f_mov @ np.linalg.inv(a_mov)


def ants_linear_to_world(transforms: str | Path | Sequence[str | Path]) -> Any:
    """
    ANTs forward transform(s) of a linear registration as a moving-world -> fixed-world 4x4.

    Parameters
    ----------
    transforms : path or sequence of paths
        ``fwdtransforms`` of ``ants.registration`` (Rigid / Similarity / Affine...): every entry
        must be a linear ``.mat``; a displacement field raises ``ValueError``.
    """
    import ants

    paths = [Path(transforms)] if isinstance(transforms, (str, Path)) else [Path(p) for p in transforms]
    if not paths:
        raise ValueError("No ANTs transform to convert.")
    nonlinear = [p.name for p in paths if p.suffix.lower() != ".mat"]
    if nonlinear:
        raise ValueError(
            f"ANTs transform(s) {nonlinear} are deformable: only a linear registration "
            "(Rigid, Similarity, Affine) has a world matrix."
        )
    with using("cpu"):
        # fwdtransforms are applied last-first to map a fixed point into the moving space.
        fixed_to_moving = np.eye(4)
        for path in paths:
            tx = ants.read_transform(str(path))
            origin = np.array(tx.apply_to_point((0.0, 0.0, 0.0)), dtype=float)
            cols = [np.array(tx.apply_to_point(tuple(e)), dtype=float) - origin for e in np.eye(3)]
            step = np.eye(4)
            step[:3, :3] = np.stack(cols, axis=1)
            step[:3, 3] = origin
            fixed_to_moving = fixed_to_moving @ step
        flip = np.diag(_LPS_RAS)
        return np.linalg.inv(flip @ fixed_to_moving @ flip)


__all__ = ["ants_linear_to_world", "flirt_matrix_to_world", "fsl_scaled_voxel_matrix"]
