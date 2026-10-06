"""Meshes back to voxels: fill a closed surface onto an image grid."""

from __future__ import annotations

from typing import Any

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.meshlab.convert import to_pyvista
from nvitk.types import Image, Mesh


def _stencil(mesh_ijk: Mesh, shape: tuple[int, int, int]) -> np.ndarray:
    """Inside test of a mesh given in voxel-index coordinates, on a ``shape`` grid."""
    import vtk
    from vtk.util import numpy_support

    nx, ny, nz = (int(v) for v in shape)
    poly = to_pyvista(mesh_ijk)
    stencil = vtk.vtkPolyDataToImageStencil()
    stencil.SetInputData(poly)
    stencil.SetOutputOrigin(0.0, 0.0, 0.0)
    stencil.SetOutputSpacing(1.0, 1.0, 1.0)
    stencil.SetOutputWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    stencil.Update()
    blank = vtk.vtkImageData()
    blank.SetDimensions(nx, ny, nz)
    blank.SetOrigin(0.0, 0.0, 0.0)
    blank.SetSpacing(1.0, 1.0, 1.0)
    blank.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    numpy_support.vtk_to_numpy(blank.GetPointData().GetScalars())[:] = 1
    cut = vtk.vtkImageStencil()
    cut.SetInputData(blank)
    cut.SetStencilConnection(stencil.GetOutputPort())
    cut.ReverseStencilOff()
    cut.SetBackgroundValue(0)
    cut.Update()
    flat = numpy_support.vtk_to_numpy(cut.GetOutput().GetPointData().GetScalars())
    # VTK stores x fastest: (z, y, x) in C order.
    return flat.reshape(nz, ny, nx).transpose(2, 1, 0).astype(bool)


def mesh_to_mask(
    mesh: Mesh,
    reference: Image | None = None,
    *,
    affine: np.ndarray | None = None,
    shape: tuple[int, int, int] | None = None,
    label: int = 1,
) -> Image:
    """Voxelise the solid a closed *mesh* encloses onto a reference image grid.

    The grid is *reference*'s (its shape and affine), or *shape* + *affine*. The
    mesh is taken in that affine's world coordinates — the space meshes from
    :func:`~nvitk.meshlab.mesh_from_image` are in. Any affine works, oblique
    ones included: the mesh is mapped into voxel indices before filling.
    """
    if reference is not None:
        shape = tuple(int(v) for v in reference.shape[:3])
        affine = reference.affine if reference.affine is not None else np.eye(4)
    if shape is None:
        raise ValueError("Give a reference image, or a shape and an affine.")
    aff = np.eye(4) if affine is None else np.asarray(to_numpy(affine), dtype=float)
    inv = np.linalg.inv(aff)
    hom = np.c_[mesh.vertices, np.ones(mesh.n_vertices)]
    ijk = (inv @ hom.T).T[:, :3]
    inside = _stencil(mesh.with_vertices(ijk), shape)
    dtype = np.uint8 if int(label) < 256 else np.int32
    data = inside.astype(dtype) * dtype(int(label))
    meta: dict[str, Any] = {"affine": aff}
    if reference is not None:
        meta = dict(reference.metadata or {})
        meta["affine"] = aff
    return Image(data=data, metadata=meta, axes="XYZ", name=f"{mesh.name}_mask")


def voxelize_to_grid(mesh: Mesh, *, spacing: float = 1.0, margin: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``(mask, affine)`` on an axis-aligned grid of *spacing* covering the mesh."""
    lo, hi = mesh.bounds
    origin = lo - margin * spacing
    shape = tuple(int(np.ceil((h - l) / spacing)) + 2 * margin + 1 for l, h in zip(lo, hi))
    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = origin
    img = mesh_to_mask(mesh, affine=affine, shape=shape)
    return np.asarray(img.data, dtype=bool), affine


__all__ = ["mesh_to_mask", "voxelize_to_grid"]
