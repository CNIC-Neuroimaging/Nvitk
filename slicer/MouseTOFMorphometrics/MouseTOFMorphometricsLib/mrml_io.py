"""MRML ↔ NIfTI / VTP marshalling for the morphometrics Slicer module.

The morphometrics pipeline resolves each vessel's proximal end from the image
affine (see ``nvitk.measure.morpho.anatomy_axes``), so the NIfTI written here
must carry Slicer's real orientation — a wrong affine silently reverses
centerlines rather than raising.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from typing import Optional

import numpy as np
import slicer
import vtk

log = logging.getLogger(__name__)

#: Same tints the Mouse TOF CoW module uses for labels 1/2/3.
TREE_COLORS_RGB = {
    1: (51, 153, 255),   # Left ICA — blue
    2: (255, 89, 89),    # Right ICA — red
    3: (89, 230, 102),   # Basilar — green
}
_FALLBACK_RGB = (200, 200, 90)


def array_from_labelmap(labelNode) -> np.ndarray:
    """Labelmap voxels as an int32 IJK-ordered array (``slicer.util`` hands back KJI)."""
    arr_kji = np.asarray(slicer.util.arrayFromVolume(labelNode))
    return np.transpose(arr_kji, (2, 1, 0)).astype(np.int32)


def ijk_to_ras_affine(volumeNode) -> np.ndarray:
    """The node's IJK→RAS 4x4 — the same convention nibabel stores as its affine."""
    m = vtk.vtkMatrix4x4()
    volumeNode.GetIJKToRASMatrix(m)
    return np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)], dtype=float)


def write_labelmap_nifti(labelNode, path: str, data: Optional[np.ndarray] = None) -> str:
    """Write *labelNode* (or *data* on its grid) to a ``.nii.gz`` with the correct affine.

    Prefers ``slicer.util.saveNode`` — it goes through Slicer's own writer and
    handles the RAS/LPS convention. Only when *data* differs from the node, or
    that write fails, does it fall back to composing the affine by hand.
    """
    if data is None:
        try:
            if slicer.util.saveNode(labelNode, path):
                return path
        except Exception as exc:  # noqa: BLE001
            log.warning("slicer.util.saveNode failed (%s); falling back to nibabel.", exc)
        data = array_from_labelmap(labelNode)

    import nibabel as nib

    arr = np.asarray(data, dtype=np.int32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D labelmap, got shape {arr.shape}.")
    nib.Nifti1Image(arr, ijk_to_ras_affine(labelNode)).to_filename(path)
    return path


def safe_filename(name: str) -> str:
    """Sanitize a node name into a filesystem-safe stem."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_")
    return cleaned or "labelmap"


def label_id_from_stem(stem: str) -> Optional[int]:
    """Leading label id in a morphometrics VTP stem (``3_BASILAR_comp01`` → ``3``)."""
    match = re.match(r"^(\d+)_", os.path.basename(stem))
    return int(match.group(1)) if match else None


def _color_for(label_id: Optional[int]) -> tuple[float, float, float]:
    rgb = TREE_COLORS_RGB.get(int(label_id)) if label_id is not None else None
    r, g, b = rgb or _FALLBACK_RGB
    return r / 255.0, g / 255.0, b / 255.0


def vtp_to_ras_matrix(nifti_path: str) -> Optional[np.ndarray]:
    """4x4 mapping the pipeline's VTP coordinates to Slicer's RAS world.

    The morphometrics pipeline works in a *scaled voxel index* frame: points are
    stored as ``voxel_index * spacing`` with the origin at ``(0, 0, 0)`` and no
    direction cosines (surfaces are built with ``SetOrigin(0, 0, 0)``). Loading
    those VTPs straight into Slicer therefore places them in a corner of the
    volume, rotated and offset from the labelmap they describe.

    Dividing out the spacing recovers the voxel index, and the NIfTI affine
    (IJK→RAS, nibabel's convention, which is also Slicer's world) puts it back
    where the anatomy is::

        RAS = affine @ diag(1/sx, 1/sy, 1/sz, 1) @ point_vtp

    The affine is read from the NIfTI that was actually handed to the pipeline,
    so this stays correct regardless of how Slicer chose to write the file.
    """
    try:
        import nibabel as nib

        img = nib.load(str(nifti_path))
        spacing = np.asarray(img.header.get_zooms()[:3], dtype=float)
        if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
            return None
        return np.asarray(img.affine, dtype=float) @ np.diag([*(1.0 / spacing), 1.0])
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not derive VTP→RAS transform from %s: %s", nifti_path, exc)
        return None


def find_case_nifti(case_dir: str) -> Optional[str]:
    """A NIfTI in *case_dir* carrying the case geometry (the Taubin-smoothed mask)."""
    for pattern in ("*_taubin.nii.gz", "*.nii.gz", "*.nii"):
        hits = sorted(glob.glob(os.path.join(case_dir, pattern)))
        if hits:
            return hits[0]
    return None


def _transformed_polydata(path: str, matrix: Optional[np.ndarray]):
    """Read a VTP and map it into RAS with *matrix* (identity when ``None``)."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        return None
    if matrix is None:
        return poly

    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, float(matrix[i][j]))
    transform = vtk.vtkTransform()
    transform.SetMatrix(vtk_matrix)

    # Bakes the transform into the points so the models are plain RAS geometry —
    # no transform node left in the scene for the user to trip over.
    filt = vtk.vtkTransformPolyDataFilter()
    filt.SetInputData(poly)
    filt.SetTransform(transform)
    filt.Update()
    return filt.GetOutput()


def load_result_models(
    case_dir: str,
    *,
    load_centerlines: bool = True,
    load_surfaces: bool = True,
    surface_opacity: float = 0.25,
    name_prefix: str = "",
    matrix: Optional[np.ndarray] = None,
) -> list:
    """Load the run's centerline / surface VTPs as MRML model nodes, in RAS.

    Returns the created nodes, coloured by label id. Vessel surfaces are loaded
    semi-transparent so the centerlines stay visible through them. *matrix* is
    the VTP→RAS transform from :func:`vtp_to_ras_matrix`; when omitted it is
    derived from a NIfTI found in *case_dir*, so the models land on top of the
    segmentation instead of in a corner of the volume.
    """
    if matrix is None:
        case_nifti = find_case_nifti(case_dir)
        if case_nifti:
            matrix = vtp_to_ras_matrix(case_nifti)
        if matrix is None:
            log.warning(
                "No case NIfTI in %s to derive the VTP→RAS transform; models will be "
                "placed in the pipeline's scaled-voxel frame.", case_dir,
            )

    nodes = []
    groups = []
    if load_centerlines:
        groups.append((os.path.join(case_dir, "centerlines"), 1.0, 3.0))
    if load_surfaces:
        groups.append((os.path.join(case_dir, "surfaces"), float(surface_opacity), 1.0))

    for directory, opacity, line_width in groups:
        if not os.path.isdir(directory):
            continue
        for path in sorted(glob.glob(os.path.join(directory, "*.vtp"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            try:
                poly = _transformed_polydata(path, matrix)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load %s: %s", path, exc)
                continue
            if poly is None:
                continue
            node = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLModelNode", f"{name_prefix}{stem}" if name_prefix else stem
            )
            node.SetAndObservePolyData(poly)
            node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            if display is not None:
                display.SetColor(*_color_for(label_id_from_stem(stem)))
                display.SetOpacity(opacity)
                display.SetLineWidth(line_width)
                display.SetVisibility(True)
                try:
                    display.SetScalarVisibility(False)
                except Exception:
                    pass
            nodes.append(node)

    if nodes:
        try:
            layoutManager = slicer.app.layoutManager()
            if layoutManager is not None and layoutManager.threeDViewCount > 0:
                layoutManager.threeDWidget(0).threeDView().resetFocalPoint()
        except Exception:
            pass
    return nodes


def remove_nodes(nodes) -> None:
    """Remove previously loaded result nodes from the scene."""
    for node in list(nodes or []):
        try:
            slicer.mrmlScene.RemoveNode(node)
        except Exception:
            pass


__all__ = [
    "TREE_COLORS_RGB",
    "array_from_labelmap",
    "find_case_nifti",
    "ijk_to_ras_affine",
    "label_id_from_stem",
    "load_result_models",
    "remove_nodes",
    "safe_filename",
    "vtp_to_ras_matrix",
    "write_labelmap_nifti",
]
