from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError

try:
    import vtk
    from vtk.util import numpy_support
except Exception:
    vtk = None
    numpy_support = None

__all__ = [
    "list_stl_files",
    "multilabel_from_stls",
    "read_nifti",
    "read_stl",
    "stl_to_numpy_binary",
    "stl_to_vtk_binary",
    "write_nifti",
]


def _require_vtk() -> None:
    if vtk is None or numpy_support is None:
        raise BackendUnavailableError('vtk is not installed. Please install it with "pip install vtk".')


def _update(filter_obj):
    filter_obj.Update()
    return filter_obj


def read_stl(stl_path: str | Path):
    _require_vtk()
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_path))
    return _update(reader)


def read_nifti(nifti_path: str | Path):
    _require_vtk()
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(nifti_path))
    return _update(reader)


def write_nifti(
    nifti_path: str | Path,
    image,
    transform_matrix,
    qfac: float,
) -> None:
    _require_vtk()
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(str(nifti_path))
    writer.SetInputData(image)
    if transform_matrix is not None:
        writer.SetQFormMatrix(transform_matrix)
        writer.SetSFormMatrix(transform_matrix)
    writer.SetQFac(float(qfac))
    out = Path(nifti_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    _update(writer)


def _get_surface_origin(
    bounds: Tuple[float, ...],
    spacing: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    # Center voxels by half-voxel from min bounds.
    return tuple(bounds[2 * i] + (s / 2.0) for i, s in enumerate(spacing))


def _get_surface_dimensions(
    bounds: Tuple[float, ...],
    spacing: Tuple[float, float, float],
) -> Tuple[int, int, int]:
    dims = [int((bounds[2 * i + 1] - bounds[2 * i]) // spacing[i]) for i in range(3)]
    return tuple(dims)


def _init_vtk_image(
    spacing: Tuple[float, float, float],
    dimensions: Tuple[int, int, int],
    origin: Tuple[float, float, float],
    direction,
    constant_value: int = 1,
):
    _require_vtk()
    image = vtk.vtkImageData()
    image.SetSpacing(spacing)
    image.SetDimensions(dimensions)
    if direction is not None:
        image.SetDirectionMatrix(direction)
    image.SetOrigin(origin)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    scalars = image.GetPointData().GetScalars()
    try:
        scalars.Fill(constant_value)
    except AttributeError:
        for i in range(scalars.GetNumberOfTuples()):
            scalars.SetTuple1(i, constant_value)
    return image


def _matrix_to_rotation_and_spacing(m) -> Tuple[object, Tuple[float, float, float]]:
    _require_vtk()
    rot = vtk.vtkMatrix3x3()
    rot.Identity()
    return rot, (1.0, 1.0, 1.0)


def _polydata_to_image_stencil(
    polydata,
    origin: Tuple[float, float, float],
    spacing: Tuple[float, float, float],
    extent: Tuple[int, int, int, int, int, int],
):
    _require_vtk()
    poly_to_stencil = vtk.vtkPolyDataToImageStencil()
    poly_to_stencil.SetInputData(polydata)
    poly_to_stencil.SetOutputOrigin(origin)
    poly_to_stencil.SetOutputSpacing(spacing)
    poly_to_stencil.SetOutputWholeExtent(extent)
    return _update(poly_to_stencil)


def _apply_image_stencil(
    vtk_image,
    poly2stencil=None,
    background_value: int = 0,
):
    _require_vtk()
    image_stencil = vtk.vtkImageStencil()
    image_stencil.SetInputData(vtk_image)
    if poly2stencil is not None:
        image_stencil.SetStencilConnection(poly2stencil.GetOutputPort())
    image_stencil.ReverseStencilOff()
    image_stencil.SetBackgroundValue(background_value)
    return _update(image_stencil)


def _translate_image(
    image,
    offset: Tuple[float, float, float],
    background_level: int = 0,
):
    _require_vtk()
    transform = vtk.vtkTransform()
    transform.Translate(*offset)
    reslice = vtk.vtkImageReslice()
    reslice.SetResliceTransform(transform)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.SetInputData(image)
    reslice.SetOutputSpacing(image.GetSpacing())
    reslice.SetOutputOrigin(image.GetOrigin())
    reslice.SetOutputExtent(image.GetExtent())
    reslice.SetBackgroundLevel(background_level)
    return _update(reslice)


def _set_image_origin(
    image,
    origin: Tuple[float, float, float],
):
    _require_vtk()
    changer = vtk.vtkImageChangeInformation()
    changer.SetInputData(image)
    changer.SetOutputOrigin(origin)
    return _update(changer)


def _get_origin_from_matrix(matrix) -> Tuple[float, float, float]:
    if matrix is None:
        return (0.0, 0.0, 0.0)
    # Match legacy behavior: apply diagonal sign to the translation terms.
    offset = [matrix.GetElement(i, 3) for i in range(3)]
    sign = [matrix.GetElement(i, i) for i in range(3)]
    return tuple(s * o for s, o in zip(sign, offset))


def stl_to_vtk_binary(
    stl_path: str | Path,
    reference_nifti_path: str | Path,
    constant_value_inside: int = 1,
):
    """
    Rasterize an STL surface into a binary vtkImageData aligned to a reference NIfTI.

    Returns `(vtkImageData, qform_matrix, qfac)`.
    """
    _require_vtk()
    surface_reader = read_stl(stl_path)
    ref_reader = read_nifti(reference_nifti_path)

    ref_img = ref_reader.GetOutput()
    ref_dims = ref_img.GetDimensions()
    ref_spacing = ref_img.GetSpacing()

    qform = ref_reader.GetQFormMatrix()
    sform = ref_reader.GetSFormMatrix()
    transform_matrix = qform if qform is not None else sform

    ref_origin = _get_origin_from_matrix(transform_matrix)
    ref_rotation = ref_img.GetDirectionMatrix()

    bounds = surface_reader.GetOutput().GetBounds()
    surface_origin = _get_surface_origin(bounds, ref_spacing)

    vtk_image = _init_vtk_image(
        spacing=ref_spacing,
        dimensions=ref_dims,
        origin=ref_origin,
        direction=ref_rotation,
        constant_value=constant_value_inside,
    )

    poly_to_stencil = _polydata_to_image_stencil(
        polydata=surface_reader.GetOutput(),
        origin=surface_origin,
        spacing=ref_spacing,
        extent=vtk_image.GetExtent(),
    )

    image_stencil = _apply_image_stencil(
        vtk_image=vtk_image,
        poly2stencil=poly_to_stencil,
        background_value=0,
    )

    offset = tuple(a - b for a, b in zip(ref_origin, surface_origin))
    translated = _translate_image(image_stencil.GetOutput(), offset=offset)
    aligned = _set_image_origin(translated.GetOutput(), origin=(0.0, 0.0, 0.0)).GetOutput()
    return aligned, transform_matrix, ref_reader.GetQFac()


def stl_to_numpy_binary(
    stl_path: str | Path,
    reference_nifti_path: str | Path,
) -> Tuple[np.ndarray, Dict]:
    _require_vtk()
    image, qform, qfac = stl_to_vtk_binary(stl_path, reference_nifti_path)
    dims = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    arr = numpy_support.vtk_to_numpy(scalars).reshape(dims[2], dims[1], dims[0]).astype(np.uint16)
    meta = {
        "dimensions": dims,
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
        "direction": image.GetDirectionMatrix(),
        "qform": qform,
        "qfac": qfac,
    }
    return arr, meta


def multilabel_from_stls(
    stl_paths: Iterable[str | Path],
    reference_nifti_path: str | Path,
    label_map: Dict[str, int] | None = None,
    label_start: int = 1,
    overwrite: bool = False,
):
    """
    Create a multi-label vtkImageData from a collection of STL files.

    Returns `(vtkImageData, qform, qfac, resolved_label_map)`.
    """
    _require_vtk()
    stl_paths = [Path(p) for p in stl_paths]
    if not stl_paths:
        raise ValueError("No STL files provided.")

    if label_map is None:
        stems = sorted(p.stem for p in stl_paths)
        label_map = {stem: idx for idx, stem in enumerate(stems, start=label_start)}

    ref_reader = read_nifti(reference_nifti_path)
    ref_img = ref_reader.GetOutput()
    dims = ref_img.GetDimensions()
    spacing = ref_img.GetSpacing()

    qform = ref_reader.GetQFormMatrix()
    sform = ref_reader.GetSFormMatrix()
    transform_matrix = qform if qform is not None else sform

    origin = _get_origin_from_matrix(transform_matrix)
    rotation = ref_img.GetDirectionMatrix()

    label_array = np.zeros((dims[2], dims[1], dims[0]), dtype=np.uint16)
    for stl_path in stl_paths:
        arr, _ = stl_to_numpy_binary(stl_path, reference_nifti_path)
        label_value = label_map.get(Path(stl_path).stem)
        if label_value is None:
            continue
        if overwrite:
            label_array[arr > 0] = label_value
        else:
            mask = (arr > 0) & (label_array == 0)
            label_array[mask] = label_value

    vtk_image = vtk.vtkImageData()
    vtk_image.SetSpacing(spacing)
    vtk_image.SetDimensions(dims)
    vtk_image.SetDirectionMatrix(rotation)
    vtk_image.SetOrigin(origin)

    vtk_data = numpy_support.numpy_to_vtk(
        num_array=label_array.ravel(order="C"),
        deep=True,
        array_type=vtk.VTK_UNSIGNED_SHORT,
    )
    vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_SHORT, 1)
    vtk_image.GetPointData().SetScalars(vtk_data)
    return vtk_image, transform_matrix, ref_reader.GetQFac(), label_map


def list_stl_files(input_path: str | Path) -> List[Path]:
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".stl":
        return [p]
    if p.is_dir():
        return sorted(item for item in p.rglob("*.stl") if item.is_file())
    raise FileNotFoundError(f"No STL files found at: {input_path}")
