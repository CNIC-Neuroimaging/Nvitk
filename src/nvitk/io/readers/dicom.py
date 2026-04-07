from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError, ValidationError

from .._common import reorder_axes

try:
    import pydicom
except Exception:
    pydicom = None


def _iter_dicom_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    out: list[Path] = []
    for item in sorted(path.rglob("*")):
        if not item.is_file():
            continue
        # DICOM files often have no extension.
        if item.suffix.lower() in {".dcm", ".dicom", ""}:
            out.append(item)
    return out


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]

    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass

    if hasattr(value, "__float__"):
        try:
            return float(value)
        except Exception:
            pass

    if hasattr(value, "__int__"):
        try:
            return int(value)
        except Exception:
            pass

    return str(value)


def _dataset_to_metadata(ds: Any, *, include_private_tags: bool) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for element in ds:
        if element.keyword == "PixelData":
            continue
        if element.tag.is_private and not include_private_tags:
            continue

        key = element.keyword if element.keyword else str(element.tag)
        value = _to_serializable(element.value)
        metadata[key] = value
        # Keep tag-code key too for compatibility with old pipelines.
        metadata[str(element.tag)] = value
    return metadata


def _slice_sort_key(ds: Any, fallback_idx: int) -> tuple[int, float, int]:
    instance = ds.get("InstanceNumber")
    if instance is None:
        instance_value = 10**9
    else:
        try:
            instance_value = int(instance)
        except Exception:
            instance_value = 10**9

    z_value = 0.0
    ipp = ds.get("ImagePositionPatient")
    if ipp is not None and len(ipp) >= 3:
        try:
            z_value = float(ipp[2])
        except Exception:
            z_value = 0.0
    elif ds.get("SliceLocation") is not None:
        try:
            z_value = float(ds.get("SliceLocation"))
        except Exception:
            z_value = 0.0

    return (instance_value, z_value, fallback_idx)


def _build_affine_from_dicom(datasets: list[Any], z_res: float | None) -> np.ndarray | None:
    if not datasets:
        return None
    first = datasets[0]

    iop = first.get("ImageOrientationPatient")
    ipp = first.get("ImagePositionPatient")
    pixel_spacing = first.get("PixelSpacing")

    if iop is None or ipp is None or pixel_spacing is None:
        return None

    try:
        row_cos = np.asarray([float(v) for v in iop[:3]], dtype=float)
        col_cos = np.asarray([float(v) for v in iop[3:6]], dtype=float)
        row_spacing = float(pixel_spacing[0])
        col_spacing = float(pixel_spacing[1])
        slice_spacing = float(z_res if z_res is not None else first.get("SliceThickness", 1.0))
        origin = np.asarray([float(v) for v in ipp[:3]], dtype=float)
    except Exception:
        return None

    slice_cos = np.cross(row_cos, col_cos)
    affine = np.eye(4, dtype=float)
    # NIfTI XYZ convention for data arranged as [X, Y, Z].
    affine[:3, 0] = col_cos * col_spacing
    affine[:3, 1] = row_cos * row_spacing
    affine[:3, 2] = slice_cos * slice_spacing
    affine[:3, 3] = origin
    return affine


def _stack_series(datasets: list[Any]) -> tuple[np.ndarray, str]:
    if not datasets:
        raise ValidationError("No decodable DICOM slices were found in selected series.")

    arrays = [np.asarray(ds.pixel_array) for ds in datasets]
    if len(arrays) == 1:
        arr = arrays[0]
        if arr.ndim == 2:
            return arr, "YX"
        if arr.ndim == 3:
            # Could be YXC or ZYX (single multi-frame). Keep conservative axis labels.
            return arr, "ZYX"
        return arr, "".join(f"D{i}" for i in range(arr.ndim))

    arr = np.stack(arrays, axis=0)
    if arr.ndim == 3:
        return arr, "ZYX"
    if arr.ndim == 4:
        return arr, "ZYXC"
    return arr, "".join(f"D{i}" for i in range(arr.ndim))


def _fill_resolution(metadata: dict[str, Any], datasets: list[Any]) -> float | None:
    if not datasets:
        return None

    first = datasets[0]
    pixel_spacing = first.get("PixelSpacing")
    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        try:
            # DICOM stores [row, col] -> [y, x]
            metadata["y_res"] = float(pixel_spacing[0])
            metadata["x_res"] = float(pixel_spacing[1])
            metadata["spacing"] = (
                metadata.get("x_res"),
                metadata.get("y_res"),
                metadata.get("z_res"),
            )
        except Exception:
            pass

    z_res = first.get("SpacingBetweenSlices", first.get("SliceThickness"))
    if z_res is None and len(datasets) > 1:
        try:
            pos0 = datasets[0].get("ImagePositionPatient")
            pos1 = datasets[1].get("ImagePositionPatient")
            if pos0 is not None and pos1 is not None:
                z_res = abs(float(pos1[2]) - float(pos0[2]))
        except Exception:
            z_res = None

    if z_res is not None:
        try:
            metadata["z_res"] = float(z_res)
        except Exception:
            pass

    if "x_res" in metadata or "y_res" in metadata or "z_res" in metadata:
        metadata["spacing"] = (
            metadata.get("x_res"),
            metadata.get("y_res"),
            metadata.get("z_res"),
        )

    frame_time = first.get("FrameTime")
    if frame_time is not None:
        try:
            metadata["temporal_resolution"] = float(frame_time) / 1000.0
            metadata["t_res"] = metadata["temporal_resolution"]
        except Exception:
            pass
    return float(metadata["z_res"]) if "z_res" in metadata else None


def _read_series(
    files: list[Path],
    *,
    include_private_tags: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    loaded: list[Any] = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), force=True)
            if getattr(ds, "PixelData", None) is None:
                continue
            loaded.append(ds)
        except Exception:
            continue

    loaded = sorted(enumerate(loaded), key=lambda pair: _slice_sort_key(pair[1], pair[0]))
    loaded = [ds for _, ds in loaded]
    data, axes_prev = _stack_series(loaded)

    metadata = _dataset_to_metadata(loaded[0], include_private_tags=include_private_tags)
    metadata["axes"] = axes_prev
    metadata["shape"] = tuple(data.shape)
    metadata["series_uid"] = str(loaded[0].get("SeriesInstanceUID", ""))
    metadata["series_description"] = str(loaded[0].get("SeriesDescription", ""))
    metadata["series_number"] = _to_serializable(loaded[0].get("SeriesNumber"))
    metadata["Modality"] = str(loaded[0].get("Modality", metadata.get("Modality", "")))
    metadata["(0008,0060)"] = metadata["Modality"]
    z_res = _fill_resolution(metadata, loaded)
    affine = _build_affine_from_dicom(loaded, z_res=z_res)
    if affine is not None:
        metadata["affine"] = affine

    return data, metadata


def read_dicom(
    path: str,
    *,
    axes: str | None = None,
    series_uid: str | None = None,
    series_index: int | None = None,
    return_all_series: bool = False,
    include_private_tags: bool = False,
    **_: Any,
):
    """
    Read DICOM source and return:
      - (data, metadata) when one series selected
      - list[(data, metadata)] when multiple series and no explicit selection
    """
    if pydicom is None:
        raise BackendUnavailableError('pydicom is not installed. Please install it with "pip install pydicom".')

    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(path)

    files = _iter_dicom_files(src)
    if not files:
        raise ValidationError(f"No DICOM files found under: {path}")

    grouped: dict[str, list[Path]] = {}
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True, specific_tags=["SeriesInstanceUID"])
            uid = str(ds.get("SeriesInstanceUID", "NO_SERIES"))
        except Exception:
            uid = "NO_SERIES"
        grouped.setdefault(uid, []).append(f)

    ordered_uids = sorted(grouped.keys())
    if series_uid is not None:
        if series_uid not in grouped:
            raise ValidationError(f"series_uid={series_uid!r} not found. Available: {ordered_uids}")
        ordered_uids = [series_uid]
    elif series_index is not None:
        if not (0 <= series_index < len(ordered_uids)):
            raise ValidationError(
                f"series_index={series_index} out of range for {len(ordered_uids)} discovered series"
            )
        ordered_uids = [ordered_uids[series_index]]

    outputs: list[tuple[np.ndarray, dict[str, Any]]] = []
    for uid in ordered_uids:
        data, metadata = _read_series(grouped[uid], include_private_tags=include_private_tags)
        if axes and axes != metadata["axes"]:
            data = reorder_axes(data, metadata["axes"], axes)
            metadata["axes"] = axes
            metadata["shape"] = tuple(data.shape)
        outputs.append((data, metadata))

    if return_all_series:
        return outputs
    if len(outputs) == 1:
        return outputs[0]
    return outputs
