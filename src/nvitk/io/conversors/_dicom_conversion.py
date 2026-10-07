"""
DICOM discovery, series selection, dicom2nifti execution, and in-memory :func:`load_dicom_series`.

RT struct, tissue, and vendor-specific hooks live in sibling ``_dicom_*`` modules.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError, ValidationError
from nvitk.core.logger import Logger
from .._common import default_nifti_axes, orientation_codes_from_affine, reorder_axes
from ..writers.tiff import write_tiff

from ._dicom_dynamic import (
    build_temporal_image,
    detect_temporal_layout,
    dicom2nifti_affine,
    rescaled_slice,
    series_label,
    split_by_geometry,
    storage_dtype,
)
from ._dicom_philips import (
    apply_heart_rate_alias,
    copy_sbi_series,
    is_spectral_base_series,
    sbi_series_summary,
    sbi_slice_positions,
    write_sbi_manifest,
)
from ._dicom_spectral import annotate_spectral
from ._dicom_rtstructs import integrate_rtstruct_processing
from ._dicom_waveform import export_waveforms, is_waveform
from ._dicom_tissue import extract_tissue_segmentation_data, is_tissue_segmentation
from ._dicom_zeiss import (
    ZeissOctAligner,
    derived_analysis_affine,
    extract_zeiss_oct_cubes,
    extract_zeiss_oct_images,
    extract_zeiss_raw_oct,
    is_zeiss_raw_storage,
    has_obfuscated_cube_payload,
    zeiss_referenced_sources,
)

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import pydicom
except Exception:
    pydicom = None

try:
    import dicom2nifti as _dicom2nifti
    from dicom2nifti.common import is_valid_imaging_dicom
except Exception:
    _dicom2nifti = None
    is_valid_imaging_dicom = None

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.ndimage import map_coordinates
    from scipy.spatial.distance import pdist, squareform

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.cluster.hierarchy")
warnings.filterwarnings(
    "ignore",
    message="The symmetric non-negative hollow observation matrix looks suspiciously like an uncondensed distance matrix",
)

__all__ = [
    "load_dicom_series",
    "run_dicom2nifti",
]

log = Logger()

METADATA_TO_SAVE = [
    "PatientID",
    "(0010,0020)",
    "PatientSex",
    "(0010,0040)",
    "PatientAge",
    "(0010,1010)",
    "PatientWeight",
    "(0010,1030)",
    "PatientSize",
    "(0010,1020)",
    "StudyDate",
    "(0008,0020)",
    "SeriesDate",
    "(0008,0021)",
    "SeriesTime",
    "(0008,0031)",
    "SeriesInstanceUID",
    "(0020,000E)",
    "SeriesDescription",
    "(0008,103E)",
    "SeriesNumber",
    "(0020,0011)",
    "AccessionNumber",
    "(0008,0050)",
    "Modality",
    "(0008,0060)",
    "ImageType",
    "(0008,0008)",
    "Manufacturer",
    "(0008,0070)",
    "Rows",
    "(0028,0010)",
    "Columns",
    "(0028,0011)",
    "PixelSpacing",
    "(0028,0030)",
    "SliceThickness",
    "(0018,0050)",
    "SpacingBetweenSlices",
    "(0018,0088)",
    "ImageOrientationPatient",
    "(0020,0037)",
    "ImagePositionPatient",
    "(0020,0032)",
    "RadiopharmaceuticalStartTime",
    "(0018,1072)",
    "RadionuclideHalfLife",
    "(0018,1075)",
    "RadionuclideTotalDose",
    "(0018,1074)",
    "DecayCorrection",
    "(0054,1102)",
    "DecayFactor",
    "(0054,1321)",
    "FrameReferenceTime",
    "(0054,1300)",
    "DoseUnits",
    "(0054,1004)",
    "Units",
    "(0054,1001)",
    "RescaleSlope",
    "(0028,1053)",
    "RescaleIntercept",
    "(0028,1052)",
    "ScaleSlope",
    "(2005,100e)",
    "InstanceNumber",
    "(0020,0013)",
    "Laterality",
    "(0020,0060)",
    "ImageLaterality",
    "(0020,0062)",
    "submodality",
    "rescale_type",
    "HeartRate",
    "(0018,1088)",
    "VENC",
    "(2001,101A)",
]

OP_SOP_CLASS_UIDS = {
    "1.2.840.10008.5.1.4.1.1.77.1.5.1",
    "1.2.840.10008.5.1.4.1.1.77.1.1",
    "1.2.840.10008.5.1.4.1.1.77.1.1.1",
}

OCT_SOP_CLASS_UIDS = {
    "1.2.840.10008.5.1.4.1.1.12.77",
    "1.2.840.10008.5.1.4.1.1.77.1.5.4",
    "1.2.840.10008.5.1.4.1.1.66",
}

RT_SOP_CLASS_UIDS = {
    "1.2.840.10008.5.1.4.1.1.48",
    "1.2.840.10008.5.1.4.1.1.481",
    "1.2.840.10008.5.1.4.1.1.481.1",
    "1.2.840.10008.5.1.4.1.1.481.2",
    "1.2.840.10008.5.1.4.1.1.481.3",
    "1.2.840.10008.5.1.4.1.1.481.4",
    "1.2.840.10008.5.1.4.1.1.481.5",
    "1.2.840.10008.5.1.4.1.1.481.6",
    "1.2.840.10008.5.1.4.1.1.481.7",
    "1.2.840.10008.5.1.4.1.1.481.8",
    "1.2.840.10008.5.1.4.1.1.481.9",
}


def _info(message: str) -> None:
    """Log an info message through the module logger."""
    log.info(message)


def _warn(message: str) -> None:
    """Log a warning through the module logger."""
    log.warning(message)


def _err(message: str) -> None:
    """Log an error through the module logger."""
    log.error(message)


def _debug(message: str) -> None:
    """No-op placeholder (debug logging is disabled for this module)."""
    return


def _require_deps() -> None:
    """Raise a clear install hint unless pydicom and nibabel are both available."""
    if pydicom is None:
        raise BackendUnavailableError('pydicom is not installed. Please install it with "pip install pydicom".')
    if nib is None:
        raise BackendUnavailableError('nibabel is not installed. Please install it with "pip install nibabel".')


def _sanitize_filename(filename: str) -> str:
    """Strip control chars, replace filesystem-illegal characters, and collapse whitespace/underscores."""
    if not isinstance(filename, str):
        filename = str(filename)
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", filename)
    for char in ["/", "\\", ":", "*", "?", '"', "<", ">", "|"]:
        cleaned = cleaned.replace(char, "_")
    cleaned = "_".join(cleaned.split()).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned


def _normalize_dicom_text(value: Any) -> str:
    """Coerce a DICOM text/multivalue element to a clean, single-value string."""
    if value is None:
        return ""
    if pydicom is not None and isinstance(value, pydicom.multival.MultiValue):
        value = value[0] if value else ""
    return str(value).strip("\x00").strip()


def _normalize_series_uid(uid: Any) -> str:
    """Coerce a SeriesInstanceUID (possibly a MultiValue) to a clean string."""
    if pydicom is not None and isinstance(uid, pydicom.multival.MultiValue):
        uid = uid[0] if uid else ""
    return _normalize_dicom_text(uid)


def _iter_candidate_file_paths(path: str | Sequence[str]) -> list[str]:
    """List every file under *path* (recursively, if a directory) as candidate DICOM files.

    *path* may also be a list of files and / or directories (e.g. the files of one
    series picked in a DICOM browser).
    """
    if isinstance(path, (list, tuple)):
        out: list[str] = []
        for item in path:
            out.extend(_iter_candidate_file_paths(str(item)))
        return out
    path = str(path)
    if os.path.isdir(path):
        fpaths: list[str] = []
        for dp, _, fns in os.walk(path):
            for fn in fns:
                fpaths.append(os.path.join(dp, fn))
        return fpaths
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(f"Invalid path provided: {path}")


def _select_candidate_file_paths(
    fpaths: list[str],
    *,
    series_number: str | None = None,
) -> list[str]:
    """Filter candidate files to one DICOM series by SeriesNumber (header-only scan, no pixel data read)."""
    if series_number is None:
        return fpaths

    target_number = str(series_number)
    header_rows: list[tuple[str, str, str]] = []
    target_uids: set[str] = set()

    for fp in fpaths:
        try:
            ds = pydicom.dcmread(
                fp,
                stop_before_pixels=True,
                force=True,
                specific_tags=["SeriesNumber", "SeriesInstanceUID"],
            )
        except Exception:
            continue

        uid = _normalize_series_uid(getattr(ds, "SeriesInstanceUID", None))
        sn = _normalize_dicom_text(getattr(ds, "SeriesNumber", None))
        header_rows.append((fp, uid, sn))
        if sn == target_number and uid:
            target_uids.add(uid)

    if target_uids:
        selected = [fp for fp, uid, sn in header_rows if sn == target_number or (uid and uid in target_uids)]
    else:
        selected = [fp for fp, _, sn in header_rows if sn == target_number]

    if selected:
        _info(
            f"Series-number header scan kept {len(selected)}/{len(fpaths)} candidate files for "
            f"series_number={series_number}"
        )
    return selected


def _ensure_output_metadata_fields(
    md: dict[str, Any],
    *,
    rescale_type: str | None = None,
) -> dict[str, Any]:
    """In-place: normalize/duplicate series identity fields (UID, number, description, modality, rescale type)
    into both their DICOM-tag and nvitk-convention metadata keys.
    """
    series_uid = _normalize_series_uid(md.get("SeriesInstanceUID", md.get("series_uid")))
    if series_uid:
        md["series_uid"] = series_uid
        md.setdefault("SeriesInstanceUID", series_uid)

    series_number = md.get("SeriesNumber", md.get("series_number"))
    if series_number not in (None, ""):
        md["series_number"] = series_number

    series_description = _normalize_dicom_text(md.get("SeriesDescription", md.get("series_description")))
    if series_description:
        md["series_description"] = series_description
        md.setdefault("SeriesDescription", series_description)
        md["submodality"] = series_description

    modality = _normalize_dicom_text(md.get("Modality", md.get("(0008,0060)")))
    if modality:
        md["Modality"] = modality
        md["(0008,0060)"] = modality

    if rescale_type is not None:
        md["rescale_type"] = str(rescale_type).upper()
    else:
        md.setdefault("rescale_type", "DV")
    return md


def _strip_array_values(value: Any, _seen: set[int] | None = None) -> tuple[bool, Any]:
    """Recursively drop array/binary/dataset values from a metadata tree (they don't belong in a JSON sidecar).

    Returns ``(keep, cleaned_value)``: ``keep=False`` tells the caller to omit
    this value entirely.
    """
    if _seen is None:
        _seen = set()

    if isinstance(value, np.ndarray):
        return False, None
    if pydicom is not None and isinstance(value, pydicom.dataset.Dataset):
        return False, None
    if isinstance(value, (bytes, bytearray)):
        return True, _convert_dicom_value(bytes(value))

    if isinstance(value, (dict, list, tuple)):
        obj_id = id(value)
        if obj_id in _seen:
            return False, None
        _seen.add(obj_id)

    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            keep, item_clean = _strip_array_values(item, _seen)
            if keep:
                cleaned[key] = item_clean
        return True, cleaned

    if isinstance(value, (list, tuple)):
        cleaned_items = []
        for item in value:
            keep, item_clean = _strip_array_values(item, _seen)
            if keep:
                cleaned_items.append(item_clean)
        return True, cleaned_items

    return True, value


def _create_ras_oriented_nifti(pixel_arrays: list[np.ndarray], output_path: str) -> bool:
    """Stack raw pixel arrays into a volume and save as RAS-oriented NIfTI (fallback path with no true DICOM geometry).

    Returns ``True`` on success, ``False`` (with a logged warning) on failure.
    """
    try:
        if isinstance(pixel_arrays, list) and len(pixel_arrays) > 1:
            array = np.array(pixel_arrays)
            if array.ndim == 3:
                volume = array.transpose(2, 1, 0)
            elif array.ndim == 4:
                volume = array.transpose(3, 2, 1, 0)
            else:
                volume = array
        elif isinstance(pixel_arrays, list) and len(pixel_arrays) == 1:
            arr = np.array(pixel_arrays[0])
            if arr.ndim == 2:
                volume = arr.T
            elif arr.ndim == 3:
                volume = arr.transpose(2, 1, 0)
            else:
                volume = arr
        else:
            arr = np.array(pixel_arrays)
            if arr.ndim == 2:
                volume = arr.T
            elif arr.ndim == 3:
                volume = arr.transpose(2, 1, 0)
            else:
                volume = arr

        image = nib.Nifti1Image(volume, np.eye(4))
        ornt_from = nib.orientations.io_orientation(image.affine)
        ornt_to = nib.orientations.axcodes2ornt(("R", "A", "S"))
        xfm = nib.orientations.ornt_transform(ornt_from, ornt_to)
        if xfm.size:
            data_ras = nib.orientations.apply_orientation(image.get_fdata(), xfm)
            aff_ras = image.affine @ nib.orientations.inv_ornt_aff(xfm, image.shape)
            image = nib.Nifti1Image(data_ras, aff_ras, header=image.header.copy())
            image.set_sform(aff_ras, code=1)
            image.set_qform(aff_ras, code=1)
        nib.save(image, output_path)
        return True
    except Exception as exc:
        _warn(f"Failed to create RAS-oriented NIfTI: {exc}")
        return False


def _reorient_pixel_array(
    pixel_array: np.ndarray,
    original_orientation: np.ndarray,
    target_orientation: np.ndarray,
) -> np.ndarray:
    """Reorient a 2-D slice from its original ImageOrientationPatient to a target orientation.

    Detects pure 90°/180°/270° rotations and applies them with cheap flip/transpose
    ops; falls back to a general affine resample (``scipy.ndimage.map_coordinates``)
    for non-axis-aligned transforms. Returns the input unchanged on any failure.
    """
    try:
        orig_orient = np.array(original_orientation)
        target_orient = np.array(target_orientation)
        orig_matrix = np.column_stack([orig_orient[0:2], orig_orient[3:5]])
        target_matrix = np.column_stack([target_orient[0:2], target_orient[3:5]])
        transform_matrix = target_matrix @ np.linalg.inv(orig_matrix)
        det = np.linalg.det(transform_matrix)
        if abs(det - 1.0) < 1e-6:
            angle = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])
            angle_deg = np.degrees(angle)
            rounded_angle = round(angle_deg / 90) * 90
            if abs(angle_deg - rounded_angle) < 5:
                if abs(rounded_angle - 90) < 1:
                    return np.flipud(pixel_array.T)
                if abs(rounded_angle - 180) < 1:
                    return np.flipud(np.fliplr(pixel_array))
                if abs(rounded_angle - 270) < 1 or abs(rounded_angle + 90) < 1:
                    return np.fliplr(pixel_array.T)
                if abs(rounded_angle) < 1:
                    return pixel_array

        if SCIPY_AVAILABLE:
            rows, cols = pixel_array.shape
            y_coords, x_coords = np.mgrid[0:rows, 0:cols]
            coords = np.stack([x_coords.flatten(), y_coords.flatten()], axis=1)
            transformed_coords = (transform_matrix @ coords.T).T
            x_transformed = transformed_coords[:, 0].reshape(rows, cols)
            y_transformed = transformed_coords[:, 1].reshape(rows, cols)
            return map_coordinates(
                pixel_array,
                [y_transformed, x_transformed],
                order=1,
                mode="constant",
                cval=0,
            )
        _warn("scipy not available, cannot perform complex pixel array reorientation")
        return pixel_array
    except Exception as exc:
        _warn(f"Failed to reorient pixel array: {exc}")
        return pixel_array


def _clean_nan_values(ds_list: list[Any]) -> list[Any]:
    """Repair NaN/Inf values in each dataset's ImageOrientationPatient, re-normalizing the row/column direction vectors."""
    cleaned_ds_list = []
    nan_fixes = 0
    for idx, ds in enumerate(ds_list):
        try:
            cleaned_ds = ds.copy()
            if hasattr(cleaned_ds, "ImageOrientationPatient"):
                iop = np.array([float(x) for x in cleaned_ds.ImageOrientationPatient])
                if np.any(np.isnan(iop)) or np.any(np.isinf(iop)):
                    iop_clean = np.where(np.isnan(iop) | np.isinf(iop), 0.0, iop)
                    if np.linalg.norm(iop_clean[:3]) > 1e-6:
                        iop_clean[:3] = iop_clean[:3] / np.linalg.norm(iop_clean[:3])
                    if np.linalg.norm(iop_clean[3:]) > 1e-6:
                        iop_clean[3:6] = iop_clean[3:6] / np.linalg.norm(iop_clean[3:6])
                    cleaned_ds.ImageOrientationPatient = list(iop_clean)
                    nan_fixes += 1
            if hasattr(cleaned_ds, "ImagePositionPatient"):
                ipp = np.array([float(x) for x in cleaned_ds.ImagePositionPatient])
                if np.any(np.isnan(ipp)) or np.any(np.isinf(ipp)):
                    ipp_clean = np.where(np.isnan(ipp) | np.isinf(ipp), 0.0, ipp)
                    cleaned_ds.ImagePositionPatient = list(ipp_clean)
                    nan_fixes += 1
            cleaned_ds_list.append(cleaned_ds)
        except Exception as exc:
            _warn(f"Failed to clean slice {idx}: {exc}")
            cleaned_ds_list.append(ds)
    if nan_fixes > 0:
        _info(f"Cleaned NaN/inf values in {nan_fixes} fields across {len(ds_list)} slices")
    return cleaned_ds_list


def _fix_orientation_inconsistencies(ds_list: list[Any]) -> tuple[list[Any], int]:
    """Detect and correct DICOM slices whose orientation/position deviates from the series' dominant geometry.

    Clusters slice orientations (hierarchical clustering when scipy is available,
    else a simple pairwise-distance heuristic) to find the reference orientation,
    then reorients any outlier slice's pixel data and repositions it onto the
    reference slice-normal plane. Returns ``(corrected_ds_list, n_corrections)``.
    """
    if not ds_list or len(ds_list) < 2:
        return ds_list, 0
    try:
        orientations = []
        positions = []
        valid_indices = []
        for idx, ds in enumerate(ds_list):
            if hasattr(ds, "ImageOrientationPatient") and hasattr(ds, "ImagePositionPatient"):
                try:
                    iop = np.array([float(x) for x in ds.ImageOrientationPatient])
                    ipp = np.array([float(x) for x in ds.ImagePositionPatient])
                    if (
                        np.all(np.isfinite(iop))
                        and np.all(np.isfinite(ipp))
                        and np.linalg.norm(iop[:3]) > 1e-6
                        and np.linalg.norm(iop[3:]) > 1e-6
                    ):
                        orientations.append(iop)
                        positions.append(ipp)
                        valid_indices.append(idx)
                    else:
                        _warn(f"Skipping slice {idx} due to invalid orientation data")
                except (ValueError, TypeError) as exc:
                    _warn(f"Skipping slice {idx} due to orientation data error: {exc}")
            else:
                _warn(f"Skipping slice {idx} due to missing orientation data")
        if len(orientations) < 2:
            if len(orientations) == 1:
                _warn("Only one valid slice found, returning original dataset for basic conversion")
            else:
                _warn("No valid orientation data found, cannot proceed with orientation fixing")
            return ds_list, 0

        orientations_array = np.array(orientations)
        positions_array = np.array(positions)
        if SCIPY_AVAILABLE:
            try:
                dist_matrix = squareform(pdist(orientations_array))
                linkage_matrix = linkage(dist_matrix, method="ward")
                n_clusters = min(3, len(orientations) // 2)
                clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
                cluster_counts = np.bincount(clusters)
                dominant_cluster = np.argmax(cluster_counts)
                dominant_indices = np.where(clusters == dominant_cluster)[0]
                reference_orientation = orientations_array[dominant_indices[0]]
                reference_position = positions_array[dominant_indices[0]]
                _info(f"Clustering analysis: {len(clusters)} slices grouped into {n_clusters} clusters")
                _info(
                    f"Dominant cluster size: {cluster_counts[dominant_cluster]} out of {len(orientations)}"
                )
            except Exception as exc:
                _warn(f"Hierarchical clustering failed, using simple orientation fixing: {exc}")
                reference_orientation = orientations_array[0]
                reference_position = positions_array[0]
        else:
            _warn("scipy not available, using simple orientation fixing")
            orientation_distances = []
            for i, orient in enumerate(orientations_array):
                total_distance = 0
                for j, other_orient in enumerate(orientations_array):
                    if i != j:
                        total_distance += np.linalg.norm(orient - other_orient)
                orientation_distances.append(total_distance)
            min_distance_idx = int(np.argmin(orientation_distances))
            reference_orientation = orientations_array[min_distance_idx]
            reference_position = positions_array[min_distance_idx]
            _info(f"Distance-based analysis: using orientation {min_distance_idx} as reference")
        _info(f"Reference orientation: {reference_orientation}")
        _info(f"Reference position: {reference_position}")

        corrected_ds_list = []
        corrections_applied = 0
        for idx, ds in enumerate(ds_list):
            if idx in valid_indices:
                idx_in_valid = valid_indices.index(idx)
                current_orientation = orientations_array[idx_in_valid]
                current_position = positions_array[idx_in_valid]
                orientation_diff = np.linalg.norm(current_orientation - reference_orientation)
                position_diff = np.linalg.norm(current_position - reference_position)
                if orientation_diff > 1e-6:
                    try:
                        corrected_ds = ds.copy()
                        corrected_ds.ImageOrientationPatient = list(reference_orientation)
                        try:
                            if hasattr(corrected_ds, "pixel_array"):
                                reoriented_pixel_array = _reorient_pixel_array(
                                    corrected_ds.pixel_array,
                                    current_orientation,
                                    reference_orientation,
                                )
                                corrected_ds.PixelData = reoriented_pixel_array.tobytes()
                                if hasattr(corrected_ds, "_pixel_array"):
                                    delattr(corrected_ds, "_pixel_array")
                        except Exception as exc:
                            _warn(f"Failed to reorient pixel array for slice {idx}: {exc}")
                        if position_diff > 1e-6:
                            r = reference_orientation[0:3]
                            c = reference_orientation[3:6]
                            n = np.cross(r, c)
                            norm_n = np.linalg.norm(n)
                            if norm_n > 1e-6:
                                n = n / norm_n
                                position_diff_vector = current_position - reference_position
                                slice_distance = np.dot(position_diff_vector, n)
                                corrected_position = reference_position + slice_distance * n
                                corrected_ds.ImagePositionPatient = list(corrected_position)
                            else:
                                position_diff_magnitude = np.linalg.norm(current_position - reference_position)
                                if position_diff_magnitude > 1e-6:
                                    corrected_position = (
                                        reference_position
                                        + position_diff_magnitude * reference_orientation[0:3]
                                    )
                                    corrected_ds.ImagePositionPatient = list(corrected_position)
                                else:
                                    corrected_ds.ImagePositionPatient = list(reference_position)
                        corrected_ds_list.append(corrected_ds)
                        corrections_applied += 1
                    except Exception as exc:
                        _warn(f"Failed to correct orientation for slice {idx}: {exc}")
                        corrected_ds_list.append(ds)
                else:
                    corrected_ds_list.append(ds)
            else:
                corrected_ds_list.append(ds)
        try:
            corrected_ds_list.sort(key=lambda ds: int(getattr(ds, "InstanceNumber", 0)))
        except Exception:
            pass
        return corrected_ds_list, corrections_applied
    except Exception as exc:
        _warn(f"Orientation fixing failed: {exc}")
        return ds_list, 0


def _is_dicom_file(path: str) -> bool:
    """True when *path* can be read as a DICOM file by pydicom."""
    try:
        pydicom.dcmread(path, stop_before_pixels=False)
        return True
    except Exception:
        return False


def _collect_tags_from_nested(obj: Any, tags_to_save: set[str], collected: dict[str, Any], _seen=None) -> None:
    """Recursively walk a metadata dict/dataset tree, collecting values for keys in *tags_to_save* into *collected*.

    Cycle-safe via an object-id *_seen* set (metadata trees can self-reference through nested sequences).
    """
    if _seen is None:
        _seen = set()
    obj_id = id(obj)
    if obj_id in _seen:
        return
    _seen.add(obj_id)
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in tags_to_save:
                if not isinstance(value, np.ndarray):
                    if pydicom is not None and isinstance(value, pydicom.dataset.Dataset):
                        for k, v in value.items():
                            if not isinstance(v, (np.ndarray, pydicom.dataset.Dataset)):
                                collected[k] = v
                    else:
                        collected[key] = value
            if isinstance(value, list):
                for item in value:
                    _collect_tags_from_nested(item, tags_to_save, collected, _seen)
            elif isinstance(value, dict):
                _collect_tags_from_nested(value, tags_to_save, collected, _seen)
            elif pydicom is not None and isinstance(value, pydicom.dataset.Dataset):
                _collect_tags_from_nested(dict(value), tags_to_save, collected, _seen)
    elif pydicom is not None and isinstance(obj, pydicom.dataset.Dataset):
        _collect_tags_from_nested(dict(obj), tags_to_save, collected, _seen)


def _filter_metadata_for_nifti(metadata: dict[str, Any], additional_tags: list[str] | None = None) -> dict[str, Any]:
    """Keep only the tags in ``METADATA_TO_SAVE`` (plus *additional_tags*) from a full DICOM metadata tree."""
    tags_to_save = set(METADATA_TO_SAVE)
    if additional_tags:
        tags_to_save.update(additional_tags)
    filtered_metadata: dict[str, Any] = {}
    _collect_tags_from_nested(metadata, tags_to_save, filtered_metadata)
    # Keys nvitk derives (timing, spectral result, geometry split) and the identifiers
    # stack grouping needs. Copied from the top level only: the recursive collection
    # above would let a value from a nested sequence overwrite the series' own.
    for key in NIFTI_TOP_LEVEL_KEYS:
        if key in metadata and metadata[key] is not None:
            filtered_metadata[key] = metadata[key]
    return filtered_metadata


#: Top-level metadata always embedded in the NIfTI extension — so a file converted
#: without ``--save-metadata`` still knows its frame times, cardiac phases, spectral
#: result, and the study / frame / recon identifiers that phase and energy stacking
#: group by.
NIFTI_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "StudyInstanceUID", "FrameOfReferenceUID", "ConvolutionKernel", "FilterType",
    "ImageComments", "AcquisitionDateTime", "HeartRateSource",
    "axes", "rgb", "dynamic", "n_timepoints", "n_slices_per_timepoint", "t_res",
    "temporal_resolution", "frame_times_s", "temporal_order_source", "temporal_time_source",
    "temporal_dropped_instances", "cardiac_phases_percent", "geometry_subseries",
    "spectral_result", "spectral_result_label", "spectral_units", "spectral_energy_kev",
    "spectral_source", "spectral_file_label",
)


def _venc_scalar_from_philips_list(raw: Any) -> float | None:
    """Collapse Philips ``(2001,101A)`` 3-float list to the non-zero encoding value."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        vals = []
        for x in raw:
            try:
                vals.append(float(x))
            except (TypeError, ValueError):
                continue
        nonzero = [abs(v) for v in vals if abs(v) > 1e-9]
        if not nonzero:
            return None
        return float(max(nonzero))
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if abs(v) > 1e-9 else None


def _enrich_venc_and_heartrate_aliases(md: dict[str, Any]) -> dict[str, Any]:
    """Ensure friendly ``VENC`` / ``HeartRate`` keys when DICOM tags are present."""
    out = dict(md)
    if "VENC" not in out:
        raw = out.get("(2001,101A)") or out.get("(2001,101a)")
        scalar = _venc_scalar_from_philips_list(raw)
        if scalar is not None:
            out["VENC"] = float(scalar)
    if "HeartRate" not in out:
        hr = out.get("(0018,1088)")
        if hr is not None:
            try:
                out["HeartRate"] = float(hr)
            except (TypeError, ValueError):
                out["HeartRate"] = hr
    return out


def _get_nifti_extension(compress: bool = False) -> str:
    """``.nii.gz`` when *compress* else ``.nii``."""
    return ".nii.gz" if compress else ".nii"


def _save_metadata_json(nifti_path: str, metadata: dict[str, Any]) -> None:
    """Write a NIfTI's metadata as a JSON sidecar (``<stem>.json``), stripping non-serializable array values."""
    try:
        json_path = nifti_path.replace(".nii.gz", ".json").replace(".nii", ".json")
        keep, clean_metadata = _strip_array_values(_enrich_venc_and_heartrate_aliases(metadata))
        if not keep or not isinstance(clean_metadata, dict):
            clean_metadata = {}
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clean_metadata, f, indent=2)
    except Exception as exc:
        _warn(f"Failed to save metadata JSON for {nifti_path}: {exc}")


def _extract_image_type_tokens(ds: Any) -> list[str]:
    """Uppercased, cleaned ``ImageType`` tokens from a DICOM dataset (empty list if absent)."""
    try:
        image_type = ds.get("ImageType", None)
        if not image_type:
            return []
        return [str(token).strip().upper() for token in image_type if str(token).strip()]
    except Exception:
        return []


def _image_type_signature(tokens: list[str]) -> tuple[str, ...]:
    """Reduce ImageType tokens to a stable signature: drop generic tokens, collapse known synonyms (PHASE→PCA, etc.)."""
    generic = {"ORIGINAL", "PRIMARY", "SECONDARY", "DERIVED", "UNSPECIFIED"}
    sig = [token for token in tokens if token not in generic]
    collapsed = []
    for token in sig:
        if token in ("PHASE", "PHASE CONTRAST", "PHASE_CONTRAST"):
            new_token = "PCA"
        elif token in ("OPP", "OPPOSED", "OPPOSED PHASE", "OPP_PHASE"):
            new_token = "OPPOSED"
        elif token in ("T2*", "T2_STAR", "T2* MAP", "T2_STAR MAP"):
            new_token = "T2STAR"
        else:
            new_token = token
        collapsed.append(new_token)
    if not collapsed:
        collapsed = tokens[:] if tokens else ["GEN"]
    return tuple(collapsed)


def _image_type_label(sig: tuple[str, ...]) -> str:
    """Human-readable label (e.g. ``\"PHASE\"``, ``\"FAT\"``) for an image-type signature, used to split multi-echo series."""
    sig_set = set(sig)
    if "OPPOSED" in sig_set and len(sig) <= 2:
        return "OPPOSED"
    if "PCA" in sig_set:
        return "PHASE"
    if "M_FFE" in sig_set:
        return "M_FFE"
    if "T2STAR" in sig_set:
        return "T2STAR"
    if "FF" in sig_set and len(sig) <= 2:
        return "FAT_FRACTION"
    if "F" in sig_set and len(sig) <= 2:
        return "FAT"
    if "W" in sig_set and len(sig) <= 2:
        return "WATER"
    return "+".join(sig)


def _generate_custom_filename(
    md: dict[str, Any],
    custom_naming: str,
    fallback_name: str | None = None,
) -> str | None:
    """Build a filename from underscore-separated DICOM tag names (e.g. ``\"AccessionNumber_Modality\"``), sanitized per part."""
    if not custom_naming:
        return fallback_name
    try:
        tags = custom_naming.split("_")
        name_parts = []
        tag_mappings = {
            "AccessionNumber": ["AccessionNumber", "(0008,0050)"],
            "PatientID": ["PatientID", "(0010,0020)"],
            "PatientName": ["PatientName", "(0010,0010)"],
            "StudyDate": ["StudyDate", "(0008,0020)"],
            "SeriesNumber": ["SeriesNumber", "(0020,0011)"],
            "SeriesDescription": ["SeriesDescription", "(0008,103E)"],
            "Modality": ["Modality", "(0008,0060)"],
            "Laterality": ["Laterality", "(0020,0060)", "ImageLaterality", "(0020,0062)"],
            "InstanceNumber": ["InstanceNumber", "(0020,0013)"],
            "StudyDescription": ["StudyDescription", "(0008,1030)"],
            "BodyPartExamined": ["BodyPartExamined", "(0018,0015)"],
        }
        for tag in tags:
            value = md.get(tag, None)
            if value is None and tag in tag_mappings:
                for possible_key in tag_mappings[tag]:
                    value = md.get(possible_key, None)
                    if value is not None:
                        break
            if value is not None:
                sanitized = _sanitize_filename(str(value))
                if sanitized:
                    name_parts.append(sanitized)
        return "_".join(name_parts) if name_parts else fallback_name
    except Exception as exc:
        _warn(f"Error generating custom filename: {exc}. Using fallback name.")
        return fallback_name


def _sanitize_dicom_dataset(ds: Any) -> Any:
    """In-place: coerce malformed numeric-VR tags (e.g. stray text in an IS/US/DS field) to their proper types."""
    tags_to_clean = {
        "NumberOfFrames": "IS",
        "Rows": "US",
        "Columns": "US",
        "SpacingBetweenSlices": "DS",
        "InstanceNumber": "IS",
        "SeriesNumber": "IS",
        "PhotometricInterpretation": "CS",
        "SamplesPerPixel": "US",
    }
    for keyword, vr in tags_to_clean.items():
        if keyword not in ds:
            continue
        elem = ds[keyword]
        original_value = elem.value
        value_to_process = (
            str(original_value[0])
            if isinstance(original_value, pydicom.multival.MultiValue)
            else str(original_value)
        )
        try:
            cleaned_value = None
            if vr in ["IS", "US"]:
                match = re.search(r"^\s*(\d+)", value_to_process)
                if match:
                    cleaned_value = int(match.group(1))
            elif vr == "DS":
                match = re.search(
                    r"^\s*([+-]?\d+\.?\d*)\s*(?:[\\,])?\s*([+-]?\d+\.?\d*)?",
                    value_to_process,
                )
                if match:
                    vals = [float(v) for v in match.groups() if v is not None]
                    cleaned_value = vals if len(vals) > 1 else vals[0]
            elif vr == "CS":
                match = re.search(r"^\s*([A-Z0-9_]+)", value_to_process)
                if match:
                    cleaned_value = match.group(1)
            if cleaned_value is not None:
                elem.value = cleaned_value
            else:
                raise ValueError(f"Regex failed to find a valid pattern in '{value_to_process}'")
        except Exception as exc:
            _warn(
                f"Could not sanitize corrupted tag '{keyword}' with value '{original_value}'. Error: {exc}"
            )
            elem.value = original_value
    return ds


def _is_custom_imaging_dicom(ds: Any) -> bool:
    """True when *ds* is a convertible imaging dataset: standard imaging, tissue segmentation, readable pixels,
    or a known OP/OCT/RT/Zeiss SOP class (covers formats the standard pixel-data check would miss).
    """
    if is_valid_imaging_dicom is not None:
        try:
            if is_valid_imaging_dicom(ds):
                return True
        except Exception:
            pass
    if is_tissue_segmentation(ds):
        return True
    if is_waveform(ds):
        # Not an image, but kept: run_dicom2nifti exports it as a signal (CSV + JSON).
        return True
    try:
        _ = ds.pixel_array
        return True
    except Exception:
        pass
    read_uid = getattr(ds, "SOPClassUID", None)
    if not read_uid:
        return False
    read_uid = str(read_uid).strip("\x00").strip()
    for valid_uid in OP_SOP_CLASS_UIDS:
        if valid_uid in read_uid:
            return True
    for valid_uid in OCT_SOP_CLASS_UIDS:
        if valid_uid in read_uid:
            return True
    for valid_uid in RT_SOP_CLASS_UIDS:
        if valid_uid in read_uid:
            return True
    if is_zeiss_raw_storage(ds):
        return True
    return False


def _has_pixel_data(ds: Any) -> bool:
    """True when *ds* has standard PixelData, or a sizeable private OB/OW/OF blob that looks like image data."""
    if "PixelData" in ds or "(07fe0,0010)" in ds:
        try:
            pixel_array = getattr(ds, "pixel_array", None)
            if pixel_array is not None:
                return True
        except Exception:
            return True
    for elem in ds:
        if elem.tag.group >= 0x0400 and elem.VR in ("OB", "OW", "OF"):
            if hasattr(elem, "value") and elem.value:
                if isinstance(elem.value, (bytes, bytearray)) and len(elem.value) > 256:
                    return True
    return False


def _convert_dicom_value(
    value: Any,
    max_bytes_preview: int = 256,
    *,
    include_private_tags: bool = True,
) -> Any:
    """Convert one pydicom element value to a JSON-friendly Python type (bytes become a hash+preview summary)."""
    if pydicom is not None and isinstance(value, (pydicom.dataset.Dataset, pydicom.sequence.Sequence)):
        return _pydicom_dataset_to_dict(value, include_private_tags=include_private_tags)
    if pydicom is not None and isinstance(value, pydicom.valuerep.DSfloat):
        return float(value)
    if pydicom is not None and isinstance(value, pydicom.valuerep.IS):
        return int(value)
    if pydicom is not None and isinstance(value, pydicom.multival.MultiValue):
        return [
            _convert_dicom_value(v, max_bytes_preview, include_private_tags=include_private_tags)
            for v in value
        ]
    if pydicom is not None and isinstance(value, pydicom.uid.UID):
        return str(value).strip("\x00").strip()
    if isinstance(value, bytes):
        digest = hashlib.md5(value).hexdigest()
        preview = value[:max_bytes_preview]
        try:
            preview_text = preview.decode("utf-8", "replace")
        except Exception:
            preview_text = preview.hex()
        return {"_type": "bytes", "length": len(value), "md5": digest, "preview": preview_text}
    if isinstance(value, str):
        return value.strip("\x00").strip()
    if isinstance(value, (int, float, list, dict, bool)) or value is None:
        return value
    return str(value).strip("\x00").strip()


def _pydicom_dataset_to_dict(ds: Any, *, include_private_tags: bool = True) -> dict[str, Any]:
    """Recursively convert a pydicom Dataset to a plain dict, keyed by both keyword and ``(group,element)`` tag string."""
    metadata_dict: dict[str, Any] = {}
    for elem in ds:
        if elem.tag.is_private and not include_private_tags:
            continue
        tag_str = f"({elem.tag.group:04X},{elem.tag.element:04X})"
        key = elem.keyword or tag_str
        try:
            if elem.VR == "SQ":
                val = [
                    _pydicom_dataset_to_dict(item, include_private_tags=include_private_tags)
                    for item in elem.value
                ]
            else:
                val = _convert_dicom_value(elem.value, include_private_tags=include_private_tags)
        except Exception as exc:
            val = f"<unserializable: {exc}>"
        metadata_dict[key] = val
        metadata_dict[tag_str] = val
    return metadata_dict


def _save_image_with_metadata(
    image: Any,
    output_path: str,
    md: dict[str, Any],
    additional_tags: list[str] | None = None,
) -> None:
    """Embed filtered/cleaned metadata as a NIfTI extension and save the image."""
    try:
        md_enriched = _enrich_venc_and_heartrate_aliases(md)
        md_filtered = _filter_metadata_for_nifti(md_enriched, additional_tags)
        keep, md_clean = _strip_array_values(md_filtered)
        if not keep or not isinstance(md_clean, dict):
            md_clean = {}
    except Exception:
        md_clean = md
    try:
        json_str = json.dumps(md_clean, indent=2)
        image.header.extensions.append(nib.nifti1.Nifti1Extension(16, json_str.encode("utf-8")))
    except Exception:
        pass
    nib.save(image, output_path)


def _pixel_arrays_to_basic_volume(pixel_arrays: list[np.ndarray]) -> np.ndarray:
    """Stack raw per-slice pixel arrays into a volume with axes transposed to NIfTI (x,y,z,...) order."""
    if not pixel_arrays:
        raise ValidationError("Series has no readable pixel arrays.")
    shapes = {tuple(np.shape(a)) for a in pixel_arrays}
    if len(shapes) > 1:
        raise ValidationError(
            f"Slices of different sizes cannot share a volume: {sorted(shapes)}."
        )
    array = np.array(pixel_arrays)
    if len(pixel_arrays) > 1:
        if array.ndim == 3:
            return array.transpose(2, 1, 0)
        if array.ndim == 4:
            return array.transpose(3, 2, 1, 0)
        try:
            return array.transpose(1, 0)
        except ValueError:
            return array
    arr0 = np.array(pixel_arrays[0])
    if arr0.ndim == 2:
        return arr0.T
    if arr0.ndim == 3:
        return arr0.transpose(2, 1, 0)
    return arr0


def _save_basic_fallback(
    ds_list: list[Any],
    final_output_path: str,
    *,
    message: str,
    md: dict[str, Any] | None,
    save_metadata: bool,
) -> str | None:
    """Fallback conversion when full geometry-aware conversion fails: save raw pixel arrays as a basic NIfTI.

    Prefers a RAS-oriented reconstruction (:func:`_create_ras_oriented_nifti`);
    falls back further to an identity-affine volume if that also fails.
    """
    pixel_arrays = []
    for ds in ds_list:
        try:
            if hasattr(ds, "pixel_array"):
                pixel_arrays.append(ds.pixel_array)
        except Exception:
            continue
    if not pixel_arrays:
        return None
    if _create_ras_oriented_nifti(pixel_arrays, final_output_path):
        _info(f"{message} (RAS-oriented) to {final_output_path}")
    else:
        volume = _pixel_arrays_to_basic_volume(pixel_arrays)
        image = nib.Nifti1Image(volume, np.eye(4))
        if md:
            _ensure_output_metadata_fields(md, rescale_type="DV")
            md_clean = {
                k: v
                for k, v in md.items()
                if not (pydicom is not None and isinstance(v, pydicom.dataset.Dataset))
            }
            try:
                json_str = json.dumps(md_clean, indent=2)
                image.header.extensions.append(nib.nifti1.Nifti1Extension(16, json_str.encode("utf-8")))
            except Exception:
                pass
        nib.save(image, final_output_path)
        _info(f"{message} (basic) to {final_output_path}")
    if save_metadata and md:
        _save_metadata_json(final_output_path, md)
    return final_output_path


def _collapse_4d_metadata_to_slices(metadata_list: list[Any], ds_list: list[Any] | None = None) -> list[Any]:
    """Collapse a flat per-instance metadata list to one entry per spatial slice, for a 4-D (multi-timepoint) series."""
    if not metadata_list:
        return []
    if ds_list and len(ds_list) == len(metadata_list):
        slice_groups: dict[Any, list[tuple[int, Any]]] = {}
        for idx, ds in enumerate(ds_list):
            try:
                if hasattr(ds, "SliceLocation") and ds.SliceLocation is not None:
                    slice_key = float(ds.SliceLocation)
                elif hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None:
                    slice_key = int(ds.InstanceNumber)
                else:
                    slice_key = idx
            except Exception:
                slice_key = idx
            slice_groups.setdefault(slice_key, []).append((idx, metadata_list[idx]))
        collapsed = []
        for slice_key in sorted(slice_groups.keys()):
            slice_values = [val for _, val in slice_groups[slice_key]]
            non_none_values = [v for v in slice_values if v is not None]
            if len(non_none_values) > 1 and len(set(non_none_values)) > 1:
                _warn(
                    f"Slice {slice_key} has varying scaling factors across time points: "
                    f"{set(non_none_values)}. Using first value."
                )
            first_val = next((v for v in slice_values if v is not None), slice_values[0] if slice_values else None)
            collapsed.append(first_val)
        return collapsed
    return metadata_list


def _apply_fp_rescale_to_nifti(
    nifti_image: Any,
    rescale_slopes: list[float],
    rescale_intercepts: list[float],
    scale_slopes: list[float | None],
    ds_list: list[Any] | None = None,
) -> tuple[Any, bool]:
    """Revert Philips 'FP' (floating-point) scaling on a NIfTI volume using per-slice RescaleSlope/Intercept/ScaleSlope.

    Applies a single constant divisor when the factors are uniform across
    slices, else a per-Z-slice divisor. Returns ``(image, applied)``; on any
    failure or missing ScaleSlope, returns the input unchanged with ``applied=False``.
    """
    try:
        data = nifti_image.get_fdata().copy()
        if data.ndim < 3:
            _warn(f"Expected at least 3D data, got {data.ndim}D. Skipping FP rescale correction.")
            return nifti_image, False
        z_axis = 2
        n_slices = data.shape[z_axis]
        if len(rescale_slopes) > n_slices or len(rescale_intercepts) > n_slices or len(scale_slopes) > n_slices:
            _info(
                f"Detected 4D image: {len(rescale_slopes)} metadata entries for {n_slices} slices. "
                "Collapsing by slice..."
            )
            rescale_slopes = _collapse_4d_metadata_to_slices(rescale_slopes, ds_list)
            rescale_intercepts = _collapse_4d_metadata_to_slices(rescale_intercepts, ds_list)
            scale_slopes = _collapse_4d_metadata_to_slices(scale_slopes, ds_list)
        has_scale_slope = any(ss is not None and ss != 0.0 for ss in scale_slopes if ss is not None)
        if not has_scale_slope:
            _warn("ScaleSlope not found or all values are None/zero. Cannot apply FP rescaling. Using DV values.")
            return nifti_image, False
        if len(rescale_slopes) != n_slices or len(rescale_intercepts) != n_slices or len(scale_slopes) != n_slices:
            _warn(
                f"Mismatch: {n_slices} slices but {len(rescale_slopes)} slopes, "
                f"{len(rescale_intercepts)} intercepts, and {len(scale_slopes)} scale slopes"
            )
            if len(rescale_slopes) > n_slices:
                rescale_slopes = rescale_slopes[:n_slices]
                rescale_intercepts = rescale_intercepts[:n_slices]
                scale_slopes = scale_slopes[:n_slices]
            elif len(rescale_slopes) < n_slices:
                last_slope = rescale_slopes[-1] if rescale_slopes else 1.0
                last_intercept = rescale_intercepts[-1] if rescale_intercepts else 0.0
                last_scale_slope = scale_slopes[-1] if scale_slopes else None
                rescale_slopes.extend([last_slope] * (n_slices - len(rescale_slopes)))
                rescale_intercepts.extend([last_intercept] * (n_slices - len(rescale_intercepts)))
                scale_slopes.extend([last_scale_slope] * (n_slices - len(scale_slopes)))
        valid_scale_slopes = [ss for ss in scale_slopes if ss is not None and ss != 0.0]
        if not valid_scale_slopes:
            _warn("No valid ScaleSlope values found. Cannot apply FP rescaling.")
            return nifti_image, False
        unique_rs = len(set(rescale_slopes)) == 1
        unique_ri = len(set(rescale_intercepts)) == 1
        unique_ss = len(set([ss for ss in scale_slopes if ss is not None])) == 1
        if unique_rs and unique_ri and unique_ss:
            rs = rescale_slopes[0]
            ri = rescale_intercepts[0]
            ss = valid_scale_slopes[0]
            denominator = rs * ss
            if denominator == 0.0:
                _warn("RescaleSlope * ScaleSlope is 0, cannot apply FP rescaling")
                return nifti_image, False
            data = data / denominator
            _info(f"Applied FP rescaling using constant factors (RS={rs}, RI={ri}, SS={ss}) to {n_slices} slices")
            return nib.Nifti1Image(data, nifti_image.affine, nifti_image.header), True
        else:
            _info(f"Scaling factors vary across slices. Applying per-slice FP rescaling to {n_slices} slices")
            corrected_count = 0
            for z in range(n_slices):
                rs = rescale_slopes[z]
                ss = scale_slopes[z]
                if ss is None or ss == 0.0:
                    continue
                denominator = rs * ss
                if denominator != 0.0:
                    if data.ndim == 3:
                        data[:, :, z] = data[:, :, z] / denominator
                    elif data.ndim == 4:
                        data[:, :, z, :] = data[:, :, z, :] / denominator
                    else:
                        indices = [slice(None)] * data.ndim
                        indices[z_axis] = z
                        data[tuple(indices)] = data[tuple(indices)] / denominator
                    corrected_count += 1
                else:
                    _warn(f"Slice {z}: RescaleSlope * ScaleSlope is 0, skipping FP rescaling")
            if corrected_count > 0:
                _info(f"Applied FP rescaling on {corrected_count}/{n_slices} slices using ScaleSlope")
                return nib.Nifti1Image(data, nifti_image.affine, nifti_image.header), True
            else:
                _warn("No slices were corrected with FP rescaling")
                return nifti_image, False
    except Exception as exc:
        _warn(f"Failed to apply FP rescaling on NIfTI: {exc}")
        return nifti_image, False


def _apply_rescale_to_nifti(
    nifti_image: Any,
    rescale_slopes: list[float],
    rescale_intercepts: list[float],
    ds_list: list[Any] | None = None,
) -> tuple[Any, bool]:
    """Revert standard DICOM RescaleSlope/RescaleIntercept scaling on a NIfTI volume, per-slice if it varies.

    Mirror of :func:`_apply_fp_rescale_to_nifti` for the 'DV' (non-floating-point) case.
    """
    try:
        data = nifti_image.get_fdata().copy()
        if data.ndim < 3:
            _warn(f"Expected at least 3D data, got {data.ndim}D. Skipping rescale correction.")
            return nifti_image, False
        z_axis = 2
        n_slices = data.shape[z_axis]
        if len(rescale_slopes) > n_slices or len(rescale_intercepts) > n_slices:
            _info(
                f"Detected 4D image: {len(rescale_slopes)} metadata entries for {n_slices} slices. "
                "Collapsing by slice..."
            )
            rescale_slopes = _collapse_4d_metadata_to_slices(rescale_slopes, ds_list)
            rescale_intercepts = _collapse_4d_metadata_to_slices(rescale_intercepts, ds_list)
        if len(rescale_slopes) != n_slices or len(rescale_intercepts) != n_slices:
            _warn(
                f"Mismatch: {n_slices} slices but {len(rescale_slopes)} slopes and "
                f"{len(rescale_intercepts)} intercepts"
            )
            if len(rescale_slopes) > n_slices:
                rescale_slopes = rescale_slopes[:n_slices]
                rescale_intercepts = rescale_intercepts[:n_slices]
            elif len(rescale_slopes) < n_slices:
                last_slope = rescale_slopes[-1] if rescale_slopes else 1.0
                last_intercept = rescale_intercepts[-1] if rescale_intercepts else 0.0
                rescale_slopes.extend([last_slope] * (n_slices - len(rescale_slopes)))
                rescale_intercepts.extend([last_intercept] * (n_slices - len(rescale_intercepts)))
        unique_slope = len(set(rescale_slopes)) == 1
        unique_intercept = len(set(rescale_intercepts)) == 1
        if unique_slope and unique_intercept:
            slope = rescale_slopes[0]
            intercept = rescale_intercepts[0]
            if slope != 1.0 or intercept != 0.0:
                if slope == 0.0:
                    _warn("RescaleSlope is 0, cannot revert scaling")
                    return nifti_image, False
                data = (data - intercept) / slope
                _info(
                    f"Reverted scanner scaling using constant factors (slope={slope}, "
                    f"intercept={intercept}) on {n_slices} slices"
                )
                return nib.Nifti1Image(data, nifti_image.affine, nifti_image.header), True
            else:
                _info("Scaling factors are identity (slope=1.0, intercept=0.0). No rescaling needed.")
                return nifti_image, False
        else:
            _info(f"Scaling factors vary across slices. Applying per-slice rescaling to {n_slices} slices")
            corrected_count = 0
            for z in range(n_slices):
                slope = rescale_slopes[z]
                intercept = rescale_intercepts[z]
                if slope != 1.0 or intercept != 0.0:
                    if slope == 0.0:
                        _warn(f"Slice {z}: RescaleSlope is 0, skipping division")
                        continue
                    if data.ndim == 3:
                        data[:, :, z] = (data[:, :, z] - intercept) / slope
                    elif data.ndim == 4:
                        data[:, :, z, :] = (data[:, :, z, :] - intercept) / slope
                    else:
                        indices = [slice(None)] * data.ndim
                        indices[z_axis] = z
                        data[tuple(indices)] = (data[tuple(indices)] - intercept) / slope
                    corrected_count += 1
            if corrected_count > 0:
                _info(f"Reverted scanner scaling on {corrected_count}/{n_slices} slices to obtain raw pixel values")
                return nib.Nifti1Image(data, nifti_image.affine, nifti_image.header), True
            else:
                _warn("No slices were corrected with rescaling")
                return nifti_image, False
    except Exception as exc:
        _warn(f"Failed to revert scanner scaling on NIfTI: {exc}")
        return nifti_image, False


def _collect_rescale_metadata(ds_list: list[Any]) -> dict[str, list[float]]:
    """Per-slice RescaleIntercept/RescaleSlope values across a series (defaults 0.0/1.0 when a tag is missing/invalid)."""
    rescale_metadata = {"RescaleIntercept": [], "RescaleSlope": []}
    for ds in ds_list:
        intercept = getattr(ds, "RescaleIntercept", None)
        if intercept is not None:
            try:
                rescale_metadata["RescaleIntercept"].append(float(intercept))
            except (ValueError, TypeError):
                _warn("Missing RescaleIntercept: setting to 0")
                rescale_metadata["RescaleIntercept"].append(0.0)
        else:
            rescale_metadata["RescaleIntercept"].append(0.0)
        slope = getattr(ds, "RescaleSlope", None)
        if slope is not None:
            try:
                rescale_metadata["RescaleSlope"].append(float(slope))
            except (ValueError, TypeError):
                _warn("Missing RescaleSlope: setting to 1")
                rescale_metadata["RescaleSlope"].append(1.0)
        else:
            rescale_metadata["RescaleSlope"].append(1.0)
    return rescale_metadata


def _collect_scale_slope_metadata(ds_list: list[Any]) -> dict[str, list[float | None]]:
    """Per-slice Philips ScaleSlope values (standard tag, else the private ``(2005,100E)`` fallback)."""
    scale_slope_metadata: dict[str, list[float | None]] = {"ScaleSlope": []}
    for ds in ds_list:
        scale_slope = None
        try:
            if hasattr(ds, "ScaleSlope"):
                scale_slope = getattr(ds, "ScaleSlope", None)
            if scale_slope is None:
                tag = (0x2005, 0x100E)
                if tag in ds:
                    scale_slope = ds[tag].value
        except Exception:
            pass
        if scale_slope is not None:
            try:
                scale_slope_metadata["ScaleSlope"].append(float(scale_slope))
            except (ValueError, TypeError):
                _warn(f"Invalid ScaleSlope value: {scale_slope}, setting to None")
                scale_slope_metadata["ScaleSlope"].append(None)
        else:
            scale_slope_metadata["ScaleSlope"].append(None)
    return scale_slope_metadata


def _read_dicom_conversion(
    path: str,
    *,
    series_number: str | None = None,
    include_private_tags: bool = True,
):
    """Read all DICOM files under *path*, filter to convertible imaging datasets, and group them into per-series datasets/metadata."""
    assert pydicom is not None, "Pydicom is not installed."

    fpaths = _iter_candidate_file_paths(path)
    if not fpaths:
        _warn(f"No DICOM files could be found in {path}")
        return [], []

    fpaths = _select_candidate_file_paths(fpaths, series_number=series_number)
    if series_number is not None and not fpaths:
        return [], []

    series: dict[Any, list[Any]] = {}
    meta: dict[Any, Any] = {}
    for fp in fpaths:
        try:
            ds = pydicom.dcmread(fp, force=True)
            ds = _sanitize_dicom_dataset(ds)
            if not _is_custom_imaging_dicom(ds):
                continue
            uid_raw = getattr(ds, "SeriesInstanceUID", "no_series")
            if isinstance(uid_raw, pydicom.multival.MultiValue):
                _warn(f"File {fp} has a multi-valued UID. Using first value: {uid_raw[0]}")
            uid = _normalize_series_uid(uid_raw) or "no_series"
            series.setdefault(uid, []).append(ds)
            if uid not in meta:
                meta[uid] = ds
        except Exception:
            continue

    if not series:
        _warn(f"Although files were found, no valid DICOM image series could be loaded from {path}")
        return [], []

    ds_lists = []
    mds = []
    for uid, dsl in series.items():
        try:
            dsl.sort(key=lambda d: int(d.InstanceNumber))
        except Exception:
            pass
        ds_lists.append(dsl)
        base_md = _pydicom_dataset_to_dict(meta[uid], include_private_tags=include_private_tags)
        base_md.update(_collect_rescale_metadata(dsl))
        # Philips leaves HeartRate empty (private tag instead); spectral results get named.
        _annotate_vendor(base_md, meta[uid])
        _ensure_output_metadata_fields(base_md, rescale_type="DV")
        mds.append(base_md)
    return ds_lists, mds


def _annotate_vendor(md: dict[str, Any], ds: Any) -> dict[str, Any]:
    """Vendor/derived metadata for one series: heart-rate alias and spectral result type."""
    apply_heart_rate_alias(md, ds)
    annotate_spectral(md, ds)
    return md


def _force_ras_nifti(image: Any) -> Any:
    """Reorient a NIfTI image to canonical RAS, updating both affine and voxel data."""
    ornt_from = nib.orientations.io_orientation(image.affine)
    ornt_to = nib.orientations.axcodes2ornt(("R", "A", "S"))
    xfm = nib.orientations.ornt_transform(ornt_from, ornt_to)
    # _info(f"Forcing canonical RAS reorientation with transform:\n {xfm}")
    if xfm.size:
        data_ras = nib.orientations.apply_orientation(image.get_fdata(), xfm)
        aff_ras = image.affine @ nib.orientations.inv_ornt_aff(xfm, image.shape)
        image = nib.Nifti1Image(data_ras, aff_ras, header=image.header.copy())
        image.set_sform(aff_ras, code=1)
        image.set_qform(aff_ras, code=1)
    return image


def _mouse_reorient_nifti(image: Any) -> Any:
    """Permute AP from Z→Y and assign canonical LAS (ANTsPy mouse gallery layout)."""
    from nvitk.transform.reorient import mouse_reorient_nifti

    _info("Applying mouse reorientation (permute 0,2,1 → LAS)")
    return mouse_reorient_nifti(image)


def _apply_nifti_reorient_flags(
    image: Any,
    *,
    force_ras: bool = False,
    mouse_reorient: bool = False,
) -> Any:
    """Apply the requested reorientation flag (mouse-gallery LAS takes precedence over plain RAS) to a NIfTI image."""
    if force_ras and not mouse_reorient:
        image = _force_ras_nifti(image)
    if mouse_reorient:
        # Mouse preset rebuilds a canonical LAS affine; RAS-first is unnecessary.
        image = _mouse_reorient_nifti(image)
    return image


def _spatial_sort_ds_list(ds_list: list[Any]) -> list[Any]:
    """Order slices like ``dicom2nifti.common.sort_dicoms`` (most-varying IPP axis, ascending).

    Matching dicom2nifti's order — not merely *an* order along the normal — is
    what makes :func:`_affine_from_dicom_series` reproduce its affine exactly.
    Datasets without ``ImagePositionPatient`` keep their incoming order.
    """
    if len(ds_list) <= 1:
        return list(ds_list)
    positions = []
    for ds in ds_list:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is None or len(ipp) < 3:
            return list(ds_list)
        try:
            positions.append([float(v) for v in ipp[:3]])
        except (TypeError, ValueError):
            return list(ds_list)
    arr = np.asarray(positions, dtype=float)
    axis = int(np.argmax(arr.max(axis=0) - arr.min(axis=0)))
    order = np.argsort(arr[:, axis], kind="stable")
    return [ds_list[int(i)] for i in order]


def _affine_from_dicom_series(ds_list: list[Any]) -> np.ndarray | None:
    """
    RAS voxel-to-world affine for a ``(columns, rows, slices)`` stack of *ds_list*.

    Same formula as ``dicom2nifti`` (:func:`._dicom_dynamic.dicom2nifti_affine`),
    so a series rescued by the fallback lands in the same world frame as its
    neighbours. *ds_list* must already be in stacking order. ``None`` when the
    orientation/position attributes are missing.
    """
    if not ds_list or nib is None:
        return None
    try:
        return dicom2nifti_affine(ds_list)
    except Exception:
        return None


def _basic_fallback_image(
    ds_list: list[Any], *, label: str = "series", rescale: bool = True
) -> Any | None:
    """Build a NIfTI image from pixel arrays with a best-effort affine (identity if geometry can't be derived).

    Grey-level slices get their ``RescaleSlope``/``RescaleIntercept`` applied (*rescale*),
    so a localizer or a locator comes out in HU like the volumes beside it — integer
    slope/intercept keep an integer dtype. Only one geometry can share a volume: when
    the slices still disagree on size (no geometry split was possible upstream), the
    largest consistent group is kept and the rest are reported rather than crashing the stack.
    """
    ordered = _spatial_sort_ds_list(ds_list)
    pairs = []
    for ds in ordered:
        try:
            if hasattr(ds, "pixel_array"):
                grey = int(ds.get("SamplesPerPixel", 1) or 1) == 1
                pairs.append((ds, rescaled_slice(ds) if (rescale and grey) else ds.pixel_array))
        except Exception:
            continue
    if not pairs:
        return None
    by_shape: dict[tuple[int, ...], list[tuple[Any, Any]]] = {}
    for ds, arr in pairs:
        by_shape.setdefault(tuple(arr.shape), []).append((ds, arr))
    if len(by_shape) > 1:
        keep = max(by_shape.values(), key=len)
        _warn(
            f"{label}: slices have {len(by_shape)} different sizes "
            f"({', '.join('x'.join(map(str, k)) for k in by_shape)}); keeping the "
            f"{len(keep)} of size {'x'.join(map(str, keep[0][1].shape))}."
        )
        pairs = keep
    arrays = [arr for _, arr in pairs]
    try:
        samples = int(pairs[0][0].get("SamplesPerPixel", 1) or 1)
    except (TypeError, ValueError):
        samples = 1
    if samples in (3, 4) and arrays[0].ndim == 3 and arrays[0].shape[-1] == samples:
        # Colour (screen captures, dose reports): keep the samples as the *last*
        # axis — (columns, rows[, slices], RGB) — which is how nvitk's reader and
        # the GUI recognise a colour image. The grey-volume transpose would put
        # them first and open a screenshot as a three-slice volume.
        if len(arrays) == 1:
            volume = np.ascontiguousarray(arrays[0].transpose(1, 0, 2))
        else:
            volume = np.ascontiguousarray(np.stack(arrays).transpose(2, 1, 0, 3))
    else:
        if rescale:
            # One dtype for the stack: the smallest that holds every rescaled value.
            dtype = storage_dtype(arrays)
            arrays = [a.astype(dtype, copy=False) for a in arrays]
        volume = _pixel_arrays_to_basic_volume(arrays)
    affine = _affine_from_dicom_series([ds for ds, _ in pairs])
    if affine is None:
        # A screen capture has no patient geometry by nature; anything else
        # without it is worth a warning.
        capture = str(getattr(pairs[0][0], "SOPClassUID", "")).startswith("1.2.840.10008.5.1.4.1.1.7")
        (_info if capture else _warn)(
            f"{label}: no orientation/position tags"
            + (" (screen capture)" if capture else "")
            + "; written with an identity affine."
        )
        affine = np.eye(4)
    return nib.Nifti1Image(volume, affine)


def _apply_requested_rescale(
    image: Any,
    *,
    md: dict[str, Any] | None,
    ds_list: list[Any],
    revert_scaling: bool = False,
    rescale_type: str = "DV",
) -> tuple[Any, str]:
    """Apply the requested scaling policy: revert to raw counts (*revert_scaling*), FP rescale, or leave as DV.

    Returns ``(image, actual_rescale_type_applied)`` — the actual type can fall
    back to ``\"DV\"`` if the requested correction couldn't be applied (e.g. no
    ScaleSlope for FP).
    """
    actual_rescale_type = "DV"
    if md is None:
        return image, actual_rescale_type

    if revert_scaling:
        rescale_slopes = md.get("RescaleSlope", [])
        rescale_intercepts = md.get("RescaleIntercept", [])
        if isinstance(rescale_slopes, list) and isinstance(rescale_intercepts, list):
            if len(rescale_slopes) > 0 and len(rescale_intercepts) > 0:
                _info(
                    "Reverting scanner scaling to obtain raw pixel values "
                    "(revert_scaling takes priority over rescale_type)"
                )
                image, applied = _apply_rescale_to_nifti(image, rescale_slopes, rescale_intercepts, ds_list)
                if applied:
                    actual_rescale_type = "REVERTED"
        return image, actual_rescale_type

    if rescale_type.upper() == "FP":
        rescale_slopes = md.get("RescaleSlope", [])
        rescale_intercepts = md.get("RescaleIntercept", [])
        scale_slopes = md.get("ScaleSlope", [])
        if not scale_slopes or all(ss is None for ss in scale_slopes):
            scale_slope_md = _collect_scale_slope_metadata(ds_list)
            scale_slopes = scale_slope_md.get("ScaleSlope", [])
            md["ScaleSlope"] = scale_slopes
        if (
            isinstance(rescale_slopes, list)
            and isinstance(rescale_intercepts, list)
            and isinstance(scale_slopes, list)
            and len(rescale_slopes) > 0
            and len(rescale_intercepts) > 0
            and len(scale_slopes) > 0
        ):
            _info("Applying FP rescaling using ScaleSlope")
            image, applied = _apply_fp_rescale_to_nifti(
                image,
                rescale_slopes,
                rescale_intercepts,
                scale_slopes,
                ds_list,
            )
            if applied:
                actual_rescale_type = "FP"
    return image, actual_rescale_type


def _convert_ds_list_to_nifti_image(
    ds_list: list[Any],
    *,
    force_ras: bool = False,
    mouse_reorient: bool = False,
    md: dict[str, Any] | None = None,
    revert_scaling: bool = False,
    rescale_type: str = "DV",
    tmp_dir: Path | None = None,
) -> tuple[Any, str] | None:
    """Convert one DICOM series (list of datasets) to a NIfTI image, with a cascade of fallbacks.

    Strategy
    --------
    1. A **dynamic** series (slice positions repeated over time — bolus tracking,
       perfusion, cine) is recognised up front (:func:`detect_temporal_layout`).
       For non-MR modalities it is built directly as ``X x Y x Z x T``: the generic
       ``dicom2nifti`` path cannot stack repeated positions at all. MR keeps
       ``dicom2nifti`` first, whose vendor converters handle Philips/GE/Siemens 4D.
    2. Otherwise ``dicom2nifti`` (proper geometry from orientation/position). Its
       known failure modes (too few slices/localizer, inconsistent orientation,
       non-cubical/gantry-tilt, NaN geometry, non-imaging files) get the targeted
       retries they always had.
    3. Any failure then falls back to the temporal builder (when the layout is
       dynamic) and finally to :func:`_basic_fallback_image`.

    Applies the requested rescale policy and reorientation flags before returning
    ``(image, actual_rescale_type)``. Every message names the series.
    """
    if not ds_list:
        _warn("Skipping a series with no images left to convert.")
        return None
    label = series_label(md if md else ds_list[0])
    modality = str((md or {}).get("Modality") or ds_list[0].get("Modality", "") or "").upper()

    try:
        layout = detect_temporal_layout(ds_list)
    except Exception as exc:  # noqa: BLE001 — detection is a hint, never a blocker
        _debug(f"{label}: temporal layout detection failed: {exc}")
        layout = None

    temp_nifti_path = None
    # Set when the image did not come from dicom2nifti: the builders below apply
    # the DICOM rescale themselves, so the DV/FP/revert step must not run again.
    custom_image = False

    def _temporal_image() -> Any | None:
        """The 3D+t volume for a dynamic layout, or ``None``."""
        nonlocal custom_image
        if layout is None:
            return None
        try:
            image, extra = build_temporal_image(layout)
        except Exception as exc:  # noqa: BLE001
            _warn(f"{label}: could not build the dynamic (3D+t) volume: {exc}")
            return None
        if md is not None:
            md.update(extra)
        unit = "s" if layout.time_source != "index" else "frame"
        dropped = (
            f"; {layout.dropped} surplus instance(s) left out" if layout.dropped else ""
        )
        _info(
            f"{label}: dynamic series → {layout.n_t} time points x {layout.n_z} slice(s), "
            f"ordered by {layout.order_source}, dt={layout.t_res:.4g} {unit}{dropped}"
        )
        if revert_scaling or str(rescale_type).upper() == "FP":
            _warn(f"{label}: revert-scaling / FP rescale are not applied to dynamic series (DV kept).")
        custom_image = True
        return image

    def _fallback_image() -> Any | None:
        """Temporal builder first (when dynamic), then the basic geometry fallback."""
        nonlocal custom_image
        image = _temporal_image()
        if image is None:
            image = _basic_fallback_image(ds_list, label=label, rescale=not revert_scaling)
            custom_image = image is not None
        return image

    # One or two single-frame images — localizers, a locator, screen captures —
    # are never a dicom2nifti volume (it needs >= 3 slices and filters captures
    # out), so they go straight to the geometry-aware basic path instead of
    # through two failed attempts and their warnings. Multi-frame objects
    # (enhanced CT/MR) still go to dicom2nifti, which understands them.
    single_images = len(ds_list) <= 2 and all(
        int(ds.get("NumberOfFrames", 1) or 1) <= 1 for ds in ds_list
    )

    try:
        image = None
        if layout is not None and modality != "MR":
            image = _temporal_image()
        if image is None and layout is None and single_images:
            image = _basic_fallback_image(ds_list, label=label, rescale=not revert_scaling)
            custom_image = image is not None
        if image is None and _dicom2nifti is None:
            image = _fallback_image()
        elif image is None:
            with tempfile.NamedTemporaryFile(suffix=".nii", delete=False, dir=tmp_dir) as tf:
                temp_nifti_path = tf.name

            res = None
            try:
                res = _dicom2nifti.convert_dicom.dicom_array_to_nifti(
                    ds_list,
                    temp_nifti_path,
                    reorient_nifti=False,
                )
            except Exception as exc:
                error_str = str(exc)
                if any(err_type in error_str for err_type in ["TOO_FEW_SLICES", "LOCALIZER"]):
                    _info(f"{label}: {error_str} — converting as single-slice / localizer images.")
                    image = _fallback_image()
                elif "IMAGE_ORIENTATION_INCONSISTENT" in error_str:
                    _warn(f"{label}: IMAGE_ORIENTATION_INCONSISTENT, fixing orientation inconsistencies: {exc}")
                    try:
                        fixed_ds_list, corrections_count = _fix_orientation_inconsistencies(ds_list)
                        if corrections_count > 0:
                            _warn(
                                f"{label}: orientation fixing applied ({corrections_count} slices "
                                "corrected), retrying conversion..."
                            )
                            try:
                                res = _dicom2nifti.convert_dicom.dicom_array_to_nifti(
                                    fixed_ds_list,
                                    temp_nifti_path,
                                    reorient_nifti=False,
                                )
                            except Exception as fix_exc:
                                fix_error_str = str(fix_exc)
                                if any(
                                    token in fix_error_str.lower()
                                    for token in ["invalid value encountered", "nan", "orthogonality check failed"]
                                ) or any(token in fix_error_str for token in ["NON_CUBICAL_IMAGE", "GANTRY_TILT"]):
                                    _warn(
                                        f"{label}: {fix_error_str} after orientation fixing, "
                                        "cleaning NaN geometry and retrying..."
                                    )
                                    cleaned_fixed_ds_list = _clean_nan_values(fixed_ds_list)
                                    try:
                                        res = _dicom2nifti.convert_dicom.dicom_array_to_nifti(
                                            cleaned_fixed_ds_list,
                                            temp_nifti_path,
                                            reorient_nifti=False,
                                        )
                                    except Exception:
                                        image = _fallback_image()
                                else:
                                    image = _fallback_image()
                        else:
                            image = _fallback_image()
                    except Exception:
                        image = _fallback_image()
                elif any(
                    err_type in error_str
                    for err_type in ["NON_CUBICAL_IMAGE", "GANTRY_TILT", "ConversionValidationError"]
                ):
                    _warn(f"{label}: {error_str}, retrying with reorientation...")
                    try:
                        res = _dicom2nifti.convert_dicom.dicom_array_to_nifti(
                            ds_list,
                            temp_nifti_path,
                            reorient_nifti=True,
                        )
                    except Exception:
                        image = _fallback_image()
                else:
                    # NON_IMAGING_DICOM_FILES, a stack of mismatched slice sizes,
                    # an empty list after dicom2nifti's own filtering (IndexError)
                    # — none of these is worth a traceback: the fallbacks below
                    # either convert the series or say why they cannot.
                    _warn(f"{label}: dicom2nifti could not convert it ({type(exc).__name__}: {exc}); using fallbacks.")
                    image = _fallback_image()

            if res is not None and image is None:
                image = res["NII"]

        if image is None:
            _warn(f"{label}: no conversion succeeded; series skipped.")
            return None

        # Colour captures keep their samples last; say so, so readers (and the
        # GUI) show an RGB image rather than a three-slice volume.
        try:
            samples = int(ds_list[0].get("SamplesPerPixel", 1) or 1)
        except (TypeError, ValueError):
            samples = 1
        if md is not None and samples in (3, 4) and image.ndim in (3, 4) and image.shape[-1] == samples:
            md["axes"] = "XYC" if image.ndim == 3 else "XYZC"
            md["rgb"] = True

        actual_rescale_type = "DV"
        if not custom_image:
            image, actual_rescale_type = _apply_requested_rescale(
                image,
                md=md,
                ds_list=ds_list,
                revert_scaling=revert_scaling,
                rescale_type=rescale_type,
            )
        if force_ras or mouse_reorient:
            image = _apply_nifti_reorient_flags(
                image, force_ras=force_ras, mouse_reorient=mouse_reorient
            )

        return image, actual_rescale_type
    except Exception as exc:
        import traceback

        _debug(traceback.format_exc())
        _warn(f"{label}: conversion with fallbacks failed: {type(exc).__name__}: {exc}")
        return None
    finally:
        try:
            if temp_nifti_path and os.path.exists(temp_nifti_path):
                os.remove(temp_nifti_path)
        except Exception:
            pass


def _prepare_array_output(
    data: Any,
    md: dict[str, Any],
    *,
    axes: str | None = None,
    affine: Any | None = None,
    rescale_type: str = "DV",
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Finalize an array + metadata dict for return: fill axes/shape/spacing (from an affine, if given) and rescale type."""
    arr = np.asarray(data)
    md_out = dict(md)
    if extra_metadata:
        md_out.update(extra_metadata)
    _ensure_output_metadata_fields(md_out, rescale_type=rescale_type)
    axes_out = axes or default_nifti_axes(arr.ndim)
    md_out["axes"] = axes_out
    md_out["shape"] = tuple(arr.shape)
    if affine is not None:
        affine_arr = np.asarray(affine, dtype=float)
        md_out["affine"] = affine_arr
        spatial_dims = min(3, sum(1 for axis_name in axes_out if axis_name in {"X", "Y", "Z"}))
        spacing_values = [float(np.linalg.norm(affine_arr[:3, axis])) for axis in range(spatial_dims)]
        if len(spacing_values) > 0:
            md_out["x_res"] = spacing_values[0]
        if len(spacing_values) > 1:
            md_out["y_res"] = spacing_values[1]
        if len(spacing_values) > 2:
            md_out["z_res"] = spacing_values[2]
    elif md_out.get("affine") is not None:
        md_out["affine"] = np.asarray(md_out["affine"], dtype=float)

    frame_time = md_out.get("FrameTime")
    if frame_time is not None and "t_res" not in md_out:
        try:
            md_out["t_res"] = float(frame_time) / 1000.0
            md_out["temporal_resolution"] = md_out["t_res"]
        except Exception:
            pass
    if "x_res" in md_out or "y_res" in md_out or "z_res" in md_out:
        md_out["spacing"] = (
            md_out.get("x_res"),
            md_out.get("y_res"),
            md_out.get("z_res"),
        )
    aff_final = md_out.get("affine")
    if aff_final is not None:
        oc = orientation_codes_from_affine(np.asarray(aff_final, dtype=float))
        if oc is not None:
            md_out["orientation"] = oc
    return arr, md_out


def _load_op_series_arrays(
    ds_list: list[Any],
    md: dict[str, Any],
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Load an Ophthalmic Photography (OP/OT) series as one 2-D(+color) array per instance, identity affine."""
    outputs: list[tuple[np.ndarray, dict[str, Any]]] = []
    for idx, ds in enumerate(ds_list):
        pixel_array_yxc = ds.pixel_array
        pixel_array_xyc = pixel_array_yxc.transpose(1, 0, 2) if pixel_array_yxc.ndim == 3 else pixel_array_yxc.T
        md_one = dict(md)
        for key in ("InstanceNumber", "ImageLaterality", "Laterality"):
            value = ds.get(key)
            if value is not None:
                md_one[key] = _convert_dicom_value(value)
        axes = "XYC" if pixel_array_xyc.ndim == 3 else default_nifti_axes(pixel_array_xyc.ndim)
        outputs.append(
            _prepare_array_output(
                pixel_array_xyc,
                md_one,
                axes=axes,
                affine=np.eye(4),
                rescale_type="DV",
                extra_metadata={"instance_index": idx},
            )
        )
    return outputs


def _load_zeiss_series_arrays(
    ds_list: list[Any],
    md: dict[str, Any],
    *,
    force_ras: bool,
    aligner: Any = None,
    j2k_decode: bool = False,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Load a Zeiss private raw-OCT series, applying reorientation and returning one output array per cube.

    An angiography acquisition returns two arrays (structural ``cube_z`` and flow
    ``FlowCube_z``); other Zeiss raw series return a single array.
    """
    cubes = _extract_zeiss_cubes_or_empty(ds_list, aligner, j2k_decode)
    if cubes:
        outputs: list[tuple[np.ndarray, dict[str, Any]]] = []
        for cube in cubes:
            image = nib.Nifti1Image(cube.array, cube.affine)
            image = _apply_nifti_reorient_flags(image, force_ras=force_ras, mouse_reorient=False)
            outputs.append(
                _prepare_array_output(
                    np.asanyarray(image.dataobj),
                    md,
                    affine=image.affine,
                    rescale_type="DV",
                    extra_metadata=cube.meta,
                )
            )
        return outputs

    try:
        vol, affine, extra = extract_zeiss_raw_oct(ds_list, md, debug_mode=False)
        affine = _apply_derived_analysis_geometry(ds_list[0], vol, affine, extra)
        image = nib.Nifti1Image(vol, affine)
        image = _apply_nifti_reorient_flags(image, force_ras=force_ras, mouse_reorient=False)
        data = np.asanyarray(image.dataobj)
        return [
            _prepare_array_output(
                data,
                md,
                affine=image.affine,
                rescale_type="DV",
                extra_metadata=extra,
            )
        ]
    except Exception as exc:
        _err(f"Error extracting Zeiss OCT data: {exc}")
        return []


def _load_standard_series_arrays(
    ds_list: list[Any],
    md: dict[str, Any],
    *,
    force_ras: bool,
    revert_scaling: bool,
    rescale_type: str,
    tmp_dir: Path | None = None,
    mouse_reorient: bool = False,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Load a standard cross-sectional series via :func:`_convert_ds_list_to_nifti_image` and prepare its output array.

    A series mixing image geometries returns one array per geometry (see
    :func:`._dicom_dynamic.split_by_geometry`), each tagged ``geometry_subseries``.
    """
    groups = split_by_geometry(ds_list)
    if len(groups) > 1:
        _info(f"{series_label(md)}: splitting into {len(groups)} sub-series by image geometry")
        outputs: list[tuple[np.ndarray, dict[str, Any]]] = []
        for suffix, sub in groups:
            md_sub = _pydicom_dataset_to_dict(sub[0], include_private_tags=True)
            _annotate_vendor(md_sub, sub[0])
            md_sub.update(_collect_rescale_metadata(sub))
            _ensure_output_metadata_fields(md_sub, rescale_type="DV")
            md_sub["geometry_subseries"] = suffix
            outputs.extend(
                _load_standard_series_arrays(
                    sub, md_sub, force_ras=force_ras, revert_scaling=revert_scaling,
                    rescale_type=rescale_type, tmp_dir=tmp_dir, mouse_reorient=mouse_reorient,
                )
            )
        return outputs
    converted = _convert_ds_list_to_nifti_image(
        ds_list,
        force_ras=force_ras,
        mouse_reorient=mouse_reorient,
        md=md,
        revert_scaling=revert_scaling,
        rescale_type=rescale_type,
        tmp_dir=tmp_dir,
    )
    if converted is None:
        return []
    image, actual_rescale_type = converted
    data = np.asanyarray(image.dataobj)
    # Axes the converter recorded (XYZT for a dynamic series, XYC for a colour capture).
    axes = md.get("axes") if isinstance(md.get("axes"), str) and len(md["axes"]) == data.ndim else None
    return [_prepare_array_output(data, md, axes=axes, affine=image.affine, rescale_type=actual_rescale_type)]


def _load_one_series_arrays(
    ds_list: list[Any],
    md: dict[str, Any],
    mod: str,
    first_ds: Any,
    *,
    force_ras: bool,
    aligner: Any = None,
    revert_scaling: bool = False,
    rescale_type: str = "DV",
    tmp_dir: Path | None = None,
    j2k_decode: bool = False,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Dispatch one series to the right loader by modality/format: OP photo, Zeiss raw-OCT, or standard cross-sectional."""
    if mod in ["OP", "OT"]:
        out = _load_op_series_arrays(ds_list, md)
        if out:
            return out

    if is_tissue_segmentation(first_ds):
        pass

    if is_zeiss_raw_storage(ds_list[0]):
        out = _load_zeiss_series_arrays(ds_list, md, force_ras=force_ras, aligner=aligner, j2k_decode=j2k_decode)
        if out:
            return out

    if mod in ["CT", "PT", "OCT", "OPT"]:
        for ds in ds_list:
            if not hasattr(ds, "RepetitionTime"):
                ds.RepetitionTime = 0
            if not hasattr(ds, "EchoTime"):
                ds.EchoTime = 0

    return _load_standard_series_arrays(
        ds_list,
        md,
        force_ras=force_ras,
        revert_scaling=revert_scaling,
        rescale_type=rescale_type,
        tmp_dir=tmp_dir,
    )


def load_dicom_series(
    input_path: str | Sequence[str],
    *,
    axes: str | None = None,
    series_number: str | None = None,
    series_uid: str | None = None,
    series_index: int | None = None,
    return_all_series: bool = False,
    include_private_tags: bool = False,
    force_ras: bool = False,
    revert_scaling: bool = False,
    rescale_type: str = "DV",
    tmp_dir: Path | None = None,
):
    """Load DICOM series from *input_path* as ``(array, metadata)`` pairs — the public array-level entry point.

    *input_path* is a file, a folder, or a list of files / folders (only those are read).

    Reads and groups all series (:func:`_read_dicom_conversion`), narrows to a
    specific series by uid/number/index when requested, then loads each via
    :func:`_load_one_series_arrays` (auto-dispatched per modality/format).
    Returns a single ``(array, metadata)`` tuple, or a list of them when
    *return_all_series* is set or multiple series remain selected.
    """
    _require_deps()

    series_all, metas = _read_dicom_conversion(
        input_path,
        series_number=series_number,
        include_private_tags=True,
    )
    if not series_all:
        if series_number is not None:
            raise ValidationError(f"series_number={series_number!r} not found.")
        raise ValidationError(f"No valid DICOM image series found at: {input_path}")

    if series_uid is not None:
        filtered = [
            (ds_list, md)
            for ds_list, md in zip(series_all, metas)
            if _normalize_series_uid(md.get("SeriesInstanceUID", md.get("series_uid")))
            == _normalize_series_uid(series_uid)
        ]
        if not filtered:
            raise ValidationError(f"series_uid={series_uid!r} not found.")
        series_all = [item[0] for item in filtered]
        metas = [item[1] for item in filtered]
    elif series_index is not None:
        if not (0 <= series_index < len(series_all)):
            raise ValidationError(f"series_index={series_index} out of range for {len(series_all)} series.")
        series_all = [series_all[series_index]]
        metas = [metas[series_index]]

    outputs: list[tuple[np.ndarray, dict[str, Any]]] = []
    for ds_list, md in zip(series_all, metas):
        if not ds_list:
            _warn(f"Skipping empty series (UID: {md.get('SeriesInstanceUID', 'N/A')}).")
            continue
        if is_spectral_base_series(ds_list):
            _info(f"{series_label(md)}: Philips spectral base images (not pixels) — skipped.")
            continue
        if all(is_waveform(ds) for ds in ds_list):
            _info(f"{series_label(md)}: waveform (ECG), not an image — skipped (dcm2nii exports it as CSV).")
            continue

        readable_ds_list = []
        for ds in ds_list:
            try:
                if _has_pixel_data(ds):
                    readable_ds_list.append(ds)
            except Exception as exc:
                _debug(
                    "Skipping unreadable DICOM file: "
                    f"{getattr(ds, 'filename', 'in-memory')}. Reason: {exc}"
                )
        if not readable_ds_list:
            _warn(f"Series {md.get('SeriesInstanceUID', 'N/A')} had no readable images after filtering. Skipping.")
            continue

        ds_list = readable_ds_list
        first_ds = ds_list[0]
        try:
            buckets: dict[tuple[str, ...], list[Any]] = {}
            for ds in ds_list:
                sig = _image_type_signature(_extract_image_type_tokens(ds))
                buckets.setdefault(sig, []).append(ds)
            if len(buckets) > 1:
                _info(f"Splitting series into {len(buckets)} subseries by ImageType")
                for _, sub in buckets.items():
                    md_sub = _pydicom_dataset_to_dict(sub[0], include_private_tags=include_private_tags)
                    _annotate_vendor(md_sub, sub[0])
                    md_sub.update(_collect_rescale_metadata(sub))
                    _ensure_output_metadata_fields(md_sub, rescale_type="DV")
                    if rescale_type.upper() == "FP":
                        md_sub.update(_collect_scale_slope_metadata(sub))
                    mod = md_sub.get("Modality", "UNK")
                    outputs.extend(
                        _load_one_series_arrays(
                            sub,
                            md_sub,
                            mod,
                            sub[0],
                            force_ras=force_ras,
                            revert_scaling=revert_scaling,
                            rescale_type=rescale_type,
                            tmp_dir=tmp_dir,
                        )
                    )
                continue
        except Exception as exc:
            _warn(f"ImageType split failed or not applicable: {exc}")

        if rescale_type.upper() == "FP":
            md.update(_collect_scale_slope_metadata(ds_list))
        _ensure_output_metadata_fields(md, rescale_type="DV")
        mod = md.get("Modality", "UNK")
        outputs.extend(
            _load_one_series_arrays(
                ds_list,
                md,
                mod,
                first_ds,
                force_ras=force_ras,
                revert_scaling=revert_scaling,
                rescale_type=rescale_type,
                tmp_dir=tmp_dir,
            )
        )

    if not outputs:
        raise ValidationError(f"No readable DICOM image series found at: {input_path}")

    if axes:
        reordered_outputs: list[tuple[np.ndarray, dict[str, Any]]] = []
        for data, md in outputs:
            axes_prev = md.get("axes", default_nifti_axes(np.asarray(data).ndim))
            data_new = reorder_axes(data, axes_prev, axes)
            md_new = dict(md)
            md_new["axes"] = axes
            md_new["shape"] = tuple(np.asarray(data_new).shape)
            reordered_outputs.append((np.asarray(data_new), md_new))
        outputs = reordered_outputs

    if return_all_series or len(outputs) != 1:
        return outputs
    return outputs[0]


def _fallback_save_pixel_arrays(
    ds_list: list[Any],
    final_output_path: str,
    *,
    force_ras: bool = False,
    mouse_reorient: bool = False,
    md: dict[str, Any] | None = None,
    revert_scaling: bool = False,
    save_metadata: bool = False,
    additional_tags: list[str] | None = None,
    rescale_type: str = "DV",
    tmp_dir: Path | None = None,
) -> str | None:
    """Convert one series to NIfTI (:func:`_convert_ds_list_to_nifti_image`) and save it with embedded/sidecar metadata."""
    converted = _convert_ds_list_to_nifti_image(
        ds_list,
        force_ras=force_ras,
        mouse_reorient=mouse_reorient,
        md=md,
        revert_scaling=revert_scaling,
        rescale_type=rescale_type,
        tmp_dir=tmp_dir,
    )
    if converted is None:
        return None

    image, actual_rescale_type = converted
    if md is not None:
        _ensure_output_metadata_fields(md, rescale_type=actual_rescale_type)
        _save_image_with_metadata(image, final_output_path, md, additional_tags)
    else:
        nib.save(image, final_output_path)
    _info(f"Saved {final_output_path}")
    if save_metadata and md:
        _save_metadata_json(final_output_path, md)
    return final_output_path


def _build_output_filename(
    md: dict[str, Any],
    first_ds: Any,
    custom_naming: str | None,
    output_folder: str,
    *,
    append_label: bool = True,
    compress: bool = False,
    skip_existing: bool = False,
    explicit_output_path: str | None = None,
    name_suffix: str = "",
) -> tuple[str, bool]:
    """Resolve the output NIfTI path for a series (custom naming, else PID/desc/accession-based), de-duplicating on collision.

    *name_suffix* (a geometry sub-series label such as ``SAG`` or ``158x512``) is
    appended after the image-type label, so sub-series of one series sort together.

    Returns ``(path, should_write)`` — *should_write* is ``False`` when
    *skip_existing* is set and the target already exists.
    """
    if explicit_output_path:
        if skip_existing and os.path.exists(explicit_output_path):
            return explicit_output_path, False
        return explicit_output_path, True

    fname = None
    mod = md.get("Modality", "UNK")
    ext = _get_nifti_extension(compress)
    if custom_naming:
        base_name = _generate_custom_filename(md, custom_naming)
        if base_name:
            fname = f"{base_name}{ext}"
    if not fname:
        pid = _sanitize_filename(md.get("PatientID", "UnknownPID"))
        sn = _sanitize_filename(md.get("SeriesNumber", "0"))
        date = _sanitize_filename(md.get("StudyDate", ""))
        acc_no = _sanitize_filename(md.get("AccessionNumber", ""))
        desc = _sanitize_filename(md.get("SeriesDescription", ""))
        if desc and mod and sn:
            base = f"{desc}_{mod}_{sn}"
        elif acc_no and pid and sn:
            base = f"{acc_no}_{mod}_{sn}"
        elif pid and sn and mod:
            base = f"{pid}_{mod}_{sn}"
        else:
            base = f"{date}_{mod}_{sn}"
        fname = f"{base}{ext}"
    if append_label and first_ds is not None:
        try:
            sig = _image_type_signature(_extract_image_type_tokens(first_ds))
            label = _sanitize_filename(_image_type_label(sig)) if sig else ""
        except Exception:
            label = ""
        if label:
            base = fname.replace(".nii.gz", "").replace(".nii", "")
            if not base.endswith(f"_{label}"):
                fname = f"{base}_{label}{ext}"
    spectral = str(md.get("spectral_file_label") or "")
    if spectral and not custom_naming and md.get("spectral_source") not in ("description", None):
        # Only when the result came from DICOM attributes: a description that matched
        # ("MonoE 70keV[HU]", "Virtual Non-Contrast") already names it; "Spectral 3" does not.
        base = fname.replace(".nii.gz", "").replace(".nii", "")
        squash = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())  # noqa: E731
        if squash(spectral) not in squash(base):
            fname = f"{base}_{_sanitize_filename(spectral)}{ext}"
    if name_suffix:
        base = fname.replace(".nii.gz", "").replace(".nii", "")
        fname = f"{base}_{_sanitize_filename(name_suffix)}{ext}"

    final_output_path = os.path.join(output_folder, fname)
    if skip_existing and os.path.exists(final_output_path):
        return final_output_path, False
    cnt = 1
    while os.path.exists(final_output_path):
        base = fname.replace(".nii.gz", "").replace(".nii", "")
        fname_try = f"{base}_{cnt}{ext}"
        final_output_path = os.path.join(output_folder, fname_try)
        cnt += 1
    return final_output_path, True


def _process_op_series(
    ds_list: list[Any],
    output_folder: str,
    md: dict[str, Any],
    mod: str,
    *,
    save_metadata: bool = False,
    additional_tags: list[str] | None = None,
    compress: bool = False,
    skip_existing: bool = False,
    explicit_output_path: str | None = None,
) -> list[str] | None:
    """Save an Ophthalmic Photography (OP/OT) series as one NIfTI file per image instance."""
    try:
        if explicit_output_path and len(ds_list) != 1:
            raise ValidationError("Explicit output file path cannot be used for multi-image OP/OT exports.")
        outputs = []
        ext = _get_nifti_extension(compress)
        for idx, ds in enumerate(ds_list):
            if explicit_output_path:
                final_output_path = explicit_output_path
                if skip_existing and os.path.exists(final_output_path):
                    outputs.append(final_output_path)
                    continue
            else:
                pid = _sanitize_filename(ds.get("PatientID", "UnknownPID"))
                sn = _sanitize_filename(ds.get("SeriesNumber", "UnknownSN"))
                laterality = _sanitize_filename(ds.get("ImageLaterality", "U"))
                inst_num = _sanitize_filename(str(ds.get("InstanceNumber", idx)))
                fname = f"{laterality}_{pid}_{mod}_{sn}_{inst_num}{ext}"
                final_output_path = os.path.join(output_folder, fname)
                if skip_existing and os.path.exists(final_output_path):
                    outputs.append(final_output_path)
                    continue
                cnt = 1
                while os.path.exists(final_output_path):
                    base = fname.replace(".nii.gz", "").replace(".nii", "")
                    final_output_path = os.path.join(output_folder, f"{base}_{cnt}{ext}")
                    cnt += 1
            pixel_array_yxc = ds.pixel_array
            pixel_array_xyc = pixel_array_yxc.transpose(1, 0, 2) if pixel_array_yxc.ndim == 3 else pixel_array_yxc.T
            image = nib.Nifti1Image(pixel_array_xyc, np.eye(4))
            md_save = dict(md)
            for key in ("InstanceNumber", "ImageLaterality", "Laterality"):
                value = ds.get(key)
                if value is not None:
                    md_save[key] = _convert_dicom_value(value)
            _ensure_output_metadata_fields(md_save, rescale_type="DV")
            _save_image_with_metadata(image, final_output_path, md_save, additional_tags)
            if save_metadata:
                _save_metadata_json(final_output_path, md_save)
            _info(f"Saved OP image to {final_output_path}")
            outputs.append(final_output_path)
        return outputs
    except Exception as exc:
        _warn(f"Error handling Ophthalmic Photography ({mod}) series: {exc}")
        return None


def _process_tissue_series(
    ds_list: list[Any],
    output_folder: str,
    md: dict[str, Any],
    mod: str,
    *,
    force_ras: bool,
    save_metadata: bool = False,
    additional_tags: list[str] | None = None,
    compress: bool = False,
    skip_existing: bool = False,
    explicit_output_path: str | None = None,
) -> list[str] | None:
    """Save a tissue-segmentation series (RTStruct-derived masks) as one label NIfTI per segmentation instance."""
    try:
        if explicit_output_path and len(ds_list) != 1:
            raise ValidationError("Explicit output file path cannot be used for multi-image TISSUE exports.")
        outputs = []
        ext = _get_nifti_extension(compress)
        for idx, ds in enumerate(ds_list):
            segmentation_array, affine_matrix, shape_info = extract_tissue_segmentation_data(ds)
            if segmentation_array is None or affine_matrix is None:
                _warn(f"Failed to extract segmentation data from TISSUE DICOM {idx}")
                continue
            if explicit_output_path:
                final_output_path = explicit_output_path
                if skip_existing and os.path.exists(final_output_path):
                    outputs.append(final_output_path)
                    continue
            else:
                desc = _sanitize_filename(ds.get("SeriesDescription", "UnknownDesc"))
                sn = _sanitize_filename(ds.get("SeriesNumber", "UnknownSN"))
                inst_num = _sanitize_filename(str(ds.get("InstanceNumber", idx)))
                tissue_comment = _sanitize_filename(ds.get("ImageComments", "Tissue"))
                fname = f"{tissue_comment}_{desc}_{mod}_{sn}_{inst_num}{ext}"
                final_output_path = os.path.join(output_folder, fname)
                if skip_existing and os.path.exists(final_output_path):
                    outputs.append(final_output_path)
                    continue
                cnt = 1
                while os.path.exists(final_output_path):
                    base = fname.replace(".nii.gz", "").replace(".nii", "")
                    final_output_path = os.path.join(output_folder, f"{base}_{cnt}{ext}")
                    cnt += 1
            image = nib.Nifti1Image(segmentation_array, affine_matrix)
            if force_ras:
                ornt_from = nib.orientations.io_orientation(image.affine)
                ornt_to = nib.orientations.axcodes2ornt(("R", "A", "S"))
                xfm = nib.orientations.ornt_transform(ornt_from, ornt_to)
                _info(f"Forcing RAS reorientation with transform:\n {xfm}")
                if xfm.size:
                    data_ras = nib.orientations.apply_orientation(image.get_fdata(), xfm)
                    aff_ras = image.affine @ nib.orientations.inv_ornt_aff(xfm, image.shape)
                    image = nib.Nifti1Image(data_ras, aff_ras, header=image.header.copy())
                    image.set_sform(aff_ras, code=1)
                    image.set_qform(aff_ras, code=1)
            md_full = {
                k: v
                for k, v in md.items()
                if not (pydicom is not None and isinstance(v, (np.ndarray, pydicom.dataset.Dataset)))
            }
            md_full["segmentation_type"] = "TISSUE"
            md_full["extracted_shape"] = shape_info
            md_header = md.copy()
            md_header["segmentation_type"] = "TISSUE"
            md_header["extracted_shape"] = shape_info
            _ensure_output_metadata_fields(md_full, rescale_type="DV")
            _ensure_output_metadata_fields(md_header, rescale_type="DV")
            _save_image_with_metadata(image, final_output_path, md_header, additional_tags)
            if save_metadata:
                _save_metadata_json(final_output_path, md_full)
            _info(f"Saved TISSUE segmentation to {final_output_path}")
            outputs.append(final_output_path)
        return outputs
    except Exception as exc:
        _debug(f"Error handling TISSUE segmentation series: {exc}")
        return None


#: Ophthalmic laterality labels used when ``map_zeiss_laterality`` is requested.
_ZEISS_LATERALITY_LABELS = {"R": "OD", "L": "OS"}


def _laterality_label(value: Any, *, map_zeiss: bool) -> str | None:
    """Filename label for a scan's laterality, optionally as the ophthalmic OD/OS pair."""
    if value in (None, ""):
        return None
    text = str(value).strip().upper()
    if map_zeiss:
        text = _ZEISS_LATERALITY_LABELS.get(text, text)
    return _sanitize_filename(text)


def _extract_zeiss_cubes_or_empty(ds_list: list[Any], aligner: Any = None, j2k_decode: bool = False) -> list[Any]:
    """Extract the private Zeiss OCT cube containers, returning ``[]`` when they are absent or undecodable."""
    try:
        return extract_zeiss_oct_cubes(ds_list, debug_mode=False, aligner=aligner, j2k_decode=j2k_decode)
    except Exception as exc:
        _warn(f"Zeiss private-container extraction failed, falling back to heuristics: {exc}")
        return []


def _apply_derived_analysis_geometry(ds: Any, vol: np.ndarray, affine: Any, extra: dict[str, Any]) -> Any:
    """Give a derived analysis volume the en-face geometry of the scan it was computed from.

    Thickness maps and report analyses share their parent acquisition's en-face grid and
    record its spacing at the top level of the dataset, so they can be written with real
    millimetres instead of the unit spacing the generic extractor falls back to.
    """
    sources = zeiss_referenced_sources(ds)
    if sources:
        extra["zeiss_referenced_sources"] = sources
        extra["zeiss_referenced_scan_type"] = sources[0].get("scan_type") or None
    derived = derived_analysis_affine(ds, tuple(vol.shape))
    if derived is None:
        return affine
    extra["zeiss_geometry_source"] = "derived_analysis_grid"
    extra["zeiss_spacing_mm"] = [float(derived[0, 0]), float(derived[1, 1])]
    return derived


def _zeiss_output_metadata(md: dict[str, Any], extra: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the ``(sidecar, header)`` metadata pair for a Zeiss output, merging the extraction diagnostics."""
    md_full = {
        k: v
        for k, v in md.items()
        if not (pydicom is not None and isinstance(v, (np.ndarray, pydicom.dataset.Dataset)))
    }
    md_full.update(extra)
    md_header = md.copy()
    md_header.update(extra)
    _ensure_output_metadata_fields(md_full, rescale_type="DV")
    _ensure_output_metadata_fields(md_header, rescale_type="DV")
    return md_full, md_header


def _split_nifti_extension(path: str) -> tuple[str, str]:
    """Split *path* into ``(stem, extension)``, treating ``.nii.gz`` as a single extension."""
    for ext in (".nii.gz", ".nii"):
        if path.endswith(ext):
            return path[: -len(ext)], ext
    return path, ""


def _insert_nifti_suffix(path: str, suffix: str) -> str:
    """Append ``_<suffix>`` to *path*'s stem, keeping its NIfTI extension."""
    stem, ext = _split_nifti_extension(path)
    return f"{stem}_{suffix}{ext}"


def _unique_nifti_path(path: str) -> str:
    """Return *path*, or the first ``_<n>``-suffixed variant that does not already exist."""
    stem, ext = _split_nifti_extension(path)
    candidate = path
    count = 1
    while os.path.exists(candidate):
        candidate = f"{stem}_{count}{ext}"
        count += 1
    return candidate


def _save_zeiss_oct_images(
    ds_list: list[Any],
    base_path: str,
    *,
    skip_existing: bool,
) -> list[str]:
    """Write the 2-D images that travel with a Zeiss cube (fundus, iris, en-face) as TIFFs beside *base_path*."""
    stem, _ = _split_nifti_extension(base_path)
    outputs: list[str] = []
    try:
        images = extract_zeiss_oct_images(ds_list, debug_mode=False)
    except Exception as exc:
        _warn(f"Zeiss 2-D image extraction failed ({exc}); no fundus/iris/en-face TIFFs written.")
        return outputs
    for image in images:
        path = f"{stem}_{image.kind}.tiff"
        if skip_existing and os.path.exists(path):
            outputs.append(path)
            continue
        try:
            write_tiff(path, image.array, axes="YX", metadata=image.meta)
        except Exception as exc:
            _err(f"Could not write Zeiss {image.kind} image to {path}: {exc}")
            continue
        _info(f"Saved Zeiss OCT {image.kind} to {path}")
        outputs.append(path)
    return outputs


def _zeiss_series_output_path(
    output_folder: str,
    md: dict[str, Any],
    mod: str,
    *,
    laterality: str | None,
    compress: bool,
    source_scan: str | None = None,
) -> str:
    """Build the default output path for a Zeiss raw-OCT series from its series identity and laterality.

    Derived analyses repeat one series description per parent acquisition, so several of them
    collide on the same name within a single study; *source_scan* keeps them apart by the scan
    each was computed from.
    """
    ext = _get_nifti_extension(compress)
    pid = _sanitize_filename(md.get("PatientID", "UnknownPID"))
    sn = _sanitize_filename(md.get("SeriesNumber", "0"))
    desc = _sanitize_filename(md.get("SeriesDescription", ""))
    if desc and mod and sn:
        base = f"{desc}_{mod}_{sn}"
    elif pid and sn and mod:
        base = f"{pid}_{mod}_{sn}"
    else:
        date = _sanitize_filename(md.get("StudyDate", ""))
        base = f"{date}_{mod}_{sn}"
    if laterality:
        base = f"{base}_{laterality}"
    if source_scan:
        base = f"{base}_from_{_sanitize_filename(source_scan)}"
    return os.path.join(output_folder, f"{base}{ext}")


def _process_zeiss_series(
    ds_list: list[Any],
    output_folder: str,
    md: dict[str, Any],
    mod: str,
    *,
    force_ras: bool,
    laterality: str | None,
    save_metadata: bool = False,
    additional_tags: list[str] | None = None,
    compress: bool = False,
    skip_existing: bool = False,
    explicit_output_path: str | None = None,
    aligner: Any = None,
    j2k_decode: bool = False,
) -> list[str] | None:
    """Save a Zeiss private raw-OCT series as NIfTI, with the extraction diagnostics merged into metadata.

    An angiography acquisition writes two volumes - the structural ``cube_z`` and the
    flow ``FlowCube_z`` - and other Zeiss raw series write a single volume.
    """
    try:
        cubes = _extract_zeiss_cubes_or_empty(ds_list, aligner, j2k_decode)
        if cubes:
            base_path = explicit_output_path or _zeiss_series_output_path(
                output_folder, md, mod, laterality=laterality, compress=compress
            )
            # A single cube with an explicitly requested path keeps that exact name;
            # otherwise every volume is tagged with the cube it holds.
            name_by_kind = len(cubes) > 1 or not explicit_output_path
            outputs: list[str] = []
            for cube in cubes:
                image = nib.Nifti1Image(cube.array, cube.affine)
                image = _apply_nifti_reorient_flags(image, force_ras=force_ras, mouse_reorient=False)
                md_full, md_header = _zeiss_output_metadata(md, cube.meta)
                # An unverified reconstruction must be impossible to mistake for real data.
                experimental = cube.meta.get("zeiss_j2k_reconstruction") == "experimental"
                if name_by_kind:
                    suffix = f"{cube.kind}_EXPERIMENTAL" if experimental else cube.kind
                    path = _insert_nifti_suffix(base_path, suffix)
                elif experimental:
                    path = _insert_nifti_suffix(base_path, "EXPERIMENTAL")
                else:
                    path = base_path
                if skip_existing and os.path.exists(path):
                    outputs.append(path)
                    continue
                path = _unique_nifti_path(path)
                _save_image_with_metadata(image, path, md_header, additional_tags)
                if save_metadata:
                    _save_metadata_json(path, md_full)
                _info(f"Saved Zeiss OCT {cube.kind} to {path}")
                outputs.append(path)
            outputs.extend(_save_zeiss_oct_images(ds_list, base_path, skip_existing=skip_existing))
            return outputs

        if has_obfuscated_cube_payload(ds_list[0]):
            # The generic heuristics cannot read these and fail with a large dump; say why once.
            detail = (
                "reconstruction did not verify"
                if j2k_decode
                else "pass --j2k-decode to attempt reconstruction"
            )
            _warn(
                f"Skipping Zeiss series {md.get('SeriesDescription', '?')} "
                f"({md.get('Laterality', '?')}): obfuscated JPEG 2000 payload, {detail}."
            )
            return []

        vol, affine, extra = extract_zeiss_raw_oct(ds_list, md, debug_mode=False)
        affine = _apply_derived_analysis_geometry(ds_list[0], vol, affine, extra)
        image = nib.Nifti1Image(vol, affine)
        image = _apply_nifti_reorient_flags(image, force_ras=force_ras, mouse_reorient=False)
        md_full, md_header = _zeiss_output_metadata(md, extra)

        if explicit_output_path:
            final_output_path = explicit_output_path
            if skip_existing and os.path.exists(final_output_path):
                return [final_output_path]
        else:
            sources = zeiss_referenced_sources(ds_list[0])
            final_output_path = _zeiss_series_output_path(
                output_folder,
                md,
                mod,
                laterality=laterality,
                compress=compress,
                source_scan=sources[0].get("scan_type") if sources else None,
            )
            if skip_existing and os.path.exists(final_output_path):
                return [final_output_path]
            final_output_path = _unique_nifti_path(final_output_path)

        _save_image_with_metadata(image, final_output_path, md_header, additional_tags)
        if save_metadata:
            _save_metadata_json(final_output_path, md_full)
        _info(f"Saved Zeiss OCT to {final_output_path}")
        return [final_output_path]
    except Exception as exc:
        debug_report_path = os.path.join(output_folder, f"debug_{md.get('SeriesInstanceUID', 'no_uid')}.json")
        try:
            with open(debug_report_path, "w", encoding="utf-8") as fh:
                json.dump({"meta": md, "note": str(exc)}, fh, indent=2)
        except Exception:
            pass
        _err(f"Error extracting Zeiss OCT data: {exc}")
        return None


def _process_one_series(
    ds_list: list[Any],
    output_folder: str,
    md: dict[str, Any],
    mod: str,
    first_ds: Any,
    *,
    custom_naming: str | None,
    force_ras: bool,
    mouse_reorient: bool = False,
    revert_scaling: bool = False,
    append_label: bool = True,
    save_metadata: bool = False,
    additional_tags: list[str] | None = None,
    compress: bool = False,
    rescale_type: str = "DV",
    skip_existing: bool = False,
    tmp_dir: Path | None = None,
    explicit_output_path: str | None = None,
    aligner: Any = None,
    map_zeiss_laterality: bool = False,
    j2k_decode: bool = False,
) -> list[str]:
    """Dispatch and save one DICOM series by modality/format: OP photo, tissue segmentation, Zeiss raw-OCT,
    or the standard cross-sectional path (with output filename resolution and the requested rescale/reorient flags).
    """
    if mod in ["OP", "OT"]:
        out = _process_op_series(
            ds_list,
            output_folder,
            md,
            mod,
            save_metadata=save_metadata,
            additional_tags=additional_tags,
            compress=compress,
            skip_existing=skip_existing,
            explicit_output_path=explicit_output_path,
        )
        if out is not None:
            return out

    if is_tissue_segmentation(first_ds):
        # ----------------------------------------
        # Temporary disabled TISSUE segmentation processing
        # ----------------------------------------
        pass
        # out = _process_tissue_series(
        #     ds_list,
        #     output_folder,
        #     md,
        #     mod,
        #     force_ras=force_ras,
        #     save_metadata=save_metadata,
        #     additional_tags=additional_tags,
        #     compress=compress,
        #     skip_existing=skip_existing,
        #     explicit_output_path=explicit_output_path,
        # )
        # if out is not None:
        #     return out

    if is_zeiss_raw_storage(ds_list[0]):
        laterality = _laterality_label(md.get("Laterality"), map_zeiss=map_zeiss_laterality)
        out = _process_zeiss_series(
            ds_list,
            output_folder,
            md,
            mod,
            force_ras=force_ras,
            laterality=laterality,
            aligner=aligner,
            j2k_decode=j2k_decode,
            save_metadata=save_metadata,
            additional_tags=additional_tags,
            compress=compress,
            skip_existing=skip_existing,
            explicit_output_path=explicit_output_path,
        )
        if out is not None:
            return out

    # ---- Standard cross-sectional path ------------------------------------------
    # One volume needs one geometry. A series mixing image sizes or orientations
    # (two surview projections, reformats of different heights, captures) is cut
    # into sub-series first; each converts — and is named — on its own.
    groups = split_by_geometry(ds_list)
    if len(groups) > 1:
        if explicit_output_path:
            raise ValidationError(
                f"{series_label(md)} mixes {len(groups)} image geometries; "
                "write to an output directory instead of one file."
            )
        _info(
            f"{series_label(md)}: splitting into {len(groups)} sub-series by image geometry "
            f"({', '.join(f'{suffix}: {len(sub)}' for suffix, sub in groups)})"
        )
        outputs: list[str] = []
        for suffix, sub in groups:
            md_sub = _pydicom_dataset_to_dict(sub[0], include_private_tags=True)
            _annotate_vendor(md_sub, sub[0])
            md_sub.update(_collect_rescale_metadata(sub))
            if rescale_type.upper() == "FP":
                md_sub.update(_collect_scale_slope_metadata(sub))
            _ensure_output_metadata_fields(md_sub, rescale_type="DV")
            md_sub["geometry_subseries"] = suffix
            outputs.extend(
                _process_standard_series(
                    sub, output_folder, md_sub, mod, sub[0],
                    custom_naming=custom_naming, force_ras=force_ras,
                    mouse_reorient=mouse_reorient, revert_scaling=revert_scaling,
                    append_label=append_label, save_metadata=save_metadata,
                    additional_tags=additional_tags, compress=compress,
                    rescale_type=rescale_type, skip_existing=skip_existing,
                    tmp_dir=tmp_dir, explicit_output_path=None, name_suffix=suffix,
                )
            )
        return outputs
    return _process_standard_series(
        ds_list, output_folder, md, mod, first_ds,
        custom_naming=custom_naming, force_ras=force_ras, mouse_reorient=mouse_reorient,
        revert_scaling=revert_scaling, append_label=append_label,
        save_metadata=save_metadata, additional_tags=additional_tags, compress=compress,
        rescale_type=rescale_type, skip_existing=skip_existing, tmp_dir=tmp_dir,
        explicit_output_path=explicit_output_path,
    )


def _process_standard_series(
    ds_list: list[Any],
    output_folder: str,
    md: dict[str, Any],
    mod: str,
    first_ds: Any,
    *,
    custom_naming: str | None,
    force_ras: bool,
    mouse_reorient: bool = False,
    revert_scaling: bool = False,
    append_label: bool = True,
    save_metadata: bool = False,
    additional_tags: list[str] | None = None,
    compress: bool = False,
    rescale_type: str = "DV",
    skip_existing: bool = False,
    tmp_dir: Path | None = None,
    explicit_output_path: str | None = None,
    name_suffix: str = "",
) -> list[str]:
    """Name, convert and save one single-geometry series (see :func:`_process_one_series`)."""
    final_output_path, should_write = _build_output_filename(
        md,
        first_ds,
        custom_naming,
        output_folder,
        append_label=append_label,
        compress=compress,
        skip_existing=skip_existing,
        explicit_output_path=explicit_output_path,
        name_suffix=name_suffix,
    )
    if not should_write:
        _info(f"Skipping existing output: {final_output_path}")
        return [final_output_path]

    if mod in ["CT", "PT", "OCT", "OPT"]:
        for ds in ds_list:
            if not hasattr(ds, "RepetitionTime"):
                ds.RepetitionTime = 0
            if not hasattr(ds, "EchoTime"):
                ds.EchoTime = 0

    out = _fallback_save_pixel_arrays(
        ds_list,
        final_output_path,
        force_ras=force_ras,
        mouse_reorient=mouse_reorient,
        md=md,
        revert_scaling=revert_scaling,
        save_metadata=save_metadata,
        additional_tags=additional_tags,
        rescale_type=rescale_type,
        tmp_dir=tmp_dir,
    )
    return [out] if out else []


def _keep_spectral_base_images(
    sbi_found: list[tuple[list[Any], dict[str, Any]]],
    series_all: list[list[Any]],
    metas: list[dict[str, Any]],
    produced: dict[str, list[str]],
    output_folder: str,
    *,
    copy: bool,
    skip_existing: bool,
) -> str:
    """Write the SBI manifest (and copy the SBI DICOM with *copy*); returns the manifest path.

    Each SBI series is matched to the conventional reconstruction it was produced with:
    same frame of reference and the very same slice positions, preferring the series whose
    description is the one named in the SBI comment (``…|IMR, 78%``).
    """
    sbi_uids = {
        _normalize_series_uid(md.get("SeriesInstanceUID", md.get("series_uid"))) for _, md in sbi_found
    }
    candidates = []
    for ds_list, md in zip(series_all, metas):
        uid = _normalize_series_uid(md.get("SeriesInstanceUID", md.get("series_uid")))
        if not ds_list or uid in sbi_uids:
            continue
        candidates.append((uid, md, sbi_slice_positions(ds_list)))

    entries = []
    for ds_list, md in sbi_found:
        entry = sbi_series_summary(ds_list)
        positions = sbi_slice_positions(ds_list)
        matches = [
            (uid, cmd) for uid, cmd, cpos in candidates
            if cpos == positions
            and str(cmd.get("FrameOfReferenceUID", "")) == entry["frame_of_reference_uid"]
        ]
        ref = entry["reference_description"].strip().lower()
        # The SBI comment may shorten the name ("Miocardio, 78%" for
        # "Miocardio, 78%, iDose (6)"), so a prefix counts as a match.
        named = [m for m in matches if ref and str(m[1].get("SeriesDescription", "")).strip().lower().startswith(ref)]
        best = (named or matches or [None])[0]
        if best is not None:
            uid, cmd = best
            entry["reference_series"] = {
                "series_number": str(cmd.get("SeriesNumber", "")),
                "series_description": str(cmd.get("SeriesDescription", "")),
                "series_instance_uid": uid,
                "nifti": [os.path.basename(p) for p in produced.get(uid, [])],
                "match": "description + slice positions" if named else "slice positions",
            }
        else:
            entry["reference_series"] = None
        if copy:
            folder = _sanitize_filename(f"{entry['series_number']}_{entry['series_description']}")
            dest = os.path.join(output_folder, "sbi", folder)
            n = copy_sbi_series(ds_list, dest, skip_existing=skip_existing)
            entry["copied_to"] = os.path.relpath(dest, output_folder)
            _info(f"{series_label(md)}: copied {n} SBI file(s) to {entry['copied_to']}")
        entries.append(entry)
    path = write_sbi_manifest(entries, output_folder)
    _info(f"Spectral base images: {len(entries)} series listed in {os.path.basename(path)}")
    return path


def _export_waveform_series(
    ds_list: list[Any],
    md: dict[str, Any],
    output_folder: str,
    *,
    custom_naming: str | None,
    skip_existing: bool,
) -> list[str]:
    """Write a waveform series (ECG) as CSV + JSON, named like the image outputs."""
    path, should_write = _build_output_filename(
        md, ds_list[0], custom_naming, output_folder, append_label=False, compress=False,
        skip_existing=skip_existing,
    )
    base = path[: -len(".nii")] if path.endswith(".nii") else path
    if not should_write or (skip_existing and os.path.exists(f"{base}.csv")):
        _info(f"Skipping existing output: {base}.csv")
        return [f"{base}.csv"]
    try:
        return export_waveforms(ds_list, base)
    except Exception as exc:  # noqa: BLE001 — a malformed waveform must not stop the study
        _warn(f"{series_label(md)}: could not export the waveform: {exc}")
        return []


def run_dicom2nifti(
    input_path: str,
    output_folder: str,
    *,
    custom_naming: str | None = None,
    force_ras: bool = False,
    mouse_reorient: bool = False,
    process_rtstruct: bool = False,
    revert_scaling: bool = False,
    save_metadata: bool = False,
    additional_tags: list[str] | None = None,
    compress: bool = False,
    rescale_type: str = "DV",
    series_number: str | None = None,
    series_index: int | None = None,
    include_private_tags: bool = False,
    skip_existing: bool = False,
    tmp_dir: Path | None = None,
    explicit_output_path: str | None = None,
    align_oct: bool = False,
    map_zeiss_laterality: bool = False,
    j2k_decode: bool = False,
    stack_phases: bool = False,
    stack_energies: bool = False,
    sbi: str = "manifest",
) -> list[str]:
    """Convert every DICOM series found at *input_path* to NIfTI files under *output_folder*.

    Philips Spectral Base Images are never converted (they hold compressed spectral data,
    not pixels); *sbi* decides what is kept of them: ``"skip"`` nothing, ``"manifest"``
    (default) a ``spectral_base_images.json`` listing each SBI series, its source files and
    the reconstruction (and NIfTI) it belongs to, ``"copy"`` that plus a copy of the SBI
    DICOM files under ``sbi/``. With *stack_energies*, monoenergetic spectral results of one
    reconstruction are stacked into ``X x Y x Z x E`` volumes.

    Dynamic series (repeated slice positions) are written as 3D+t / 2D+t volumes,
    and series mixing image geometries are split into one file per geometry. With
    *stack_phases*, cardiac phase reconstructions of one acquisition are also
    stacked into one 3D+t file each (:func:`._dicom_phases.stack_cardiac_phases`).

    The top-level library entry point behind the ``dcm2nii`` CLI: reads and
    groups series (:func:`_read_dicom_conversion`), narrows by series number/
    index if requested, then processes and saves each via
    :func:`_process_one_series`. Returns the list of written file paths.

    With *align_oct*, Zeiss OCT cubes are placed in-plane into a world frame shared by every
    scan of one eye, by registering their fundus images, instead of each starting at the
    origin. Depth is untouched. Without it every cube keeps a zero origin.
    """
    _require_deps()
    sbi_mode = str(sbi or "manifest").strip().lower()
    if sbi_mode not in ("skip", "manifest", "copy"):
        raise ValidationError(f"sbi={sbi!r}: use 'skip', 'manifest' or 'copy'.")
    os.makedirs(output_folder, exist_ok=True)

    series_all, metas = _read_dicom_conversion(
        input_path,
        series_number=series_number,
        include_private_tags=include_private_tags,
    )
    if not series_all:
        if series_number is not None:
            raise ValidationError(f"series_number={series_number!r} not found.")
        raise ValidationError(f"No valid DICOM image series found at: {input_path}")

    if series_number is not None:
        _info(f"Selected series number: {series_number}")
    elif series_index is not None:
        if not (0 <= series_index < len(series_all)):
            raise ValidationError(f"series_index={series_index} out of range for {len(series_all)} series.")
        series_all = [series_all[series_index]]
        metas = [metas[series_index]]

    if explicit_output_path and len(series_all) > 1:
        raise ValidationError("Input contains multiple series. Use output directory or select one series.")

    if process_rtstruct:
        try:
            rtstruct_results = integrate_rtstruct_processing(input_path, output_folder, (series_all, metas))
            if rtstruct_results:
                successful_rtstruct = sum(1 for success in rtstruct_results.values() if success)
                total_rtstruct = len(rtstruct_results)
                _info(f"Processed {successful_rtstruct}/{total_rtstruct} RT-Struct files")
        except Exception as exc:
            _warn(f"Error processing RT-Struct files: {exc}")

    aligner = None
    if align_oct:
        aligner = ZeissOctAligner(debug_mode=False)
        for ds_list in series_all:
            try:
                aligner.add_reference_candidates(ds_list)
            except Exception as exc:
                _debug(f"Zeiss alignment reference scan failed: {exc}")

    outputs: list[str] = []
    # What each series produced (by SeriesInstanceUID), so the SBI manifest can point
    # at the NIfTI of the reconstruction a spectral base series belongs to.
    produced: dict[str, list[str]] = {}
    sbi_found: list[tuple[list[Any], dict[str, Any]]] = []
    for ds_list, md in zip(series_all, metas):
        start = len(outputs)
        uid = _normalize_series_uid(md.get("SeriesInstanceUID", md.get("series_uid"))) or str(id(ds_list))
        try:
            if not ds_list:
                _warn(f"Skipping empty series (UID: {md.get('SeriesInstanceUID', 'N/A')}).")
                continue
            if is_spectral_base_series(ds_list):
                sbi_found.append((ds_list, md))
                kept = {
                    "skip": "nothing kept",
                    "manifest": "listed in spectral_base_images.json",
                    "copy": "DICOM copied to sbi/ and listed in spectral_base_images.json",
                }[sbi_mode]
                _info(
                    f"{series_label(md)}: Philips spectral base images (compressed spectral "
                    f"data, not pixels) — not converted; {kept}. Generate monoE / iodine / "
                    "Z-effective series on a Philips spectral workstation to convert them."
                )
                continue
            if all(is_waveform(ds) for ds in ds_list):
                outputs.extend(
                    _export_waveform_series(
                        ds_list, md, output_folder, custom_naming=custom_naming,
                        skip_existing=skip_existing,
                    )
                )
                continue
            readable_ds_list = []
            for ds in ds_list:
                try:
                    if _has_pixel_data(ds):
                        readable_ds_list.append(ds)
                except Exception as exc:
                    _debug(f"Skipping unreadable DICOM file: {getattr(ds, 'filename', 'in-memory')}. Reason: {exc}")
            if not readable_ds_list:
                _warn(f"Series {md.get('SeriesInstanceUID', 'N/A')} had no readable images after filtering. Skipping.")
                continue

            ds_list = readable_ds_list
            first_ds = ds_list[0]
            try:
                buckets: dict[tuple[str, ...], list[Any]] = {}
                for ds in ds_list:
                    sig = _image_type_signature(_extract_image_type_tokens(ds))
                    buckets.setdefault(sig, []).append(ds)
                if len(buckets) > 1:
                    if explicit_output_path:
                        raise ValidationError("Explicit output file cannot be used when a series splits into multiple subseries.")
                    _info(f"Splitting series into {len(buckets)} subseries by ImageType")
                    for sig, sub in buckets.items():
                        md_sub = _pydicom_dataset_to_dict(sub[0], include_private_tags=True)
                        _annotate_vendor(md_sub, sub[0])
                        md_sub.update(_collect_rescale_metadata(sub))
                        _ensure_output_metadata_fields(md_sub, rescale_type="DV")
                        if rescale_type.upper() == "FP":
                            md_sub.update(_collect_scale_slope_metadata(sub))
                        mod = md_sub.get("Modality", "UNK")
                        outputs.extend(
                            _process_one_series(
                                sub,
                                output_folder,
                                md_sub,
                                mod,
                                sub[0],
                                custom_naming=custom_naming,
                                force_ras=force_ras,
                                mouse_reorient=mouse_reorient,
                                revert_scaling=revert_scaling,
                                append_label=True,
                                save_metadata=save_metadata,
                                additional_tags=additional_tags,
                                compress=compress,
                                rescale_type=rescale_type,
                                skip_existing=skip_existing,
                                tmp_dir=tmp_dir,
                                explicit_output_path=None,
                                aligner=aligner,
                                map_zeiss_laterality=map_zeiss_laterality,
                                j2k_decode=j2k_decode,
                            )
                        )
                    continue
            except Exception as exc:
                _warn(f"ImageType split failed or not applicable: {exc}")

            if rescale_type.upper() == "FP":
                md.update(_collect_scale_slope_metadata(ds_list))
            _ensure_output_metadata_fields(md, rescale_type="DV")
            mod = md.get("Modality", "UNK")
            outputs.extend(
                _process_one_series(
                    ds_list,
                    output_folder,
                    md,
                    mod,
                    first_ds,
                    custom_naming=custom_naming,
                    force_ras=force_ras,
                    mouse_reorient=mouse_reorient,
                    revert_scaling=revert_scaling,
                    append_label=False,
                    save_metadata=save_metadata,
                    additional_tags=additional_tags,
                    compress=compress,
                    rescale_type=rescale_type,
                    skip_existing=skip_existing,
                    tmp_dir=tmp_dir,
                    explicit_output_path=explicit_output_path,
                    aligner=aligner,
                    map_zeiss_laterality=map_zeiss_laterality,
                    j2k_decode=j2k_decode,
                )
            )
        except Exception as exc:
            _err(f"Error during conversion for {series_label(md)}: {exc}")
            continue
        finally:
            produced[uid] = outputs[start:]

    if sbi_found and sbi_mode != "skip":
        try:
            outputs.append(
                _keep_spectral_base_images(
                    sbi_found, series_all, metas, produced, output_folder,
                    copy=sbi_mode == "copy", skip_existing=skip_existing,
                )
            )
        except Exception as exc:  # noqa: BLE001 — the image outputs are already written
            _warn(f"Could not write the spectral base image manifest: {exc}")

    if stack_phases and outputs:
        try:
            from ._dicom_phases import stack_cardiac_phases

            stacks = stack_cardiac_phases(
                outputs,
                output_folder,
                compress=compress,
                save_metadata=save_metadata,
                skip_existing=skip_existing,
            )
            if not stacks:
                _info("No cardiac phase siblings to stack (need >= 2 phases of one reconstruction).")
            outputs.extend(stacks)
        except Exception as exc:  # noqa: BLE001 — the per-series outputs are already written
            _warn(f"Cardiac phase stacking failed: {exc}")

    if stack_energies and outputs:
        try:
            from ._dicom_spectral import stack_monoenergetic

            stacks = stack_monoenergetic(
                [p for p in outputs if p.endswith((".nii", ".nii.gz"))],
                output_folder,
                compress=compress,
                save_metadata=save_metadata,
                skip_existing=skip_existing,
            )
            if not stacks:
                _info("No monoenergetic levels to stack (need >= 2 keV levels of one reconstruction).")
            outputs.extend(stacks)
        except Exception as exc:  # noqa: BLE001
            _warn(f"Monoenergetic stacking failed: {exc}")
    return outputs
