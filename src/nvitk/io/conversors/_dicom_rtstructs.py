"""DICOM RT-STRUCT contours → masks / overlays (optional scikit-image)."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
from nvitk.core.logger import Logger

try:
    import pydicom
except Exception:
    pydicom = None

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    from skimage.draw import polygon2mask

    SKIMAGE_AVAILABLE = True
except Exception:
    polygon2mask = None
    SKIMAGE_AVAILABLE = False

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

__all__ = [
    "extract_rtstruct_rois",
    "integrate_rtstruct_processing",
    "is_rtstruct_file",
    "process_rtstruct_to_masks",
]

log = Logger()


def _sanitize_filename(value: str) -> str:
    """Sanitize an RTStruct ROI name into a filesystem-safe filename fragment (falls back to ``\"roi\"``)."""
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe.strip("_") or "roi"


def _warn(message: str) -> None:
    """Log a warning through the module logger."""
    log.warning(message)


def _err(message: str) -> None:
    """Log an error through the module logger."""
    log.error(message)


def _info(message: str) -> None:
    """Log an info message through the module logger."""
    log.info(message)


def _debug(message: str) -> None:
    """No-op placeholder (debug logging is disabled for this module)."""
    return


if not SKIMAGE_AVAILABLE:
    _warn(
        "scikit-image not available - RT-Struct contour processing will use rectangular "
        "approximation. Install with: pip install scikit-image"
    )


def is_rtstruct_file(ds: Any) -> bool:
    """Check if a DICOM dataset is an RT-Struct file."""
    if not pydicom:
        return False

    try:
        sop_class_uid = getattr(ds, "SOPClassUID", None)
        if not sop_class_uid:
            return False

        sop_class_uid = str(sop_class_uid).strip("\x00").strip()
        for rt_uid in RT_SOP_CLASS_UIDS:
            if rt_uid in sop_class_uid:
                return True

        return "1.2.840.10008.5.1.4.1.1.481.3" in sop_class_uid
    except Exception as exc:
        _debug(f"Error checking RT-Struct file: {exc}")
        return False


def extract_rtstruct_rois(rtstruct_ds: Any) -> list[dict[str, Any]]:
    """Extract ROI information from RT-Struct dataset."""
    rois: list[dict[str, Any]] = []

    try:
        if hasattr(rtstruct_ds, "StructureSetROISequence"):
            for roi_item in rtstruct_ds.StructureSetROISequence:
                roi_info = {
                    "roi_number": getattr(roi_item, "ROINumber", "Unknown"),
                    "roi_name": getattr(roi_item, "ROIName", "Unknown"),
                    "roi_description": getattr(rtstruct_ds, "SeriesDescription", "Unknown"),
                    "referenced_frame_of_reference_uid": getattr(
                        roi_item,
                        "ReferencedFrameOfReferenceUID",
                        "Unknown",
                    ),
                    "roi_generation_algorithm": getattr(
                        roi_item,
                        "ROIGenerationAlgorithm",
                        "Unknown",
                    ),
                }
                rois.append(roi_info)
        return rois
    except Exception as exc:
        _err(f"Error extracting ROI information: {exc}")
        return []


def process_rtstruct_to_masks(
    rtstruct_path: str,
    reference_dicom_path: str | None,
    output_path: str,
    method: str = "manual",
    ds_res: tuple | None = None,
) -> bool:
    """Process RT-Struct file to create NIfTI masks."""
    try:
        _debug(f"Using {method} method for RT-Struct processing")
        if ds_res and ds_res[0]:
            return _process_manually(rtstruct_path, None, output_path, ds_res=ds_res)
        return _process_manually(rtstruct_path, reference_dicom_path, output_path)
    except Exception as exc:
        _err(f"Error processing RT-Struct file {rtstruct_path}: {exc}")
        return False


def _process_manually(
    rtstruct_path: str,
    reference_dicom_path: str | None,
    output_path: str,
    ds_res: tuple | None = None,
) -> bool:
    """Process RT-Struct using manual contour processing."""
    try:
        rtstruct_ds = pydicom.dcmread(rtstruct_path, force=True)

        rois = extract_rtstruct_rois(rtstruct_ds)
        if not rois:
            _warn(f"No ROIs found in RT-Struct file: {rtstruct_path}")
            return False

        contour_data = _extract_contour_data(rtstruct_ds)
        if not contour_data:
            _warn(f"No contour data found in RT-Struct file: {rtstruct_path}")
            return False

        if ds_res and ds_res[0]:
            series_all, metas = ds_res
            masks = _create_masks_from_contours(
                contour_data,
                None,
                rtstruct_ds,
                series_all=series_all,
                metas=metas,
            )
        else:
            masks = _create_masks_from_contours(contour_data, reference_dicom_path, rtstruct_ds)

        if not masks:
            _warn(f"No masks could be created from RT-Struct file: {rtstruct_path}")
            return False

        os.makedirs(output_path, exist_ok=True)
        combined_mask = np.zeros_like(list(masks.values())[0], dtype=np.uint16)
        roi_info: dict[Any, dict[str, Any]] = {}
        saved_files: list[str] = []

        for roi_number, mask in masks.items():
            roi_name = "Unknown"
            roi_description = ""
            for roi in rois:
                if roi["roi_number"] == roi_number:
                    roi_name = roi["roi_name"]
                    roi_description = roi.get("roi_description", roi_name)
                    break

            roi_info[roi_number] = {
                "name": roi_name,
                "description": roi_description,
                "label_id": roi_number,
            }

            combined_mask[mask > 0] = roi_number

            safe_description = _sanitize_filename(roi_description)
            if not safe_description or safe_description == "Unknown":
                safe_description = f"ROI_{roi_number}"

            individual_filename = f"{safe_description}_{roi_name}_{roi_number}.nii"
            individual_filepath = os.path.join(output_path, individual_filename)

            if os.path.exists(individual_filepath):
                idx = 1
                while os.path.exists(individual_filepath):
                    individual_filename = f"{safe_description}_{idx}.nii"
                    individual_filepath = os.path.join(output_path, individual_filename)
                    idx += 1

            mask_xyz = np.transpose(mask, (2, 1, 0))
            individual_image = nib.Nifti1Image(mask_xyz, np.eye(4))
            individual_header = individual_image.header
            individual_header["descrip"] = f"{roi_description}_{roi_number}"
            individual_header["cal_max"] = float(np.max(mask_xyz))
            individual_header["cal_min"] = float(np.min(mask_xyz))
            nib.save(individual_image, individual_filepath)
            saved_files.append(individual_filepath)

            _debug(f"Saved individual ROI {roi_number} ({roi_description}) to {individual_filepath}")
            _debug(f"Added ROI {roi_number} ({roi_description}) to combined mask with label ID {roi_number}")

        rtstruct_basename = os.path.splitext(os.path.basename(rtstruct_path))[0]
        combined_filename = f"{rtstruct_basename}_rtstruct_combined_masks.nii"
        combined_filepath = os.path.join(output_path, combined_filename)

        if os.path.exists(combined_filepath):
            idx = 1
            while os.path.exists(combined_filepath):
                combined_filename = f"{rtstruct_basename}_rtstruct_combined_masks_{idx}.nii"
                combined_filepath = os.path.join(output_path, combined_filename)
                idx += 1

        combined_mask_xyz = np.transpose(combined_mask, (2, 1, 0))
        combined_image = nib.Nifti1Image(combined_mask_xyz, np.eye(4))
        combined_header = combined_image.header
        combined_header["descrip"] = f"RT-Struct combined segmentation masks from {rtstruct_basename}"
        combined_header["cal_max"] = float(np.max(combined_mask_xyz))
        combined_header["cal_min"] = float(np.min(combined_mask_xyz))

        metadata = {
            "source_file": rtstruct_path,
            "rtstruct_basename": rtstruct_basename,
            "total_rois": len(rois),
            "roi_info": roi_info,
            "mask_dimensions": combined_mask_xyz.shape,
            "label_ids": list(roi_info.keys()),
            "individual_files": saved_files,
            "axis_ordering": "XYZ",
        }

        try:
            json_str = json.dumps(metadata, indent=2)
            combined_image.header.extensions.append(
                nib.nifti1.Nifti1Extension(16, json_str.encode("utf-8"))
            )
        except Exception as exc:
            _debug(f"Could not add metadata to NIfTI header: {exc}")

        nib.save(combined_image, combined_filepath)

        roi_info_file = os.path.join(output_path, f"{rtstruct_basename}_roi_info.json")
        if os.path.exists(roi_info_file):
            idx = 1
            while os.path.exists(roi_info_file):
                roi_info_file = os.path.join(output_path, f"{rtstruct_basename}_roi_info_{idx}.json")
                idx += 1
        try:
            with open(roi_info_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            _debug(f"Saved ROI information to {roi_info_file}")
        except Exception as exc:
            _debug(f"Could not save ROI information file: {exc}")

        _info(f"Successfully processed RT-Struct using manual method: {os.path.basename(rtstruct_path)}")
        _info(f"Saved {len(rois)} individual ROI masks and 1 combined mask")
        _info(f"Combined mask saved to: {combined_filepath}")
        return True
    except Exception as exc:
        _err(f"Manual processing failed for RT-Struct {rtstruct_path}: {exc}")
        return False


def _extract_contour_data(rtstruct_ds: Any) -> dict[Any, dict[str, Any]]:
    """Extract contour data from RT-Struct dataset."""
    contour_data: dict[Any, dict[str, Any]] = {}

    try:
        if hasattr(rtstruct_ds, "ROIContourSequence"):
            for roi_contour in rtstruct_ds.ROIContourSequence:
                current_roi_number = getattr(roi_contour, "ReferencedROINumber", None)
                if current_roi_number is None:
                    continue

                roi_data = {
                    "roi_number": current_roi_number,
                    "display_color": getattr(roi_contour, "ROIDisplayColor", [255, 255, 255]),
                    "contours": [],
                }

                if hasattr(roi_contour, "ContourSequence"):
                    for contour in roi_contour.ContourSequence:
                        contour_info = {
                            "geometric_type": getattr(contour, "ContourGeometricType", "UNKNOWN"),
                            "number_of_points": getattr(contour, "NumberOfContourPoints", 0),
                            "contour_number": getattr(contour, "ContourNumber", 0),
                            "contour_data": getattr(contour, "ContourData", []),
                            "referenced_images": [],
                        }

                        if hasattr(contour, "ContourImageSequence"):
                            for img_ref in contour.ContourImageSequence:
                                contour_info["referenced_images"].append(
                                    {
                                        "sop_class_uid": getattr(img_ref, "ReferencedSOPClassUID", ""),
                                        "sop_instance_uid": getattr(img_ref, "ReferencedSOPInstanceUID", ""),
                                    }
                                )

                        roi_data["contours"].append(contour_info)

                contour_data[current_roi_number] = roi_data

        return contour_data
    except Exception as exc:
        _err(f"Error extracting contour data: {exc}")
        return {}


def _create_masks_from_contours(
    contour_data: dict[Any, dict[str, Any]],
    reference_dicom_path: str | None,
    rtstruct_ds: Any | None = None,
    series_all: list | None = None,
    metas: list | None = None,
) -> dict[Any, np.ndarray]:
    """Create binary masks from contour data."""
    masks: dict[Any, np.ndarray] = {}

    try:
        if series_all and metas:
            reference_series_uid = None
            if rtstruct_ds and hasattr(rtstruct_ds, "ReferencedFrameOfReferenceSequence"):
                try:
                    ref_frame = rtstruct_ds.ReferencedFrameOfReferenceSequence[0]
                    if hasattr(ref_frame, "RTReferencedStudySequence"):
                        rt_ref_study = ref_frame.RTReferencedStudySequence[0]
                        if hasattr(rt_ref_study, "RTReferencedSeriesSequence"):
                            rt_ref_series = rt_ref_study.RTReferencedSeriesSequence[0]
                            reference_series_uid = getattr(rt_ref_series, "SeriesInstanceUID", None)
                            _debug(f"Found reference series UID: {reference_series_uid}")
                except Exception as exc:
                    _debug(f"Could not extract reference series UID from RT-Struct: {exc}")

            reference_series = None
            reference_meta = None
            for series, meta in zip(series_all, metas):
                series_uid = meta.get("SeriesInstanceUID", None)
                if reference_series_uid and series_uid == reference_series_uid:
                    reference_series = series
                    reference_meta = meta
                    _debug(f"Found matching reference series: {series_uid}")
                    break
                if not reference_series and not is_rtstruct_file(series[0]):
                    reference_series = series
                    reference_meta = meta
                    _debug(f"Using first available imaging series: {series_uid}")

            if not reference_series:
                _warn("No valid reference imaging series found in loaded data")
                return {}

            ref_ds = reference_series[0]
            rows = getattr(ref_ds, "Rows", 512)
            cols = getattr(ref_ds, "Columns", 512)
            depth = len(reference_series)
        else:
            reference_series_uid = None
            if rtstruct_ds and hasattr(rtstruct_ds, "ReferencedFrameOfReferenceSequence"):
                try:
                    ref_frame = rtstruct_ds.ReferencedFrameOfReferenceSequence[0]
                    if hasattr(ref_frame, "RTReferencedStudySequence"):
                        rt_ref_study = ref_frame.RTReferencedStudySequence[0]
                        if hasattr(rt_ref_study, "RTReferencedSeriesSequence"):
                            rt_ref_series = rt_ref_study.RTReferencedSeriesSequence[0]
                            reference_series_uid = getattr(rt_ref_series, "SeriesInstanceUID", None)
                            _debug(f"Found reference series UID: {reference_series_uid}")
                except Exception as exc:
                    _debug(f"Could not extract reference series UID from RT-Struct: {exc}")

            reference_ds_list = []
            if reference_dicom_path and os.path.isdir(reference_dicom_path):
                for root, _, files in os.walk(reference_dicom_path):
                    for file in files:
                        filepath = os.path.join(root, file)
                        if os.path.isfile(filepath):
                            try:
                                ds = pydicom.dcmread(filepath, force=True)
                                if not is_rtstruct_file(ds) and hasattr(ds, "pixel_array"):
                                    if reference_series_uid:
                                        series_uid = getattr(ds, "SeriesInstanceUID", None)
                                        if series_uid == reference_series_uid:
                                            reference_ds_list.append(ds)
                                            _debug(f"Added reference image: {file}")
                                    else:
                                        reference_ds_list.append(ds)
                            except Exception:
                                continue
            elif reference_dicom_path:
                try:
                    ds = pydicom.dcmread(reference_dicom_path, force=True)
                    if not is_rtstruct_file(ds) and hasattr(ds, "pixel_array"):
                        reference_ds_list.append(ds)
                except Exception:
                    pass

            if not reference_ds_list and reference_dicom_path:
                parent_dir = os.path.dirname(reference_dicom_path)
                if parent_dir != reference_dicom_path and os.path.isdir(parent_dir):
                    _debug(
                        f"No reference images found in {reference_dicom_path}, trying parent directory: {parent_dir}"
                    )
                    for root, _, files in os.walk(parent_dir):
                        for file in files:
                            filepath = os.path.join(root, file)
                            if os.path.isfile(filepath):
                                try:
                                    ds = pydicom.dcmread(filepath, force=True)
                                    if not is_rtstruct_file(ds) and hasattr(ds, "pixel_array"):
                                        if reference_series_uid:
                                            series_uid = getattr(ds, "SeriesInstanceUID", None)
                                            if series_uid == reference_series_uid:
                                                reference_ds_list.append(ds)
                                                _debug(f"Added reference image from parent directory: {file}")
                                        else:
                                            reference_ds_list.append(ds)
                                except Exception:
                                    continue

            if not reference_ds_list:
                _warn("No valid reference DICOM files found for RT-Struct processing")
                return {}

            reference_ds_list.sort(key=lambda ds: getattr(ds, "InstanceNumber", 0))
            ref_ds = reference_ds_list[0]
            rows = getattr(ref_ds, "Rows", 512)
            cols = getattr(ref_ds, "Columns", 512)
            depth = len(reference_ds_list)

        _debug(f"Creating masks with dimensions: {depth}x{rows}x{cols}")
        if SKIMAGE_AVAILABLE:
            _debug("Using scikit-image polygon2mask for accurate contour processing")
        else:
            _debug("Using rectangular approximation for contour processing (scikit-image not available)")

        for roi_number, roi_data in contour_data.items():
            mask = np.zeros((depth, rows, cols), dtype=np.uint8)
            _debug(f"Processing ROI {roi_number} with {len(roi_data['contours'])} contours")

            for contour in roi_data["contours"]:
                if contour["geometric_type"] != "CLOSED_PLANAR":
                    _debug(f"Skipping non-planar contour: {contour['geometric_type']}")
                    continue

                contour_points = np.array(contour["contour_data"]).reshape(-1, 3)
                if len(contour_points) == 0:
                    continue

                x_coords = contour_points[:, 0]
                y_coords = contour_points[:, 1]
                z_coords = contour_points[:, 2]

                slice_idx = 0
                if series_all and metas:
                    if len(reference_series) > 1:
                        slice_positions = []
                        slice_normals = []

                        for ds in reference_series:
                            if hasattr(ds, "ImagePositionPatient") and hasattr(ds, "ImageOrientationPatient"):
                                ipp = np.array([float(x) for x in ds.ImagePositionPatient])
                                iop = np.array([float(x) for x in ds.ImageOrientationPatient])
                                r = iop[0:3]
                                c = iop[3:6]
                                n = np.cross(r, c)
                                n = n / np.linalg.norm(n)
                                slice_positions.append(ipp)
                                slice_normals.append(n)

                        if slice_positions and slice_normals:
                            slice_normal = slice_normals[0]
                            z_center = np.mean(z_coords)
                            contour_position = np.array([0, 0, z_center])

                            min_distance = float("inf")
                            best_slice_idx = 0
                            for idx, slice_pos in enumerate(slice_positions):
                                vec_to_contour = contour_position - slice_pos
                                distance_along_normal = np.dot(vec_to_contour, slice_normal)
                                if abs(distance_along_normal) < min_distance:
                                    min_distance = abs(distance_along_normal)
                                    best_slice_idx = idx

                            slice_idx = best_slice_idx
                            _debug(
                                f"Found best slice {slice_idx} for contour at z={z_center:.2f} "
                                f"(distance: {min_distance:.2f})"
                            )
                        else:
                            z_center = np.mean(z_coords)
                            slice_positions_z = [pos[2] for pos in slice_positions if len(pos) > 2]
                            if slice_positions_z:
                                slice_idx = int(np.argmin(np.abs(np.array(slice_positions_z) - z_center)))
                else:
                    z_min, z_max = np.min(z_coords), np.max(z_coords)
                    if z_max > z_min:
                        z_normalized = (z_coords - z_min) / (z_max - z_min)
                        slice_idx = int(np.mean(z_normalized) * (depth - 1))
                    else:
                        slice_idx = 0

                slice_idx = max(0, min(slice_idx, depth - 1))

                if series_all and metas:
                    ref_ds = reference_series[slice_idx] if slice_idx < len(reference_series) else reference_series[0]
                else:
                    ref_ds = reference_ds_list[slice_idx] if slice_idx < len(reference_ds_list) else reference_ds_list[0]

                if hasattr(ref_ds, "ImageOrientationPatient") and hasattr(ref_ds, "ImagePositionPatient"):
                    iop = np.array([float(x) for x in ref_ds.ImageOrientationPatient])
                    ipp = np.array([float(x) for x in ref_ds.ImagePositionPatient])

                    if hasattr(ref_ds, "PixelSpacing"):
                        ps = np.array([float(x) for x in ref_ds.PixelSpacing])
                        row_spacing, col_spacing = ps[0], ps[1]
                    else:
                        row_spacing, col_spacing = 1.0, 1.0

                    r = iop[0:3]
                    c = iop[3:6]
                    world_coords = np.column_stack([x_coords, y_coords, z_coords])
                    image_coords = []

                    for world_coord in world_coords:
                        vec = world_coord - ipp
                        row_idx = np.dot(vec, r) / row_spacing
                        col_idx = np.dot(vec, c) / col_spacing
                        image_coords.append([col_idx, row_idx])

                    image_coords = np.array(image_coords)
                    x_scaled = np.clip(image_coords[:, 0], 0, cols - 1)
                    y_scaled = np.clip(image_coords[:, 1], 0, rows - 1)
                else:
                    _warn("Missing DICOM transformation parameters, using simplified coordinate mapping")
                    x_scaled = np.clip(x_coords, 0, cols - 1).astype(int)
                    y_scaled = np.clip(y_coords, 0, rows - 1).astype(int)

                if SKIMAGE_AVAILABLE:
                    try:
                        polygon_vertices = np.column_stack([x_scaled, y_scaled])
                        valid_mask = np.isfinite(polygon_vertices).all(axis=1)
                        if np.sum(valid_mask) < 3:
                            _debug(f"Not enough valid vertices for polygon creation on slice {slice_idx}")
                            continue

                        polygon_vertices = polygon_vertices[valid_mask]
                        slice_mask = polygon2mask((rows, cols), polygon_vertices)
                        mask[slice_idx] = np.logical_or(mask[slice_idx], slice_mask).astype(np.uint8)
                        _debug(f"Added polygon contour to slice {slice_idx} using skimage.polygon2mask")
                    except Exception as exc:
                        _debug(
                            "Failed to create polygon mask with skimage: "
                            f"{exc}, falling back to rectangular mask"
                        )
                        x_min, x_max = int(np.min(x_scaled)), int(np.max(x_scaled))
                        y_min, y_max = int(np.min(y_scaled)), int(np.max(y_scaled))
                        if x_max > x_min and y_max > y_min:
                            mask[slice_idx, y_min : y_max + 1, x_min : x_max + 1] = 1
                            _debug(
                                f"Added rectangular contour to slice {slice_idx}, "
                                f"region ({x_min}:{x_max}, {y_min}:{y_max})"
                            )
                else:
                    _debug("skimage not available, using rectangular mask approximation")
                    x_min, x_max = int(np.min(x_scaled)), int(np.max(x_scaled))
                    y_min, y_max = int(np.min(y_scaled)), int(np.max(y_scaled))
                    if x_max > x_min and y_max > y_min:
                        mask[slice_idx, y_min : y_max + 1, x_min : x_max + 1] = 1
                        _debug(
                            f"Added rectangular contour to slice {slice_idx}, "
                            f"region ({x_min}:{x_max}, {y_min}:{y_max})"
                        )

            masks[roi_number] = mask

        return masks
    except Exception as exc:
        _err(f"Error creating masks from contours: {exc}")
        return {}


def integrate_rtstruct_processing(
    input_path: str,
    output_folder: str,
    ds_res: tuple | None = None,
) -> dict[str, bool]:
    """
    Integrate RT-Struct processing into the main DICOM conversion pipeline.
    """
    rtstruct_results: dict[str, bool] = {}
    _info("Processing RT-Struct files...")

    try:
        rtstruct_files = []
        for root, _, files in os.walk(input_path):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.isfile(filepath):
                    try:
                        ds = pydicom.dcmread(filepath, force=True)
                        if is_rtstruct_file(ds):
                            rtstruct_files.append(filepath)
                    except Exception:
                        continue

        if not rtstruct_files:
            _info("No RT-Struct files found in input directory")
            return rtstruct_results

        _info(f"Found {len(rtstruct_files)} RT-Struct files to process")
        rtstruct_output_dir = os.path.join(output_folder, "rtstruct_masks")

        for rtstruct_file in rtstruct_files:
            try:
                file_output_path = rtstruct_output_dir
                if ds_res and ds_res[0]:
                    success = process_rtstruct_to_masks(
                        rtstruct_file,
                        None,
                        file_output_path,
                        method="manual",
                        ds_res=ds_res,
                    )
                else:
                    reference_path = os.path.dirname(rtstruct_file)
                    if os.path.basename(reference_path) in ["rtstruct", "RTSTRUCT", "structures"]:
                        reference_path = os.path.dirname(reference_path)

                    _debug(f"RT-Struct file: {rtstruct_file}")
                    _debug(f"Reference path: {reference_path}")
                    success = process_rtstruct_to_masks(
                        rtstruct_file,
                        reference_path,
                        file_output_path,
                        method="manual",
                    )

                rtstruct_results[rtstruct_file] = success
                if success:
                    _debug(f"Successfully processed RT-Struct: {os.path.basename(rtstruct_file)}")
                else:
                    _warn(f"Failed to process RT-Struct: {os.path.basename(rtstruct_file)}")
            except Exception as exc:
                _err(f"Error processing RT-Struct {rtstruct_file}: {exc}")
                rtstruct_results[rtstruct_file] = False

        return rtstruct_results
    except Exception as exc:
        _err(f"Error in RT-Struct integration: {exc}")
        return rtstruct_results
