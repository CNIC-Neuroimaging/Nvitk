from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import pydicom
except Exception:
    pydicom = None

try:
    from skimage.draw import polygon2mask

    _HAS_SKIMAGE = True
except Exception:
    polygon2mask = None
    _HAS_SKIMAGE = False

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


def _sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe.strip("_") or "roi"


def is_rtstruct_file(ds: Any) -> bool:
    if pydicom is None:
        return False
    sop = str(getattr(ds, "SOPClassUID", "") or "").strip("\x00").strip()
    if not sop:
        return False
    if sop in RT_SOP_CLASS_UIDS:
        return True
    return "1.2.840.10008.5.1.4.1.1.481.3" in sop


def _iter_dicom_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    out: list[Path] = []
    for item in sorted(input_path.rglob("*")):
        if not item.is_file():
            continue
        out.append(item)
    return out


def _read_all_reference_slices(input_path: Path, series_uid: str | None = None) -> list[Any]:
    if pydicom is None:
        return []
    series: list[Any] = []
    for fp in _iter_dicom_files(input_path):
        try:
            ds = pydicom.dcmread(str(fp), force=True)
        except Exception:
            continue
        if is_rtstruct_file(ds):
            continue
        if getattr(ds, "PixelData", None) is None:
            continue
        if series_uid is not None and str(ds.get("SeriesInstanceUID", "")) != str(series_uid):
            continue
        series.append(ds)

    def _sort_key(ds: Any) -> tuple[int, float]:
        try:
            inst = int(ds.get("InstanceNumber", 10**9))
        except Exception:
            inst = 10**9
        try:
            ipp = ds.get("ImagePositionPatient", [0, 0, 0])
            z = float(ipp[2]) if ipp is not None and len(ipp) > 2 else 0.0
        except Exception:
            z = 0.0
        return inst, z

    return sorted(series, key=_sort_key)


def _extract_referenced_series_uid(rt_ds: Any) -> str | None:
    try:
        ref = rt_ds.ReferencedFrameOfReferenceSequence[0]
        study = ref.RTReferencedStudySequence[0]
        series = study.RTReferencedSeriesSequence[0]
        return str(series.SeriesInstanceUID)
    except Exception:
        return None


def _build_sop_index(reference_series: list[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for idx, ds in enumerate(reference_series):
        uid = str(ds.get("SOPInstanceUID", ""))
        if uid:
            out[uid] = idx
    return out


def _world_to_image_pixels(contour_xyz: np.ndarray, ref_ds: Any) -> np.ndarray:
    if contour_xyz.size == 0:
        return np.empty((0, 2), dtype=float)
    iop = np.asarray([float(v) for v in ref_ds.ImageOrientationPatient], dtype=float)
    ipp = np.asarray([float(v) for v in ref_ds.ImagePositionPatient], dtype=float)
    spacing = np.asarray([float(v) for v in ref_ds.PixelSpacing], dtype=float)  # row, col
    row_dir = iop[0:3]
    col_dir = iop[3:6]

    out: list[list[float]] = []
    for xyz in contour_xyz:
        vec = xyz - ipp
        row_idx = float(np.dot(vec, row_dir) / spacing[0])
        col_idx = float(np.dot(vec, col_dir) / spacing[1])
        out.append([col_idx, row_idx])
    return np.asarray(out, dtype=float)


def _contour_slice_index(contour: Any, ref_series: list[Any], sop_to_idx: dict[str, int]) -> int:
    # Preferred route: referenced SOP UID inside contour.
    try:
        img_seq = contour.ContourImageSequence
        if img_seq:
            sop_uid = str(img_seq[0].ReferencedSOPInstanceUID)
            if sop_uid in sop_to_idx:
                return sop_to_idx[sop_uid]
    except Exception:
        pass

    # Fallback route: nearest z in patient space.
    try:
        points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
        z = float(np.mean(points[:, 2]))
    except Exception:
        return 0

    best_idx = 0
    best_dist = float("inf")
    for idx, ds in enumerate(ref_series):
        try:
            ipp = ds.get("ImagePositionPatient")
            if ipp is None or len(ipp) < 3:
                continue
            z_ds = float(ipp[2])
            d = abs(z_ds - z)
            if d < best_dist:
                best_dist = d
                best_idx = idx
        except Exception:
            continue
    return best_idx


def _polygon_mask(shape: tuple[int, int], vertices_xy: np.ndarray) -> np.ndarray:
    if vertices_xy.size == 0:
        return np.zeros(shape, dtype=np.uint8)

    x = np.clip(vertices_xy[:, 0], 0, shape[1] - 1)
    y = np.clip(vertices_xy[:, 1], 0, shape[0] - 1)
    pts = np.column_stack([x, y])
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 3:
        return np.zeros(shape, dtype=np.uint8)

    if _HAS_SKIMAGE:
        try:
            return polygon2mask(shape, pts).astype(np.uint8)
        except Exception:
            pass

    # Rectangular fallback when skimage is unavailable.
    x0, x1 = int(np.floor(np.min(pts[:, 0]))), int(np.ceil(np.max(pts[:, 0])))
    y0, y1 = int(np.floor(np.min(pts[:, 1]))), int(np.ceil(np.max(pts[:, 1])))
    out = np.zeros(shape, dtype=np.uint8)
    out[max(0, y0) : min(shape[0], y1 + 1), max(0, x0) : min(shape[1], x1 + 1)] = 1
    return out


def process_rtstruct_file(rtstruct_path: str, input_path: str, output_folder: str) -> list[str]:
    if pydicom is None or nib is None:
        return []

    rt_fp = Path(rtstruct_path)
    input_fp = Path(input_path)
    out_root = Path(output_folder) / "rtstruct_masks"
    out_root.mkdir(parents=True, exist_ok=True)

    rt_ds = pydicom.dcmread(str(rt_fp), force=True)
    ref_uid = _extract_referenced_series_uid(rt_ds)
    ref_series = _read_all_reference_slices(input_fp, series_uid=ref_uid)
    if not ref_series:
        ref_series = _read_all_reference_slices(input_fp, series_uid=None)
    if not ref_series:
        return []

    rows = int(getattr(ref_series[0], "Rows", 0) or 0)
    cols = int(getattr(ref_series[0], "Columns", 0) or 0)
    depth = len(ref_series)
    if rows <= 0 or cols <= 0 or depth <= 0:
        return []

    sop_to_idx = _build_sop_index(ref_series)

    roi_defs: dict[int, dict[str, Any]] = {}
    for item in getattr(rt_ds, "StructureSetROISequence", []):
        roi_number = int(getattr(item, "ROINumber", 0))
        if roi_number == 0:
            continue
        roi_defs[roi_number] = {
            "name": str(getattr(item, "ROIName", f"ROI_{roi_number}")),
            "description": str(getattr(rt_ds, "SeriesDescription", "")),
        }

    masks: dict[int, np.ndarray] = {}
    for roi_contour in getattr(rt_ds, "ROIContourSequence", []):
        roi_number = int(getattr(roi_contour, "ReferencedROINumber", 0) or 0)
        if roi_number == 0:
            continue
        mask = masks.setdefault(roi_number, np.zeros((depth, rows, cols), dtype=np.uint8))

        for contour in getattr(roi_contour, "ContourSequence", []):
            if str(getattr(contour, "ContourGeometricType", "UNKNOWN")) != "CLOSED_PLANAR":
                continue
            try:
                contour_xyz = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            except Exception:
                continue
            if contour_xyz.shape[0] < 3:
                continue

            z_idx = _contour_slice_index(contour, ref_series, sop_to_idx)
            z_idx = max(0, min(depth - 1, z_idx))
            ref_ds = ref_series[z_idx]
            try:
                vertices_xy = _world_to_image_pixels(contour_xyz, ref_ds)
            except Exception:
                continue

            poly = _polygon_mask((rows, cols), vertices_xy)
            mask[z_idx] = np.logical_or(mask[z_idx], poly).astype(np.uint8)

    if not masks:
        return []

    written: list[str] = []
    combined = np.zeros((depth, rows, cols), dtype=np.uint16)
    roi_json: dict[str, Any] = {
        "source_file": str(rt_fp),
        "referenced_series_uid": ref_uid,
        "axis_ordering": "XYZ",
        "rois": {},
    }

    for roi_number, mask in sorted(masks.items(), key=lambda x: x[0]):
        meta = roi_defs.get(roi_number, {"name": f"ROI_{roi_number}", "description": ""})
        name = str(meta["name"])
        desc = str(meta["description"])
        safe = _sanitize_filename(f"{desc}_{name}_{roi_number}")
        out_file = out_root / f"{safe}.nii"

        mask_xyz = np.transpose(mask, (2, 1, 0))  # ZYX -> XYZ
        image = nib.Nifti1Image(mask_xyz, np.eye(4))
        image.header["descrip"] = f"{desc}_{name}_{roi_number}"[:79]
        nib.save(image, str(out_file))
        written.append(str(out_file))

        combined[mask > 0] = np.uint16(roi_number)
        roi_json["rois"][str(roi_number)] = {
            "name": name,
            "description": desc,
            "label_id": int(roi_number),
            "file": str(out_file),
        }

    base = _sanitize_filename(rt_fp.stem)
    combined_xyz = np.transpose(combined, (2, 1, 0))
    combined_path = out_root / f"{base}_rtstruct_combined_masks.nii"
    combined_img = nib.Nifti1Image(combined_xyz, np.eye(4))
    combined_img.header["descrip"] = f"RTStruct masks from {base}"[:79]
    try:
        payload = json.dumps(roi_json, ensure_ascii=True).encode("utf-8")
        combined_img.header.extensions.append(nib.nifti1.Nifti1Extension(16, payload))
    except Exception:
        pass
    nib.save(combined_img, str(combined_path))
    written.append(str(combined_path))

    json_path = out_root / f"{base}_roi_info.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(roi_json, f, indent=2, ensure_ascii=True)
    except Exception:
        pass

    return written


def integrate_rtstruct_processing(input_path: str, output_folder: str) -> dict[str, bool]:
    if pydicom is None:
        return {}

    src = Path(input_path)
    files = _iter_dicom_files(src)
    rt_files: list[Path] = []
    for fp in files:
        try:
            ds = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
        except Exception:
            continue
        if is_rtstruct_file(ds):
            rt_files.append(fp)

    results: dict[str, bool] = {}
    for rt_fp in rt_files:
        try:
            written = process_rtstruct_file(str(rt_fp), input_path, output_folder)
            results[str(rt_fp)] = bool(written)
        except Exception:
            results[str(rt_fp)] = False
    return results

