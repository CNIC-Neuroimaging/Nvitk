"""Dixon anatomical region assignment from DICOM slice locations.

Regions (HEAD, THORAX, LEGS) are ordered by ascending slice coordinate:
lowest centroid → HEAD, middle → THORAX, highest → LEGS.
"""

from __future__ import annotations

import io
import os
import re
import statistics
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.logger import Logger

try:
    import pydicom
except ImportError:  
    pydicom = None  

log = Logger()

DIXON_REGIONS: tuple[str, ...] = ("HEAD", "THORAX", "LEGS")

_REGION_BY_LAST_DIGIT: dict[str, str] = {"1": "HEAD", "2": "THORAX", "3": "LEGS"}

_DICOM_HEADER_TAGS = [
    "SeriesNumber",
    "Modality",
    "SliceLocation",
    "ImagePositionPatient",
    "ImageOrientationPatient",
]


def region_from_scan_id_fallback(scan_id: str | None) -> str | None:
    """Last-digit fallback when fewer than three slice-location clusters exist."""
    if not scan_id:
        return None
    digits = re.sub(r"\D", "", str(scan_id))
    if not digits:
        return None
    return _REGION_BY_LAST_DIGIT.get(digits[-1])


def _slice_coordinate(ds: Any) -> float | None:
    """Scalar slice position: SliceLocation preferred, else IPP projected on slice normal."""
    if hasattr(ds, "SliceLocation") and ds.SliceLocation is not None:
        try:
            return float(ds.SliceLocation)
        except (TypeError, ValueError):
            pass

    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    if iop is None or ipp is None or len(iop) < 6:
        return None
    try:
        row_dir = np.array([float(x) for x in iop[:3]], dtype=float)
        col_dir = np.array([float(x) for x in iop[3:6]], dtype=float)
        slice_dir = np.cross(row_dir, col_dir)
        nrm = np.linalg.norm(slice_dir)
        if nrm < 1e-8:
            return None
        slice_dir = slice_dir / nrm
        ipp_arr = np.array([float(x) for x in ipp], dtype=float)
        return float(np.dot(ipp_arr, slice_dir))
    except (TypeError, ValueError):
        return None


def slice_centroid_from_datasets(datasets: list[Any]) -> float | None:
    """Median slice coordinate across *datasets* (one per slice)."""
    coords: list[float] = []
    for ds in datasets:
        coord = _slice_coordinate(ds)
        if coord is not None:
            coords.append(coord)
    if not coords:
        return None
    return float(statistics.median(coords))


def _read_dicom_header(path: str | Path) -> Any | None:
    if pydicom is None:
        return None
    try:
        return pydicom.dcmread(
            str(path),
            stop_before_pixels=True,
            force=True,
            specific_tags=_DICOM_HEADER_TAGS,
        )
    except Exception:
        return None


def _read_dicom_header_bytes(data: bytes) -> Any | None:
    if pydicom is None:
        return None
    try:
        return pydicom.dcmread(
            io.BytesIO(data),
            stop_before_pixels=True,
            force=True,
            specific_tags=_DICOM_HEADER_TAGS,
        )
    except Exception:
        return None


def _normalize_series_number(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def series_centroids_from_dicom_dir(
    dicom_dir: Path | str,
    series_numbers: set[str] | frozenset[str] | None = None,
) -> dict[str, float]:
    """Return median slice centroid per SeriesNumber under *dicom_dir*."""
    root = Path(dicom_dir)
    if not root.exists():
        return {}

    wanted = {str(s).strip() for s in series_numbers} if series_numbers else None
    per_series: dict[str, list[float]] = {}

    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            fp = Path(dirpath) / fn
            ds = _read_dicom_header(fp)
            if ds is None:
                continue
            modality = str(getattr(ds, "Modality", "") or "").strip().upper()
            if modality and modality != "MR":
                continue
            sn = _normalize_series_number(getattr(ds, "SeriesNumber", None))
            if sn is None:
                continue
            if wanted is not None and sn not in wanted:
                continue
            coord = _slice_coordinate(ds)
            if coord is None:
                continue
            per_series.setdefault(sn, []).append(coord)

    return {
        sn: float(statistics.median(coords))
        for sn, coords in per_series.items()
        if coords
    }


def _cluster_keys_by_largest_gaps(
    sorted_items: list[tuple[str, float]],
) -> list[list[str]]:
    """Split sorted (key, centroid) items into three clusters using two largest gaps."""
    if len(sorted_items) <= 3:
        return [[key] for key, _ in sorted_items]

    gaps: list[tuple[float, int]] = []
    for i in range(len(sorted_items) - 1):
        gap = sorted_items[i + 1][1] - sorted_items[i][1]
        gaps.append((gap, i))

    split_indices = sorted(idx for _, idx in sorted(gaps, reverse=True)[:2])
    clusters: list[list[str]] = []
    start = 0
    for split_idx in split_indices:
        clusters.append([key for key, _ in sorted_items[start : split_idx + 1]])
        start = split_idx + 1
    clusters.append([key for key, _ in sorted_items[start:]])
    return clusters


def assign_dixon_regions(centroids: dict[str, float]) -> dict[str, str]:
    """Map series/scan keys to HEAD, THORAX, or LEGS by ascending slice centroid."""
    if not centroids:
        return {}

    if len(centroids) < 3:
        log.warning(
            f"Only {len(centroids)} Dixon series with slice locations; "
            "falling back to last-digit mapping where possible."
        )
        out: dict[str, str] = {}
        for key in centroids:
            region = region_from_scan_id_fallback(key)
            if region:
                out[key] = region
        return out

    sorted_items = sorted(centroids.items(), key=lambda item: item[1])

    if len(sorted_items) == 3:
        clusters = [[key] for key, _ in sorted_items]
    else:
        clusters = _cluster_keys_by_largest_gaps(sorted_items)
        if len(clusters) != 3:
            log.warning(
                f"Expected 3 Dixon region clusters, got {len(clusters)}; "
                "falling back to last-digit mapping where possible."
            )
            out = {}
            for key in centroids:
                region = region_from_scan_id_fallback(key)
                if region:
                    out[key] = region
            return out

    region_map: dict[str, str] = {}
    for region, cluster in zip(DIXON_REGIONS, clusters):
        for key in cluster:
            region_map[key] = region
    return region_map


def build_series_region_map(
    dicom_dir: Path | str,
    series_numbers: set[str],
) -> dict[str, str]:
    """Compute series_number → region from DICOM slice locations."""
    centroids = series_centroids_from_dicom_dir(dicom_dir, series_numbers)
    missing = series_numbers - set(centroids)
    for sn in sorted(missing):
        log.warning(f"Dixon series {sn}: no slice location in DICOM; skipping region assignment.")

    assigned = assign_dixon_regions(centroids)
    for sn, region in sorted(assigned.items()):
        centroid = centroids.get(sn)
        if centroid is not None:
            log.info(f"  Dixon series {sn} → {region} (centroid={centroid:.2f})")
        else:
            log.info(f"  Dixon series {sn} → {region}")
    return assigned


def _is_dicom_zip_member(name: str) -> bool:
    base = Path(name).name.lower()
    if base.endswith(".nii") or base.endswith(".nii.gz"):
        return False
    if base.endswith((".json", ".bval", ".bvec", ".xml")):
        return False
    if base.endswith((".png", ".jpg", ".jpeg", ".gif", ".tif", ".tiff")):
        return False
    return True


def peek_scan_slice_centroid(scan: Any) -> float | None:
    """Download an XNAT scan DICOM zip to temp and return median slice centroid."""
    # Lazy import avoids db.xnat ↔ dixon_regions circular dependency at import time.
    from nvitk.db.xnat import _download_scan_bundle

    datasets: list[Any] = []
    with tempfile.TemporaryDirectory(prefix="nvitk_dixon_peek_") as tmp_dir:
        zip_path = Path(tmp_dir) / "scan.zip"
        try:
            bundle_path = _download_scan_bundle(scan, zip_path, resource_label="DICOM")
        except Exception as exc:
            log.warning(f"XNAT DICOM peek failed: {exc}")
            return None

        try:
            with zipfile.ZipFile(bundle_path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    if not _is_dicom_zip_member(member.filename):
                        continue
                    try:
                        with archive.open(member) as handle:
                            data = handle.read()
                    except Exception:
                        continue
                    ds = _read_dicom_header_bytes(data)
                    if ds is None:
                        continue
                    modality = str(getattr(ds, "Modality", "") or "").strip().upper()
                    if modality and modality != "MR":
                        continue
                    datasets.append(ds)
        except Exception as exc:
            log.warning(f"XNAT DICOM zip read failed: {exc}")
            return None

    return slice_centroid_from_datasets(datasets)


def classify_dixon_scans_by_slice_location(
    scan_ids: list[str],
    scans_by_id: dict[str, Any],
) -> dict[str, str]:
    """Assign DIXON_HEAD / DIXON_THORAX / DIXON_LEGS per XNAT scan_id."""
    centroids: dict[str, float] = {}
    for scan_id in scan_ids:
        scan = scans_by_id.get(scan_id)
        if scan is None:
            continue
        centroid = peek_scan_slice_centroid(scan)
        if centroid is not None:
            centroids[scan_id] = centroid
        else:
            log.warning(f"Dixon scan {scan_id}: could not read slice location from XNAT DICOM.")

    region_by_key = assign_dixon_regions(centroids)

    # Last-digit fallback for scans missing centroids when assign left them out.
    out: dict[str, str] = {}
    for scan_id in scan_ids:
        region = region_by_key.get(scan_id)
        if region:
            out[scan_id] = f"DIXON_{region}"
            continue
        fallback = region_from_scan_id_fallback(scan_id)
        if fallback:
            log.warning(f"Dixon scan {scan_id}: using last-digit fallback → {fallback}.")
            out[scan_id] = f"DIXON_{fallback}"

    for scan_id, seq in sorted(out.items()):
        centroid = centroids.get(scan_id)
        if centroid is not None:
            log.info(f"  Dixon scan {scan_id} → {seq} (centroid={centroid:.2f})")

    return out


__all__ = [
    "DIXON_REGIONS",
    "assign_dixon_regions",
    "build_series_region_map",
    "classify_dixon_scans_by_slice_location",
    "peek_scan_slice_centroid",
    "region_from_scan_id_fallback",
    "series_centroids_from_dicom_dir",
    "slice_centroid_from_datasets",
]
