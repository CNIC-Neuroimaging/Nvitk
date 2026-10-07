"""Index a DICOM folder without loading any pixels: studies → series → files.

:func:`scan_dicom` reads only the headers (``stop_before_pixels``), in parallel,
and groups the files the way :func:`~nvitk.io.conversors._dicom_conversion.load_dicom_series`
would split them into volumes: by study, by ``SeriesInstanceUID``, then — when a
series mixes them — by ``ImageType`` (magnitude / phase, water / fat…). Every
file says what it is (an image, a waveform, a structured report, a Philips
spectral base image that cannot be shown…), so a browser can list everything and
offer only what can become a volume.

Example::

    from nvitk.io.dicom_index import scan_dicom

    for study in scan_dicom("/data/patient01"):
        for series in study.series:
            print(series.label, len(series.files), series.loadable)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

#: File kinds that become volumes when loaded.
LOADABLE_KINDS = ("image",)


@dataclass
class DicomFile:
    """One DICOM file of a series (its header only)."""

    path: str
    sop_uid: str = ""
    instance: int | None = None
    image_type: tuple[str, ...] = ()
    rows: int | None = None
    columns: int | None = None
    frames: int = 1
    position: tuple[float, float, float] | None = None
    slice_location: float | None = None
    time: str = ""
    sop_class: str = ""
    #: ``image``, ``waveform``, ``report``, ``spectral base``, ``presentation``, ``directory`` or ``other``.
    kind: str = "image"
    size: int = 0

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def matrix(self) -> str:
        """``512×512`` (``×40`` frames when multi-frame)."""
        if not self.rows or not self.columns:
            return ""
        return f"{self.columns}×{self.rows}" + (f"×{self.frames}" if self.frames > 1 else "")


@dataclass
class DicomSeries:
    """The files of one series (or of one ``ImageType`` part of a mixed series)."""

    uid: str
    number: str = ""
    description: str = ""
    modality: str = ""
    #: The ``ImageType`` part this is, when the series was split (else ``""``).
    subseries: str = ""
    study_uid: str = ""
    date: str = ""
    time: str = ""
    protocol: str = ""
    body_part: str = ""
    manufacturer: str = ""
    files: list[DicomFile] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Unique within a scan: the series UID plus the part."""
        return f"{self.uid}|{self.subseries}"

    @property
    def paths(self) -> list[str]:
        return [f.path for f in self.files]

    @property
    def kind(self) -> str:
        """The most common file kind of the series."""
        kinds = [f.kind for f in self.files]
        return max(set(kinds), key=kinds.count) if kinds else "other"

    @property
    def loadable(self) -> bool:
        """Whether the series can be loaded as a volume."""
        return self.kind in LOADABLE_KINDS

    @property
    def label(self) -> str:
        """``#401 IMR, 83%`` (``[PHASE]`` for a split part)."""
        text = f"#{self.number} " if self.number else ""
        text += self.description or self.protocol or self.modality or "series"
        if self.subseries:
            text += f" [{self.subseries}]"
        return text

    @property
    def matrix(self) -> str:
        """The image size and slice count, e.g. ``512×512×300``."""
        sizes = {f.matrix for f in self.files if f.matrix}
        if not sizes:
            return ""
        if len(sizes) > 1:
            return "mixed sizes"
        size = next(iter(sizes))
        n = len(self.files)
        return size if n == 1 else f"{size} ×{n}"


@dataclass
class DicomStudy:
    """The series of one study."""

    uid: str
    description: str = ""
    date: str = ""
    time: str = ""
    patient_name: str = ""
    patient_id: str = ""
    accession: str = ""
    series: list[DicomSeries] = field(default_factory=list)

    @property
    def label(self) -> str:
        date = self.date
        if len(date) == 8 and date.isdigit():
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        parts = [p for p in (self.patient_name, self.patient_id, date, self.description) if p]
        return " · ".join(parts) or "study"

    @property
    def files(self) -> list[DicomFile]:
        return [f for s in self.series for f in s.files]


def _text(ds: Any, keyword: str) -> str:
    try:
        value = ds.get(keyword, "")
    except Exception:  # noqa: BLE001
        return ""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "\\".join(str(v) for v in value).strip()
    return str(value).strip()


def _int(ds: Any, keyword: str) -> int | None:
    try:
        value = ds.get(keyword, None)
        return None if value in (None, "") else int(float(str(value).split("\\")[0]))
    except Exception:  # noqa: BLE001
        return None


def _float(ds: Any, keyword: str) -> float | None:
    try:
        value = ds.get(keyword, None)
        return None if value in (None, "") else float(value)
    except Exception:  # noqa: BLE001
        return None


def _kind(ds: Any) -> tuple[str, str]:
    """``(kind, SOP class name)`` of a header."""
    from nvitk.io.conversors._dicom_philips import is_philips_spectral_base_image
    from nvitk.io.conversors._dicom_waveform import is_waveform

    sop = _text(ds, "SOPClassUID")
    try:
        from pydicom.uid import UID

        name = UID(sop).name if sop else ""
    except Exception:  # noqa: BLE001
        name = sop
    low = name.lower()
    if "directory" in low or "MediaStorageDirectoryStorage" in name:
        return "directory", name
    try:
        if is_waveform(ds):
            return "waveform", name
    except Exception:  # noqa: BLE001
        pass
    if "structured report" in low or " sr " in f" {low} " or low.endswith(" sr") or "dose sr" in low:
        return "report", name
    if "presentation state" in low or "key object" in low:
        return "presentation", name
    if is_philips_spectral_base_image(ds):
        return "spectral base", name
    if _int(ds, "Rows") and _int(ds, "Columns"):
        return "image", name
    return "other", name


def _read_one(path: str) -> tuple[str, Any] | None:
    """The header of *path*, or ``None`` when it is not DICOM."""
    import pydicom

    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:  # noqa: BLE001
        return None
    if "SOPInstanceUID" not in ds and "SeriesInstanceUID" not in ds and "DirectoryRecordSequence" not in ds:
        return None  # a stray file pydicom forced its way through
    return path, ds


def _candidate_files(paths: Iterable[str | Path]) -> list[str]:
    out: list[str] = []
    for p in paths:
        p = str(p)
        if os.path.isdir(p):
            for root, _dirs, names in os.walk(p):
                out.extend(os.path.join(root, n) for n in sorted(names) if not n.startswith("."))
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def _file_info(path: str, ds: Any) -> DicomFile:
    from nvitk.io.conversors._dicom_conversion import _extract_image_type_tokens

    kind, sop_name = _kind(ds)
    pos = None
    try:
        ipp = ds.get("ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            pos = tuple(float(v) for v in ipp)
    except Exception:  # noqa: BLE001
        pos = None
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    return DicomFile(
        path=path, sop_uid=_text(ds, "SOPInstanceUID"), instance=_int(ds, "InstanceNumber"),
        image_type=tuple(_extract_image_type_tokens(ds)), rows=_int(ds, "Rows"), columns=_int(ds, "Columns"),
        frames=_int(ds, "NumberOfFrames") or 1, position=pos, slice_location=_float(ds, "SliceLocation"),
        time=_text(ds, "ContentTime") or _text(ds, "AcquisitionTime"), sop_class=sop_name, kind=kind, size=size,
    )


def _sort_files(files: list[DicomFile]) -> None:
    files.sort(key=lambda f: (f.instance if f.instance is not None else 10**9,
                              f.slice_location if f.slice_location is not None else 0.0, f.name))


def scan_dicom(
    paths: str | Path | Sequence[str | Path],
    *,
    workers: int = 8,
    split_image_type: bool = True,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[DicomStudy]:
    """Studies → series → files under *paths* (folders and / or files), headers only.

    Series that mix ``ImageType`` values (magnitude / phase, Dixon water / fat…)
    are split into parts, as the loader splits them into volumes (with
    *split_image_type*). *progress* is called with ``(files read, files found)``;
    *cancelled* stops the scan early when it returns ``True``.
    """
    from nvitk.io.conversors._dicom_conversion import (
        _extract_image_type_tokens,
        _image_type_label,
        _image_type_signature,
    )

    if isinstance(paths, (str, Path)):
        paths = [paths]
    files = _candidate_files(paths)
    headers: list[tuple[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        for result in pool.map(_read_one, files, chunksize=16):
            done += 1
            if result is not None:
                headers.append(result)
            if progress is not None and (done % 64 == 0 or done == len(files)):
                progress(done, len(files))
            if cancelled is not None and cancelled():
                break

    studies: dict[str, DicomStudy] = {}
    series: dict[str, list[tuple[DicomFile, Any]]] = {}
    for path, ds in headers:
        if "DirectoryRecordSequence" in ds:
            continue  # DICOMDIR: an index of the files, which are scanned themselves
        uid = _text(ds, "SeriesInstanceUID") or "no-series"
        series.setdefault(uid, []).append((_file_info(path, ds), ds))

    for uid, members in series.items():
        first = members[0][1]
        study_uid = _text(first, "StudyInstanceUID") or "no-study"
        study = studies.get(study_uid)
        if study is None:
            study = studies[study_uid] = DicomStudy(
                uid=study_uid, description=_text(first, "StudyDescription"), date=_text(first, "StudyDate"),
                time=_text(first, "StudyTime"), patient_name=_text(first, "PatientName"),
                patient_id=_text(first, "PatientID"), accession=_text(first, "AccessionNumber"))
        parts: dict[tuple[str, ...], list[tuple[DicomFile, Any]]] = {}
        if split_image_type:
            for info, ds in members:
                sig = _image_type_signature(_extract_image_type_tokens(ds)) if info.kind == "image" else ("",)
                parts.setdefault(sig, []).append((info, ds))
        else:
            parts[("",)] = members
        for sig, part in parts.items():
            ds0 = part[0][1]
            s = DicomSeries(
                uid=uid, number=_text(ds0, "SeriesNumber"), description=_text(ds0, "SeriesDescription"),
                modality=_text(ds0, "Modality"), subseries=_image_type_label(sig) if len(parts) > 1 and sig != ("",) else "",
                study_uid=study_uid, date=_text(ds0, "SeriesDate"), time=_text(ds0, "SeriesTime"),
                protocol=_text(ds0, "ProtocolName"), body_part=_text(ds0, "BodyPartExamined"),
                manufacturer=_text(ds0, "Manufacturer"), files=[info for info, _ds in part])
            _sort_files(s.files)
            study.series.append(s)

    return _sorted_studies(studies.values())


def _sorted_studies(studies: Iterable[DicomStudy]) -> list[DicomStudy]:
    """Studies by date, their series by number."""

    def _number(s: DicomSeries) -> tuple[int, str]:
        try:
            return int(float(s.number)), s.subseries
        except ValueError:
            return 10**9, s.label

    out = sorted(studies, key=lambda st: (st.date, st.time, st.uid))
    for st in out:
        st.series.sort(key=_number)
    return out


def merge_studies(*scans: Sequence[DicomStudy]) -> list[DicomStudy]:
    """Several scans as one: the same study (or series) found in several folders is
    one study (series), its files gathered once."""
    import copy

    studies: dict[str, DicomStudy] = {}
    for scan in scans:
        for st in scan:
            target = studies.get(st.uid)
            if target is None:
                target = studies[st.uid] = copy.copy(st)
                target.series = []
            for s in st.series:
                same = next((x for x in target.series if x.key == s.key), None)
                if same is None:
                    s = copy.copy(s)
                    s.files = list(s.files)
                    target.series.append(s)
                    continue
                known = {os.path.abspath(f.path) for f in same.files}
                same.files.extend(f for f in s.files if os.path.abspath(f.path) not in known)
                _sort_files(same.files)
    return _sorted_studies(studies.values())


def read_header(path: str | Path) -> Any:
    """The header of one DICOM file (a pydicom ``Dataset`` without the pixels)."""
    import pydicom

    return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)


__all__ = ["DicomFile", "DicomSeries", "DicomStudy", "LOADABLE_KINDS", "merge_studies", "read_header", "scan_dicom"]
