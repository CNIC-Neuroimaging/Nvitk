"""Edit DICOM headers and export a chosen subset of files — anonymized if wanted.

A :class:`DicomEditor` is an ordered list of operations, each scoped to a set of
files: set a tag, delete a tag, or de-identify (:class:`Anonymize`, a subset of
the DICOM PS3.15 *Basic Application Level Confidentiality Profile*). Nothing is
written until :func:`export_dicom`; :meth:`DicomEditor.header` shows a file's
header as it will be written.

De-identification here replaces the patient's name and ID, removes the other
identifying attributes (addresses, physicians, institution, device serial…), can
shift or blank the dates, drops private tags, and gives every study / series /
instance a new UID — consistently across files, so the series still fit together.
It does not look inside the pixels: text burned into the image (secondary
captures, dose screens) stays.

Example::

    from nvitk.io.dicom_edit import DicomEditor, export_dicom
    from nvitk.io.dicom_index import scan_dicom

    study = scan_dicom("/data/patient01")[0]
    paths = study.series[3].paths
    ed = DicomEditor()
    ed.anonymize(paths, patient_name="SUBJ-001", patient_id="SUBJ-001", dates="shift", shift_days=-100)
    ed.set(paths, "InstitutionName", "")          # any tag, by keyword or (gggg,eeee)
    export_dicom(paths, "/data/out", editor=ed)
"""

from __future__ import annotations

import datetime as _dt
import os
import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

#: How :func:`export_dicom` lays the files out.
EXPORT_LAYOUTS = ("series folders", "patient / study / series", "flat")
#: How :func:`export_dicom` names the files.
EXPORT_NAMES = ("auto", "original", "numbered")
#: What :class:`Anonymize` does with dates.
DATE_MODES = ("keep", "shift", "remove")

#: Attributes the de-identification removes (PS3.15 basic profile, the ones that
#: occur in practice on CT / MR / PET).
_REMOVE = (
    "OtherPatientIDs", "OtherPatientIDsSequence", "OtherPatientNames", "PatientAddress",
    "PatientTelephoneNumbers", "PatientMotherBirthName", "PatientBirthName", "PatientBirthTime",
    "MilitaryRank", "BranchOfService", "MedicalRecordLocator", "Occupation", "AdditionalPatientHistory",
    "PatientComments", "CountryOfResidence", "RegionOfResidence", "PatientInsurancePlanCodeSequence",
    "ResponsiblePerson", "ResponsibleOrganization", "IssuerOfPatientID", "InstitutionName",
    "InstitutionAddress", "InstitutionalDepartmentName", "InstitutionCodeSequence",
    "ReferringPhysicianAddress", "ReferringPhysicianTelephoneNumbers",
    "ReferringPhysicianIdentificationSequence", "PhysiciansOfRecord", "PhysiciansOfRecordIdentificationSequence",
    "PerformingPhysicianName", "PerformingPhysicianIdentificationSequence", "NameOfPhysiciansReadingStudy",
    "PhysiciansReadingStudyIdentificationSequence", "OperatorsName", "OperatorIdentificationSequence",
    "RequestingPhysician", "RequestingService", "RequestAttributesSequence", "StationName",
    "DeviceSerialNumber", "ScheduledPerformingPhysicianName", "AdmittingDiagnosesDescription",
    "PerformedLocation", "ScheduledStudyLocation", "CurrentPatientLocation", "PatientState",
    "MedicalAlerts", "Allergies", "SmokingStatus", "PregnancyStatus", "LastMenstrualDate",
    "SpecialNeeds", "EthnicGroup", "ReferencedPatientSequence", "ReferencedStudySequence",
    "PerformedProcedureStepID", "ScheduledProcedureStepID", "RequestedProcedureID", "ImageComments",
    "FillerOrderNumberImagingServiceRequest", "PlacerOrderNumberImagingServiceRequest",
)
#: Type-2 attributes kept but emptied.
_EMPTY = ("ReferringPhysicianName", "AccessionNumber", "StudyID", "PatientBirthDate")
#: Kept unless asked otherwise: what analyses need (dose, body size, sex, age).
_DEMOGRAPHICS = ("PatientSex", "PatientAge", "PatientSize", "PatientWeight")
#: UIDs that name a class or a coding scheme, not an instance: never remapped.
_CLASS_UID = re.compile(r"(ClassUID|TransferSyntaxUID|CodingSchemeUID|ContextUID|MappingResourceUID|"
                        r"PrivateInformationCreatorUID|ImplementationClassUID)$")


def parse_tag(text: str | int | tuple[int, int]) -> int:
    """A tag from a keyword (``PatientName``), ``(0010,0010)``, ``0010,0010`` or ``00100010``."""
    from pydicom.datadict import tag_for_keyword
    from pydicom.tag import Tag

    if isinstance(text, int):
        return int(Tag(text))
    if isinstance(text, tuple):
        return int(Tag(*text))
    s = str(text).strip()
    tag = tag_for_keyword(s)
    if tag is not None:
        return int(tag)
    hexes = re.findall(r"[0-9A-Fa-f]{4}", s.replace("x", ""))
    if len(hexes) == 2:
        return int(Tag(int(hexes[0], 16), int(hexes[1], 16)))
    if re.fullmatch(r"[0-9A-Fa-f]{8}", s):
        return int(Tag(int(s, 16)))
    raise ValueError(f"Not a DICOM tag: {text!r} (a keyword like PatientName, or (gggg,eeee)).")


def tag_vr(tag: int, ds: Any = None) -> str:
    """The VR to use for *tag*: the existing element's, else the dictionary's, else ``LO``."""
    from pydicom.datadict import dictionary_VR

    if ds is not None and tag in ds:
        return str(ds[tag].VR)
    try:
        vr = dictionary_VR(tag)
        return vr.split(" or ")[0] if vr else "LO"
    except KeyError:
        return "LO"


def coerce_value(text: Any, vr: str) -> Any:
    """A value typed in as text, as pydicom wants it for *vr* (``\\`` separates multiple values)."""
    if not isinstance(text, str):
        return text
    parts = text.split("\\") if text else []
    if vr in ("US", "UL", "SS", "SL", "UV", "SV", "AT"):
        nums = [int(float(p)) for p in parts]
    elif vr in ("FL", "FD"):
        nums = [float(p) for p in parts]
    else:
        return text
    return nums[0] if len(nums) == 1 else nums


@dataclass
class TagEdit:
    """Set (``value``) or delete (``value`` is ``None`` and *delete*) one tag on some files."""

    files: frozenset[str]
    tag: int
    value: Any = None
    vr: str = ""
    delete: bool = False

    def applies(self, path: str) -> bool:
        return not self.files or os.path.abspath(path) in self.files

    def apply(self, ds: Any) -> None:
        if self.delete:
            if self.tag in ds:
                del ds[self.tag]
            return
        vr = self.vr or tag_vr(self.tag, ds)
        value = coerce_value(self.value, vr)
        if self.tag in ds:
            ds[self.tag].value = value
        else:
            ds.add_new(self.tag, vr, value)

    @property
    def tags(self) -> set[int]:
        return {self.tag}


@dataclass
class Anonymize:
    """De-identify the files (see the module notes for what this covers)."""

    files: frozenset[str]
    patient_name: str = "ANONYMOUS"
    patient_id: str = "ANON"
    #: ``keep``, ``shift`` (by *shift_days*) or ``remove``.
    dates: str = "keep"
    shift_days: int = 0
    keep_demographics: bool = True
    remove_private: bool = True
    new_uids: bool = True
    remove_descriptions: bool = False
    #: Mixed into the new UIDs: the same original UID always maps to the same new
    #: one within this operation, and to a different one in another.
    salt: str = field(default_factory=lambda: secrets.token_hex(8))

    def applies(self, path: str) -> bool:
        return not self.files or os.path.abspath(path) in self.files

    def _uid(self, uid: str) -> str:
        from pydicom.uid import generate_uid

        return str(generate_uid(entropy_srcs=[self.salt, str(uid)]))

    def _shift_date(self, text: str) -> str:
        text = str(text)
        if len(text) < 8 or not text[:8].isdigit():
            return text
        try:
            day = _dt.datetime.strptime(text[:8], "%Y%m%d") + _dt.timedelta(days=int(self.shift_days))
        except ValueError:
            return text
        return day.strftime("%Y%m%d") + text[8:]

    def _walk(self, ds: Any) -> None:
        """UIDs and dates, nested sequences included."""
        for elem in list(ds):
            if elem.VR == "SQ":
                for item in elem.value or []:
                    self._walk(item)
                continue
            keyword = elem.keyword or ""
            if self.new_uids and elem.VR == "UI" and elem.value and not _CLASS_UID.search(keyword):
                if isinstance(elem.value, (list, tuple)) or type(elem.value).__name__ == "MultiValue":
                    elem.value = [self._uid(v) for v in elem.value]
                else:
                    elem.value = self._uid(elem.value)
            elif elem.VR in ("DA", "DT") and elem.value and self.dates != "keep":
                if self.dates == "remove":
                    elem.value = ""
                elif isinstance(elem.value, str):
                    elem.value = self._shift_date(elem.value)
            elif elem.VR == "TM" and elem.value and self.dates == "remove":
                elem.value = ""

    def apply(self, ds: Any) -> None:
        ds.PatientName = self.patient_name
        ds.PatientID = self.patient_id
        for keyword in _REMOVE:
            if keyword in ds:
                delattr(ds, keyword)
        for keyword in _EMPTY:
            if keyword in ds:
                ds[keyword].value = ""
        if not self.keep_demographics:
            for keyword in _DEMOGRAPHICS:
                if keyword in ds:
                    delattr(ds, keyword)
        if self.remove_descriptions:
            for keyword in ("StudyDescription", "SeriesDescription", "ProtocolName", "PerformedProcedureStepDescription",
                            "RequestedProcedureDescription"):
                if keyword in ds:
                    ds[keyword].value = ""
        if self.remove_private:
            ds.remove_private_tags()
        self._walk(ds)
        meta = getattr(ds, "file_meta", None)
        if meta is not None and "MediaStorageSOPInstanceUID" in meta and "SOPInstanceUID" in ds:
            meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "nvitk: PS3.15 basic profile (subset)" + (
            f"; dates shifted {int(self.shift_days):+d} d" if self.dates == "shift" else "")

    @property
    def tags(self) -> set[int]:
        """The top-level tags this touches (for highlighting; private and UIDs not listed)."""
        from pydicom.datadict import tag_for_keyword

        names = ["PatientName", "PatientID", *_REMOVE, *_EMPTY, "PatientIdentityRemoved", "DeidentificationMethod"]
        if not self.keep_demographics:
            names += list(_DEMOGRAPHICS)
        return {int(t) for t in (tag_for_keyword(n) for n in names) if t is not None}


class DicomEditor:
    """Pending header changes over a set of DICOM files, applied on export."""

    def __init__(self) -> None:
        self.ops: list[TagEdit | Anonymize] = []

    @staticmethod
    def _scope(files: Iterable[str] | None) -> frozenset[str]:
        return frozenset(os.path.abspath(str(f)) for f in (files or ()))

    def set(self, files: Iterable[str] | None, tag: str | int, value: Any, vr: str = "") -> TagEdit:
        """Set *tag* to *value* on *files* (``None``: every file)."""
        op = TagEdit(self._scope(files), parse_tag(tag), value=value, vr=vr)
        self.ops.append(op)
        return op

    def delete(self, files: Iterable[str] | None, tag: str | int) -> TagEdit:
        """Remove *tag* from *files*."""
        op = TagEdit(self._scope(files), parse_tag(tag), delete=True)
        self.ops.append(op)
        return op

    def anonymize(self, files: Iterable[str] | None, **options: Any) -> Anonymize:
        """De-identify *files* (options: see :class:`Anonymize`)."""
        op = Anonymize(self._scope(files), **options)
        self.ops.append(op)
        return op

    def undo(self) -> TagEdit | Anonymize | None:
        """Drop the last operation."""
        return self.ops.pop() if self.ops else None

    def clear(self) -> None:
        self.ops.clear()

    def ops_for(self, path: str) -> list[TagEdit | Anonymize]:
        return [op for op in self.ops if op.applies(path)]

    def edited(self, path: str) -> bool:
        return bool(self.ops_for(path))

    def anonymized(self, path: str) -> bool:
        return any(isinstance(op, Anonymize) for op in self.ops_for(path))

    def changed_tags(self, path: str) -> set[int]:
        out: set[int] = set()
        for op in self.ops_for(path):
            out |= op.tags
        return out

    def apply(self, ds: Any, path: str) -> Any:
        """Apply the operations for *path* to *ds* (in place); returns *ds*."""
        for op in self.ops_for(path):
            op.apply(ds)
        return ds

    def header(self, path: str) -> Any:
        """*path*'s header as it will be exported (no pixels)."""
        from nvitk.io.dicom_index import read_header

        return self.apply(read_header(path), path)


def _safe(text: str, fallback: str) -> str:
    text = re.sub(r"[^\w\-. ]+", "_", str(text or "")).strip(" ._")
    return text[:60] or fallback


def _series_folder(ds: Any) -> str:
    number = str(ds.get("SeriesNumber", "") or "").strip()
    desc = _safe(str(ds.get("SeriesDescription", "") or ds.get("ProtocolName", "") or ds.get("Modality", "")), "series")
    return f"{int(float(number)):04d}_{desc}" if number.replace(".", "", 1).isdigit() else desc


def export_dicom(
    paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    editor: DicomEditor | None = None,
    layout: str = "series folders",
    names: str = "auto",
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[str]:
    """Write *paths* (with *editor*'s changes) under *out_dir*; returns the files written.

    *layout*: ``series folders`` (``0401_IMR, 83%/``), ``patient / study / series``
    or ``flat``; folder names come from the exported (edited) headers, so an
    anonymized export does not carry the original names. *names*: ``original``,
    ``numbered`` (``IM00001.dcm`` per folder) or ``auto`` (numbered when the files
    are anonymized — original names often hold UIDs or names — else original).
    Pixel data and transfer syntax are kept as they are.
    """
    import pydicom

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    counters: dict[Path, int] = {}
    total = len(paths)
    for k, src in enumerate(paths, 1):
        if cancelled is not None and cancelled():
            break
        src = str(src)
        ds = pydicom.dcmread(src, force=True)
        if editor is not None:
            editor.apply(ds, src)
        if layout == "flat":
            folder = out_root
        elif layout == "patient / study / series":
            patient = _safe(str(ds.get("PatientID", "") or ds.get("PatientName", "")), "patient")
            date = str(ds.get("StudyDate", "") or "")
            study = _safe(f"{date}_{ds.get('StudyDescription', '') or ''}".strip("_"), "study")
            folder = out_root / patient / study / _series_folder(ds)
        else:
            folder = out_root / _series_folder(ds)
        folder.mkdir(parents=True, exist_ok=True)
        anonymized = editor is not None and editor.anonymized(src)
        numbered = names == "numbered" or (names == "auto" and anonymized)
        if numbered:
            counters[folder] = counters.get(folder, 0) + 1
            target = folder / f"IM{counters[folder]:05d}.dcm"
        else:
            target = folder / os.path.basename(src)
            stem, ext = os.path.splitext(target.name)
            n = 1
            while target.exists() or str(target) in written:
                target = folder / f"{stem}_{n}{ext}"
                n += 1
        ds.save_as(str(target), enforce_file_format=False)
        written.append(str(target))
        if progress is not None:
            progress(k, total)
    return written


__all__ = [
    "Anonymize",
    "DATE_MODES",
    "DicomEditor",
    "EXPORT_LAYOUTS",
    "EXPORT_NAMES",
    "TagEdit",
    "coerce_value",
    "export_dicom",
    "parse_tag",
    "tag_vr",
]
