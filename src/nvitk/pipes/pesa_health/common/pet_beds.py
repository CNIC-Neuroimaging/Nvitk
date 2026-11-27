"""Bed geometry of a step-and-shoot whole-body PET, and the plane below which its
inferior end is too noisy to measure.

A whole-body PET is acquired bed by bed: the scanner's axial field (164 mm on the
PESA Philips scanner) moves in steps shorter than the field (100 mm), so
neighbouring beds overlap. Under a triangular sensitivity profile (1 at the bed
centre, 0 at half a field away) every point between two bed centres collects about
the same counts (0.78 of one bed centre with 164/100 mm), except at both ends of the
scan, where only one bed contributes and the sensitivity falls to 0. Noise scales as
``1/sqrt(sensitivity)``, so the slices furthest below the lowest bed centre are the
noisiest of the whole volume. On the PESA scans those are mid-thigh, right where the
quadriceps SUV masks end.

What the DICOM headers give and what is inferred
(see ``placeholder/Auxiliary/informe_bordes_pet.md`` for the evidence):

* ``FrameReferenceTime (0054,1300)`` is constant within a bed, so its runs along z
  group the slices by bed. Slices are 2 mm, groups are 50 slices (one 100 mm step)
  except the two end groups. [read from the headers]
* The bed centre sits :data:`CENTRE_ABOVE_GROUP_MM` (21 mm) above the first slice of
  its group: it is the middle of the constant run of the Philips per-bed value
  ``(7053,10D5)`` on all five subjects explored. [inferred, to confirm with the
  scanner technologists] When ``(7053,10D5)`` is present, its run centres are used to
  cross-check the leg-bed centre; a disagreement skips the cut.
* The leg (lowest) bed's group starts at the first slice of the volume, which is not
  its real start (the PET is reconstructed only within the CT coverage), so its centre
  is taken one step below the second bed's. [assumed: equal steps]
* ``FieldOfViewDimensions (0018,1149)`` gives the axial field (second value).

:func:`read_pet_beds` reads the geometry from a PET DICOM series and
:func:`save_pet_beds` / :func:`load_pet_beds` keep it in ``PT_beds.json`` next to
``PT.nii.gz`` (stage 0 writes it, stage 2 reads it, since the NIfTI drops these tags).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.logger import Logger

try:
    import pydicom
except ImportError:
    pydicom = None

log = Logger()

PET_BEDS_JSON = "PT_beds.json"

DEFAULT_AXIAL_FOV_MM = 164.0
CENTRE_ABOVE_GROUP_MM = 21.0
CENTRE_CHECK_TOLERANCE_MM = 10.0

_TAG_D5 = (0x7053, 0x10D5)
_HEADER_TAGS = [
    "Modality",
    "SeriesInstanceUID",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "FrameReferenceTime",
    "FieldOfViewDimensions",
    _TAG_D5,
]


@dataclass
class PetBeds:
    """Bed layout of one PET series along the patient z axis (mm, head = +z).

    Groups are ordered by ascending z, so group 0 is the inferior (leg) end.
    """

    series_uid: str
    n_slices: int
    z_first: float
    z_last: float
    slice_spacing: float
    axial_fov: float
    group_starts: list[float]
    group_sizes: list[int]
    step: float
    d5_centres: list[float] | None = None
    notes: list[str] = field(default_factory=list)

    def leg_bed_centre(self, centre_above_group: float = CENTRE_ABOVE_GROUP_MM) -> float:
        """z of the lowest bed's centre: one step below the second group's bed centre."""
        return self.group_starts[1] + centre_above_group - self.step

    def sensitivity(self, z: Any, centre_above_group: float = CENTRE_ABOVE_GROUP_MM) -> np.ndarray:
        """Relative sensitivity at *z*: the sum over beds of the triangular profile."""
        half = 0.5 * self.axial_fov
        c0 = self.leg_bed_centre(centre_above_group)
        centres = c0 + self.step * np.arange(len(self.group_starts))
        z = np.asarray(z, dtype=float)
        return np.clip(1.0 - np.abs(z[..., None] - centres) / half, 0.0, None).sum(-1)

    def overlap_sensitivity(self) -> float:
        """Sensitivity midway between two interior bed centres (0.78 for 164/100 mm)."""
        half = 0.5 * self.axial_fov
        k = np.arange(-8, 9)
        return float(np.clip(1.0 - np.abs(0.5 * self.step + k * self.step) / half, 0.0, None).sum())

    def cut_distance(self, noise_limit: float) -> float:
        """Distance below the leg-bed centre at which the noise reaches *noise_limit* times
        the overlap level (41 mm for 164/100 mm and x1.25)."""
        half = 0.5 * self.axial_fov
        target = self.overlap_sensitivity() / noise_limit**2
        k = np.arange(0, 9)

        def sens(d: float) -> float:
            return float(np.clip(1.0 - (d + k * self.step) / half, 0.0, None).sum())

        lo, hi = 0.0, half
        if sens(lo) <= target:
            return 0.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if sens(mid) > target else (lo, mid)
        return 0.5 * (lo + hi)

    def inferior_cut(
        self, noise_limit: float, centre_above_group: float = CENTRE_ABOVE_GROUP_MM
    ) -> float:
        """z below which the noise exceeds *noise_limit* times the overlap level."""
        return self.leg_bed_centre(centre_above_group) - self.cut_distance(noise_limit)

    def slices_below(self, z: float) -> int:
        """Number of PET slices whose centre lies below *z*."""
        z_slices = self.z_first + self.slice_spacing * np.arange(self.n_slices)
        return int(np.count_nonzero(z_slices < z))

    def slice_boundary(self, n_below: int) -> float:
        """z of the boundary between PET slice ``n_below - 1`` and slice ``n_below``, so a
        cut there removes whole PET slices whichever grid the mask is on."""
        return self.z_first + (n_below - 0.5) * self.slice_spacing


def _is_axial(ds: Any) -> bool:
    """True if *ds*'s slices are axial (row and column directions with no z component)."""
    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is None or len(iop) != 6:
        return False
    return abs(float(iop[2])) < 1e-3 and abs(float(iop[5])) < 1e-3


def _headers(paths: list[Path]) -> list[Any]:
    """Header-only datasets for *paths* (unreadable files skipped)."""
    out = []
    for p in paths:
        try:
            out.append(pydicom.dcmread(str(p), stop_before_pixels=True, force=True, specific_tags=_HEADER_TAGS))
        except Exception:
            continue
    return out


def _pet_files(dicom_dir: Path) -> list[Path]:
    """PET DICOM files of a subject: ``<dicom_dir>/PET`` when present, else every folder
    under *dicom_dir* whose first readable file has ``Modality == 'PT'``."""
    pet_dir = dicom_dir / "PET"
    if pet_dir.is_dir():
        return sorted(p for p in pet_dir.rglob("*") if p.is_file())
    found: list[Path] = []
    for folder in sorted({p.parent for p in dicom_dir.rglob("*") if p.is_file()}):
        files = sorted(p for p in folder.iterdir() if p.is_file())
        first = _headers(files[:1])
        if first and str(getattr(first[0], "Modality", "")) == "PT":
            found.extend(files)
    return found


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    """``(first, last)`` index pairs of the constant runs of *values*."""
    runs, start = [], 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            runs.append((start, i - 1))
            start = i
    return runs


def pet_beds_from_headers(headers: list[Any]) -> PetBeds | None:
    """Bed geometry from the header datasets of one PET series, or ``None`` (with a
    warning) when the series does not look like a step-and-shoot whole-body PET."""
    headers = [d for d in headers if hasattr(d, "ImagePositionPatient")]
    if not headers:
        log.warning("PET beds: no slice with ImagePositionPatient.")
        return None
    by_series: dict[str, list[Any]] = {}
    for d in headers:
        by_series.setdefault(str(getattr(d, "SeriesInstanceUID", "")), []).append(d)
    uid, ds = max(by_series.items(), key=lambda kv: len(kv[1]))
    if len(by_series) > 1:
        log.warning(f"PET beds: {len(by_series)} PET series found; using the largest ({len(ds)} slices).")
    if not all(_is_axial(d) for d in ds):
        log.warning("PET beds: slices are not axial; no bed geometry.")
        return None
    if not all(hasattr(d, "FrameReferenceTime") for d in ds):
        log.warning("PET beds: FrameReferenceTime (0054,1300) missing; no bed geometry.")
        return None

    ds = sorted(ds, key=lambda d: float(d.ImagePositionPatient[2]))
    z = np.array([float(d.ImagePositionPatient[2]) for d in ds])
    frt = np.array([float(d.FrameReferenceTime) for d in ds])
    spacing = float(np.median(np.diff(z))) if len(z) > 1 else 0.0
    groups = _runs(frt)
    if len(groups) < 3:
        log.warning(f"PET beds: {len(groups)} FrameReferenceTime group(s); need at least 3 beds.")
        return None
    if len(set(frt[a] for a, _ in groups)) != len(groups):
        log.warning("PET beds: a FrameReferenceTime value repeats in separate runs; slices are not grouped by bed.")
        return None

    starts = [float(z[a]) for a, _ in groups]
    sizes = [int(b - a + 1) for a, b in groups]
    steps = np.diff(starts[1:])
    step = float(np.median(steps)) if steps.size else float(starts[1] - starts[0])
    if steps.size and float(np.max(np.abs(steps - step))) > 1.01 * spacing:
        log.warning(f"PET beds: uneven bed steps {np.round(steps, 1).tolist()} mm; no bed geometry.")
        return None

    fov = DEFAULT_AXIAL_FOV_MM
    fovd = getattr(ds[0], "FieldOfViewDimensions", None)
    if fovd is not None and len(fovd) >= 2 and float(fovd[1]) > 0:
        fov = float(fovd[1])
    if not 0.5 * fov < step < fov:
        log.warning(f"PET beds: step {step:.1f} mm does not overlap a {fov:.0f} mm axial field; no bed geometry.")
        return None

    beds = PetBeds(
        series_uid=uid,
        n_slices=len(ds),
        z_first=float(z[0]),
        z_last=float(z[-1]),
        slice_spacing=spacing,
        axial_fov=fov,
        group_starts=starts,
        group_sizes=sizes,
        step=step,
    )

    if all(_TAG_D5 in d for d in ds):
        d5 = np.array([float(d[_TAG_D5].value) for d in ds])
        plateaus = [(a, b) for a, b in _runs(d5) if b > a]
        if len(plateaus) == len(groups):
            beds.d5_centres = [0.5 * float(z[a] + z[b]) for a, b in plateaus]
    return beds


def read_pet_beds(dicom_dir: str | Path) -> PetBeds | None:
    """Bed geometry of the PET series under a subject's DICOM directory (see :func:`_pet_files`)."""
    if pydicom is None:
        log.warning("PET beds: pydicom is not installed.")
        return None
    files = _pet_files(Path(dicom_dir))
    if not files:
        log.warning(f"PET beds: no PET DICOM under {dicom_dir}.")
        return None
    headers = [d for d in _headers(files) if str(getattr(d, "Modality", "PT")) == "PT"]
    return pet_beds_from_headers(headers)


def check_leg_centre(beds: PetBeds, centre_above_group: float = CENTRE_ABOVE_GROUP_MM) -> str | None:
    """Why the leg-bed centre is not trusted, or ``None``.

    The centre comes from the FrameReferenceTime groups; when the Philips per-bed value
    ``(7053,10D5)`` gives plateau centres, the second bed's must match within
    :data:`CENTRE_CHECK_TOLERANCE_MM`.
    """
    if beds.d5_centres is None or len(beds.d5_centres) < 2:
        return None
    from_groups = beds.group_starts[1] + centre_above_group
    diff = beds.d5_centres[1] - from_groups
    if abs(diff) > CENTRE_CHECK_TOLERANCE_MM:
        return (
            f"bed centre from FrameReferenceTime ({from_groups:.1f} mm) and from (7053,10D5) "
            f"({beds.d5_centres[1]:.1f} mm) differ by {diff:+.1f} mm"
        )
    return None


def save_pet_beds(beds: PetBeds, nifti_dir: str | Path) -> Path:
    """Write *beds* to ``<nifti_dir>/PT_beds.json``."""
    path = Path(nifti_dir) / PET_BEDS_JSON
    path.write_text(json.dumps(asdict(beds), indent=2))
    return path


def load_pet_beds(nifti_dir: str | Path) -> PetBeds | None:
    """``<nifti_dir>/PT_beds.json`` as :class:`PetBeds`, or ``None`` if absent."""
    path = Path(nifti_dir) / PET_BEDS_JSON
    if not path.exists():
        return None
    return PetBeds(**json.loads(path.read_text()))


def ensure_pet_beds(nifti_dir: str | Path, dicom_dir: str | Path | None) -> PetBeds | None:
    """Bed geometry from ``PT_beds.json``, else read from *dicom_dir* and cached there
    (a read-only NIfTI folder only loses the cache)."""
    beds = load_pet_beds(nifti_dir)
    if beds is not None or dicom_dir is None:
        return beds
    try:
        if not Path(dicom_dir).is_dir():
            return None
        beds = read_pet_beds(dicom_dir)
    except OSError as exc:
        log.warning(f"PET beds: could not read the PET DICOM under {dicom_dir}: {exc}")
        return None
    if beds is not None:
        try:
            save_pet_beds(beds, nifti_dir)
        except OSError as exc:
            log.warning(f"PET beds: could not write {PET_BEDS_JSON} in {nifti_dir}: {exc}")
    return beds


__all__ = [
    "CENTRE_ABOVE_GROUP_MM",
    "DEFAULT_AXIAL_FOV_MM",
    "PET_BEDS_JSON",
    "PetBeds",
    "check_leg_centre",
    "ensure_pet_beds",
    "load_pet_beds",
    "pet_beds_from_headers",
    "read_pet_beds",
    "save_pet_beds",
]
