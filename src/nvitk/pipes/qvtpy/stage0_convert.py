"""qvtpy stage 0 (convert): DICOM → NIfTI conversion and reorganization.

**Inputs**

- Per-subject DICOM tree under ``<dicom_root>/<subject>/`` (from stage 0 download).

**Outputs**

- ``<nifti_root>/<subject>/4DFlow/{AP,RL,FH}/`` — ``*_m.nii.gz`` / ``*_ph.nii.gz`` + JSON sidecars.
- ``<nifti_root>/<subject>/TOF/TOF.nii.gz``.
- Optional derived volumes via :func:`nvitk.io.conversors.phase2volume.phase2volume`
  (``Angiography_3D``, ``ComplexDifference_3D/4D``, …) when ``--compute-phase-derived``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable

import click

from nvitk.core.click_backend import backend_click_option
from nvitk.core.logger import Logger
from nvitk.io.conversors.dcm2nii import dcm2nii
from nvitk.io.conversors.phase2volume import phase2volume

from . import config as cfg

log = Logger()

# ---------------------------------------------------------------------------
# Constants (expected layout / derivatives)
# ---------------------------------------------------------------------------

REQUIRED_FLOW_DIRS: tuple[str, ...] = ("AP", "RL", "FH")
DERIVED_FILES: tuple[str, ...] = (
    "Angiography_3D",
    "Angiography_4D",
    "ComplexDifference_3D",
    "ComplexDifference_4D",
    "VelocityMagnitude_3D",
    "VelocityMagnitude_4D",
    "VelocityMeanComponents",
)

# ---------------------------------------------------------------------------
# DICOM stem classification + NIfTI export
# ---------------------------------------------------------------------------


def _iter_subjects(dicom_root: Path) -> list[str]:
    """Subject ids: one folder per subject under *dicom_root*."""
    if not dicom_root.exists():
        return []
    return sorted(p.name for p in dicom_root.iterdir() if p.is_dir())


def _iter_nifti(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    return sorted([*folder.glob("*.nii"), *folder.glob("*.nii.gz")])


def _nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    if name.endswith(".nii"):
        return name[: -len(".nii")]
    return path.stem


def _flow_direction(stem: str) -> str | None:
    """Infer 4DFlow encoding direction from the dcm2nii series stem.

    Supports two naming styles seen in the field:

    - **Embedded** (Philips-style): ``..._AP_...``, ``..._RL_...``, ``..._FH_...``
    - **Prefix** (some sites): ``AP_<series>_...``, ``RL_<series>_...``, ``FH_<series>_...``

    Embedded tokens are checked first so mixed strings still resolve predictably.
    Prefix detection is strict (``AP_`` at start, etc.) to avoid substring hits
    like ``ref DTI AP-P`` in unrelated series descriptions.
    """
    u = stem.upper()
    if "_AP_" in u:
        return "AP"
    if "_RL_" in u:
        return "RL"
    if "_FH_" in u:
        return "FH"
    if u.startswith("AP_"):
        return "AP"
    if u.startswith("RL_"):
        return "RL"
    if u.startswith("FH_"):
        return "FH"
    return None


def _classify_flow_stem(stem: str) -> tuple[str, str] | None:
    """Return (AP|RL|FH, 'm'|'ph') for 4DFlow magnitude / phase stems, else None."""
    direction = _flow_direction(stem)
    if direction is None:
        return None
    su = stem.upper()
    if su.endswith("_M_FFE"):
        return (direction, "m")
    if su.endswith("_PHASE"):
        return (direction, "ph")
    return None


def _flow_dest_nifti_name(stem: str, kind: str) -> str:
    su = stem.upper()
    if kind == "m":
        if not su.endswith("_M_FFE"):
            raise ValueError(stem)
        base = stem[: len(stem) - len("_M_FFE")]
    else:
        if not su.endswith("_PHASE"):
            raise ValueError(stem)
        base = stem[: len(stem) - len("_PHASE")]
    return f"{base}_{kind}.nii.gz"


def _is_tof_stem(stem: str) -> bool:
    """Heuristic for TOF / angio series (not 4DFlow)."""
    if _classify_flow_stem(stem) is not None:
        return False
    u = stem.upper()
    return any(
        k in u
        for k in (
            "TOF",
            "MRA",
            "ANGIO",
            "TOF3D",
            "CS3DI",
            "3DI",
        )
    )


def _export_nifti_move(src: Path, dst: Path) -> None:
    """Write *dst* as a NIfTI file (compression inferred from extension, e.g. ``.nii.gz``),
    then remove *src* if different path."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        log.warning(f"Replacing existing file: {dst}")
        dst.unlink()
    try:
        import nibabel as nib

        img = nib.load(str(src))
        nib.save(img, str(dst))
    except Exception as exc:
        log.exception(f"Failed to convert NIfTI {src} -> {dst}: {exc}")
        raise RuntimeError(f"Failed to convert NIfTI {src} -> {dst}: {exc}") from exc
    if src.resolve() != dst.resolve():
        src.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Per-subject convert + reorganize + phase2volume
# ---------------------------------------------------------------------------


def convert_subject(
    subject_dicom_dir: Path,
    subject_out_dir: Path,
    *,
    skip_existing: bool = False,
) -> None:
    """Run DICOM→NIfTI conversion (dcm2nii) into a flat subject output folder."""
    subject_out_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and any(_iter_nifti(subject_out_dir)):
        log.info(f"[{subject_dicom_dir.name}] stage0: skipping existing conversion")
        return

    dcm2nii(
        str(subject_dicom_dir),
        str(subject_out_dir),
        custom_naming="AccessionNumber_SeriesDescription_SeriesNumber",
        rescale_type="FP",
        force_ras=True,
        compress=True,
        save_metadata=True,
    )


def reorganize_subject(subject_out_dir: Path) -> None:
    """Move flat dcm2nii outputs into ``4DFlow/{AP,RL,FH}/`` and ``TOF/`` (QVT naming)."""
    flow_root = subject_out_dir / "4DFlow"
    tof_dir = subject_out_dir / "TOF"
    for sub in ("AP", "RL", "FH"):
        (flow_root / sub).mkdir(parents=True, exist_ok=True)
    tof_dir.mkdir(parents=True, exist_ok=True)

    top_files = sorted(p for p in subject_out_dir.iterdir() if p.is_file())
    niftis = [p for p in top_files if p.name.endswith(".nii.gz") or p.name.endswith(".nii")]
    jsons = [p for p in top_files if p.suffix == ".json"]

    for src in niftis:
        stem = _nifti_stem(src)
        flow = _classify_flow_stem(stem)
        if flow is not None:
            direction, kind = flow
            dest_name = _flow_dest_nifti_name(stem, kind)
            dest = flow_root / direction / dest_name
            _export_nifti_move(src, dest)
            continue

        if _is_tof_stem(stem):
            dest = tof_dir / "TOF.nii.gz"
            if dest.exists() and src.resolve() != dest.resolve():
                log.warning(f"Multiple TOF-like series; overwriting {dest} with {src.name}")
            _export_nifti_move(src, dest)
            continue

        log.warning(f"[{subject_out_dir.name}] leaving unclassified NIfTI at subject root: {src.name}")

    for js in jsons:
        stem = js.stem
        flow = _classify_flow_stem(stem)
        if flow is not None:
            direction, _kind = flow
            dest = flow_root / direction / js.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                log.warning(f"Replacing existing JSON: {dest}")
                dest.unlink()
            shutil.move(str(js), str(dest))
            continue

        if _is_tof_stem(stem):
            dest = tof_dir / js.name
            if dest.exists():
                log.warning(f"Replacing existing JSON: {dest}")
                dest.unlink()
            shutil.move(str(js), str(dest))
            continue

        log.warning(f"[{subject_out_dir.name}] leaving unclassified JSON at subject root: {js.name}")


# ---------------------------------------------------------------------------
# Stage 0 convert entrypoint
# ---------------------------------------------------------------------------


def run_subject(
    subject: str,
    *,
    dicom_root: Path,
    nifti_root: Path,
    compute_phase_derived: bool = True,
    skip_existing: bool = False,
    phase_background_correction: bool = True,
    phase_bg_poly_order: int = 2,
    phase_bg_static_percentile: float = 25.0,
) -> Path:
    """Convert + reorganize for one subject. Returns the subject NIfTI folder."""
    subj_dicom = dicom_root / subject
    if not subj_dicom.exists():
        raise FileNotFoundError(f"DICOM subject dir not found: {subj_dicom}")

    subj_nifti = nifti_root / subject
    log.info(f"qvtpy stage0 | subject={subject}")
    log.info(f"  dicom: {subj_dicom}")
    log.info(f"  nifti: {subj_nifti}")

    convert_subject(subj_dicom, subj_nifti, skip_existing=skip_existing)
    reorganize_subject(subj_nifti)

    if compute_phase_derived:
        try:
            phase2volume(
                subj_nifti,
                multifile=False,
                dicom_search_dir=subj_dicom,
                background_phase_correction=phase_background_correction,
                bg_poly_order=phase_bg_poly_order,
                bg_static_percentile=phase_bg_static_percentile,
            )
        except Exception as exc:
            log.exception(f"[{subject}] phase2volume failed: {exc}")

    return subj_nifti


def _glob_first(directory: Path, *patterns: str) -> Path | None:
    """Return the first sorted glob hit under *directory* across *patterns*, else None."""
    if not directory.is_dir():
        return None
    for pat in patterns:
        hits = sorted(directory.glob(pat))
        if hits:
            return hits[0]
    return None


def _iter_subjects_nifti(nifti_root: Path) -> list[str]:
    """Subject ids found under *nifti_root* (one folder per subject)."""
    if not nifti_root.exists():
        return []
    return sorted(p.name for p in nifti_root.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# QC report
# ---------------------------------------------------------------------------


def print_nifti_qc_report(
    nifti_root: str | Path,
    subjects: Iterable[str],
    *,
    check_derived: bool = False,
) -> dict[str, Any]:
    """Print a brief QC report of the stage0_convert NIfTI layout.

    For every subject id under *nifti_root* (each ``{nifti_root}/{subject}``), check:

    - ``4DFlow/{AP,RL,FH}/`` exists and contains both ``*_m.nii*`` and ``*_ph.nii*``
    - ``TOF/TOF.nii*`` is present

    When ``check_derived=True``, additionally list per-subject missing optional
    derived images under ``4DFlow/`` (informational only — derived misses do not
    flag a subject as incomplete).

    Returns a summary dict with ``root``, ``complete``, ``incomplete``,
    ``derived_missing`` and ``total`` keys.
    """
    root = Path(nifti_root)
    subj_list = sorted({s for s in subjects if s})

    complete: list[str] = []
    incomplete: dict[str, list[str]] = {}
    derived_missing: dict[str, list[str]] = {}

    for subj in subj_list:
        subj_dir = root / subj
        missing: list[str] = []

        flow_root = subj_dir / "4DFlow"
        for d in REQUIRED_FLOW_DIRS:
            dd = flow_root / d
            if not dd.is_dir():
                missing.extend([f"{d}_m[missing dir]", f"{d}_ph[missing dir]"])
                continue
            if _glob_first(dd, "*_m.nii.gz", "*_m.nii") is None:
                missing.append(f"{d}_m[missing]")
            if _glob_first(dd, "*_ph.nii.gz", "*_ph.nii") is None:
                missing.append(f"{d}_ph[missing]")

        tof_dir = subj_dir / "TOF"
        if not tof_dir.is_dir():
            missing.append("TOF[missing dir]")
        elif _glob_first(tof_dir, "TOF.nii.gz", "TOF.nii") is None:
            missing.append("TOF[missing]")

        if check_derived:
            d_missing: list[str] = []
            for f in DERIVED_FILES:
                if not ((flow_root / f"{f}.nii.gz").is_file() or (flow_root / f"{f}.nii").is_file()):
                    d_missing.append(f"{f}[missing]")
            if d_missing:
                derived_missing[subj] = d_missing

        if missing:
            incomplete[subj] = missing
        else:
            complete.append(subj)

    total = len(subj_list)

    print("=" * 60)
    print("NIfTI completeness report")
    print(f"  root          : {root}")
    print(
        "  required      : "
        + " ".join(f"{d}_m+ph" for d in REQUIRED_FLOW_DIRS)
        + " TOF.nii*"
    )
    if check_derived:
        print(f"  derived check : {' '.join(DERIVED_FILES)}")
    print("=" * 60)
    print(f"Subjects scanned : {total}")
    print(f"Complete         : {len(complete)}")
    print(f"Incomplete       : {len(incomplete)}")
    print()

    if incomplete:
        print("-- Incomplete subjects (missing required) --")
        for subj in sorted(incomplete):
            print(f"  {subj:<20s}  ->  {' '.join(incomplete[subj])}")
        print()

    if complete:
        print("-- Complete subjects --")
        for subj in complete:
            print(f"  {subj}")
        print()

    if check_derived and derived_missing:
        print("-- Derived (optional) missing --")
        for subj in sorted(derived_missing):
            print(f"  {subj:<20s}  ->  {' '.join(derived_missing[subj])}")
        print()

    return {
        "root": str(root),
        "complete": complete,
        "incomplete": incomplete,
        "derived_missing": derived_missing if check_derived else {},
        "total": total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("qvtpy-stage0")
@backend_click_option()
@click.option("--dicom-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_DICOM_ROOT)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=cfg.DEFAULT_NIFTI_ROOT)
@click.option(
    "--subject",
    default=None,
    help="Single subject id. If omitted, all subfolders of --dicom-root are processed.",
)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--compute-phase-derived", is_flag=True, default=False)
@click.option(
    "--phase-background-correction/--no-phase-background-correction",
    "phase_background_correction",
    is_flag=True,
    default=True,
    show_default=True,
    help="Polynomial spatial background on PC velocity (QVTplus-style; default on).",
)
@click.option("--phase-bg-poly-order", type=int, default=2, show_default=True)
@click.option("--phase-bg-static-percentile", type=float, default=25.0, show_default=True)
@click.option(
    "--report",
    is_flag=True,
    default=False,
    help="After conversion, print a NIfTI completeness report against --nifti-root.",
)
@click.option(
    "--report-derived",
    is_flag=True,
    default=False,
    help="When combined with --report, also list missing optional derived 4DFlow images.",
)
def main(
    dicom_root: Path,
    nifti_root: Path,
    subject: str | None,
    skip_existing: bool,
    compute_phase_derived: bool,
    phase_background_correction: bool,
    phase_bg_poly_order: int,
    phase_bg_static_percentile: float,
    report: bool,
    report_derived: bool,
) -> None:
    Logger()
    if subject:
        run_subject(
            subject,
            dicom_root=dicom_root,
            nifti_root=nifti_root,
            compute_phase_derived=compute_phase_derived,
            skip_existing=skip_existing,
            phase_background_correction=phase_background_correction,
            phase_bg_poly_order=phase_bg_poly_order,
            phase_bg_static_percentile=phase_bg_static_percentile,
        )
        report_subjects = [subject]
    else:
        subjects = _iter_subjects(dicom_root)
        if not subjects:
            if report:
                report_subjects = _iter_subjects_nifti(nifti_root)
                if not report_subjects:
                    raise click.ClickException(
                        f"No subject folders found under dicom_root={dicom_root} "
                        f"or nifti_root={nifti_root}."
                    )
            else:
                raise click.ClickException(
                    f"No subject folders found under dicom_root={dicom_root}. "
                    "Pass --subject or point --dicom-root at a directory of subject subfolders."
                )
        else:
            for subj in subjects:
                run_subject(
                    subj,
                    dicom_root=dicom_root,
                    nifti_root=nifti_root,
                    compute_phase_derived=compute_phase_derived,
                    skip_existing=skip_existing,
                    phase_background_correction=phase_background_correction,
                    phase_bg_poly_order=phase_bg_poly_order,
                    phase_bg_static_percentile=phase_bg_static_percentile,
                )
            report_subjects = subjects

    if report:
        print_nifti_qc_report(nifti_root, report_subjects, check_derived=report_derived)


__all__ = [
    "REQUIRED_FLOW_DIRS",
    "DERIVED_FILES",
    "run_subject",
    "print_nifti_qc_report",
    "main",
]


if __name__ == "__main__":
    main()
