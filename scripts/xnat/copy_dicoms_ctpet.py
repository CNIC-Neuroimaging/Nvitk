#!/usr/bin/env python3
"""Copy each PET-MR subject's CT and PET folders into ``<output>/<subject_uid>/{CT,PET,PET2}``.

Description
-----------
The input root holds one folder per subject named ``<pet_id>-<mr_id>``, e.g.
``BPETX-BMRIX``. ``mr_id`` is the MR id the dataset already knows, and resolves the
``subject_uid`` that names the output folder. ``pet_id`` is new: with ``--write`` it is registered
in ``subject_ids`` under the ``pet_id`` namespace, next to the subject's other ids.

Inside a subject folder the series folders are named inconsistently::

    BPETX-BMRIX/          BPETX-BMRIX/
    ├── BPETXPET               ├── CT
    ├── BPETXPET2              ├── NIFTIS_3
    ├── CT                          ├── NIFTIS_4
    ├── NIFTIS_3                    ├── studyid_…
    └── NIFTIS_4                    └── studyid_…

Roles come from the folder names alone — no file is opened to decide them:

* ``NIFTI*`` folders, and folders with no files, are ignored.
* **CT** — the folder whose name ends in ``CT``.
* **PET / PET2** — every other folder, at most two. A ``…PET`` / ``…PET2`` name pair sets the order;
  otherwise the names are sorted with their numbers compared as numbers and the first is ``PET``.
  For ``studyid_<UID>`` folders that is acquisition order: Philips study UIDs embed the study time
  (``…63779574296…`` is 2022-02-04 12:24, ``…63779580193…`` the same day at 14:03).

A subject that cannot be assigned without guessing — two CT folders, three PET candidates, an
unknown or ambiguous ``mr_id``, two input folders for one subject — is skipped and reported.

Copy semantics
--------------
The input is only read. Every file of a chosen folder is copied, keeping the folder's inner layout
below its new name. A file already at the destination with the same size is left alone, so
re-running resumes an interrupted copy; a same-named file with a different size is a conflict,
reported and not overwritten unless ``--overwrite``. Each file is written under a temporary name and
renamed into place, so an interrupted copy never leaves a truncated file under the real name. File
contents are copied; timestamps and permissions are not.

Subject by subject
------------------
Folder names are resolved up front (names only). Then each subject is listed, assigned and copied
before the next is touched, one directory at a time, so memory holds one directory's file names and
a one-line summary per series — never the whole input. On a network share every file operation is
a round trip, so none is spent outside the copy itself: directories are listed without stat-ing
their files, and a destination directory is listed once rather than probed file by file.

Default mode is dry-run.  Use ``--write`` to copy and register the pet_ids.

Examples::

    # Show what each subject folder maps to, and what would be copied
    python scripts/xnat/copy_dicoms_ctpet.py --input /path/to/ctpet --output /path/to/sorted

    # Copy two subjects and register their pet_ids
    python scripts/xnat/copy_dicoms_ctpet.py --input ... --output ... \
        --folder BPETX-BMRIX --folder BPET109532-BMRIX --write
"""

from __future__ import annotations

import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import pandas as pd

from nvitk.core.logger import Logger
from nvitk.db.storage import utc_now_iso

log = Logger()

#: Fill these in to make them the defaults; until then ``--input`` / ``--output`` are required.
DEFAULT_INPUT = ""
DEFAULT_OUTPUT = ""

SOURCE_BATCH_ID = "copy_dicoms_ctpet"
PET_ID_NAMESPACE = "pet_id"
#: ``subject_ids`` namespaces that hold the MR id (``BMRI…``).
MR_ID_NAMESPACES = ("mri_id", "mr_id", "mrid", "session")
ROLES = ("CT", "PET", "PET2")

#: Two ids, each letters then digits (``BPET000001``), joined by ``-`` or ``_``.
_FOLDER_NAME = re.compile(r"^\s*([A-Za-z]+\d+)\s*[-_]\s*([A-Za-z]+\d+)\s*$")
_MISSING_TEXT = {"", "<na>", "nan", "none", "null"}
_PET_SUFFIX = re.compile(r"PET(\d*)$", re.IGNORECASE)
PARTIAL_SUFFIX = ".partial"
COPY_BUFFER = 1 << 20


# ---------------------------------------------------------------------------
# Subject resolution (folder names only)
# ---------------------------------------------------------------------------
def parse_folder_name(name: str) -> tuple[str, str] | None:
    """
    ``(pet_id, mr_id)`` from a subject folder name, or ``None`` when it is not two ids.

    The order is ``pet_id-mr_id``; if the ids are recognisably swapped (``BMRI…-BPET…``) they are
    put back.

    >>> parse_folder_name("BPET000001-BMRI000002")
    ('BPET000001', 'BMRI000002')
    >>> parse_folder_name("BMRI000002-BPET000001")
    ('BPET000001', 'BMRI000002')
    """
    match = _FOLDER_NAME.match(name)
    if match is None:
        return None
    first, second = (part.upper() for part in match.groups())
    if "MR" in first and "PET" in second:
        first, second = second, first
    return first, second


def mr_to_subject(
    registry: pd.DataFrame, sessions: pd.DataFrame | None
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """
    ``({mr_id: subject_uid}, {mr_id: {subject_uid, …}})`` — the second holds the MR ids that
    resolve to more than one subject, which are left out of the first.
    """
    pairs: dict[str, set[str]] = {}

    def _add(mr_id: Any, subject: Any) -> None:
        if mr_id is None or subject is None or pd.isna(mr_id) or pd.isna(subject):
            return
        key, value = str(mr_id).strip().upper(), str(subject).strip()
        # subject_ids stores some unresolved rows with the literal text "<NA>" as subject_uid.
        if key and value and value.lower() not in _MISSING_TEXT:
            pairs.setdefault(key, set()).add(value)

    if not registry.empty:
        rows = registry.loc[registry["id_namespace"].astype(str).str.lower().isin(MR_ID_NAMESPACES)]
        for mr_id, subject in zip(rows["id_value"], rows["subject_uid"]):
            _add(mr_id, subject)
    if sessions is not None and not sessions.empty and "experiment_label" in sessions.columns:
        for mr_id, subject in zip(sessions["experiment_label"], sessions["subject_uid"]):
            _add(mr_id, subject)

    mapping = {mr_id: next(iter(subjects)) for mr_id, subjects in pairs.items() if len(subjects) == 1}
    ambiguous = {mr_id: subjects for mr_id, subjects in pairs.items() if len(subjects) > 1}
    if not mapping:
        raise ValueError("No MR id -> subject_uid mapping in this dataset (subject_ids / sessions).")
    log.info("Resolved %d MR id -> subject_uid pair(s).", len(mapping))
    return mapping, ambiguous


@dataclass
class Subject:
    """One input subject folder: its ids, and why it would be skipped."""

    folder: Path
    pet_id: str = ""
    mr_id: str = ""
    subject_uid: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def resolve_subjects(
    folders: list[Path], mapping: dict[str, str], ambiguous: dict[str, set[str]]
) -> list[Subject]:
    """Ids and subject_uid of every input folder, from the names alone."""
    subjects: list[Subject] = []
    for folder in folders:
        subject = Subject(folder)
        ids = parse_folder_name(folder.name)
        if ids is None:
            subject.problems.append("folder name is not '<pet_id>-<mr_id>'")
        else:
            subject.pet_id, subject.mr_id = ids
            if subject.mr_id in ambiguous:
                subject.problems.append(
                    f"{subject.mr_id} resolves to several subjects: {', '.join(sorted(ambiguous[subject.mr_id]))}"
                )
            else:
                subject.subject_uid = mapping.get(subject.mr_id, "")
                if not subject.subject_uid:
                    subject.problems.append(f"{subject.mr_id} is not in the dataset")
        subjects.append(subject)

    # Two input folders resolving to one subject would write into one output folder: skip both.
    by_uid: dict[str, list[Subject]] = {}
    for subject in subjects:
        if subject.subject_uid:
            by_uid.setdefault(subject.subject_uid, []).append(subject)
    for uid, group in by_uid.items():
        if len(group) > 1:
            names = ", ".join(s.folder.name for s in group)
            for subject in group:
                subject.problems.append(f"{uid} is also the subject of another folder ({names})")
    return subjects


# ---------------------------------------------------------------------------
# Roles (folder names only)
# ---------------------------------------------------------------------------
def _pet_number(name: str) -> int | None:
    """``…PET`` -> 1, ``…PET2`` -> 2; ``None`` when the name does not end that way."""
    match = _PET_SUFFIX.search(name)
    if match is None:
        return None
    return int(match.group(1) or 1)


def _natural_key(name: str) -> list[Any]:
    """Sort key comparing digit runs as numbers: ``…33.1.9…`` before ``…33.1.10…``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _has_files(path: Path) -> bool:
    """True as soon as one file turns up anywhere under *path* (stops there)."""
    for _, _, files in os.walk(path):
        if files:
            return True
    return False


def assign_roles(subject_folder: Path) -> tuple[dict[str, Path], list[str], list[str], list[str]]:
    """``({role: folder}, ignored, problems, warnings)`` for one subject folder; a problem skips it."""
    ignored: list[str] = []
    problems: list[str] = []
    warnings: list[str] = []
    ct: list[Path] = []
    pet: list[Path] = []
    with os.scandir(subject_folder) as entries:
        children = sorted((Path(e.path) for e in entries if e.is_dir()), key=lambda p: _natural_key(p.name))
    for child in children:
        upper = child.name.strip().upper()
        if "NIFTI" in upper:
            ignored.append(child.name)
        elif not _has_files(child):
            ignored.append(f"{child.name} (empty)")
        elif upper.endswith("CT"):
            ct.append(child)
        else:
            pet.append(child)

    roles: dict[str, Path] = {}
    if len(ct) > 1:
        problems.append(f"{len(ct)} CT folders: {', '.join(p.name for p in ct)}")
    elif ct:
        roles["CT"] = ct[0]
    else:
        warnings.append("no CT folder")

    if len(pet) > 2:
        problems.append(f"{len(pet)} PET candidates: {', '.join(p.name for p in pet)}")
    elif len(pet) == 2:
        numbers = sorted(n for n in (_pet_number(p.name) for p in pet) if n is not None)
        if numbers == [1, 2]:
            first, second = sorted(pet, key=lambda p: _pet_number(p.name) or 0)
        else:
            first, second = pet  # already in natural name order
        roles["PET"], roles["PET2"] = first, second
    elif len(pet) == 1:
        role = "PET2" if _pet_number(pet[0].name) == 2 else "PET"
        roles[role] = pet[0]
        warnings.append(f"only one PET folder ({pet[0].name}), copied as {role}")
    else:
        warnings.append("no PET folder")
    if not roles and not problems:
        problems.append("no CT or PET folder with files")
    return roles, ignored, problems, warnings


# ---------------------------------------------------------------------------
# Copy, one directory at a time
# ---------------------------------------------------------------------------
def _copy_file(job: tuple[str, Path]) -> None:
    """Copy one file under a temporary name and rename it into place."""
    source, target = job
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    try:
        # Plain streams rather than shutil.copyfile, whose same-file and special-file checks cost
        # four stats per file — four network round trips on a share.
        with open(source, "rb") as src, open(partial, "wb") as dst:
            shutil.copyfileobj(src, dst, COPY_BUFFER)
        os.replace(partial, target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _listing(directory: Path) -> dict[str, os.DirEntry]:
    """``{name: entry}`` of the files already in *directory*; empty when it does not exist."""
    try:
        with os.scandir(directory) as entries:
            return {e.name: e for e in entries if not e.name.endswith(PARTIAL_SUFFIX) and e.is_file()}
    except FileNotFoundError:
        return {}


def copy_folder(
    source: Path,
    destination: Path,
    pool: ThreadPoolExecutor,
    *,
    overwrite: bool,
    write: bool,
) -> dict[str, Any]:
    """
    Copy every file under *source* to the same relative place under *destination*.

    Works one directory at a time: list it, list its destination once, copy what is missing in
    parallel, move on. Sizes are only read for names that already exist at the destination.
    """
    counts: dict[str, Any] = {"files": 0, "copied": 0, "present": 0, "conflict": 0, "example": ""}
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames.sort(key=_natural_key)
        if not filenames:
            continue
        relative = os.path.relpath(dirpath, source)
        target_dir = destination if relative == "." else destination / relative
        existing = _listing(target_dir)
        jobs: list[tuple[str, Path]] = []
        for name in sorted(filenames, key=_natural_key):
            counts["files"] += 1
            src = os.path.join(dirpath, name)
            entry = existing.get(name)
            if entry is not None:
                if entry.stat().st_size == os.stat(src).st_size:
                    counts["present"] += 1
                    continue
                if not overwrite:
                    counts["conflict"] += 1
                    counts["example"] = counts["example"] or str(target_dir / name)
                    continue
            jobs.append((src, target_dir / name))
        counts["copied"] += len(jobs)
        if write and jobs:
            target_dir.mkdir(parents=True, exist_ok=True)
            for _ in pool.map(_copy_file, jobs):
                pass
    return counts


# ---------------------------------------------------------------------------
# pet_id registration
# ---------------------------------------------------------------------------
def pet_id_rows(subjects: list[Subject], registry: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """``(subject_ids rows, warnings)`` for the resolved subjects' pet_ids, minus contradictions."""
    existing = (
        registry.loc[registry["id_namespace"].astype(str) == PET_ID_NAMESPACE, ["subject_uid", "id_value"]].astype(str)
        if not registry.empty
        else pd.DataFrame(columns=["subject_uid", "id_value"])
    )
    owner = dict(zip(existing["id_value"], existing["subject_uid"]))
    held: dict[str, set[str]] = {}
    for uid, value in zip(existing["subject_uid"], existing["id_value"]):
        held.setdefault(uid, set()).add(value)

    now = utc_now_iso()
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for subject in subjects:
        if not subject.ok:
            continue
        current = owner.get(subject.pet_id)
        if current is not None and current != subject.subject_uid:
            warnings.append(f"{subject.folder.name}: {subject.pet_id} already belongs to {current}; not registered")
            continue
        others = held.get(subject.subject_uid, set()) - {subject.pet_id}
        if others:
            warnings.append(
                f"{subject.folder.name}: {subject.subject_uid} already has pet_id "
                f"{', '.join(sorted(others))}; adding {subject.pet_id}"
            )
        rows.append(
            {
                "subject_uid": subject.subject_uid,
                "id_namespace": PET_ID_NAMESPACE,
                "id_value": subject.pet_id,
                "id_source": "ctpet_dicom_folder",
                "source_file": subject.folder.name,
                "source_sheet": pd.NA,
                "is_primary": False,
                "source_batch_id": SOURCE_BATCH_ID,
                "updated_at": now,
            }
        )
    return pd.DataFrame(rows), warnings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
REPORT_COLUMNS = (
    "input_folder", "pet_id", "mr_id", "subject_uid", "role", "source",
    "files", "copied", "already_present", "conflicts", "status", "notes",
)


def _subject_folders(input_root: Path, wanted: tuple[str, ...]) -> list[Path]:
    """The subject folders under *input_root* (only *wanted* ones, when given)."""
    with os.scandir(input_root) as entries:
        folders = sorted(Path(e.path) for e in entries if e.is_dir())
    if not wanted:
        return folders
    names = {name.strip() for name in wanted}
    missing = sorted(names - {p.name for p in folders})
    if missing:
        raise click.ClickException(f"Not in {input_root}: {', '.join(missing)}")
    return [p for p in folders if p.name in names]


@click.command("copy-dicoms-ctpet")
@click.option(
    "--input", "input_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_INPUT or None,
    required=not DEFAULT_INPUT,
    help="Root holding one '<pet_id>-<mr_id>' folder per subject. Only read.",
)
@click.option(
    "--output", "output_root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT or None,
    required=not DEFAULT_OUTPUT,
    help="Root to create '<subject_uid>/{CT,PET,PET2}' under.",
)
@click.option("--folder", "folders", multiple=True, help="Only this input subject folder (repeatable).")
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    default=None,
    help="Dataset root. Omit to use the path configured in .nvitk/settings.json.",
)
@click.option("--overwrite", is_flag=True, help="Replace destination files whose size differs from the source.")
@click.option("--workers", default=16, show_default=True, help="Files copied in parallel.")
@click.option("--register/--no-register", default=True, show_default=True, help="Register pet_ids in subject_ids on --write.")
@click.option(
    "--manifest",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Also save the per-series report as CSV (written on --write only).",
)
@click.option(
    "--write/--dry-run",
    default=False,
    show_default="--dry-run",
    help="Actually copy and register; the default only reports.",
)
def main(
    input_root: Path,
    output_root: Path,
    folders: tuple[str, ...],
    dataset: Path | None,
    overwrite: bool,
    workers: int,
    register: bool,
    manifest: Path | None,
    write: bool,
) -> None:
    """Copy CT/PET folders into <output>/<subject_uid>/{CT,PET,PET2} and register pet_ids."""
    from nvitk.pipes.qvtpy.stage9_autoqc import _open_repo

    input_root, output_root = input_root.resolve(), output_root.resolve()
    if output_root == input_root or input_root in output_root.parents:
        raise click.ClickException("--output must not be inside --input: the input is never written to.")

    try:
        repo = _open_repo(dataset)
        registry = repo.get("subject_ids", cohort_id=False)
        registry = registry if registry is not None else pd.DataFrame()
        sessions = repo.get("sessions", cohort_id=False) if repo.catalog.table_exists("sessions") else None
        mapping, ambiguous = mr_to_subject(registry, sessions)
    except (OSError, ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc

    subjects = resolve_subjects(_subject_folders(input_root, folders), mapping, ambiguous)
    resolved = sum(s.ok for s in subjects)
    log.info("%d subject folder(s), %d resolved to a subject.", len(subjects), resolved)

    # The pet_id link comes from the folder name alone, so it is registered before the (long) copy.
    rows, registration_warnings = pet_id_rows(subjects, registry) if register else (pd.DataFrame(), [])
    for warning in registration_warnings:
        log.warning("%s.", warning)
    if write and not rows.empty:
        repo.upsert_table(
            "subject_ids", rows,
            provenance={"importer": SOURCE_BATCH_ID, "rows": len(rows)},
            build_sqlite_index=True,
        )
        log.ok("Registered %d pet_id(s) in subject_ids.", len(rows))

    report: list[dict[str, Any]] = []
    totals = {"copied": 0, "present": 0, "conflict": 0, "done": 0}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, subject in enumerate(subjects, start=1):
            base = {
                "input_folder": subject.folder.name, "pet_id": subject.pet_id,
                "mr_id": subject.mr_id, "subject_uid": subject.subject_uid,
            }
            if subject.ok:
                roles, ignored, problems, warnings = assign_roles(subject.folder)
                subject.problems.extend(problems)
            if not subject.ok:
                log.warning("[%d/%d] %s skipped: %s.", index, len(subjects), subject.folder.name, "; ".join(subject.problems))
                report.append({**base, "status": "skipped: " + "; ".join(subject.problems)})
                continue

            notes = "; ".join(warnings)
            parts = []
            for role in ROLES:
                if role not in roles:
                    continue
                counts = copy_folder(
                    roles[role], output_root / subject.subject_uid / role, pool,
                    overwrite=overwrite, write=write,
                )
                totals["copied"] += counts["copied"]
                totals["present"] += counts["present"]
                totals["conflict"] += counts["conflict"]
                parts.append(f"{role}={roles[role].name} ({counts['copied']}/{counts['files']})")
                report.append({
                    **base, "role": role, "source": roles[role].name, "files": counts["files"],
                    "copied": counts["copied"], "already_present": counts["present"],
                    "conflicts": counts["conflict"],
                    "status": "ok" if not counts["conflict"] else f"conflicts, e.g. {counts['example']}",
                    "notes": notes,
                })
            totals["done"] += 1
            log.info(
                "[%d/%d] %s -> %s: %s%s",
                index, len(subjects), subject.folder.name, subject.subject_uid, ", ".join(parts),
                f" | {notes}" if notes else "",
            )

    frame = pd.DataFrame(report, columns=list(REPORT_COLUMNS))
    counts_columns = ["files", "copied", "already_present", "conflicts"]
    frame[counts_columns] = frame[counts_columns].astype("Int64")
    attention = frame.loc[frame["status"].fillna("").ne("ok") | frame["notes"].fillna("").ne("")]
    if not attention.empty:
        with pd.option_context("display.max_rows", None, "display.width", 250, "display.max_colwidth", 100):
            click.echo("\nNeeds attention:\n" + attention.to_string(index=False))
    if write and manifest is not None:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(manifest, index=False)

    action = "Copied" if write else "Dry run — would copy"
    click.echo(
        f"\n{action} {totals['copied']} file(s) for {totals['done']} of {len(subjects)} subject folder(s) "
        f"({totals['present']} already present, {totals['conflict']} conflict(s), "
        f"{len(subjects) - totals['done']} skipped). "
        f"pet_id rows {'registered' if write else 'to register'}: {len(rows)}."
    )
    if not write:
        click.echo("Re-run with --write to apply.")


if __name__ == "__main__":
    main()
