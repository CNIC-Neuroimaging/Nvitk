#!/usr/bin/env python3
"""Copy each subject's main CT — found among its PET2 DICOM files — into ``<output>/<subject_uid>/CT``.

Description
-----------
The input is the output of ``copy_dicoms_ctpet.py``: one ``<subject_uid>`` folder per subject with
``CT``, ``PET`` and ``PET2`` inside. The CT wanted here is not the one in ``CT``: it is the CT of
the second PET study, stored in ``PET2`` next to the PET series, the localizers (projections) and
the dose reports. Only that CT is copied; everything else is left aside. A subject with no
``PET2`` folder — a single PET study — is searched in ``PET`` instead (``--source-folder`` sets
the folders and their order; the first one that exists is used, a later one is not tried when an
earlier one exists but holds no iDose CT).

Selection
---------
Every file of the source folder has its header read — only up to the series tags, never the pixel
data (well under 1 KB per file) — and the file is kept when:

* its ``SeriesDescription`` contains ``iDose`` (any case; ``--series-pattern`` changes it),
* its ``Modality`` is ``CT``, and
* its ``ImageType`` is not ``LOCALIZER``.

The last two are guards: a projection or a report whose description also says iDose is left out,
and counted in the report. The kept files must form one series (one ``SeriesInstanceUID``). When
several iDose series match, they are ranked by :data:`PREFERENCES` (``--prefer``), wildcard patterns
on the description tried in order, case-insensitive:

1. ``*CT-AC Image, iDose*``
2. ``*Head-Low Dose CT, iDose*``
3. ``*FUSION, iDose*``

The first pattern that matches exactly one series picks it. When none matches — or the first that
matches fits several series — ``--on-multiple`` decides among those: ``largest`` takes the series
with the most files (skipping on a tie), ``skip`` skips the subject. A subject with no source
folder, or no iDose CT in it, is skipped. Skipped subjects are listed with their candidates.

Target slice count (``--target-slices``)
----------------------------------------
Off by default. ``--target-slices`` alone means 82; ``--target-slices N`` sets another count. When on:

* A subject whose output ``CT`` already holds between 1 and N slices is skipped without being read.
* When the selected CT has more than N slices, an iDose CT of exactly N slices is looked for in the
  source folders in order — ``PET2``, then ``PET`` — and taken instead (preferences break ties
  within a folder). If neither has one, the originally selected CT is copied.
* If the output ``CT`` holds the bigger CT from an earlier run, it is replaced by the N-slice one:
  its files are removed once the new CT is fully in place.

Report
------
With ``--write``, ``<output>/ct_scans.csv`` (``--report``) gets one row per subject: the source
folder, the series copied (number, description, UID), its number of slices, why it was chosen, the
other candidates, the copy counts and the status — or why the subject was skipped. A re-run updates
the rows of the subjects it processed and keeps the others.

Copy semantics
--------------
The input is only read. The series is copied flat into ``<output>/<subject_uid>/CT`` under its own
file names; a name that occurs in two subfolders gets its subfolder path as a prefix instead
(``sub__IM_0001``). A new CT is copied into ``CT.partial`` and renamed to ``CT`` once complete, so a
``CT`` folder is never half-copied and an interrupted copy resumes from ``CT.partial``. Files are
copied under a temporary name and renamed into place. File contents are copied; timestamps and
permissions are not.

A ``CT`` that already holds the chosen series is topped up: files with the same name and size are
left alone, a same-named file with a different size is a conflict unless ``--overwrite``. A ``CT``
that holds another series (told by the SeriesInstanceUID of its first file) is never mixed with the
new one: it is a conflict unless ``--overwrite`` — or the ``--target-slices`` switch — replaces it,
in which case the new CT is staged in full before the old folder is removed.

Subject by subject
------------------
Each subject is read and copied before the next is touched, one directory at a time; memory holds
one directory's file names and the selected series' paths. Headers are read in parallel
(``--workers``) because on a network share each read is a round trip.

This script needs no dataset access: the input folders are already named by ``subject_uid``.

Default mode is dry-run.  Use ``--write`` to copy.

Examples::

    # Show which series would be taken for each subject
    python scripts/xnat/copy_dicoms_ct.py --input /path/to/sorted --output /path/to/ct

    # Copy two subjects
    python scripts/xnat/copy_dicoms_ct.py --input ... --output ... --subject PESAX --subject PESAY --write
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import click
import pandas as pd
from pydicom.filereader import read_partial
from pydicom.tag import Tag

# Plain logging rather than nvitk's Logger: importing the nvitk package loads its imaging/GPU stack
# (~700 MB), which a file copy does not need.
logging.basicConfig(format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("copy_dicoms_ct")

DEFAULT_INPUT = ""
DEFAULT_OUTPUT = ""

#: Subject subfolders searched for the CT, in order; the first that exists is used.
SOURCE_FOLDERS = ("PET2", "PET")
TARGET_FOLDER = "CT"
SERIES_PATTERN = "iDose"
#: Which iDose series wins when there are several: wildcard patterns on SeriesDescription, in order.
PREFERENCES = ("*CT-AC Image, iDose*", "*Head-Low Dose CT, iDose*", "*FUSION, iDose*")
REPORT_NAME = "ct_scans.csv"
#: Slice count ``--target-slices`` looks for when given without a value.
TARGET_SLICES = 82
PARTIAL_SUFFIX = ".partial"
COPY_BUFFER = 1 << 20
#: Read size for a header: enough for the tags below in one request.
HEADER_BUFFER = 16 * 1024
#: Header parsing stops after SeriesNumber (0020,0011), the last tag the selection uses.
_LAST_TAG = Tag(0x0020, 0x0011)


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Header:
    """The tags the selection needs from one file."""

    modality: str
    description: str
    image_type: tuple[str, ...]
    series_uid: str
    series_number: str


def _past_series_tags(tag: Any, vr: Any, length: int) -> bool:
    return tag > _LAST_TAG


def read_header(path: str) -> Header | None:
    """The selection tags of *path*, read up to SeriesNumber only; ``None`` when unreadable."""
    try:
        with open(path, "rb", buffering=HEADER_BUFFER) as handle:
            ds = read_partial(handle, stop_when=_past_series_tags, force=True)
    except Exception:  # not DICOM, truncated, permission — reported as unreadable
        return None
    image_type = ds.get("ImageType", ())
    if isinstance(image_type, str):
        image_type = (image_type,)
    return Header(
        modality=str(ds.get("Modality", "") or "").strip().upper(),
        description=str(ds.get("SeriesDescription", "") or "").strip(),
        image_type=tuple(str(part).strip().upper() for part in image_type),
        series_uid=str(ds.get("SeriesInstanceUID", "") or "").strip(),
        series_number=str(ds.get("SeriesNumber", "") or "").strip(),
    )


def classify(header: Header | None, pattern: str) -> str:
    """
    ``match`` for a file of the wanted CT, otherwise why not.

    >>> from dataclasses import replace
    >>> h = Header("CT", "CT 3.0 iDose(4)", ("ORIGINAL", "PRIMARY", "AXIAL"), "1.2", "3")
    >>> classify(h, "idose"), classify(replace(h, modality="SR"), "idose")
    ('match', 'pattern_not_ct')
    >>> classify(replace(h, image_type=("ORIGINAL", "PRIMARY", "LOCALIZER")), "idose")
    'pattern_localizer'
    """
    if header is None:
        return "unreadable"
    if pattern not in header.description.lower():
        return "other_series"
    if header.modality != "CT":
        return "pattern_not_ct"
    if "LOCALIZER" in header.image_type:
        return "pattern_localizer"
    return "match"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def _natural_key(name: str) -> list[Any]:
    """Sort key comparing digit runs as numbers: ``IM_9`` before ``IM_10``."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


@dataclass
class Series:
    """One matching series: its description and the paths of its files."""

    uid: str
    number: str
    description: str
    files: list[str] = field(default_factory=list)

    def label(self) -> str:
        return f"#{self.number or '?'} '{self.description}' ({len(self.files)} files)"


def scan_source(source: Path, pool: ThreadPoolExecutor, pattern: str) -> tuple[dict[str, Series], Counter]:
    """
    ``({series_uid: Series}, {class: count})`` for every file under *source*.

    One directory at a time: list it, read its headers in parallel, keep only the matching paths.
    """
    series: dict[str, Series] = {}
    counts: Counter = Counter()
    needle = pattern.lower()
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames.sort(key=_natural_key)
        paths = [
            os.path.join(dirpath, name)
            for name in sorted(filenames, key=_natural_key)
            if not name.endswith(PARTIAL_SUFFIX)
        ]
        for path, header in zip(paths, pool.map(read_header, paths)):
            kind = classify(header, needle)
            counts[kind] += 1
            if kind != "match":
                continue
            hit = series.get(header.series_uid)
            if hit is None:
                hit = series[header.series_uid] = Series(header.series_uid, header.series_number, header.description)
            hit.files.append(path)
    return series, counts


def choose_series(
    series: dict[str, Series], preferences: tuple[str, ...], on_multiple: str
) -> tuple[Series | None, str]:
    """
    ``(series, reason)`` — the series to copy and why it was chosen, or ``None`` and why not.

    >>> ac, head = Series("1", "201", "CT-AC Image, iDose (4)"), Series("2", "401", "Body-Low Dose CT, iDose (4)")
    >>> choose_series({"1": ac, "2": head}, PREFERENCES, "skip")
    (Series(uid='1', number='201', description='CT-AC Image, iDose (4)', files=[]), 'preference *CT-AC Image, iDose*')
    """
    if not series:
        return None, "no iDose CT series"
    if len(series) == 1:
        return next(iter(series.values())), "only iDose series"

    candidates = list(series.values())
    for pattern in preferences:
        hits = [s for s in candidates if fnmatch.fnmatchcase(s.description.lower(), pattern.lower())]
        if len(hits) == 1:
            return hits[0], f"preference {pattern}"
        if hits:  # the preferred kind exists more than once: decide among those alone
            candidates = hits
            break

    ranked = sorted(candidates, key=lambda s: len(s.files), reverse=True)
    listing = "; ".join(s.label() for s in ranked)
    if on_multiple == "largest":
        if len(ranked[0].files) == len(ranked[1].files):
            return None, f"several iDose series, no preference and none largest: {listing}"
        return ranked[0], "largest"
    return None, f"several iDose series, no preference (--on-multiple largest takes the biggest): {listing}"


def find_target_series(
    subject_folder: Path,
    source_folders: tuple[str, ...],
    scanned: dict[str, dict[str, Series]],
    slices: int,
    pool: ThreadPoolExecutor,
    *,
    pattern: str,
    preferences: tuple[str, ...],
    on_multiple: str,
) -> tuple[Series, Path, str, dict[str, Series]] | None:
    """
    The first iDose CT of exactly *slices* files in the source folders, taken in order.

    Returns ``(series, folder, reason, every series of that folder)``, or ``None``. *scanned* caches
    the folders already read (``{folder name: series}``) so none is read twice.
    """
    for name in source_folders:
        path = subject_folder / name
        if not path.is_dir():
            continue
        if name not in scanned:
            scanned[name] = scan_source(path, pool, pattern)[0]
        exact = {uid: s for uid, s in scanned[name].items() if len(s.files) == slices}
        if not exact:
            continue
        pick, why = choose_series(exact, preferences, on_multiple)
        if pick is not None:
            return pick, path, ("only one" if len(exact) == 1 else why), scanned[name]
    return None


# ---------------------------------------------------------------------------
# Copy
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


def target_names(files: list[str], source: Path) -> list[str]:
    """
    Flat destination names: the file's own name, or its path below *source* joined with ``__``
    when that name occurs more than once.

    >>> target_names(["/s/a/IM_1", "/s/b/IM_1", "/s/b/IM_2"], Path("/s"))
    ['a__IM_1', 'b__IM_1', 'IM_2']
    """
    seen = Counter(os.path.basename(path) for path in files)
    return [
        os.path.basename(path) if seen[os.path.basename(path)] == 1
        else os.path.relpath(path, source).replace(os.sep, "__")
        for path in files
    ]


def copy_series(
    series: Series, source: Path, destination: Path, pool: ThreadPoolExecutor, *, overwrite: bool, write: bool
) -> dict[str, Any]:
    """Copy *series* flat into *destination*; the destination is listed once, not probed per file."""
    counts: dict[str, Any] = {"copied": 0, "present": 0, "conflict": 0, "example": ""}
    existing = _listing(destination)
    jobs: list[tuple[str, Path]] = []
    for path, name in zip(series.files, target_names(series.files, source)):
        entry = existing.get(name)
        if entry is not None:
            if entry.stat().st_size == os.stat(path).st_size:
                counts["present"] += 1
                continue
            if not overwrite:
                counts["conflict"] += 1
                counts["example"] = counts["example"] or str(destination / name)
                continue
        jobs.append((path, destination / name))
    counts["copied"] = len(jobs)
    if write and jobs:
        destination.mkdir(parents=True, exist_ok=True)
        for _ in pool.map(_copy_file, jobs):
            pass
    return counts


def _series_uid_in(directory: Path, existing: dict[str, os.DirEntry]) -> str:
    """SeriesInstanceUID of the CT in *directory*, from its first file; ``""`` when empty or unreadable."""
    if not existing:
        return ""
    header = read_header(str(directory / min(existing, key=_natural_key)))
    return header.series_uid if header is not None else ""


def _swap_in(staging: Path, ct_dir: Path) -> None:
    """Make the complete *staging* folder the new *ct_dir*, removing the old one only afterwards."""
    old = ct_dir.with_name(ct_dir.name + ".old")
    if old.exists():  # left by an interrupted swap
        shutil.rmtree(old)
    if ct_dir.exists():
        os.replace(ct_dir, old)
    os.replace(staging, ct_dir)
    if old.exists():
        shutil.rmtree(old)


def install_series(
    series: Series,
    source: Path,
    ct_dir: Path,
    pool: ThreadPoolExecutor,
    *,
    replace_other: bool,
    overwrite: bool,
    write: bool,
) -> dict[str, Any]:
    """
    Put *series* in *ct_dir*; returns the copy counts plus ``replaced`` (files of a previous CT removed).

    * *ct_dir* already holds this series: top it up in place.
    * *ct_dir* holds another series: a conflict, unless *replace_other* — then the new series is
      staged in full in ``CT.partial`` and swapped in, and only then is the old folder removed.
    * *ct_dir* is empty or missing: staged and swapped in the same way, so it appears complete.
    """
    existing = _listing(ct_dir)
    if existing and set(existing) <= set(target_names(series.files, source)) \
            and _series_uid_in(ct_dir, existing) == series.uid:
        return {**copy_series(series, source, ct_dir, pool, overwrite=overwrite, write=write), "replaced": 0}
    if existing and not replace_other:
        return {
            "copied": 0, "present": 0, "conflict": len(existing), "replaced": 0,
            "example": f"{ct_dir} holds another CT ({len(existing)} files); --overwrite replaces it",
        }

    staging = ct_dir.with_name(ct_dir.name + PARTIAL_SUFFIX)
    staged = _listing(staging)
    if staged and _series_uid_in(staging, staged) != series.uid:
        # A staged copy of another series (an interrupted earlier choice) must not count as present.
        if not write:
            return {"copied": len(series.files), "present": 0, "conflict": 0, "replaced": len(existing), "example": ""}
        shutil.rmtree(staging)
    counts = copy_series(series, source, staging, pool, overwrite=True, write=write)
    if write:
        _swap_in(staging, ct_dir)
    return {**counts, "replaced": len(existing)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
REPORT_COLUMNS = (
    "subject_uid", "source_folder", "series_number", "series_description", "series_uid", "slices",
    "chosen_by", "other_candidates", "copied", "already_present", "conflicts", "replaced", "status",
    "headers_read", "pattern_not_ct", "pattern_localizer", "unreadable",
)
_COUNT_COLUMNS = [
    "slices", "copied", "already_present", "conflicts", "replaced", "headers_read",
    "pattern_not_ct", "pattern_localizer", "unreadable",
]


def _subject_folders(input_root: Path, wanted: tuple[str, ...]) -> list[Path]:
    """The subject folders under *input_root* (only *wanted* ones, when given)."""
    with os.scandir(input_root) as entries:
        folders = sorted((Path(e.path) for e in entries if e.is_dir()), key=lambda p: p.name)
    if not wanted:
        return folders
    names = {name.strip() for name in wanted}
    missing = sorted(names - {p.name for p in folders})
    if missing:
        raise click.ClickException(f"Not in {input_root}: {', '.join(missing)}")
    return [p for p in folders if p.name in names]


def save_report(frame: pd.DataFrame, path: Path, kept: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    """
    Write *frame* to *path*, keeping the rows of subjects this run did not process.

    *kept* are subjects skipped because their output was already done: their earlier row is kept,
    and the *kept* row only stands in when the report has none. Written under a temporary name and
    renamed, so an interrupted save never truncates the report.
    """
    stand_ins = pd.DataFrame(kept or [], columns=list(REPORT_COLUMNS))
    if path.exists():
        previous = pd.read_csv(path, dtype={"subject_uid": str, "series_number": str, "series_uid": str})
        stand_ins = stand_ins.loc[~stand_ins["subject_uid"].isin(previous["subject_uid"])]
        previous = previous.loc[~previous["subject_uid"].isin(frame["subject_uid"])]
        frame = pd.concat([previous.reindex(columns=REPORT_COLUMNS), stand_ins, frame], ignore_index=True)
    else:
        frame = pd.concat([stand_ins, frame], ignore_index=True)
    frame = frame.sort_values("subject_uid", kind="stable").reset_index(drop=True)
    frame[_COUNT_COLUMNS] = frame[_COUNT_COLUMNS].astype("Int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    frame.to_csv(partial, index=False)
    os.replace(partial, path)
    return frame


@click.command("copy-dicoms-ct")
@click.option(
    "--input", "input_root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=DEFAULT_INPUT or None,
    required=not DEFAULT_INPUT,
    help="Output of copy_dicoms_ctpet.py: '<subject_uid>/{CT,PET,PET2}'. Only read.",
)
@click.option(
    "--output", "output_root",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT or None,
    required=not DEFAULT_OUTPUT,
    help="Root to create '<subject_uid>/CT' under.",
)
@click.option("--subject", "subjects", multiple=True, help="Only this subject folder (repeatable).")
@click.option(
    "--source-folder", "source_folders",
    multiple=True,
    default=SOURCE_FOLDERS,
    show_default=True,
    help="Subject subfolders searched for the CT, in order; the first that exists is used (repeatable).",
)
@click.option("--series-pattern", default=SERIES_PATTERN, show_default=True, help="Text the SeriesDescription must contain (any case).")
@click.option(
    "--prefer", "preferences",
    multiple=True,
    default=PREFERENCES,
    show_default=True,
    help="With several iDose series: description patterns tried in order (repeatable, * wildcards).",
)
@click.option(
    "--on-multiple",
    type=click.Choice(["skip", "largest"]),
    default="largest",
    show_default=True,
    help="When no preference decides: skip the subject, or take the series with the most files.",
)
@click.option(
    "--target-slices",
    type=int,
    is_flag=False,
    flag_value=TARGET_SLICES,
    default=None,
    help=(
        f"Prefer a CT of exactly N slices ({TARGET_SLICES} when given without a value): skip outputs that "
        "already hold 1..N slices, and replace a bigger selection with an N-slice CT from PET2, else PET."
    ),
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Replace destination files whose size differs, and an output CT that holds another series.",
)
@click.option("--workers", default=16, show_default=True, help="Headers read / files copied in parallel.")
@click.option(
    "--report", "report_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help=f"CSV of the CT copied per subject and its slices (default: <output>/{REPORT_NAME}; written on --write).",
)
@click.option(
    "--write/--dry-run",
    default=False,
    show_default="--dry-run",
    help="Actually copy; the default only reports.",
)
def main(
    input_root: Path,
    output_root: Path,
    subjects: tuple[str, ...],
    source_folders: tuple[str, ...],
    series_pattern: str,
    preferences: tuple[str, ...],
    on_multiple: str,
    target_slices: int | None,
    overwrite: bool,
    workers: int,
    report_path: Path | None,
    write: bool,
) -> None:
    """Copy the iDose CT series of each subject's PET2 (else PET) folder into <output>/<subject_uid>/CT."""
    input_root, output_root = input_root.resolve(), output_root.resolve()
    if output_root == input_root or input_root in output_root.parents:
        raise click.ClickException("--output must not be inside --input: the input is never written to.")
    if target_slices is not None and target_slices < 1:
        raise click.ClickException("--target-slices must be a positive number of slices.")
    report_path = report_path or output_root / REPORT_NAME

    folders = _subject_folders(input_root, subjects)
    log.info(
        "%d subject folder(s) under %s%s.", len(folders), input_root,
        f"; preferring {target_slices}-slice CTs" if target_slices else "",
    )

    report: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    totals = {"copied": 0, "present": 0, "conflict": 0, "replaced": 0, "done": 0, "switched": 0}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, folder in enumerate(folders, start=1):
            row: dict[str, Any] = {"subject_uid": folder.name}
            ct_dir = output_root / folder.name / TARGET_FOLDER
            if target_slices:
                done_slices = len(_listing(ct_dir))
                if 0 < done_slices <= target_slices:
                    kept.append({**row, "slices": done_slices, "status": f"kept: output CT already has {done_slices} slice(s)"})
                    log.info(
                        "[%d/%d] %s: output CT already has %d slice(s) (<= %d), skipped.",
                        index, len(folders), folder.name, done_slices, target_slices,
                    )
                    continue

            source = next((folder / name for name in source_folders if (folder / name).is_dir()), None)
            if source is None:
                row["status"] = f"skipped: no {' or '.join(source_folders)} folder"
                log.warning("[%d/%d] %s %s.", index, len(folders), folder.name, row["status"])
                report.append(row)
                continue

            series, counts = scan_source(source, pool, series_pattern)
            row.update(
                headers_read=sum(counts.values()),
                pattern_not_ct=counts["pattern_not_ct"],
                pattern_localizer=counts["pattern_localizer"],
                unreadable=counts["unreadable"],
            )
            chosen, reason = choose_series(series, preferences, on_multiple)
            if chosen is None:
                row.update(source_folder=source.name, status=f"skipped ({source.name}): {reason}")
                log.warning("[%d/%d] %s %s.", index, len(folders), folder.name, row["status"])
                report.append(row)
                continue

            switched = False
            if target_slices and len(chosen.files) > target_slices:
                found = find_target_series(
                    folder, source_folders, {source.name: series}, target_slices, pool,
                    pattern=series_pattern, preferences=preferences, on_multiple=on_multiple,
                )
                if found is not None:
                    target, target_source, why, target_folder_series = found
                    reason = (
                        f"{target_slices} slices in {target_source.name} ({why}), "
                        f"instead of {source.name} {chosen.label()} ({reason})"
                    )
                    chosen, source, series, switched = target, target_source, target_folder_series, True
                    totals["switched"] += 1
                else:
                    reason = f"{reason}; no {target_slices}-slice CT in {' or '.join(source_folders)}"

            outcome = install_series(
                chosen, source, ct_dir, pool,
                replace_other=overwrite or switched, overwrite=overwrite, write=write,
            )
            for key in ("copied", "present", "conflict", "replaced"):
                totals[key] += outcome[key]
            totals["done"] += 1
            others = [s for s in series.values() if s is not chosen]
            row.update(
                source_folder=source.name,
                series_number=chosen.number,
                series_description=chosen.description,
                series_uid=chosen.uid,
                slices=len(chosen.files),
                chosen_by=reason,
                other_candidates="; ".join(s.label() for s in others),
                copied=outcome["copied"],
                already_present=outcome["present"],
                conflicts=outcome["conflict"],
                replaced=outcome["replaced"],
                status="ok" if not outcome["conflict"] else f"conflicts: {outcome['example']}",
            )
            report.append(row)
            left_out = counts["pattern_not_ct"] + counts["pattern_localizer"]
            log.info(
                "[%d/%d] %s: %s/%s (%s) -> %s %d/%d%s%s",
                index, len(folders), folder.name, source.name, chosen.label(), reason,
                "copied" if write else "to copy", outcome["copied"], len(chosen.files),
                f", {'replacing' if write else 'would replace'} a previous {outcome['replaced']}-file CT"
                if outcome["replaced"] else "",
                f" | {left_out} iDose file(s) left out as non-CT or localizer" if left_out else "",
            )

    frame = pd.DataFrame(report, columns=list(REPORT_COLUMNS))
    frame[_COUNT_COLUMNS] = frame[_COUNT_COLUMNS].astype("Int64")
    attention = frame.loc[frame["status"].fillna("").ne("ok")]
    if not attention.empty:
        with pd.option_context("display.max_rows", None, "display.width", 250, "display.max_colwidth", 160):
            click.echo("\nNeeds attention:\n" + attention[["subject_uid", "status"]].to_string(index=False))
    if write:
        saved = save_report(frame, report_path, kept)
        click.echo(f"\nReport: {report_path} ({len(saved)} subject row(s), {int(saved['slices'].notna().sum())} with a CT).")

    action = "Copied" if write else "Dry run — would copy"
    click.echo(
        f"\n{action} {totals['copied']} file(s) for {totals['done']} of {len(folders)} subject(s) "
        f"({totals['present']} already present, {totals['conflict']} conflict(s), "
        f"{len(folders) - totals['done'] - len(kept)} skipped"
        + (
            f"; {len(kept)} kept as already done, {totals['switched']} switched to a {target_slices}-slice CT, "
            f"{totals['replaced']} file(s) of previous CTs {'removed' if write else 'to remove'}"
            if target_slices else ""
        )
        + ")."
    )
    if not write:
        click.echo(f"Re-run with --write to apply and save the report to {report_path}.")


if __name__ == "__main__":
    main()
