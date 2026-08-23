"""
Mass-univariate voxelwise analysis with FSL ``randomise``.

Every other analysis in this toolkit is region-wise: a measurement is averaged inside a parcel or a
vessel and modelled as one number per subject × region. That answers *which region* and cannot
answer *where in this region*, and it presumes the parcellation is the right unit — which for an
effect that straddles a boundary, or sits in part of a territory, it is not.

This module fits the same GLM at every voxel instead. A cohort's spatially normalised images are
stacked into one 4D volume, a design matrix is built from database measurements, and
``randomise`` estimates the null by permutation, giving family-wise-error-corrected p-values with
TFCE. Nothing here assumes the images and the design come from the same modality: the motivating
question — *does 4D-flow mean flow in the left MCA predict ASL perfusion, voxel by voxel?* — takes
its images from one pipeline and its design from another.

The ordering contract
---------------------
Row *i* of the design matrix must be volume *i* of the 4D stack. :func:`resolve_cohort_images`
returns an ordered list and every step downstream is indexed off that one list — the merge, the
design matrix, and the manifest. A transposed or mis-ordered design still runs and still produces a
plausible-looking map, so alignment is never inferred, only carried.

Backend
-------
``fslmerge`` and ``randomise`` ship in *different* conda packages (``fsl-avwutils`` and
``fsl-randomise``), so :func:`fsl_backend_status` probes them separately: an image that has one and
not the other fails halfway through, and the status has to be able to say which half.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import fnmatch
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from nvitk.core.logger import Logger

log = Logger()

#: Binaries this module shells out to, and the conda package each one ships in. The mapping is what
#: lets the status report name the missing package rather than just the missing command.
FSL_BINARIES: dict[str, str] = {
    "fslmerge": "fsl-avwutils",
    "randomise": "fsl-randomise",
    "randomise_parallel": "fsl-randomise",
}

#: Binaries without which nothing can run at all.
FSL_REQUIRED: tuple[str, ...] = ("fslmerge", "randomise")

INSTALL_HINT = (
    "Install FSL and put its bin/ on PATH, e.g.\n"
    "    conda install -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public "
    "fsl-avwutils fsl-randomise\n"
    "or source an existing installation:\n"
    "    export FSLDIR=/path/to/fsl && . $FSLDIR/etc/fslconf/fsl.sh"
)

#: Optional explicit regex override for pulling a session id out of a filename
#: (``…_BMRI123456_s8_…nii.gz``). Only needed when the default token scan (see
#: :func:`resolve_cohort_images`) is not enough — e.g. the id is glued onto neighbouring text with
#: no delimiter, or the filename contains a decoy that also happens to be a registered id.
DEFAULT_ID_PATTERN: str | None = None

#: ``subject_ids`` namespaces searched, by default, to map a filename token onto a ``subject_uid``.
#: These three carry the same *kind* of code — an acquisition/session id, not a subject id like
#: ``pesa_id`` — and different cohorts within one dataset are free to use different ones for the
#: same physical session (a 4D-flow export named by ``mr_id``, an ASL export named by ``session``),
#: which is exactly why all three are searched at once rather than requiring the caller to know
#: which kind a given directory uses.
DEFAULT_ID_NAMESPACES: tuple[str, ...] = ("mr_id", "mri_id", "session")

#: A maximal run of letters-then-digits, e.g. ``BMRI100102`` or ``IA004754``. This is the *shape*
#: every id in :data:`DEFAULT_ID_NAMESPACES` happens to share regardless of project prefix, so
#: splitting a filename into these tokens and checking each one against the database recognises a
#: mixed cohort (``BMRI*``, ``IA*``, …) without the caller enumerating every prefix in use.
_ID_TOKEN_RX = re.compile(r"[A-Za-z]{2,}\d+")

#: What to do when two files resolve to the same subject — the same person scanned twice, or the
#: same session exported under two different id namespaces. ``error`` is the default because which
#: session is kept changes the result, and that is the caller's call to make, not a default's.
DUPLICATE_POLICIES: tuple[str, ...] = ("error", "skip", "first", "last")

#: Affine agreement tolerance in mm. Images resampled onto a shared grid by different tools differ
#: in the last bits of the affine; a real space mismatch is orders of magnitude larger than this.
AFFINE_TOL = 1e-3

#: ``randomise`` writes corrected p-values as 1-p, so p < 0.05 is a map value above this.
CORRP_THRESHOLD = 0.95


# ──────────────────────────────────────────────────────────────────────────────
# Backend probe
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FslBackendStatus:
    """What is available to run a voxelwise analysis."""

    available: bool
    binaries: dict[str, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    reason: str = ""

    def summary(self) -> str:
        """One-line description for a status bar."""
        if not self.available:
            return f"unavailable — {self.reason}"
        version = self.versions.get("fsl", "?")
        extra = "" if "randomise_parallel" in self.binaries else " (no randomise_parallel)"
        return f"FSL {version} · {len(self.binaries)} binaries{extra}"

    def install_hint(self) -> str:
        """What to install to make the engine work."""
        if not self.missing:
            return INSTALL_HINT
        packages = sorted({FSL_BINARIES[name] for name in self.missing if name in FSL_BINARIES})
        return (
            f"Missing {', '.join(self.missing)} — ships in {', '.join(packages)}.\n" + INSTALL_HINT
        )

    def supports_parallel(self) -> bool:
        """True when ``randomise_parallel`` is on PATH."""
        return "randomise_parallel" in self.binaries


def _fsl_version(fsldir: str | None) -> str:
    """Read the FSL version string from ``$FSLDIR/etc/fslversion``, or ``""`` when unreadable."""
    if not fsldir:
        return ""
    for rel in ("etc/fslversion", "VERSION", "share/fsl/etc/fslversion"):
        candidate = Path(fsldir) / rel
        try:
            if candidate.is_file():
                return candidate.read_text(encoding="utf-8").strip().splitlines()[0]
        except OSError:
            continue
    return ""


def fsl_backend_status() -> FslBackendStatus:
    """Probe FSL. Never raises, and never runs anything as an import side effect.

    ``FSLDIR``, ``fslmerge`` and ``randomise`` are probed separately because they can genuinely
    disagree: the nvitk Singularity image installs ``fsl-avwutils`` but not ``fsl-randomise``, so
    ``fslmerge`` resolves and ``randomise`` does not. A single "FSL missing" would send someone
    looking for the wrong problem.
    """
    fsldir = os.environ.get("FSLDIR", "").strip() or None
    found: dict[str, str] = {}
    for name in FSL_BINARIES:
        exe = shutil.which(name)
        if exe:
            found[name] = exe

    missing = tuple(name for name in FSL_REQUIRED if name not in found)
    versions: dict[str, str] = {}
    version = _fsl_version(fsldir)
    if version:
        versions["fsl"] = version

    if missing:
        packages = sorted({FSL_BINARIES[name] for name in missing})
        where = f"FSLDIR={fsldir}" if fsldir else "FSLDIR is unset"
        return FslBackendStatus(
            available=False,
            binaries=found,
            versions=versions,
            missing=missing,
            reason=(
                f"{', '.join(missing)} not on PATH ({where}); "
                f"ships in {', '.join(packages)}"
            ),
        )
    return FslBackendStatus(available=True, binaries=found, versions=versions)


def require_fsl() -> FslBackendStatus:
    """Return the backend status, raising ``RuntimeError`` with the install hint if unusable."""
    status = fsl_backend_status()
    if not status.available:
        raise RuntimeError(f"{status.reason}\n{status.install_hint()}")
    return status


def _ensure_fsl_env() -> None:
    """FSL defaults to uncompressed NIFTI; pin the output type so discovered paths are predictable.

    ``randomise`` writes a dozen files whose extension follows ``FSLOUTPUTTYPE``, and the result
    loader globs for them by name. Same reconciliation as
    :mod:`nvitk.registration.fsl.flirt`.
    """
    os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"
    os.environ.setdefault("FSLMULTIFILEQUIT", "TRUE")


def _resolve_nifti_path(path: Path) -> Path | None:
    """Return *path* or its ``.nii`` / ``.nii.gz`` sibling if FSL wrote a different extension."""
    if path.is_file():
        return path
    if path.name.endswith(".nii.gz"):
        alt = path.with_name(path.name[: -len(".gz")])
        if alt.is_file():
            return alt
    if path.suffix == ".nii":
        alt = Path(f"{path}.gz")
        if alt.is_file():
            return alt
    return None


def _abbreviate(cmd: Sequence[str], *, keep: int = 12) -> str:
    """Command line for the log, with a long input list elided.

    ``fslmerge`` takes one absolute path per subject; printing 500 of them buries every other line
    in the log without telling anyone anything the subject list in ``manifest.json`` does not.
    """
    if len(cmd) <= keep:
        return " ".join(cmd)
    head = " ".join(cmd[: keep - 1])
    return f"{head} … (+{len(cmd) - keep + 1} more) {cmd[-1]}"


def _run_checked(argv: Sequence[str], *, what: str) -> subprocess.CompletedProcess:
    """Run *argv*, logging the command, and raise ``RuntimeError`` naming *what* on failure."""
    cmd = [str(a) for a in argv]
    log.info(f"Running {what}: {_abbreviate(cmd)}")
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("%s failed rc=%s stderr=%s", what, proc.returncode, (proc.stderr or "").strip())
        raise RuntimeError(
            f"{what} failed (rc={proc.returncode}): "
            f"{(proc.stderr or proc.stdout or '').strip()[-2000:]}"
        )
    return proc


# ──────────────────────────────────────────────────────────────────────────────
# Cohort assembly
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CohortImage:
    """One subject's normalised volume, and the ids that got it there."""

    subject_uid: str
    session_id: str
    path: Path

    def label(self) -> str:
        """``subject (session)`` — what an error message or a manifest row should say."""
        return f"{self.subject_uid} ({self.session_id})"


def _read_exclusion_patterns(exclude_csv: str | Path | None) -> list[str]:
    """Read one id-glob per line from *exclude_csv* (``BMRI12345*``), skipping blanks and ``#``.

    A CSV is accepted as much as a bare list: only the first field of each row is read, so a sheet
    exported with a comment column still works.
    """
    if exclude_csv is None:
        return []
    path = Path(exclude_csv).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Exclusion list not found: {path}")
    patterns: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        token = raw.split(",")[0].strip().strip('"').strip("'")
        if not token or token.startswith("#"):
            continue
        patterns.append(token)
    return patterns


def _id_to_subject_map(
    repo: Any,
    namespaces: Sequence[str],
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Build ``id_value -> {subject_uid}`` and ``id_value -> namespace`` from ``subject_ids``.

    Rows whose ``subject_uid`` is null are dropped: some id namespaces are harvested from sheets
    that carry the code without ever resolving it to a subject, and keeping those would turn a
    clean lookup into a spurious "unknown id".
    """
    import pandas as pd

    frame = repo.get("subject_ids")
    if frame is None or frame.empty:
        raise ValueError("The dataset has no 'subject_ids' table; cannot map filenames to subjects.")

    wanted = [str(n).strip() for n in namespaces if str(n).strip()]
    subset = frame[frame["id_namespace"].astype("string").isin(wanted)]
    subset = subset[subset["subject_uid"].notna()]
    subset = subset[subset["subject_uid"].astype("string").str.strip() != ""]
    if subset.empty:
        raise ValueError(
            f"No 'subject_ids' rows in namespace(s) {wanted!r}. Available: "
            f"{sorted(pd.unique(frame['id_namespace'].astype('string').dropna()))}"
        )

    mapping: dict[str, set[str]] = {}
    origin: dict[str, str] = {}
    for id_value, subject_uid, namespace in zip(
        subset["id_value"].astype("string"),
        subset["subject_uid"].astype("string"),
        subset["id_namespace"].astype("string"),
        strict=False,
    ):
        key = str(id_value).strip()
        if not key:
            continue
        mapping.setdefault(key, set()).add(str(subject_uid).strip())
        origin.setdefault(key, str(namespace))
    return mapping, origin


def _strip_nifti_suffix(name: str) -> str:
    """Drop a trailing ``.nii`` or ``.nii.gz`` from a file name."""
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _candidate_tokens(stem: str) -> list[str]:
    """Every maximal letters-then-digits run in *stem*, in order of appearance.

    These are the tokens checked against the database when no ``--id-pattern`` is given — the
    generic shape (``BMRI100102``, ``IA004754``, ``PESA150544``, …) every id in
    :data:`DEFAULT_ID_NAMESPACES` happens to share, regardless of which project's prefix it uses.
    """
    return _ID_TOKEN_RX.findall(stem)


def _resolve_session_id(
    path: Path,
    *,
    rx: re.Pattern[str] | None,
    mapping: dict[str, set[str]],
    namespaces: Sequence[str],
) -> tuple[str, set[str]]:
    """Return ``(session_id, subject_uids)`` for *path*, raising a message naming the file.

    With an explicit ``id_pattern`` (*rx* set), extraction is exactly the id the regex names —
    unchanged from before, for the cases where a delimiter-free filename or a decoy token needs a
    caller-supplied rule rather than a guess.

    With no pattern, every letters-then-digits token in the filename is checked against *mapping*
    (built once, from *namespaces*) and the token(s) that are registered ids win. This is what lets
    a mixed directory — some files named by an ``mr_id``, others by a ``session`` from a different
    project prefix — resolve without the caller enumerating every prefix in use.
    """
    if rx is not None:
        match = rx.search(path.name)
        if match is None:
            raise ValueError(
                f"No session id in {path.name!r} using pattern {rx.pattern!r}. "
                "Pass --id-pattern with a regex whose first group is the id."
            )
        session_id = match.group(1) if match.groups() else match.group(0)
        subjects = mapping.get(session_id)
        if not subjects:
            raise ValueError(
                f"Session id {session_id!r} (from {path.name}) is not in 'subject_ids' "
                f"namespace(s) {list(namespaces)!r}. Exclude it with --exclude-csv, or widen "
                "--namespaces."
            )
        return session_id, subjects

    tokens = _candidate_tokens(_strip_nifti_suffix(path.name))
    hits = {tok: mapping[tok] for tok in tokens if tok in mapping}
    if not hits:
        raise ValueError(
            f"No registered subject id found in {path.name!r} (looked for a "
            f"letters-then-digits token — e.g. 'BMRI100102' — in 'subject_ids' namespace(s) "
            f"{list(namespaces)!r}). Widen --namespace, exclude the file with --exclude-csv, or "
            "pass --id-pattern if the id is not a plain letters+digits token."
        )
    distinct = {subject for group in hits.values() for subject in group}
    if len(distinct) > 1:
        raise ValueError(
            f"{path.name} contains tokens for {len(distinct)} different subjects — "
            f"{ {tok: sorted(subs) for tok, subs in hits.items()} }. Narrow --namespace or use "
            "--id-pattern to pick one."
        )
    # Several tokens can legitimately name the *same* subject (a run number glued next to the real
    # id, say); report the longest, since a longer code is the more specific match.
    session_id = max(hits, key=len)
    return session_id, hits[session_id]


def _apply_duplicate_policy(
    images: Sequence[CohortImage],
    policy: str = "error",
) -> list[CohortImage]:
    """Enforce one image per subject, according to *policy*.

    A voxelwise design has exactly one row per subject, so two files claiming the same
    ``subject_uid`` — a repeat scan, or one session exported under both an ``mr_id`` and a
    ``session`` code — have to be reconciled before the stack is built.

    ``error`` reports **every** conflict at once rather than the first. With hundreds of files a
    one-at-a-time error costs one full re-run per duplicate, and the fix is a single edit once the
    whole list is visible.
    """
    key = str(policy or "error").strip().lower()
    if key not in DUPLICATE_POLICIES:
        raise ValueError(
            f"Unknown duplicate policy {policy!r}. Choose one of {list(DUPLICATE_POLICIES)}."
        )

    grouped: dict[str, list[CohortImage]] = {}
    for image in images:
        grouped.setdefault(image.subject_uid, []).append(image)
    duplicates = {uid: group for uid, group in grouped.items() if len(group) > 1}
    if not duplicates:
        return list(images)

    if key == "error":
        shown = sorted(duplicates.items())[:10]
        listing = "\n".join(
            f"  {uid}: " + ", ".join(f"{im.path.name} ({im.session_id})" for im in group)
            for uid, group in shown
        )
        if len(duplicates) > len(shown):
            listing += f"\n  … and {len(duplicates) - len(shown)} more"

        # Nearly every subject duplicated the same number of times is not a cohort with repeat
        # scans — it is one include pattern matching several derivatives of each image (an
        # unsmoothed and two smoothed copies, say). Saying "pick a duplicate policy" there would
        # have someone silently analyse an arbitrary smoothing level.
        widespread = len(duplicates) > 0.5 * len(grouped)
        sizes = Counter(len(group) for group in duplicates.values())
        common_size, common_count = sizes.most_common(1)[0]
        # A handful of subjects with an odd count (a repeat scan on top of the variants) should not
        # mask the pattern, so this asks whether *most* duplicates share one size, not all of them.
        if widespread and common_count > 0.8 * len(duplicates):
            example = ", ".join(im.path.name for im in shown[0][1]) if shown else ""
            hint = (
                f"Most subjects have the same {common_size} files, so this is one image in several "
                f"variants rather than repeat scans — narrow --include to pick one "
                f"(e.g. '*_s8*'). Files for the first subject: {example}."
            )
        else:
            hint = (
                "Choose with --on-duplicate skip (drop these subjects), first / last (keep one by "
                "filename order), or list the sessions to drop in --exclude-csv."
            )
        raise ValueError(
            f"{len(duplicates)} of {len(grouped)} subject(s) have more than one image, but a "
            f"voxelwise design has one row per subject:\n{listing}\n{hint}"
        )

    kept: list[CohortImage] = []
    dropped: list[str] = []
    for image in images:
        group = grouped[image.subject_uid]
        if len(group) == 1:
            kept.append(image)
            continue
        if key == "skip":
            dropped.append(image.path.name)
            continue
        # ``images`` arrives in filename order, so the group preserves it.
        chosen = group[0] if key == "first" else group[-1]
        if image is chosen:
            kept.append(image)
        else:
            dropped.append(image.path.name)

    log.warning(
        f"--on-duplicate {key}: {len(duplicates)} subject(s) had more than one image; "
        f"dropped {len(dropped)} file(s), kept {len(kept)}."
    )
    log.debug("Dropped as duplicates: %s", ", ".join(sorted(dropped)))
    return kept


def resolve_cohort_images(
    image_dir: str | Path,
    *,
    repo: Any = None,
    include: str = "*",
    exclude_csv: str | Path | None = None,
    id_pattern: str | None = None,
    namespaces: Sequence[str] = DEFAULT_ID_NAMESPACES,
    on_duplicate: str = "error",
) -> list[CohortImage]:
    """Scan a **flat** directory of normalised volumes and resolve each one to a subject.

    Parameters
    ----------
    image_dir
        One directory, no recursion. Cohort images produced by a normalisation step land side by
        side; a recursive walk would silently pick up derivatives and QC copies living under it.
    include
        Glob applied to the file name (``'*_s8_*'`` for 8 mm-smoothed volumes). A bare pattern with
        no NIfTI extension still only matches NIfTI files. Leave it as the default ``'*'`` to
        include every NIfTI in the directory — narrowing is opt-in, not required.
    exclude_csv
        Optional file of id-globs, one per line — subjects to drop before anything is read.
    id_pattern
        Optional regex override whose first group is the session id inside the file name. Leave
        unset (the default) to auto-detect: every letters-then-digits token in each filename
        (``BMRI100102``, ``IA004754``, …) is checked against the database, so a directory mixing
        several projects' naming conventions resolves without the caller supplying a pattern per
        prefix. Only needed for an id with no delimiter separating it from surrounding text, or
        where auto-detection would be ambiguous.
    namespaces
        ``subject_ids`` namespaces the token is checked against. Whichever *kind* of code (an
        ``mr_id``, an ``mri_id``, a ``session`` label — never a subject id like ``pesa_id``) the
        directory happens to use, it resolves to the same ``subject_uid`` — the standardisation
        every downstream step (in particular ``--cohort``) is written against.
    on_duplicate
        What to do when two files resolve to the same subject: ``error`` (default, listing every
        conflict), ``skip`` (drop those subjects entirely), or ``first`` / ``last`` (keep one, by
        filename order).

    Returns
    -------
    list[CohortImage]
        Ordered by file name, and **that order is the contract**: it becomes the volume order of the
        4D stack and therefore the row order of the design matrix.

    Raises
    ------
    ValueError
        If a file name carries no recognisable id, an id resolves to no subject or to more than
        one, or two files claim the same subject. Each message names the offending file.
    """
    directory = Path(image_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory not found: {directory}")

    pattern = str(include or "*").strip() or "*"
    candidates: list[Path] = []
    for entry in sorted(directory.iterdir()):
        if not entry.is_file():
            continue
        # Dotfiles are never cohort images: a network share picks up macOS AppleDouble forks
        # (``._subject.nii``), and a botched export can leave a file whose whole name is the
        # extension. Neither has a stem that could carry an id.
        if entry.name.startswith("."):
            log.debug("Skipping hidden file %s", entry.name)
            continue
        if not (entry.name.endswith(".nii") or entry.name.endswith(".nii.gz")):
            continue
        if not fnmatch.fnmatch(entry.name, pattern):
            continue
        candidates.append(entry)

    log.info(f"Image scan: {len(candidates)} file(s) in {directory} matching {pattern!r}")
    if not candidates:
        raise ValueError(
            f"No NIfTI files in {directory} match include pattern {pattern!r}. "
            "The scan is flat — subdirectories are not searched."
        )

    rx: re.Pattern[str] | None = None
    if id_pattern:
        try:
            rx = re.compile(id_pattern)
        except re.error as exc:
            raise ValueError(f"Invalid --id-pattern {id_pattern!r}: {exc}") from None

    exclusions = _read_exclusion_patterns(exclude_csv)

    if repo is None:
        from nvitk.db.repo import get_repo

        repo = get_repo()
    mapping, origin = _id_to_subject_map(repo, namespaces)

    resolved: list[CohortImage] = []
    excluded: list[str] = []
    for path in candidates:
        # An exclusion glob is checked against the file name too, since with no --id-pattern the
        # id is not known until after the (possibly expensive-to-read) database lookup below —
        # excluding by file name lets a bad file be skipped without ever needing to resolve it.
        if any(fnmatch.fnmatch(path.name, pat) for pat in exclusions):
            excluded.append(path.name)
            continue

        session_id, subjects = _resolve_session_id(
            path, rx=rx, mapping=mapping, namespaces=namespaces
        )

        if any(fnmatch.fnmatch(session_id, pat) for pat in exclusions):
            excluded.append(session_id)
            continue

        if len(subjects) > 1:
            raise ValueError(
                f"Session id {session_id!r} (from {path.name}) maps to {len(subjects)} subjects "
                f"{sorted(subjects)!r}. Resolve the duplicate in 'subject_ids' or exclude the id."
            )
        subject_uid = next(iter(subjects))
        log.debug(
            "%s -> id %r (namespace %s) -> subject %s",
            path.name, session_id, origin.get(session_id, "?"), subject_uid,
        )
        resolved.append(CohortImage(subject_uid=subject_uid, session_id=session_id, path=path))

    if excluded:
        log.info(f"Excluded {len(excluded)} image(s) by --exclude-csv")
        log.debug("Excluded: %s", ", ".join(sorted(excluded)))
    if id_pattern:
        log.info(f"Resolved {len(resolved)} image(s) to subjects using --id-pattern {id_pattern!r}")
    else:
        log.info(
            f"Resolved {len(resolved)} image(s) to subjects "
            f"(auto-detected against namespace(s) {list(namespaces)!r})"
        )
    return _apply_duplicate_policy(resolved, on_duplicate)


def cohort_subjects(repo: Any, pipeline_id: str) -> set[str]:
    """Subjects with any measurement under *pipeline_id*.

    This is the ``--cohort`` filter: the subject set is defined by *what data exists*, not by
    whatever images happen to sit in the directory. Distinct from ``--cohort-id``, which is
    :class:`~nvitk.db.repo.DataRepo`'s named-membership filter over ``cohort_membership``; the two
    can disagree, and naming them the same would hide that.

    Raises
    ------
    ValueError
        If *pipeline_id* is not registered, listing the ids that are — a plain "not found" would not
        reveal that the real ids are ``qvtpy`` and ``4dflow_v3`` rather than ``qvtpy_v3``.
    """
    token = str(pipeline_id).strip()
    if not token:
        raise ValueError("cohort pipeline id is empty.")

    known = sorted(repo.catalog.all_pipeline_ids())
    try:
        resolved = repo.catalog.resolve_pipeline_selector(token)
    except Exception as exc:
        raise ValueError(str(exc)) from None
    if not resolved:
        raise ValueError(f"Unknown pipeline id {token!r}. Registered pipeline ids: {known}")
    if list(resolved) != [token]:
        log.info(f"Cohort {token!r} resolves to pipeline id(s) {resolved}")

    frame = repo.image(pipeline=resolved, wide=False, cohort_id=False)
    if frame.empty or "subject_uid" not in frame.columns:
        raise ValueError(
            f"Pipeline {', '.join(resolved)} has no image measurements in this dataset. "
            f"Registered pipeline ids: {known}"
        )
    subjects = {
        str(s).strip()
        for s in frame["subject_uid"].dropna().astype(str)
        if str(s).strip()
    }
    log.info(f"Cohort {token!r} → {len(subjects)} subject(s) with measurements")
    return subjects


@dataclass(frozen=True)
class CohortOption:
    """One value ``--cohort`` can take, and how many subjects it would select."""

    pipeline_id: str
    n_subjects: int
    registered: bool = True
    aliases: tuple[str, ...] = ()
    label: str = ""

    def describe(self) -> str:
        """``4dflow_v3  513 subjects  (qvtpy, latest, v3)`` — one line for a listing or a combo."""
        text = f"{self.pipeline_id}  ·  {self.n_subjects} subject(s)"
        if self.aliases:
            text += f"  ·  aka {', '.join(self.aliases)}"
        if not self.registered:
            text += "  ·  NOT in the pipeline manifest — not selectable by --cohort"
        return text


def cohort_id_subjects(repo: Any, cohort_id: str) -> set[str] | None:
    """Subjects enrolled in the named cohort *cohort_id*, or ``None`` when membership is unavailable.

    The other filter. ``--cohort`` asks *which pipeline produced measurements for this subject*;
    ``--cohort-id`` asks *which named cohort is this subject enrolled in*, out of
    ``cohort_membership`` — the table :class:`~nvitk.db.repo.DataRepo` already applies by default.
    They can disagree, which is why they are two flags and not one.

    ``None`` rather than an empty set when the table is missing: "no membership data" and "nobody is
    a member" would otherwise both silently empty the cohort.
    """
    token = str(cohort_id).strip()
    if not token:
        return None
    if not repo.catalog.table_exists("cohort_membership"):
        log.warning("No 'cohort_membership' table in this dataset; --cohort-id has nothing to filter on.")
        return None
    try:
        frame = repo.get("cohort_membership", cohort_id=False)
    except Exception as exc:  # noqa: BLE001
        log.warning(f"Could not read cohort membership for {token!r}: {exc}")
        return None
    if frame.empty or {"cohort_id", "subject_uid"} - set(frame.columns):
        log.warning(f"'cohort_membership' has no rows to filter on; ignoring --cohort-id {token!r}.")
        return None
    # Filtered here rather than through ``repo.get(filters={"cohort_id": ...})``: that call pops
    # ``cohort_id`` out of *filters* and reads it as the membership selector, so passing it as a
    # filter silently matches everything instead of one cohort.
    rows = frame[frame["cohort_id"].astype("string").str.strip() == token]
    subjects = {str(s).strip() for s in rows["subject_uid"].dropna().astype(str) if str(s).strip()}
    if not subjects:
        known = sorted(str(c) for c in frame["cohort_id"].dropna().astype(str).unique())
        raise ValueError(f"Cohort id {token!r} has no members. Registered cohort ids: {known}")
    log.info(f"Cohort id {token!r} → {len(subjects)} enrolled subject(s)")
    return subjects


def available_cohorts(repo: Any) -> list[CohortOption]:
    """Every value ``--cohort`` can take, most subjects first.

    Counts are what ``--cohort <id>`` would actually select, which is not the same as counting
    ``image_measurements.pipeline_id`` values: the manifest resolves aliases, so ``qvtpy`` is
    ``4dflow_v3``. Pipeline ids that appear in the data but are *not* registered are listed too and
    marked unselectable — silently omitting them would make a 514-row group look like it never
    existed, and silently counting them would promise a ``--cohort`` value that raises.

    Backs both ``nvitk-voxelwise cohorts`` and the GUI's cohort picker.
    """
    # The raw table, not ``repo.image()``: with no modality and no pipeline that call restricts rows
    # to each modality's *default* pipeline, which is exactly the set this listing exists to show
    # alternatives to.
    frame = repo.get("image_measurements")
    counts: dict[str, int] = {}
    if frame is not None and not frame.empty and {"pipeline_id", "subject_uid"} <= set(frame.columns):
        grouped = frame.groupby(frame["pipeline_id"].astype(str))["subject_uid"].nunique()
        counts = {str(k): int(v) for k, v in grouped.items()}

    options: list[CohortOption] = []
    registered: set[str] = set()
    for entry in repo.catalog.pipelines_manifest.get("pipelines", []):
        pid = str(entry.get("pipeline_id") or "").strip()
        if not pid:
            continue
        registered.add(pid)
        options.append(
            CohortOption(
                pipeline_id=pid,
                n_subjects=int(counts.get(pid, 0)),
                registered=True,
                aliases=tuple(str(a) for a in (entry.get("aliases") or [])),
                label=str(entry.get("label") or entry.get("name") or entry.get("pipeline_name") or ""),
            )
        )
    for pid, n in counts.items():
        if pid not in registered:
            options.append(CohortOption(pipeline_id=pid, n_subjects=int(n), registered=False))

    return sorted(options, key=lambda o: (not o.registered, -o.n_subjects, o.pipeline_id))


# ──────────────────────────────────────────────────────────────────────────────
# Geometry validation
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class SpaceInfo:
    """The grid every volume in a stack has to share."""

    shape: tuple[int, int, int]
    affine: np.ndarray
    zooms: tuple[float, float, float]

    def describe(self) -> str:
        """``96x114x96 @ 2.0×2.0×2.0 mm`` — enough to see a mismatch at a glance."""
        z = "×".join(f"{v:g}" for v in self.zooms)
        return f"{'x'.join(str(v) for v in self.shape)} @ {z} mm"


def _space_of(path: Path) -> SpaceInfo:
    """Read *path*'s grid from its header alone — no voxel data is loaded."""
    import nibabel as nib

    img = nib.load(str(path))
    shape = tuple(int(v) for v in img.shape[:3])
    affine = np.asarray(img.affine, dtype=float)
    zooms = tuple(float(v) for v in img.header.get_zooms()[:3])
    return SpaceInfo(shape=shape, affine=affine, zooms=zooms)  # type: ignore[arg-type]


def validate_common_space(
    images: Sequence[CohortImage],
    *,
    mask: str | Path | None = None,
    tol: float = AFFINE_TOL,
) -> SpaceInfo:
    """Assert every image (and the mask) sits on one grid; return that grid.

    Voxel *i* of subject A and voxel *i* of subject B have to be the same anatomical location or the
    whole model is meaningless — and nothing downstream would notice, because a stack of
    mismatched-but-same-shaped volumes merges and permutes perfectly happily. This is the guard.

    Raises
    ------
    ValueError
        Naming the **first** offending subject and showing its grid against the reference.
    """
    if not images:
        raise ValueError("No images to validate.")

    reference = _space_of(images[0].path)
    for image in images[1:]:
        space = _space_of(image.path)
        if space.shape != reference.shape:
            raise ValueError(
                f"Image space mismatch: {image.label()} is {space.describe()} but "
                f"{images[0].label()} is {reference.describe()} ({images[0].path.name}). "
                "Voxelwise analysis needs every input on one grid — normalise them to a common "
                "template first."
            )
        if not np.allclose(space.affine, reference.affine, atol=tol, rtol=0.0):
            delta = float(np.max(np.abs(space.affine - reference.affine)))
            raise ValueError(
                f"Image affine mismatch: {image.label()} differs from {images[0].label()} by up to "
                f"{delta:g} mm (tolerance {tol:g}).\n"
                f"  {images[0].path.name}:\n{np.array2string(reference.affine, precision=4)}\n"
                f"  {image.path.name}:\n{np.array2string(space.affine, precision=4)}"
            )

    if mask is not None:
        mask_path = Path(mask).expanduser().resolve()
        mask_space = _space_of(mask_path)
        if mask_space.shape != reference.shape or not np.allclose(
            mask_space.affine, reference.affine, atol=tol, rtol=0.0
        ):
            raise ValueError(
                f"Mask {mask_path.name} is {mask_space.describe()} but the images are "
                f"{reference.describe()}. Resample the mask onto the image grid "
                "(nilearn.image.resample_to_img) or omit it to use the default MNI mask."
            )

    log.info(f"Common space verified across {len(images)} image(s): {reference.describe()}")
    return reference


# ──────────────────────────────────────────────────────────────────────────────
# Design and contrasts
# ──────────────────────────────────────────────────────────────────────────────
#: Comparison operators ``--prefilter`` accepts, longest first so ``>=`` is not read as ``>``.
PREFILTER_OPERATORS: tuple[str, ...] = (">=", "<=", "!=", "==", ">", "<", "=")


@dataclass(frozen=True)
class PreFilter:
    """One subject-level inclusion rule, e.g. ``flow_mean__LICA >= 15``.

    Applied to the design frame *before* the design is validated, so it narrows the cohort rather
    than the model: the column it tests need not be an EV, which is the point — keeping only
    subjects with a patent left ICA while modelling the right one is a cohort decision, not a
    covariate.
    """

    column: str
    operator: str
    value: float | str

    @classmethod
    def parse(cls, token: str) -> "PreFilter":
        """Parse ``'flow_mean__LICA>=15'`` (spaces optional) into a rule.

        A bare ``=`` is accepted as ``==``: it is what people type, and there is no assignment
        syntax here for it to be confused with.
        """
        text = str(token).strip()
        if not text:
            raise ValueError("Empty --prefilter.")
        for operator in PREFILTER_OPERATORS:
            head, found, tail = text.partition(operator)
            if not found:
                continue
            column, raw = head.strip(), tail.strip()
            if not column:
                raise ValueError(f"--prefilter {token!r} has no column before {operator!r}.")
            if not raw:
                raise ValueError(f"--prefilter {token!r} has no value after {operator!r}.")
            try:
                value: float | str = float(raw)
            except ValueError:
                # A non-numeric right-hand side is compared as text, so `sex==Female` works on a
                # column that was never recoded; only equality is meaningful there.
                value = raw.strip("'\"")
                if operator not in ("==", "!=", "="):
                    raise ValueError(
                        f"--prefilter {token!r} compares {value!r} with {operator!r}, but a "
                        "non-numeric value only supports == and !=."
                    ) from None
            return cls(column=column, operator="==" if operator == "=" else operator, value=value)
        raise ValueError(
            f"--prefilter {token!r} has no comparison operator. Write it as "
            f"'column>=value', e.g. 'flow_mean__LICA>=15'. Operators: "
            f"{', '.join(o for o in PREFILTER_OPERATORS if o != '=')}."
        )

    def describe(self) -> str:
        """``flow_mean__LICA >= 15`` — what a log line or a manifest should say."""
        return f"{self.column} {self.operator} {self.value!r}"

    def mask(self, frame: Any) -> Any:
        """Boolean mask over *frame*'s rows: True for the subjects this rule keeps.

        A missing value never satisfies a comparison, so a subject with no measurement in the
        tested column is excluded rather than silently kept — the same rule complete-case handling
        applies to the EVs.
        """
        import pandas as pd

        if self.column not in frame.columns:
            available = sorted(str(c) for c in frame.columns)
            raise ValueError(
                f"--prefilter column {self.column!r} is not in the design frame. "
                f"Available columns: {available}"
            )
        column = frame[self.column]
        if isinstance(self.value, str):
            left = column.astype("string").str.strip()
            right = self.value
        else:
            left = pd.to_numeric(column, errors="coerce")
            right = float(self.value)

        if self.operator == ">=":
            keep = left >= right
        elif self.operator == "<=":
            keep = left <= right
        elif self.operator == ">":
            keep = left > right
        elif self.operator == "<":
            keep = left < right
        elif self.operator == "==":
            keep = left == right
        elif self.operator == "!=":
            keep = left != right
        else:  # unreachable: parse() rejects anything else
            raise ValueError(f"Unsupported operator {self.operator!r}.")
        return keep.fillna(False).astype(bool)


def parse_prefilters(tokens: Sequence[str]) -> tuple[PreFilter, ...]:
    """Parse every ``--prefilter`` token, in the order given."""
    return tuple(PreFilter.parse(t) for t in tokens if str(t).strip())


def apply_prefilters(frame: Any, prefilters: Sequence[PreFilter]) -> Any:
    """Narrow *frame* to the subjects passing every rule, logging each step's count.

    Rules combine with AND, and each is logged separately: a filter that removes almost everything
    is indistinguishable from a mis-typed column once only the final count is visible.
    """
    if not prefilters:
        return frame
    out = frame
    for rule in prefilters:
        before = len(out)
        out = out.loc[rule.mask(out)]
        log.info(f"Prefilter {rule.describe()}: {before} → {len(out)} subject(s)")
    if out.empty:
        raise ValueError(
            "No subject passes the --prefilter rule(s): "
            + "; ".join(r.describe() for r in prefilters)
        )
    return out


@dataclass(frozen=True)
class Contrast:
    """One t-contrast: a name and one weight per column of the design matrix."""

    name: str
    weights: tuple[float, ...]

    @classmethod
    def parse(cls, token: str, evs: Sequence[str], *, add_intercept: bool = True) -> "Contrast":
        """Parse a CLI contrast token into weights over the design's columns.

        Two spellings, because two things are natural to write::

            '+flow_mean__LMCA:mca_positive'      one EV, sign only
            '1,0,-1:late_minus_early'            explicit weights, one per column

        The name after ``:`` is optional and defaults to the EV name (or ``c1``-style numbering,
        assigned by the caller). Weights are over the *design columns*, so with
        ``add_intercept=True`` the intercept is column 0 and gets weight 0 unless named.
        """
        raw, _, label = str(token).partition(":")
        raw = raw.strip()
        label = label.strip()
        if not raw:
            raise ValueError(f"Empty contrast in {token!r}.")

        columns = (["intercept"] if add_intercept else []) + list(evs)

        if "," in raw:
            try:
                weights = [float(part) for part in raw.split(",")]
            except ValueError:
                raise ValueError(
                    f"Contrast {token!r} looks like explicit weights but is not all numbers."
                ) from None
            if len(weights) != len(columns):
                raise ValueError(
                    f"Contrast {token!r} has {len(weights)} weight(s) but the design has "
                    f"{len(columns)} column(s): {columns}."
                )
            return cls(name=label or "contrast", weights=tuple(weights))

        sign = 1.0
        name = raw
        if raw[0] in "+-":
            sign = -1.0 if raw[0] == "-" else 1.0
            name = raw[1:].strip()
        if name not in columns:
            raise ValueError(
                f"Contrast {token!r} names {name!r}, which is not a design column: {columns}. "
                "Add it with --ev, or give explicit comma-separated weights."
            )
        weights = [0.0] * len(columns)
        weights[columns.index(name)] = sign
        default = f"{'neg' if sign < 0 else 'pos'}_{name}"
        return cls(name=label or default, weights=tuple(weights))


@dataclass(frozen=True)
class VoxelwiseDesign:
    """The GLM fitted at every voxel: which columns, and which contrasts over them."""

    evs: tuple[str, ...]
    contrasts: tuple[Contrast, ...]
    add_intercept: bool = True
    #: Centre continuous EVs. ``randomise`` expects a demeaned design when an intercept is present:
    #: without it the intercept and the EV share variance and the EV's t-statistic is not the
    #: partial effect anyone means by it.
    demean: bool = True

    def columns(self) -> tuple[str, ...]:
        """Design-matrix column names, intercept first when present."""
        return (("intercept",) if self.add_intercept else ()) + tuple(self.evs)

    def _numeric(self, frame: Any) -> Any:
        """The EV columns of *frame* coerced to float, in design order."""
        import pandas as pd

        missing = [ev for ev in self.evs if ev not in frame.columns]
        if missing:
            available = sorted(str(c) for c in frame.columns)
            raise KeyError(
                f"EV(s) not in the frame: {missing}. Available columns: {available}"
            )
        out = {}
        for ev in self.evs:
            column = frame[ev]
            if pd.api.types.is_numeric_dtype(column):
                out[ev] = pd.to_numeric(column, errors="coerce").astype(float)
            else:
                coerced = pd.to_numeric(column, errors="coerce")
                out[ev] = coerced.astype(float)
        return pd.DataFrame(out, index=frame.index)

    def complete_cases(self, frame: Any) -> Any:
        """Boolean mask over *frame*'s rows: True where every EV is present and finite."""
        numeric = self._numeric(frame)
        return numeric.notna().all(axis=1) & np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)

    def matrix(self, frame: Any) -> np.ndarray:
        """The design matrix for *frame*, one row per row of *frame*, in that order.

        Continuous EVs are centred when ``demean`` is set. A column with two or fewer distinct
        values is treated as a group indicator and centred too — ``randomise`` needs it demeaned for
        the same reason a continuous EV does, and a 0/1 sex column is the common case.
        """
        numeric = self._numeric(frame)
        values = numeric.to_numpy(dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if self.demean and values.size:
            values = values - values.mean(axis=0, keepdims=True)
        if self.add_intercept:
            values = np.column_stack([np.ones(len(values), dtype=float), values])
        return np.ascontiguousarray(values, dtype=float)

    def contrast_matrix(self) -> np.ndarray:
        """One row per contrast, one column per design column."""
        return np.asarray([c.weights for c in self.contrasts], dtype=float)

    def validate(self, frame: Any) -> str:
        """Empty string when the design is fittable on *frame*; otherwise everything wrong with it.

        Reported together rather than one at a time: a design with a missing column *and* a
        mis-sized contrast should say so in one pass, because fixing them is one edit.
        """
        problems: list[str] = []
        n_columns = len(self.columns())

        if not self.evs:
            problems.append("No EVs — pass at least one --ev naming a column of the design frame.")
        if not self.contrasts:
            problems.append("No contrasts — pass at least one --contrast.")

        missing = [ev for ev in self.evs if ev not in getattr(frame, "columns", [])]
        if missing:
            available = sorted(str(c) for c in getattr(frame, "columns", []))
            problems.append(
                f"EV(s) not in the design frame: {missing}. Available columns: {available[:40]}"
                + (" …" if len(available) > 40 else "")
            )

        for contrast in self.contrasts:
            if len(contrast.weights) != n_columns:
                problems.append(
                    f"Contrast {contrast.name!r} has {len(contrast.weights)} weight(s) but the "
                    f"design has {n_columns} column(s) {list(self.columns())}."
                )

        if not missing and self.evs:
            numeric = self._numeric(frame)
            non_numeric = [
                ev for ev in self.evs
                if numeric[ev].notna().sum() == 0 and frame[ev].notna().sum() > 0
            ]
            if non_numeric:
                problems.append(
                    f"EV(s) that are not numeric and could not be coerced: {non_numeric}. "
                    "Recode a categorical covariate as 0/1 before using it as an EV."
                )
            keep = self.complete_cases(frame)
            n_complete = int(keep.sum())
            if n_complete <= n_columns:
                problems.append(
                    f"Only {n_complete} complete case(s) for {n_columns} design column(s) — "
                    "the model has no residual degrees of freedom."
                )
            elif not non_numeric:
                design = self.matrix(frame.loc[keep])
                rank = int(np.linalg.matrix_rank(design))
                if rank < design.shape[1]:
                    problems.append(
                        f"Design matrix is rank-deficient: rank {rank} < {design.shape[1]} "
                        "column(s). Two EVs are collinear, or one is constant after demeaning."
                    )
        return "\n".join(problems)

    # -- VEST writers ---------------------------------------------------------
    # Written directly rather than shelling to Text2Vest so design construction stays unit-testable
    # with no FSL present, and so the contrast names travel into the file.
    def write_mat(self, path: str | Path, frame: Any) -> Path:
        """Write the design matrix in FSL VEST format (``design.mat``)."""
        matrix = self.matrix(frame)
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        heights = np.ptp(matrix, axis=0)
        heights[heights == 0] = 1.0
        lines = [
            f"/NumWaves {matrix.shape[1]}",
            f"/NumPoints {matrix.shape[0]}",
            "/PPheights " + " ".join(f"{v:.6f}" for v in heights),
            "/Matrix",
        ]
        lines += [" ".join(f"{v:.12f}" for v in row) for row in matrix]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def write_con(self, path: str | Path) -> Path:
        """Write the t-contrast matrix in FSL VEST format (``design.con``)."""
        matrix = self.contrast_matrix()
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"/ContrastName{i + 1} {c.name}" for i, c in enumerate(self.contrasts)]
        lines += [
            f"/NumWaves {matrix.shape[1] if matrix.size else len(self.columns())}",
            f"/NumContrasts {matrix.shape[0]}",
            "/PPheights " + " ".join("1.000000" for _ in self.contrasts),
            "/Matrix",
        ]
        lines += [" ".join(f"{v:.12f}" for v in row) for row in matrix]
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out


def parse_contrasts(
    tokens: Sequence[str],
    evs: Sequence[str],
    *,
    add_intercept: bool = True,
) -> tuple[Contrast, ...]:
    """Parse ``--contrast`` tokens, defaulting a blank name to ``c1``, ``c2``, ….

    With no tokens at all, emit one positive contrast per EV — the thing almost every caller wants
    from a one-EV design, and harmless to override.
    """
    if not tokens:
        return tuple(
            Contrast.parse(f"+{ev}:pos_{ev}", evs, add_intercept=add_intercept) for ev in evs
        )
    out: list[Contrast] = []
    for i, token in enumerate(tokens):
        contrast = Contrast.parse(token, evs, add_intercept=add_intercept)
        if contrast.name in {"contrast", ""}:
            contrast = Contrast(name=f"c{i + 1}", weights=contrast.weights)
        out.append(contrast)
    return tuple(out)


def align_design_to_images(
    images: Sequence[CohortImage],
    frame: Any,
    design: VoxelwiseDesign,
    *,
    subject_column: str = "subject_uid",
) -> tuple[list[CohortImage], Any]:
    """Intersect the image list with the design frame and return both, in the same order.

    Subjects with a missing value in any EV are dropped **from the image list too**. The two must
    never fall out of step, which is why this returns them together rather than filtering either
    one in isolation.
    """
    if subject_column not in frame.columns:
        raise ValueError(
            f"The design frame has no {subject_column!r} column; cannot match it to images."
        )

    by_subject = frame.drop_duplicates(subset=[subject_column]).set_index(
        frame.drop_duplicates(subset=[subject_column])[subject_column].astype(str)
    )
    keep_mask = design.complete_cases(by_subject)
    usable = set(by_subject.index[keep_mask].astype(str))

    kept: list[CohortImage] = []
    dropped_missing: list[str] = []
    dropped_incomplete: list[str] = []
    for image in images:
        if image.subject_uid not in by_subject.index:
            dropped_missing.append(image.subject_uid)
        elif image.subject_uid not in usable:
            dropped_incomplete.append(image.subject_uid)
        else:
            kept.append(image)

    if dropped_missing:
        log.info(f"Dropped {len(dropped_missing)} image(s) with no row in the design frame")
        log.debug("No design row: %s", ", ".join(sorted(dropped_missing)))
    if dropped_incomplete:
        log.info(f"Dropped {len(dropped_incomplete)} image(s) with a missing EV value")
        log.debug("Incomplete EVs: %s", ", ".join(sorted(dropped_incomplete)))

    ordered = by_subject.loc[[image.subject_uid for image in kept]].reset_index(drop=True)
    log.info(f"Design aligned to {len(kept)} subject(s)")
    return kept, ordered


# ──────────────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────────────
def merge_4d(images: Sequence[CohortImage], out_path: str | Path) -> Path:
    """Concatenate *images* into one 4D volume with ``fslmerge -t``, in list order.

    Volume *i* of the result is ``images[i]``. Nothing checks that afterwards, so this function is
    the only place the order is established and it never sorts, filters, or dedupes.
    """
    if not images:
        raise ValueError("No images to merge.")
    require_fsl()
    _ensure_fsl_env()

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    argv = ["fslmerge", "-t", str(out)] + [str(image.path) for image in images]
    log.info(f"Merging {len(images)} volume(s) → {out.name}")
    _run_checked(argv, what="fslmerge")

    resolved = _resolve_nifti_path(out)
    if resolved is None:
        raise RuntimeError(f"fslmerge did not produce: {out}")
    if resolved != out:
        log.warning(
            f"fslmerge wrote {resolved.name} (expected {out.name}); "
            "set FSLOUTPUTTYPE=NIFTI_GZ for consistent outputs."
        )
    return resolved


def default_mni_mask(reference: str | Path, out_path: str | Path) -> Path:
    """Nilearn's MNI152 brain mask resampled onto *reference*'s grid.

    Without a mask ``randomise`` tests every voxel in the bounding box, which means it spends most
    of its permutations on air and pays for it in the family-wise correction.
    """
    from nilearn.datasets import load_mni152_brain_mask
    from nilearn.image import resample_to_img
    import nibabel as nib

    out = Path(out_path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    reference_img = nib.load(str(Path(reference).expanduser().resolve()))
    if reference_img.ndim > 3:
        reference_img = nib.Nifti1Image(
            np.asarray(reference_img.dataobj[..., 0]),
            reference_img.affine,
            reference_img.header,
        )
    mask = load_mni152_brain_mask()
    resampled = resample_to_img(
        mask, reference_img, interpolation="nearest", force_resample=True, copy_header=True
    )
    data = (np.asarray(resampled.dataobj) > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(data, resampled.affine), str(out))
    log.info(f"Default MNI152 brain mask → {out.name} ({int(data.sum())} voxel(s))")
    return out


def run_randomise(
    stack: str | Path,
    mat_path: str | Path,
    con_path: str | Path,
    out_root: str | Path,
    *,
    mask: str | Path | None = None,
    n_perm: int = 5000,
    tfce: bool = True,
    voxelwise_corrp: bool = True,
    uncorrected_p: bool = False,
    parallel: bool = False,
    seed: int | None = None,
    demean_data: bool = False,
    extra_args: Sequence[str] = (),
) -> Path:
    """Run ``randomise`` on *stack* and return its output root path prefix.

    ``randomise_parallel`` takes the same argv with a different binary — it splits the permutations
    across jobs and recombines them — so it is a flag rather than a separate function.
    """
    status = require_fsl()
    _ensure_fsl_env()
    if parallel and not status.supports_parallel():
        raise RuntimeError(
            "randomise_parallel is not on PATH; run without --parallel or install fsl-randomise."
        )

    root = Path(out_root).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    argv: list[str] = [
        "randomise_parallel" if parallel else "randomise",
        "-i", str(Path(stack).expanduser().resolve()),
        "-o", str(root),
        "-d", str(Path(mat_path).expanduser().resolve()),
        "-t", str(Path(con_path).expanduser().resolve()),
        "-n", str(int(n_perm)),
    ]
    if mask is not None:
        argv += ["-m", str(Path(mask).expanduser().resolve())]
    if tfce:
        argv.append("-T")
    if voxelwise_corrp:
        argv.append("-x")
    if uncorrected_p:
        argv.append("--uncorrp")
    if demean_data:
        argv.append("-D")
    if seed is not None:
        argv.append(f"--seed={int(seed)}")
    argv += [str(a) for a in extra_args]

    log.info(f"randomise: {n_perm} permutation(s), TFCE={'on' if tfce else 'off'}")
    _run_checked(argv, what="randomise_parallel" if parallel else "randomise")
    return root


# ──────────────────────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────────────────────
#: ``randomise`` output kinds, keyed by the infix it puts before ``_tstat<N>``.
STAT_KINDS: dict[str, str] = {
    "tstat": "t-statistic (unpermuted)",
    "tfce_corrp_tstat": "TFCE, FWE-corrected 1−p",
    "vox_corrp_tstat": "voxelwise FWE-corrected 1−p",
    "tfce_p_tstat": "TFCE, uncorrected 1−p",
    "p_tstat": "uncorrected 1−p",
    "tfce_tstat": "TFCE-enhanced statistic",
    "clustere_corrp_tstat": "cluster-extent FWE-corrected 1−p",
    "clusterm_corrp_tstat": "cluster-mass FWE-corrected 1−p",
}

#: Kinds whose values are 1−p, so a map value above :data:`CORRP_THRESHOLD` means p < 0.05.
CORRP_KINDS: frozenset[str] = frozenset(
    {"tfce_corrp_tstat", "vox_corrp_tstat", "clustere_corrp_tstat", "clusterm_corrp_tstat"}
)


@dataclass(frozen=True)
class VoxelwiseResult:
    """A finished analysis: where it lives, what it produced, and what it was run on."""

    out_dir: Path
    out_root: Path
    maps: dict[str, dict[int, Path]] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def contrast_names(self) -> list[str]:
        """Contrast names in contrast order, from the manifest (``c1``… if it is absent)."""
        names = [str(c.get("name")) for c in self.manifest.get("contrasts", []) if c.get("name")]
        if names:
            return names
        indices = sorted({i for by_index in self.maps.values() for i in by_index})
        return [f"c{i}" for i in indices]

    def contrast_index(self, name: str) -> int:
        """1-based ``randomise`` index of the contrast called *name*."""
        names = self.contrast_names
        if name in names:
            return names.index(name) + 1
        raise KeyError(f"No contrast named {name!r}. Available: {names}")

    def map_path(self, kind: str, contrast: str | int) -> Path:
        """Path to one statistical map, addressed by contrast name or 1-based index."""
        index = contrast if isinstance(contrast, int) else self.contrast_index(str(contrast))
        by_index = self.maps.get(kind) or {}
        if index not in by_index:
            available = sorted(self.maps)
            raise KeyError(
                f"No {kind!r} map for contrast {contrast!r}. Kinds present: {available}"
            )
        return by_index[index]

    def primary_kind(self) -> str:
        """The corrected map a viewer should show by default, preferring TFCE."""
        for kind in ("tfce_corrp_tstat", "vox_corrp_tstat", "clustere_corrp_tstat", "tstat"):
            if self.maps.get(kind):
                return kind
        return next(iter(self.maps), "")

    def summary(self) -> str:
        """One line per contrast naming its corrected map and the suprathreshold voxel count."""
        kind = self.primary_kind()
        if not kind:
            return f"{self.out_root.name}: no maps found"
        lines = [f"{self.out_root.name} — {STAT_KINDS.get(kind, kind)}"]
        for name in self.contrast_names:
            try:
                path = self.map_path(kind, name)
            except KeyError:
                continue
            lines.append(f"  {name}: {path.name} · {count_significant(path)} voxel(s) p<0.05")
        return "\n".join(lines)


def count_significant(path: str | Path, *, threshold: float = CORRP_THRESHOLD) -> int:
    """Voxels above *threshold* in a 1−p map (0.95 → p < 0.05)."""
    import nibabel as nib

    data = np.asarray(nib.load(str(path)).dataobj)
    return int(np.count_nonzero(data > float(threshold)))


_STAT_RX = re.compile(r"^(?P<root>.+?)_(?P<kind>[a-z0-9_]*tstat)(?P<index>\d+)$")


def _discover_maps(out_root: Path) -> dict[str, dict[int, Path]]:
    """Glob ``<root>_*tstat<N>`` next to *out_root* and bucket them by kind and contrast index."""
    maps: dict[str, dict[int, Path]] = {}
    for path in sorted(out_root.parent.glob(f"{out_root.name}_*")):
        if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
            continue
        stem = path.name
        for suffix in (".nii.gz", ".nii"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        match = _STAT_RX.match(stem)
        if match is None or match.group("root") != out_root.name:
            continue
        maps.setdefault(match.group("kind"), {})[int(match.group("index"))] = path
    return maps


def write_manifest(
    out_dir: str | Path,
    *,
    out_root: str | Path,
    images: Sequence[CohortImage],
    design: VoxelwiseDesign,
    n_perm: int,
    mask: str | Path | None,
    cohort: str | None = None,
    include: str = "*",
    image_dir: str | Path | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``manifest.json`` beside the maps: the ordered subject list, design, and provenance.

    The ordered subject list is the part that matters. ``randomise``'s outputs carry no record of
    which volume was which subject, so without this the result cannot be re-checked, re-plotted
    against a covariate, or trusted after the image directory changes.
    """
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    status = fsl_backend_status()
    payload: dict[str, Any] = {
        "schema": "nvitk.voxelwise.v1",
        "out_root": Path(out_root).name,
        "n_subjects": len(images),
        "subjects": [
            {"order": i, "subject_uid": im.subject_uid, "session_id": im.session_id,
             "image": str(im.path)}
            for i, im in enumerate(images)
        ],
        "evs": list(design.evs),
        "columns": list(design.columns()),
        "add_intercept": bool(design.add_intercept),
        "demean": bool(design.demean),
        "contrasts": [
            {"index": i + 1, "name": c.name, "weights": list(c.weights)}
            for i, c in enumerate(design.contrasts)
        ],
        "n_perm": int(n_perm),
        "mask": str(mask) if mask else None,
        "cohort": cohort,
        "include": include,
        "image_dir": str(image_dir) if image_dir else None,
        "fsl_version": status.versions.get("fsl", ""),
    }
    if extra:
        payload.update(dict(extra))
    path = out / "manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_voxelwise_result(out_dir: str | Path) -> VoxelwiseResult:
    """Load a finished results folder — no FSL, no re-run, no database.

    Both viewers depend on this: the napari tool and the Statmodels display each open a folder
    somebody else produced, possibly on the cluster, possibly months earlier.
    """
    directory = Path(out_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"Results folder not found: {directory}")

    manifest: dict[str, Any] = {}
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(f"Unreadable manifest.json in {directory}: {exc}")

    root_name = str(manifest.get("out_root") or "").strip()
    candidates: list[Path] = []
    if root_name:
        candidates = [directory / root_name]
    else:
        # No manifest: recover the root from the file names themselves. Every randomise output is
        # ``<root>_<kind>tstat<N>``, so the roots are whatever survives stripping that suffix.
        roots: set[str] = set()
        for path in directory.glob("*tstat*"):
            stem = path.name
            for suffix in (".nii.gz", ".nii"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            match = _STAT_RX.match(stem)
            if match:
                roots.add(match.group("root"))
        candidates = [directory / r for r in sorted(roots)]

    for root in candidates:
        maps = _discover_maps(root)
        if maps:
            log.info(
                f"Loaded voxelwise result {root.name}: "
                f"{sum(len(v) for v in maps.values())} map(s), {len(maps)} kind(s)"
            )
            return VoxelwiseResult(out_dir=directory, out_root=root, maps=maps, manifest=manifest)

    raise FileNotFoundError(
        f"No randomise outputs (<root>_*tstat<N>.nii.gz) in {directory}. "
        "Point at the folder that holds the maps, not its parent."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Design frame + one-call entry point
# ──────────────────────────────────────────────────────────────────────────────
#: Columns the analysis frame *derives* rather than loads, mapped to what they are derived from.
#: ``age_c`` is ``age_at_mri`` mean-centred, computed by ``finalize_analysis_frame`` only when its
#: source is in the frame — so naming it as an EV has to pull the source in, or it silently is not
#: there and the design fails validation for a reason that reads like a typo.
DERIVED_COVARIATES: dict[str, str] = {"age_c": "age_at_mri"}


def build_design_frame(
    repo: Any,
    *,
    pipeline: str,
    pipeline_kind: str = "qvtpy",
    feature: str = "flow_mean",
    grouping: str = "vessel",
    atlas: str | None = None,
    covariates: Sequence[str] = (),
) -> tuple[Any, dict[str, Any]]:
    """One row per subject: the named measurement spread by region, plus subject covariates.

    ``grain="subject"`` is the only grain that can relate measurements whose parcellations name
    different kinds of region — a 4D-flow vessel against an ASL voxel grid — which is exactly the
    cross-modal case this module exists for. Each region becomes its own column
    (``flow_mean__LMCA``), and those column names are what ``--ev`` refers to.

    Covariate names that do not resolve are simply absent from the frame;
    :meth:`VoxelwiseDesign.validate` then names them against the columns that *are* there, which
    is a better error than a lookup failure here would be.
    """
    from nvitk.stats._statmodels_frames import (
        MeasurementSpec,
        build_multi_feature_analysis_frame,
    )

    spec = MeasurementSpec(
        pipeline_kind=str(pipeline_kind),
        pipeline=str(pipeline),
        feature=str(feature),
        grouping=str(grouping),
        atlas=atlas,
    )
    requested = [str(c).strip() for c in covariates if str(c).strip()]
    for name in list(requested):
        source = DERIVED_COVARIATES.get(name)
        if source and source not in requested:
            log.debug("Requesting %r so %r can be derived.", source, name)
            requested.append(source)

    frame, meta = build_multi_feature_analysis_frame(
        repo,
        measurements=[spec],
        clinical_vars=requested or None,
        grain="subject",
        attach_qc=False,
    )
    log.info(
        f"Design frame: {len(frame)} subject(s) × {len(frame.columns)} column(s) "
        f"from {pipeline_kind}/{pipeline}/{feature}"
    )
    return frame, meta


def run_voxelwise(
    image_dir: str | Path,
    out_dir: str | Path,
    *,
    evs: Sequence[str],
    contrasts: Sequence[str] = (),
    prefilters: Sequence[str] = (),
    repo: Any = None,
    frame: Any = None,
    include: str = "*",
    exclude_csv: str | Path | None = None,
    id_pattern: str | None = DEFAULT_ID_PATTERN,
    namespaces: Sequence[str] = DEFAULT_ID_NAMESPACES,
    on_duplicate: str = "error",
    cohort: str | None = None,
    cohort_id: str | bool | None = None,
    pipeline_kind: str = "qvtpy",
    feature: str = "flow_mean",
    grouping: str = "vessel",
    atlas: str | None = None,
    mask: str | Path | None = None,
    n_perm: int = 5000,
    tfce: bool = True,
    voxelwise_corrp: bool = True,
    uncorrected_p: bool = False,
    parallel: bool = False,
    seed: int | None = None,
    add_intercept: bool = True,
    demean: bool = True,
    out_name: str = "randomise",
    keep_stack: bool = True,
) -> VoxelwiseResult:
    """Resolve → validate → design → merge → ``randomise``. The whole analysis in one call.

    The subject set is ``images found ∩ cohort ∩ prefilters ∩ design-frame complete cases``,
    applied in that order with a count logged at each step. A cohort that silently removes half the images is
    otherwise indistinguishable from a bad include pattern, which is the single most common way to
    get a plausible-looking wrong answer out of this.
    """
    require_fsl()
    if repo is None:
        from nvitk.db.repo import get_repo

        repo = get_repo()

    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 1. Images found -----------------------------------------------------------
    images = resolve_cohort_images(
        image_dir,
        repo=repo,
        include=include,
        exclude_csv=exclude_csv,
        id_pattern=id_pattern,
        namespaces=namespaces,
        on_duplicate=on_duplicate,
    )
    n_found = len(images)

    # 2. ∩ cohort ---------------------------------------------------------------
    if cohort:
        allowed = cohort_subjects(repo, cohort)
        dropped = [im.subject_uid for im in images if im.subject_uid not in allowed]
        images = [im for im in images if im.subject_uid in allowed]
        log.info(f"Cohort {cohort!r}: {n_found} → {len(images)} image(s)")
        if dropped:
            log.debug("Not in cohort: %s", ", ".join(sorted(dropped)))
        if not images:
            raise ValueError(
                f"No image matched cohort {cohort!r}. The directory holds {n_found} resolvable "
                "image(s), none of which has a measurement from that pipeline."
            )
    n_cohort = len(images)

    if cohort_id not in (None, False):
        members = cohort_id_subjects(repo, str(cohort_id))
        if members is not None:
            images = [im for im in images if im.subject_uid in members]
            log.info(f"Cohort id {cohort_id!r}: {n_cohort} → {len(images)} image(s)")
            n_cohort = len(images)

    # 3. ∩ design-frame complete cases -------------------------------------------
    if frame is None:
        if not cohort:
            raise ValueError(
                "Pass --cohort <pipeline_id> (or a ready-made frame) so the design matrix has a "
                "measurement source. `nvitk-voxelwise cohorts` lists the registered ids."
            )
        # A prefilter column need not be an EV, but it still has to be *in* the frame — so it is
        # requested alongside them. A name that is already a measurement column is a no-op here.
        rules = parse_prefilters(list(prefilters))
        frame, _meta = build_design_frame(
            repo,
            pipeline=cohort,
            pipeline_kind=pipeline_kind,
            feature=feature,
            grouping=grouping,
            atlas=atlas,
            covariates=list(evs) + [r.column for r in rules],
        )
    else:
        rules = parse_prefilters(list(prefilters))

    # Before the design is validated: a prefilter narrows the cohort, and rank or degrees-of-freedom
    # checks should see the subjects that will actually be fitted.
    n_before_prefilter = len(frame)
    frame = apply_prefilters(frame, rules)

    design = VoxelwiseDesign(
        evs=tuple(str(e) for e in evs),
        contrasts=parse_contrasts(list(contrasts), [str(e) for e in evs], add_intercept=add_intercept),
        add_intercept=bool(add_intercept),
        demean=bool(demean),
    )
    problems = design.validate(frame)
    if problems:
        raise ValueError(f"Design is not fittable:\n{problems}")

    images, aligned = align_design_to_images(images, frame, design)
    if len(images) <= len(design.columns()):
        raise ValueError(
            f"Only {len(images)} subject(s) survive the intersection for "
            f"{len(design.columns())} design column(s). "
            f"Found {n_found} image(s), {n_cohort} in cohort."
        )
    log.info(
        f"Intersection: {n_found} image(s) found · {n_cohort} in cohort"
        f"{f' {cohort}' if cohort else ''}"
        + (f" · {len(frame)} of {n_before_prefilter} pass the prefilter(s)" if rules else "")
        + f" · {len(images)} complete case(s)"
    )

    # 4. One grid ----------------------------------------------------------------
    validate_common_space(images, mask=mask)

    # 5. Merge, mask, run ---------------------------------------------------------
    mat_path = design.write_mat(out / "design.mat", aligned)
    con_path = design.write_con(out / "design.con")
    stack = merge_4d(images, out / "stack_4d.nii.gz")
    mask_path = Path(mask).expanduser().resolve() if mask else default_mni_mask(stack, out / "mask.nii.gz")

    out_root = run_randomise(
        stack,
        mat_path,
        con_path,
        out / out_name,
        mask=mask_path,
        n_perm=n_perm,
        tfce=tfce,
        voxelwise_corrp=voxelwise_corrp,
        uncorrected_p=uncorrected_p,
        parallel=parallel,
        seed=seed,
    )

    write_manifest(
        out,
        out_root=out_root,
        images=images,
        design=design,
        n_perm=n_perm,
        mask=mask_path,
        cohort=cohort,
        include=include,
        image_dir=image_dir,
        extra={
            "n_images_found": n_found,
            "n_in_cohort": n_cohort,
            "prefilters": [r.describe() for r in rules],
            "n_before_prefilter": int(n_before_prefilter),
            "feature": feature,
            "grouping": grouping,
            "pipeline_kind": pipeline_kind,
            "tfce": bool(tfce),
        },
    )
    if not keep_stack:
        for path in (stack,):
            try:
                path.unlink()
            except OSError as exc:
                log.debug("Could not remove %s: %s", path, exc)

    result = load_voxelwise_result(out)
    log.info(result.summary())
    return result


__all__ = [
    "AFFINE_TOL",
    "CORRP_KINDS",
    "CORRP_THRESHOLD",
    "DEFAULT_ID_NAMESPACES",
    "DEFAULT_ID_PATTERN",
    "DUPLICATE_POLICIES",
    "PREFILTER_OPERATORS",
    "DERIVED_COVARIATES",
    "FSL_BINARIES",
    "STAT_KINDS",
    "CohortImage",
    "CohortOption",
    "Contrast",
    "PreFilter",
    "FslBackendStatus",
    "SpaceInfo",
    "VoxelwiseDesign",
    "VoxelwiseResult",
    "align_design_to_images",
    "apply_prefilters",
    "available_cohorts",
    "build_design_frame",
    "cohort_id_subjects",
    "cohort_subjects",
    "count_significant",
    "default_mni_mask",
    "fsl_backend_status",
    "load_voxelwise_result",
    "merge_4d",
    "parse_contrasts",
    "parse_prefilters",
    "require_fsl",
    "resolve_cohort_images",
    "run_randomise",
    "run_voxelwise",
    "validate_common_space",
    "write_manifest",
]
