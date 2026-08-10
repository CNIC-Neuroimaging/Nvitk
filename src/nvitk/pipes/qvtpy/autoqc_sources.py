"""
Recover 4D-flow measurements the dataset is missing, from the pipeline's results tree.

Description
-----------
:mod:`~nvitk.pipes.qvtpy.stage9_autoqc` scores what stage 6 **published**. When the import has not
run, or has run only partially, the numbers exist on disk but not in the dataset — and the QC then
either refuses to run or scores every vessel as patent because the areas are missing.

This module closes that gap. It reads ``loc_measurements.csv`` straight out of each subject's stage-6
directory and returns the same long ``(subject_uid, region_id, value)`` frames the dataset would have
produced, so the scoring path does not care where the numbers came from.

Where the results live
----------------------
``--submit local``  the results root on this machine.
``--submit sge``    the cluster's results root, fetched over SFTP into a temporary directory that is
                    removed afterwards. Only the stage-6 CSVs are pulled, not whole subject trees —
                    a QC pass has no business moving gigabytes of NIfTI.
``--submit xnat``   each session's ``qvtpy`` resource, downloaded into a temporary directory. XNAT
                    serves a resource as one archive, so unlike the SFTP path this cannot fetch a
                    single file — the whole resource comes down and is discarded afterwards.

Units
-----
Stage 6 writes ``loc_mean_flow_ml_s`` per **second**. The literature bands are per minute. The
conversion is left to :func:`~nvitk.pipes.qvtpy.stage9_autoqc.infer_flow_scale`, which decides from
the magnitude rather than from the column name — so a results tree written by an older stage that
used different units still scores correctly.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────────────────────
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy import config as cfg

log = Logger()

#: Stage-6 CSV holding one row per localized cross-section.
LOC_MEASUREMENTS = "loc_measurements.csv"

#: Stage-6 CSV holding one row per dense PITC/PWV station (along-segment CV input).
PITC_PROFILE = "pitc_profile.csv"

#: Stage-6 CSVs the remote fetch mirrors locally.
STAGE6_FETCH_FILES: tuple[str, ...] = (LOC_MEASUREMENTS, PITC_PROFILE)

#: Dataset variable → the stage-6 column it is written from.
VARIABLE_TO_LOC_COLUMN: dict[str, str] = {
    "flow_mean": "loc_mean_flow_ml_s",
    "cross_section_area": "loc_cross_section_area_mm2",
    "velocity_mean": "loc_mean_velocity_mm_s",
    "pi": "loc_pi",
    "ri": "loc_ri",
}

#: How a LOC row is identified as a vessel. ``vessel_name`` is the published spelling.
REGION_COLUMNS: tuple[str, ...] = ("vessel_name", "vessel_id")


@dataclass
class ResultsSource:
    """Where to look for stage-6 outputs, and how to reach them."""

    submit: str = "local"
    results_root: Path | None = None
    host: str = ""
    user: str = ""
    password: str = ""
    port: int = 22
    #: XNAT connection config, for ``submit="xnat"``.
    xnat_config: Any = None
    xnat_project: str = ""
    #: Populated when a remote fetch staged files locally; removed by :meth:`cleanup`.
    _staged: Path | None = field(default=None, repr=False)

    def mode(self) -> str:
        """Normalized submit mode."""
        return str(self.submit).strip().lower()

    def is_remote(self) -> bool:
        """Whether the results have to be fetched before they can be read."""
        return self.mode() in {"sge", "xnat"}

    def is_xnat(self) -> bool:
        """Whether the results come from XNAT session resources."""
        return self.mode() == "xnat"

    def root(self) -> Path:
        """The results root to read from, defaulting to the configured one for this submit mode."""
        if self.results_root is not None:
            return Path(self.results_root)
        if self.is_xnat():
            # Nothing on a filesystem to point at; the staging directory becomes the root.
            return Path(self._staged) if self._staged else Path(".")
        return Path(
            cfg.DEFAULT_RESULTS_ROOT if self.mode() == "sge" else cfg.LOCAL_DEFAULT_RESULTS_ROOT
        )

    def cleanup(self) -> None:
        """Remove anything a remote fetch staged locally."""
        if self._staged is not None and self._staged.exists():
            log.info("Removing the staged results: %s", self._staged)
            shutil.rmtree(self._staged, ignore_errors=True)
        self._staged = None


def stage6_dir(results_root: Path, subject: str) -> Path:
    """``<results_root>/<subject>/qvtpy/stage6_measure``."""
    return Path(results_root) / subject / cfg.QVT_SUBDIR / cfg.STAGE6_MEASURE_DIR


def discover_subjects(results_root: Path) -> list[str]:
    """Subjects under *results_root* that actually have a stage-6 measurements CSV."""
    root = Path(results_root)
    if not root.is_dir():
        return []
    return sorted(
        entry.name for entry in root.iterdir()
        if entry.is_dir() and (stage6_dir(root, entry.name) / LOC_MEASUREMENTS).is_file()
    )


def read_loc_measurements(
    results_root: Path, subjects: Sequence[str] | None = None
) -> pd.DataFrame:
    """
    Concatenate every subject's ``loc_measurements.csv`` into one frame.

    Returns
    -------
    pandas.DataFrame
        The stage-6 columns plus ``subject_uid``. Empty when nothing was found — reported by the
        caller rather than raised here, since a partially-populated tree is normal mid-run.
    """
    root = Path(results_root)
    wanted = list(subjects) if subjects is not None else discover_subjects(root)
    frames: list[pd.DataFrame] = []
    for subject in wanted:
        path = stage6_dir(root, subject) / LOC_MEASUREMENTS
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            log.warning("Could not read %s (%s) — skipping this subject.", path, exc)
            continue
        frame["subject_uid"] = subject
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info(
        "Read %d LOC row(s) for %d subject(s) from %s", len(out), len(frames), root
    )
    return out


def load_pitc_profiles(
    results_root: Path, subjects: Sequence[str] | None = None
) -> pd.DataFrame:
    """
    Concatenate every subject's ``pitc_profile.csv`` into one frame with ``subject_uid``.

    Returns an empty frame when nothing is found — along-segment CV is optional and must not
    take the rest of autoQC down with it.
    """
    root = Path(results_root)
    wanted = list(subjects) if subjects is not None else discover_subjects(root)
    frames: list[pd.DataFrame] = []
    for subject in wanted:
        path = stage6_dir(root, subject) / PITC_PROFILE
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            log.warning("Could not read %s (%s) — skipping this subject.", path, exc)
            continue
        frame["subject_uid"] = subject
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info(
        "Read %d PITC-profile station(s) for %d subject(s) from %s",
        len(out), len(frames), root,
    )
    return out


def long_measurements(
    loc: pd.DataFrame, variable_id: str
) -> pd.DataFrame:
    """
    One variable as long ``(subject_uid, region_id, value)`` rows, matching the dataset's shape.

    Several LOC rows can share a vessel — stage 6 writes one per localized cross-section — so they
    are averaged per vessel, which is the same time-averaged quantity the dataset publishes.
    """
    empty = pd.DataFrame(columns=["subject_uid", "region_id", "value"])
    column = VARIABLE_TO_LOC_COLUMN.get(variable_id)
    if loc is None or loc.empty or column is None or column not in loc.columns:
        return empty
    region = next((c for c in REGION_COLUMNS if c in loc.columns), "")
    if not region or "subject_uid" not in loc.columns:
        return empty

    out = pd.DataFrame({
        "subject_uid": loc["subject_uid"].astype(str),
        "region_id": loc[region].astype(str),
        "value": pd.to_numeric(loc[column], errors="coerce"),
    }).dropna(subset=["value"])
    if out.empty:
        return empty
    return out.groupby(["subject_uid", "region_id"], as_index=False)["value"].mean()


# ---------------------------------------------------------------------------
# Remote retrieval
# ---------------------------------------------------------------------------
def fetch_stage6_csvs(
    source: ResultsSource, subjects: Sequence[str] | None = None
) -> Path:
    """
    Pull each subject's stage-6 CSV from the cluster into a temporary tree, and return its root.

    Only the stage-6 CSVs in :data:`STAGE6_FETCH_FILES` are transferred, mirrored into the same
    ``<subject>/qvtpy/stage6_measure/`` layout so the local reader is unchanged. A QC pass has no
    reason to move the NIfTIs, and on a full cohort that is the difference between seconds and
    hours.

    The caller must call :meth:`ResultsSource.cleanup` — or use :func:`open_results` — to remove it.
    """
    from nvitk.cluster.remote_transfer import remote_path_exists, sftp_session

    root = source.root()
    source.cleanup()      # a second fetch on the same source must not orphan the first staging
    staged = Path(tempfile.mkdtemp(prefix="nvitk-autoqc-results-"))
    source._staged = staged
    n_files = 0

    # ``sftp_session`` yields ``(ssh_client, sftp)``; binding it as one name gave the tuple, whose
    # ``listdir`` does not exist — the cluster looked empty rather than erroring.
    with sftp_session(
        host=source.host, user=source.user, password=source.password, port=source.port
    ) as (_ssh, sftp):
        names = list(subjects) if subjects is not None else _remote_subjects(sftp, root)
        log.info("Fetching stage-6 measurements for %d subject(s) from %s", len(names), root)
        for subject in names:
            for filename in STAGE6_FETCH_FILES:
                remote = (
                    f"{str(root).rstrip('/')}/{subject}/{cfg.QVT_SUBDIR}/"
                    f"{cfg.STAGE6_MEASURE_DIR}/{filename}"
                )
                if not remote_path_exists(sftp, remote):
                    continue
                local = stage6_dir(staged, subject) / filename
                local.parent.mkdir(parents=True, exist_ok=True)
                try:
                    sftp.get(remote, str(local))
                except Exception as exc:
                    log.warning("Could not fetch %s (%s) — skipping.", remote, exc)
                    continue
                n_files += 1

    log.ok("Staged %d stage-6 CSV(s) under %s", n_files, staged)
    if not n_files:
        log.warning(
            "No stage-6 measurements found under %s. Check --results-root and that stage 6 has run.",
            root,
        )
    return staged




def fetch_stage6_xnat(
    source: ResultsSource, subjects: Sequence[str] | None = None
) -> Path:
    """
    Download each session's ``qvtpy`` resource from XNAT and keep only the stage-6 measurements.

    XNAT serves a resource as a single archive, so — unlike the SFTP path — there is no way to ask
    for one file. The whole ``qvtpy`` resource comes down per session into a scratch directory, the
    stage-6 CSV is copied into the mirrored layout, and the rest is deleted immediately. That keeps
    peak disk to one subject rather than a cohort.

    Returns the staging root, which the caller must clean up.
    """
    import shutil as _shutil
    import tempfile as _tempfile

    from nvitk.db.xnat import connect_xnat
    from nvitk.db.xnat_pipeline_resources import download_experiment_resource

    source.cleanup()      # as above
    staged = Path(_tempfile.mkdtemp(prefix="nvitk-autoqc-xnat-"))
    source._staged = staged
    wanted = {str(s) for s in subjects} if subjects is not None else None
    n_files = 0

    with connect_xnat(source.xnat_config) as session:
        projects = (
            [session.projects[source.xnat_project]] if source.xnat_project
            else list(session.projects.values())
        )
        for project in projects:
            for subject_label, subject in dict(project.subjects).items():
                if wanted is not None and str(subject_label) not in wanted:
                    continue
                for experiment in dict(subject.experiments).values():
                    scratch = Path(_tempfile.mkdtemp(prefix="nvitk-autoqc-xnat-one-"))
                    try:
                        try:
                            resource_dir = download_experiment_resource(
                                experiment, cfg.QVT_SUBDIR, scratch, overwrite=True
                            )
                        except LookupError:
                            continue          # this session has no qvtpy resource
                        except Exception as exc:
                            log.warning(
                                "Could not download the qvtpy resource for %s (%s).",
                                subject_label, exc,
                            )
                            continue
                        found = list(Path(resource_dir).rglob(LOC_MEASUREMENTS))
                        if not found:
                            continue
                        local_dir = stage6_dir(staged, str(subject_label))
                        local_dir.mkdir(parents=True, exist_ok=True)
                        _shutil.copy2(found[0], local_dir / LOC_MEASUREMENTS)
                        # Prefer a pitc_profile next to the LOC CSV when the resource has one.
                        profile_candidates = list(Path(resource_dir).rglob(PITC_PROFILE))
                        if profile_candidates:
                            _shutil.copy2(profile_candidates[0], local_dir / PITC_PROFILE)
                        n_files += 1
                    finally:
                        # One subject's worth of scratch at a time, whatever happened above.
                        _shutil.rmtree(scratch, ignore_errors=True)

    log.ok("Staged %d stage-6 CSV(s) from XNAT under %s", n_files, staged)
    if not n_files:
        log.warning(
            "No qvtpy resource carried %s on XNAT. Check the project and that stage 8 has uploaded.",
            LOC_MEASUREMENTS,
        )
    return staged


def _remote_subjects(sftp: Any, results_root: Path) -> list[str]:
    """Subject directories directly under the remote results root."""
    try:
        return sorted(sftp.listdir(str(results_root)))
    except Exception as exc:
        log.warning("Could not list %s (%s).", results_root, exc)
        return []


def open_results(
    source: ResultsSource, subjects: Sequence[str] | None = None
) -> tuple[pd.DataFrame, Path]:
    """
    Read the stage-6 measurements for *source*, fetching them first when they are remote.

    Returns
    -------
    (loc, root)
        The concatenated LOC frame and the root it was read from. For a remote source the root is
        the temporary staging directory; call :meth:`ResultsSource.cleanup` when done with it.
    """
    if source.is_xnat():
        root = fetch_stage6_xnat(source, subjects)
    elif source.is_remote():
        root = fetch_stage6_csvs(source, subjects)
    else:
        root = source.root()
    if not source.is_remote() and not Path(root).is_dir():
        raise FileNotFoundError(
            f"Results root does not exist: {root}. Pass --results-root, or --submit sge to read "
            f"them from the cluster."
        )
    return read_loc_measurements(root, subjects), Path(root)


def recover_missing(
    needed: Iterable[str],
    source: ResultsSource,
    *,
    subjects: Sequence[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """
    Load whichever of *needed* the results tree can supply.

    Parameters
    ----------
    needed : iterable of str
        Dataset variable ids that are missing — typically ``flow_mean`` and/or
        ``cross_section_area``.

    Returns
    -------
    (frames, report)
        ``frames`` maps each recovered variable to its long frame; a variable the tree cannot
        supply is simply absent. ``report`` carries ``root``, ``n_subjects``, ``recovered`` and
        ``unavailable`` for the log.
    """
    wanted = [str(v) for v in needed]
    unknown = [v for v in wanted if v not in VARIABLE_TO_LOC_COLUMN]
    if unknown:
        log.info(
            "autoqc: %s cannot come from stage 6 — it writes %s.",
            ", ".join(unknown), ", ".join(sorted(VARIABLE_TO_LOC_COLUMN)),
        )

    loc, root = open_results(source, subjects)
    frames: dict[str, pd.DataFrame] = {}
    for variable in wanted:
        recovered = long_measurements(loc, variable)
        if not recovered.empty:
            frames[variable] = recovered

    report = {
        "root": str(root),
        "n_subjects": int(loc["subject_uid"].nunique()) if not loc.empty else 0,
        "recovered": sorted(frames),
        "unavailable": sorted(set(wanted) - set(frames)),
    }
    if frames:
        log.ok(
            "autoqc: recovered %s from the results tree (%d subject(s)).",
            ", ".join(report["recovered"]), report["n_subjects"],
        )
    return frames, report


__all__ = [
    "LOC_MEASUREMENTS",
    "PITC_PROFILE",
    "REGION_COLUMNS",
    "STAGE6_FETCH_FILES",
    "VARIABLE_TO_LOC_COLUMN",
    "ResultsSource",
    "discover_subjects",
    "fetch_stage6_csvs",
    "fetch_stage6_xnat",
    "load_pitc_profiles",
    "long_measurements",
    "open_results",
    "read_loc_measurements",
    "recover_missing",
    "stage6_dir",
]
