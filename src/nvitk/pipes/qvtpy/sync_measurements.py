"""Publish qvtpy stage-6 measurements into ``image_measurements`` (pipeline ``4dflow_v3``).

Scans an ``--output-root`` for subjects with stage-6 CSV outputs and upserts them into
the DB. This is a measurement-sync CLI and is distinct from the DICOM ``--from-source``
flag on the pipeline runner: here ``--from-source`` selects which DataRepo to write to
(``local`` settings repo vs. the ``sge`` cluster dataset root).
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.pipes.qvtpy import config as cfg
from nvitk.pipes.qvtpy.common.db_publish import publish_stage6, resolve_repo

log = Logger()


def _stage6_dir(output_root: Path, subject: str) -> Path:
    return output_root / subject / cfg.QVT_SUBDIR / cfg.STAGE6_MEASURE_DIR


def _subjects_with_stage6(output_root: Path) -> list[str]:
    if not output_root.is_dir():
        return []
    out: list[str] = []
    for subj_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        s6 = _stage6_dir(output_root, subj_dir.name)
        if (s6 / "loc_measurements.csv").is_file() or (s6 / "vessel_hemodynamics.csv").is_file():
            out.append(subj_dir.name)
    return out


@click.command("qvtpy-sync-measurements")
@click.option("--output-root", type=click.Path(path_type=Path), required=True)
@click.option("--subjects", default="", help="Comma-separated subject ids (default: all with stage6).")
@click.option(
    "--from-source",
    type=click.Choice(["local", "sge"], case_sensitive=False),
    default="local",
    show_default=True,
    help="Target DataRepo: local settings repo or the SGE cluster dataset root.",
)
@click.option("--build-sqlite-index/--no-build-sqlite-index", default=True, show_default=True)
def main(output_root: Path, subjects: str, from_source: str, build_sqlite_index: bool) -> None:
    Logger()
    subject_list = [s.strip() for s in subjects.split(",") if s.strip()] or _subjects_with_stage6(
        output_root
    )
    if not subject_list:
        log.warning(f"No subjects with stage6 measurements under {output_root}")
        return
    repo = resolve_repo(prefer_sge=(from_source.lower() == "sge"))
    total = 0
    for subject in subject_list:
        rows = publish_stage6(
            subject_uid=subject,
            stage6_dir=_stage6_dir(output_root, subject),
            repo=repo,
            build_sqlite_index=False,
        )
        total += int(len(rows))
        log.info(f"[{subject}] published {len(rows)} image_measurements row(s)")
    if build_sqlite_index:
        repo.build_sqlite_index()
    log.info(f"qvtpy sync: {total} row(s) across {len(subject_list)} subject(s)")


__all__ = ["main"]


if __name__ == "__main__":
    main()
