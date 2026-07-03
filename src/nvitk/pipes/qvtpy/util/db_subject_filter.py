"""Pre-filter XNAT subject lists using the catalog ``scans`` table."""

from __future__ import annotations

from pathlib import Path

from nvitk.core.logger import Logger
from nvitk.db.repo import DataRepo
from nvitk.db.xnat import (
    filter_subjects_by_asset_slots,
    project_subject_asset_slots,
    xnat_sequence_to_asset_slot,
)
from nvitk.pipes.qvtpy.stage0_download import DEFAULT_SEQUENCES

log = Logger()

QVT_REQUIRED_ASSET_SLOTS: frozenset[str] = frozenset(
    xnat_sequence_to_asset_slot(seq) for seq in DEFAULT_SEQUENCES
)


def filter_subjects_by_qvtpy_scan_availability(
    subject_labels: list[str],
    *,
    database_root: Path | str,
    project_id: str,
) -> list[str]:
    """Keep only subjects indexed with all qvtpy sequences (TOF + 3× 4DFlow).

    Uses the same ``asset_slot`` logic as the GUI data browser XNAT scan filter.
    """
    root = Path(database_root).expanduser().resolve()
    repo = DataRepo(root, auto_scaffold=False, use_sqlite=True)
    if not repo.catalog.table_exists("scans") or not repo.catalog.table_exists("sessions"):
        raise ValueError(
            f"Database at {root} has no scans/sessions tables; run nvitk-xnat-sync first."
        )

    subject_slots = project_subject_asset_slots(repo, str(project_id))
    if not subject_slots:
        raise ValueError(
            f"No indexed scans for XNAT project {project_id!r} under {root}."
        )

    eligible = set(
        filter_subjects_by_asset_slots(
            subject_slots,
            set(QVT_REQUIRED_ASSET_SLOTS),
            match_all=True,
        )
    )
    requested = list(subject_labels)
    filtered = [s for s in requested if s in eligible]
    skipped = [s for s in requested if s not in eligible]

    log.info(
        f"DB scan pre-filter ({project_id}): "
        f"{len(filtered)}/{len(requested)} subject(s) have "
        f"{', '.join(sorted(QVT_REQUIRED_ASSET_SLOTS))}"
    )
    if skipped:
        preview = ", ".join(skipped[:8])
        suffix = f" ... (+{len(skipped) - 8} more)" if len(skipped) > 8 else ""
        log.info(f"  excluded: {preview}{suffix}")

    return filtered


__all__ = [
    "QVT_REQUIRED_ASSET_SLOTS",
    "filter_subjects_by_qvtpy_scan_availability",
]
