"""SGE submission helpers for publishing PESA-Health measurements to the cluster DB."""

from __future__ import annotations

from pathlib import Path

from nvitk.db.settings_paths import configured_sge_dataset_root


def pesa_health_sge_db_submission() -> tuple[dict[str, str], tuple[tuple[Path, str], ...]]:
    """Return ``extra_env`` and Singularity ``extra_host_binds`` for cluster DB writes.

    Uses ``dataset.sge_root`` from ``.nvitk/settings.json`` when set (path need not exist on
    the submission host). Workers set ``NVITK_DATASET_ROOT`` and ``NVITK_SGE=1``.
    """
    root = configured_sge_dataset_root()
    if root is None:
        return {}, ()
    root_s = str(root)
    return (
        {"NVITK_SGE": "1", "NVITK_DATASET_ROOT": root_s},
        ((root, root_s),),
    )
