"""Concatenate per-subject stage-3 Excel rows into one batch SummaryCodebook."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nvitk.core.logger import Logger
from nvitk.pipes.pesa_fat.common.paths import BatchLayout

log = Logger()


def aggregate_stage3_summary(
    lay: BatchLayout,
    subjects: list[str],
    pipeline: str,
) -> Path | None:
    """Merge ``per_subject/<subj>.xlsx`` into ``<batch>_SummaryCodebook.xlsx``.

    Parameters
    ----------
    pipeline
        ``ct-pet-v5`` or ``dixon-v5``.
    """
    pl = pipeline.strip().lower()
    if pl == "ct-pet-v5":
        from nvitk.pipes.pesa_fat.ct_pet_v5 import config as cfg
        from nvitk.pipes.pesa_fat.ct_pet_v5 import stage3_measure
    elif pl == "dixon-v5":
        from nvitk.pipes.pesa_fat.dixon_v5 import config as cfg
        from nvitk.pipes.pesa_fat.dixon_v5 import stage3_measure
    else:
        raise ValueError(f"Unknown pipeline {pipeline!r}; expected ct-pet-v5 or dixon-v5")

    per_subject = lay.results_dir / cfg.STAGE3_DIR / "per_subject"
    if not per_subject.exists():
        log.warning("Stage 3 aggregation skipped: %s not found", per_subject)
        return None

    rows: list[dict[str, Any]] = []
    for subj in subjects:
        f = per_subject / f"{subj}.xlsx"
        if not f.exists():
            log.warning("Stage 3 aggregation: %s missing", f)
            continue
        df = pd.read_excel(f)
        if not df.empty:
            rows.append(df.iloc[0].to_dict())

    if not rows:
        log.warning("Stage 3 aggregation: no per-subject rows collected (%s)", pl)
        return None

    out = lay.results_dir / cfg.STAGE3_DIR / f"{lay.batch}_SummaryCodebook.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    df_all = pd.DataFrame(rows, columns=stage3_measure.column_order())
    df_all.to_excel(out, index=False)
    log.info("Stage 3 aggregate written: %s (%s)", out, pl)
    return out


__all__ = ["aggregate_stage3_summary"]
