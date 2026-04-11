#!/usr/bin/env python3
"""One-time migration: rename pipeline_version -> pipeline_id in Parquet tables (image_measurements, assets).

Run from repo root:

  PYTHONPATH=src python scripts/migrate_image_measurements_pipeline_version_to_pipeline_id.py

Then rebuild the SQLite index: ``nvitk-build-sqlite-index`` or DataRepo.build_sqlite_index().
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from nvitk.db import DataRepo  # noqa: E402


def _migrate_table(repo: DataRepo, table_name: str) -> bool:
    definition = repo.catalog.get_table(table_name)
    path = definition.path
    if not path.exists():
        print(f"No Parquet at {path}; skip {table_name}.")
        return False
    df = pd.read_parquet(path)
    if "pipeline_id" in df.columns:
        print(f"{table_name}: column pipeline_id already present; skip.")
        return False
    if "pipeline_version" not in df.columns:
        print(f"{table_name}: no pipeline_version column; skip (add pipeline_id manually if needed).")
        return False
    df = df.rename(columns={"pipeline_version": "pipeline_id"})
    df.to_parquet(path, index=False)
    repo.catalog.update_table_schema(
        table_name, df, provenance={"migration": "pipeline_version_to_pipeline_id"}
    )
    print(f"Migrated {table_name}: {path} ({len(df)} rows).")
    return True


def main() -> None:
    repo = DataRepo()
    any_done = False
    for table_name in ("image_measurements", "assets"):
        if table_name not in repo.catalog.list_tables():
            continue
        if _migrate_table(repo, table_name):
            any_done = True
    if any_done:
        print("Rebuild SQLite index if you use the query cache.")
    else:
        print("Nothing migrated.")


if __name__ == "__main__":
    main()
