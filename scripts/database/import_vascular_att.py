#!/usr/bin/env python3
"""
Import vascular-atlas ASL ATT (``att_mean`` / ``att_median``) into ``image_measurements``.

The Desikan ATT long CSV (``ATT_native_results.csv``) is handled by ``import_new_vars.py`` /
``import_att_csv``. Vascular parcels live in two wide Excel sheets (one row per ``mri_id``,
one column per parcel such as ``Left_ACA-0``). Those sheets have no usable ``patient_id``,
so ``mri_id`` is mapped to ``subject_uid`` via the sessions table (same lookup as the
Desikan ATT importer).

Re-running is idempotent: upsert keys include ``variable_id``, ``region_id``, ``source_file``,
and ``source_column`` (the region label), so a second run replaces the same rows.

Example::

    python scripts/database/import_vascular_att.py --dry-run
    python scripts/database/import_vascular_att.py
    python scripts/database/import_vascular_att.py --dataset-root ~/nvitk/dataset/nvitk-dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from nvitk.db.repo import DataRepo, get_repo, get_repo_from_settings

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from import_new_vars import (  # noqa: E402
    DEFAULT_BATCH,
    DEFAULT_PATHS,
    import_att_vascular,
)

DEFAULT_MEAN = DEFAULT_PATHS["att_vascular_mean"]
DEFAULT_MEDIAN = DEFAULT_PATHS["att_vascular_median"]


def _open_repo(dataset_root: Path | None) -> DataRepo:
    """Open the dataset at *dataset_root*, or the one configured in ``.nvitk/settings.json``."""
    if dataset_root is not None:
        return get_repo(root=dataset_root)
    got = get_repo_from_settings()
    return got[0] if isinstance(got, tuple) else got


def main(argv: Iterable[str] | None = None) -> int:
    """Parse arguments, import vascular ATT mean/median, and rebuild the SQLite index."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Dataset root (default: the one in .nvitk/settings.json)",
    )
    parser.add_argument(
        "--att-vascular-mean",
        type=Path,
        default=DEFAULT_MEAN,
        help=f"Wide Excel for att_mean (default: {DEFAULT_MEAN})",
    )
    parser.add_argument(
        "--att-vascular-median",
        type=Path,
        default=DEFAULT_MEDIAN,
        help=f"Wide Excel for att_median (default: {DEFAULT_MEDIAN})",
    )
    parser.add_argument("--source-batch-id", type=str, default=DEFAULT_BATCH)
    parser.add_argument(
        "--full-index",
        action="store_true",
        help="Rebuild the SQLite index for every table instead of just image_measurements",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the plan without writing")
    args = parser.parse_args(list(argv) if argv is not None else None)

    for label, path in (
        ("att_mean", args.att_vascular_mean),
        ("att_median", args.att_vascular_median),
    ):
        if not path.exists():
            print(f"Missing {label} source: {path}", file=sys.stderr)
            return 1

    repo = _open_repo(args.dataset_root)
    print(f"Dataset: {repo.root}")
    print(f"  att_mean   : {args.att_vascular_mean}")
    print(f"  att_median : {args.att_vascular_median}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    out = import_att_vascular(
        repo,
        mean_path=args.att_vascular_mean,
        median_path=args.att_vascular_median,
        source_batch_id=args.source_batch_id,
        log=print,
    )
    print(f"Upserted {len(out)} vascular ATT rows into image_measurements.")

    tables = None if args.full_index else ["image_measurements"]
    repo.build_sqlite_index(tables=tables)
    print(f"Rebuilt SQLite index ({'all tables' if tables is None else 'image_measurements'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
