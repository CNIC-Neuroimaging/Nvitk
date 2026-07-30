#!/usr/bin/env python3
"""Index qvtpy / eICAB pipeline outputs from XNAT and optional local results trees.

Updates the dataset ``assets`` table with experiment resources ``eicab`` and
``qvtpy`` (same labels used by ``nvitk-qvtpy-xnat-upload``). Optionally
downloads bundles to a local cache while syncing.

Examples::

    # Metadata only (no download) using subjects already in the local DB sessions table
    python scripts/pesa_brain/db/sync_db_xnat_resources.py \\
        --dataset-root dataset/nvitk-dataset \\
        --config .nvitk/xnat.json \\
        --build-sqlite-index

    # Download + index for explicit subjects
    python scripts/pesa_brain/db/sync_db_xnat_resources.py \\
        --dataset-root dataset/nvitk-dataset \\
        --config .nvitk/xnat.json \\
        --subjects PESA5745609,PESA123 \\
        --download-root /data/RESULTS/QVTPy \\
        --download

    # Merge XNAT availability with an on-disk results tree
    python scripts/pesa_brain/db/sync_db_xnat_resources.py \\
        --dataset-root dataset/nvitk-dataset \\
        --with-local /data/RESULTS/QVTPy \\
        --build-sqlite-index
"""

from __future__ import annotations

import sys

from nvitk.db.xnat_pipeline_resources import main

if __name__ == "__main__":
    try:
        main(standalone_mode=False)
    except SystemExit as exc:
        raise SystemExit(exc.code) from None
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
