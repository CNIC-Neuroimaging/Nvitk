#!/usr/bin/env python3
"""Compare qvtpy ``pitc_profile.csv`` against a legacy QVT export CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_profile(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def summarize(rows: list[dict[str, str]], region: str | None = None) -> dict[str, float]:
    if region is not None:
        rows = [r for r in rows if r.get("root_region_id") == region]
    pis = [float(r["pi"]) for r in rows if r.get("pi")]
    quals = [float(r["quality"]) for r in rows if r.get("quality")]
    dists = [float(r["distance_mm"]) for r in rows if r.get("distance_mm")]
    return {
        "n": float(len(rows)),
        "pi_mean": float(np.mean(pis)) if pis else float("nan"),
        "q_mean": float(np.mean(quals)) if quals else float("nan"),
        "d_max": float(np.max(dists)) if dists else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qvtpy_profile", type=Path)
    parser.add_argument("legacy_profile", type=Path, help="CSV with pi, quality, distance_mm columns")
    parser.add_argument("--region", default=None)
    args = parser.parse_args()

    a = summarize(_load_profile(args.qvtpy_profile), region=args.region)
    b = summarize(_load_profile(args.legacy_profile), region=args.region)
    print("qvtpy:", a)
    print("legacy:", b)


if __name__ == "__main__":
    main()
