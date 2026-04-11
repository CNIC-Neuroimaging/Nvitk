#!/usr/bin/env python3
"""Convert QVT result CSVs (taverage + timeseries) into three Excel workbooks.

Inputs:
  - taverage CSV: per-MRI summary (flows and PI per territory).
  - timeseries CSV: per-frame flow (and metadata) by vessel.

Outputs:
  - flow_mean.xlsx: patient_id, mri_id, territory flow columns.
  - pi.xlsx: patient_id, mri_id, territory PI columns.
  - flow_tseries.xlsx: patient_id, vessel, vessel_code, frames 0..14 as columns (flow).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


FLOW_MEAN_COLS: list[str] = [
    "patient_id",
    "mri_id",
    "Left ICA",
    "Right ICA",
    "Basilar",
    "Left MCA",
    "Right MCA",
    "Left PCA",
    "Right PCA",
    "Left ACA",
    "Right ACA",
    "Straight Sinus",
    "Right Transverse",
    "TCBF",
    "Sagital Sinus",
    "Left Transverse",
    "Left Communicating",
    "Right Communicating",
]

PI_COLS: list[str] = [
    "patient_id",
    "mri_id",
    "Left ICA_PI",
    "Right ICA_PI",
    "Basilar_PI",
    "Left MCA_PI",
    "Right MCA_PI",
    "Left PCA_PI",
    "Right PCA_PI",
    "Left ACA_PI",
    "Right ACA_PI",
    "Straight Sinus_PI",
    "Right Transverse_PI",
    "Sagital Sinus_PI",
    "Left Transverse_PI",
    "Left Communicating_PI",
    "Right Communicating_PI",
]

TSERIES_KEYS = ["patient_id", "vessel", "vessel_code"]
FRAME_COLUMNS = list(range(15))  # 0 .. 14


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _subset_columns(df: pd.DataFrame, wanted: list[str], *, label: str) -> pd.DataFrame:
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        print(f"Warning [{label}]: missing columns (filled with NA): {missing}", file=sys.stderr)
    data = {c: df[c] if c in df.columns else pd.NA for c in wanted}
    return pd.DataFrame(data, columns=wanted)


def build_flow_mean(taverage: pd.DataFrame) -> pd.DataFrame:
    return _subset_columns(taverage, FLOW_MEAN_COLS, label="flow_mean")


def build_pi(taverage: pd.DataFrame) -> pd.DataFrame:
    return _subset_columns(taverage, PI_COLS, label="pi")


def build_flow_tseries(timeseries: pd.DataFrame) -> pd.DataFrame:
    ts = _strip_columns(timeseries)
    for key in TSERIES_KEYS + ["frame", "flow"]:
        if key not in ts.columns:
            raise ValueError(f"Timeseries CSV must contain column {key!r}; got: {list(ts.columns)}")

    ts = ts.copy()
    ts["frame"] = pd.to_numeric(ts["frame"], errors="coerce")
    ts = ts.loc[ts["frame"].notna()].copy()
    ts["frame"] = ts["frame"].astype(int)

    # One row per (patient, vessel, vessel_code, frame); duplicates → mean(flow)
    piv = ts.pivot_table(
        index=TSERIES_KEYS,
        columns="frame",
        values="flow",
        aggfunc="mean",
    )
    piv = piv.reindex(columns=FRAME_COLUMNS)
    piv.columns = [int(c) for c in piv.columns]
    out = piv.reset_index()
    # Ensure column order: keys then 0..14
    ordered = TSERIES_KEYS + FRAME_COLUMNS
    return out[ordered]


def run(
    taverage_path: Path,
    timeseries_path: Path,
    out_flow_mean: Path,
    out_pi: Path,
    out_flow_tseries: Path,
) -> None:
    ta = _strip_columns(_read_csv(taverage_path))
    ts = _read_csv(timeseries_path)

    build_flow_mean(ta).to_excel(out_flow_mean, index=False)
    build_pi(ta).to_excel(out_pi, index=False)
    build_flow_tseries(ts).to_excel(out_flow_tseries, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export QVT taverage / timeseries CSVs to three Excel files.",
    )
    parser.add_argument(
        "--taverage",
        required=True,
        type=Path,
        help="Path to taverage CSV (per-MRI summary).",
    )
    parser.add_argument(
        "--timeseries",
        required=True,
        type=Path,
        help="Path to timeseries CSV (per-frame flow).",
    )
    parser.add_argument(
        "--out-flow-mean",
        required=True,
        type=Path,
        help="Output Excel path for territory mean flow.",
    )
    parser.add_argument(
        "--out-pi",
        required=True,
        type=Path,
        help="Output Excel path for territory PI.",
    )
    parser.add_argument(
        "--out-flow-tseries",
        required=True,
        type=Path,
        help="Output Excel path for flow timeseries (frames 0–14 as columns).",
    )
    args = parser.parse_args(argv)

    for p in (args.taverage, args.timeseries):
        if not p.is_file():
            print(f"Input not found: {p}", file=sys.stderr)
            return 1

    run(
        args.taverage,
        args.timeseries,
        args.out_flow_mean,
        args.out_pi,
        args.out_flow_tseries,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
