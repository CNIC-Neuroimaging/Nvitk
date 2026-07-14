#!/usr/bin/env python3
"""Build synthetic golden arrays for StdvFromMean / PI regression tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nvitk.measure.hemodynamics import pulsatility_index_qvt, stdv_from_mean_branch


def main() -> None:
    rng = np.random.default_rng(42)
    n, nt = 7, 12
    area = rng.uniform(8.0, 14.0, size=n)
    diam = rng.uniform(0.5, 0.95, size=n)
    flow_pulsatile = rng.uniform(50.0, 120.0, size=(n, nt))
    flow_per_cycle = flow_pulsatile.mean(axis=1)
    pi = np.array([pulsatility_index_qvt(flow_pulsatile[i]) for i in range(n)])
    quality = stdv_from_mean_branch(flow_per_cycle, area, diam, flow_pulsatile)

    out = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "pitc_golden" / "synthetic_branch.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        flow_per_cycle=flow_per_cycle,
        area=area,
        diam=diam,
        flow_pulsatile=flow_pulsatile,
        pi=pi,
        stdv_from_mean=quality,
        nframes=nt,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
