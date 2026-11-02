#!/usr/bin/env python
"""Calibrate the siphon (high-curvature) mask of the ``eicab_cw`` morphometrics profile.

cow_morpho's siphon mask flagged ~100 % of every vessel (curvature measured on the 0.1 mm
resampled centerline is dominated by sampling noise), so its taper reference collapsed to a flat
median. The fix measures curvature on the centerline smoothed at ``SIPHON_CURVATURE_SMOOTH_MM``;
this script picks that smoothing, the curvature threshold and the dilation on real eICAB CW masks:

1. run stage-7 morphometrics (VMTK, cow's legacy siphon mask) on N subjects (reused if present);
2. sweep σ × κ × dilation on the measured root->terminal paths (``points/*.csv``) and measure the
   mask's coverage of the ICAs (whose CW crop is mostly cavernous / supraclinoid siphon) and of
   every other vessel;
3. keep the settings whose non-ICA coverage is <= ``--max-other`` (each taper fit keeps most of its
   points: the purpose of the fix) and ICA coverage >= ``--min-ica``, and choose the one that best
   separates the two (ICA - other coverage; ties: closest to cow's constants). Curvature alone
   cannot confine the mask to the ICA (the PCAs and SCAs bend as tightly), so the separation is
   maximised under a coverage cap rather than required to be clean. For parity the choice keeps
   cow's thresholds (κ >= 0.20 mm⁻¹ (its 0.10 on the half-valued curvature), and 5 mm dilation)
   and tunes only the smoothing σ cow lacked; ``--free-thresholds`` lets κ and the dilation move
   too (the free optimum is reported either way);
4. re-run with the chosen values and report how stenosis / enlargement calls change per vessel
   (recounted from each case's per-path detections, so both runs use the same counting).

Outputs in ``--out``: ``sweep.csv``, ``chosen.json`` (paste the values into the profile in
``nvitk/measure/morphometrics_config.py``), ``lesion_changes.csv`` and ``report.md``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS = Path("/data_lab_MCC/imarcoss/LabMCC/RESULTS/QVTPy")
ICA_NAMES = ("LICA", "RICA")
#: cow_morpho's siphon thresholds in true curvature units (its 0.10 is on κ/2) and its dilation.
COW_KAPPA = 0.20
COW_DILATION_MM = 5.0


def pick_subjects(root: Path, n: int, seed: int) -> list[str]:
    """*n* random subjects of *root* that have an eICAB CW mask."""
    subjects = sorted(p.parent.parent.name for p in root.glob("*/eicab/TOF_eICAB_CW.nii.gz"))
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(subjects, size=min(n, len(subjects)), replace=False).tolist())


def run_subject(seg: Path, out: Path, cfg, workers: int, reuse: bool) -> Path:
    from nvitk.measure.morphometrics import run_morphometrics_case

    if not (reuse and (out / "case_metrics_donut_tree.xlsx").is_file()):
        run_morphometrics_case(seg, out, config=cfg, n_workers=workers)
    return out


def vessel_of(path_id: str, names: set[str]) -> str:
    """Vessel name of a point-table key: ``"13_LPCA_P2_comp01_arm02"`` or ``"LPCA_P2_M1"``."""
    rest = path_id.split("_comp")[0]
    head, _, tail = rest.partition("_")
    if head.isdigit():
        rest = tail
    matches = [n for n in names if rest == n or rest.startswith(n + "_")]
    return max(matches, key=len) if matches else rest


def path_index(case_dir: Path) -> pd.DataFrame:
    """``Sheet_Index`` with a ``vessel_name`` column (resolved from the key for older runs)."""
    book = case_dir / "case_metrics_donut_tree.xlsx"
    index = pd.read_excel(book, sheet_name="Sheet_Index")
    if "vessel_name" not in index.columns or index["vessel_name"].isna().any():
        names = set(pd.read_excel(book, sheet_name="07_Root_To_Terminal_Paths")["vessel_name"].astype(str))
        index["vessel_name"] = [vessel_of(str(k), names) for k in index["path_id"]]
    return index


def load_paths(case_dir: Path) -> list[tuple[str, np.ndarray]]:
    """``(vessel_name, points)`` of every measured root->terminal path of a case."""
    out = []
    for name, csv in zip(path_index(case_dir)["vessel_name"], path_index(case_dir)["points_csv"]):
        df = pd.read_csv(case_dir / csv)
        out.append((str(name), df[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)))
    return out


def coverage(paths, sigma: float, kappa: float, dilation: float) -> tuple[float, float]:
    """Mean siphon-mask coverage of ICA paths and of all other paths."""
    from nvitk.measure import morphometrics_config as mc
    from nvitk.measure.morpho.caliber import compute_siphon_mask
    from nvitk.measure.morpho.geometry import cumulative_s

    cfg = mc.morphometrics_config("eicab_cw").with_params(
        SIPHON_LEGACY_MASK=False, SIPHON_CURVATURE_SMOOTH_MM=sigma,
        SIPHON_KAPPA_THRESHOLD=kappa, SIPHON_DILATION_MM=dilation,
    )
    ica, other = [], []
    with mc.active_config(cfg):
        for name, pts in paths:
            if len(pts) < 5:
                continue
            frac = float(compute_siphon_mask(pts, cumulative_s(pts)).mean())
            (ica if name in ICA_NAMES else other).append(frac)
    return (float(np.mean(ica)) if ica else np.nan, float(np.mean(other)) if other else np.nan)


def vessel_lesions(case_dir: Path) -> pd.DataFrame:
    """Stenosis / enlargement calls per vessel, recounted from the per-path detections
    (root->terminal path runs + ``points/*.csv``), each lesion once where overlapping paths share it."""
    book = case_dir / "case_metrics_donut_tree.xlsx"
    index = path_index(case_dir)
    csv_of = dict(zip(index["path_id"].astype(str), index["points_csv"]))
    name_of = dict(zip(index["path_id"].astype(str), index["vessel_name"].astype(str)))
    summary = pd.read_excel(book, sheet_name="07_Root_To_Terminal_Paths")
    if "path_id" not in summary.columns:
        # Runs written before 07 kept path_id: match each row to the path of the same vessel whose
        # points have its length (both come from the same centerline).
        lengths = {pid: float(pd.read_csv(case_dir / csv, usecols=["s_mm"])["s_mm"].iloc[-1])
                   for pid, csv in csv_of.items()}
        ids = []
        for _, row in summary.iterrows():
            cands = [pid for pid in lengths if name_of[pid] == str(row["vessel_name"])]
            ids.append(min(cands, key=lambda pid: abs(lengths[pid] - float(row["length_mm"]))) if cands else "")
        summary = summary.assign(path_id=ids)
    found: dict[tuple[str, str], list[tuple[np.ndarray, float]]] = {}
    for _, row in summary.iterrows():
        path_id = str(row.get("path_id", ""))
        if path_id not in csv_of:
            continue
        name = name_of[path_id]
        pts = pd.read_csv(case_dir / csv_of[path_id])
        xyz = pts[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=float)
        for kind in ("stenosis", "enlargement"):
            raw = row.get(f"{kind}_segments_point_idx")
            try:
                runs = json.loads(raw) if isinstance(raw, str) else []
            except ValueError:
                runs = []
            pct = pts[f"{kind}_percent_point"].to_numpy(dtype=float)
            for a, b in runs:
                a, b = sorted((max(0, int(a)), min(len(xyz) - 1, int(b))))
                mid = xyz[(a + b) // 2]
                seen = found.setdefault((name, kind), [])
                if any(np.linalg.norm(mid - m) < 1.0 for m, _ in seen):
                    continue
                seen.append((mid, float(np.nanmax(pct[a:b + 1])) if np.isfinite(pct[a:b + 1]).any() else np.nan))
    rows: dict[str, dict] = {}
    for (name, kind), items in found.items():
        r = rows.setdefault(name, {"vessel_name": name})
        r[f"n_{kind}"] = len(items)
        r[f"{kind}_taper_percent_max"] = float(np.nanmax([p for _, p in items])) if items else np.nan
    cols = ["vessel_name", "n_stenosis", "n_enlargement", "stenosis_taper_percent_max", "enlargement_taper_percent_max"]
    return pd.DataFrame(list(rows.values()), columns=cols).fillna({"n_stenosis": 0, "n_enlargement": 0})


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
    ap.add_argument("--kappas", type=float, nargs="+", default=[0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25])
    ap.add_argument("--dilations", type=float, nargs="+", default=[2.0, 3.0, 5.0])
    ap.add_argument("--min-ica", type=float, default=0.25)
    ap.add_argument("--max-other", type=float, default=0.15)
    ap.add_argument("--free-thresholds", action="store_true",
                    help="choose κ and the dilation too (default: keep cow's 0.20 mm⁻¹ / 5 mm, tune σ only)")
    args = ap.parse_args(argv)

    from nvitk.measure.morphometrics_config import morphometrics_config

    subjects = args.subjects or pick_subjects(args.results_root, args.n_random, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    legacy_cfg = morphometrics_config("eicab_cw").with_params(SIPHON_LEGACY_MASK=True)
    paths = []
    for subj in subjects:
        seg = args.results_root / subj / "eicab" / "TOF_eICAB_CW.nii.gz"
        case = run_subject(seg, args.out / "legacy" / subj, legacy_cfg, args.workers, args.reuse)
        paths.extend(load_paths(case))
    legacy_ica, legacy_other = coverage_legacy(paths)

    rows = []
    for sigma, kappa, dil in itertools.product(args.sigmas, args.kappas, args.dilations):
        ica, other = coverage(paths, sigma, kappa, dil)
        rows.append({"sigma_mm": sigma, "kappa_1_per_mm": kappa, "dilation_mm": dil,
                     "ica_coverage": ica, "other_coverage": other})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(args.out / "sweep.csv", index=False)
    ok = sweep[(sweep["ica_coverage"] >= args.min_ica) & (sweep["other_coverage"] <= args.max_other)]
    if ok.empty:
        print(sweep.sort_values("other_coverage").head(10).to_string(index=False))
        raise SystemExit("No setting satisfies the coverage criteria; relax --min-ica / --max-other.")
    # Closest to cow's constants (σ unknown -> 2 mm, κ 0.10 half-valued = 0.20 true, dilation 5 mm).
    ok = ok.assign(
        separation=(ok["ica_coverage"] - ok["other_coverage"]).round(3),
        distance=np.abs(ok["sigma_mm"] - 2.0) / 2.0 + np.abs(ok["kappa_1_per_mm"] - 0.20) / 0.20
        + np.abs(ok["dilation_mm"] - 5.0) / 5.0,
    )
    ranked = ok.sort_values(["separation", "distance"], ascending=[False, True])
    free_best = ranked.iloc[0]
    parity = ranked[np.isclose(ranked["kappa_1_per_mm"], COW_KAPPA) & np.isclose(ranked["dilation_mm"], COW_DILATION_MM)]
    if args.free_thresholds:
        best = free_best
    elif parity.empty:
        raise SystemExit(f"No σ with κ = {COW_KAPPA} and dilation {COW_DILATION_MM} mm meets the criteria; "
                         "add them to --kappas / --dilations or pass --free-thresholds.")
    else:
        best = parity.iloc[0]
    chosen = {
        "SIPHON_CURVATURE_SMOOTH_MM": float(best["sigma_mm"]),
        "SIPHON_KAPPA_THRESHOLD": float(best["kappa_1_per_mm"]),
        "SIPHON_DILATION_MM": float(best["dilation_mm"]),
        "ica_coverage": float(best["ica_coverage"]),
        "other_coverage": float(best["other_coverage"]),
        "rule": "free" if args.free_thresholds else "cow thresholds, σ tuned",
        "free_optimum": {k: float(free_best[k]) for k in ("sigma_mm", "kappa_1_per_mm", "dilation_mm",
                                                         "ica_coverage", "other_coverage")},
        "legacy_ica_coverage": legacy_ica,
        "legacy_other_coverage": legacy_other,
        "subjects": subjects,
    }
    (args.out / "chosen.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")

    fixed_cfg = morphometrics_config("eicab_cw").with_params(
        SIPHON_LEGACY_MASK=False,
        SIPHON_CURVATURE_SMOOTH_MM=chosen["SIPHON_CURVATURE_SMOOTH_MM"],
        SIPHON_KAPPA_THRESHOLD=chosen["SIPHON_KAPPA_THRESHOLD"],
        SIPHON_DILATION_MM=chosen["SIPHON_DILATION_MM"],
    )
    changes = []
    for subj in subjects:
        seg = args.results_root / subj / "eicab" / "TOF_eICAB_CW.nii.gz"
        fixed = run_subject(seg, args.out / "fixed" / subj, fixed_cfg, args.workers, args.reuse)
        a = vessel_lesions(args.out / "legacy" / subj).set_index("vessel_name")
        b = vessel_lesions(fixed).set_index("vessel_name")
        both = a.join(b, lsuffix="_legacy", rsuffix="_fixed", how="outer")
        both.insert(0, "subject", subj)
        changes.append(both.reset_index())
    ch = pd.concat(changes, ignore_index=True)
    ch.to_csv(args.out / "lesion_changes.csv", index=False)
    per_vessel = ch.groupby("vessel_name")[[c for c in ch.columns if c.startswith("n_")]].sum()
    report = [
        "# Siphon-mask calibration (eicab_cw)",
        "",
        f"Subjects: {len(subjects)} ({', '.join(subjects)})",
        "",
        f"Legacy (cow_morpho) mask coverage: ICA {legacy_ica:.1%}, other vessels {legacy_other:.1%}.",
        f"Chosen: σ = {chosen['SIPHON_CURVATURE_SMOOTH_MM']} mm, κ ≥ {chosen['SIPHON_KAPPA_THRESHOLD']} mm⁻¹, "
        f"dilation {chosen['SIPHON_DILATION_MM']} mm → ICA {chosen['ica_coverage']:.1%}, "
        f"other {chosen['other_coverage']:.1%} ({chosen['rule']}).",
        f"Free optimum (σ, κ and dilation all tuned): σ = {free_best['sigma_mm']} mm, "
        f"κ ≥ {free_best['kappa_1_per_mm']} mm⁻¹, dilation {free_best['dilation_mm']} mm → "
        f"ICA {free_best['ica_coverage']:.1%}, other {free_best['other_coverage']:.1%}.",
        "",
        "## Lesion calls per vessel (summed over subjects)",
        "",
        per_vessel.to_markdown() if hasattr(per_vessel, "to_markdown") else per_vessel.to_string(),
    ]
    (args.out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))
    return 0


def coverage_legacy(paths) -> tuple[float, float]:
    """Coverage of cow_morpho's mask (``SIPHON_LEGACY_MASK``) on the same paths."""
    from nvitk.measure import morphometrics_config as mc
    from nvitk.measure.morpho.caliber import compute_siphon_mask
    from nvitk.measure.morpho.geometry import cumulative_s

    ica, other = [], []
    with mc.active_config(mc.morphometrics_config("eicab_cw").with_params(SIPHON_LEGACY_MASK=True)):
        for name, pts in paths:
            if len(pts) < 5:
                continue
            frac = float(compute_siphon_mask(pts, cumulative_s(pts)).mean())
            (ica if name in ICA_NAMES else other).append(frac)
    return float(np.mean(ica)) if ica else np.nan, float(np.mean(other)) if other else np.nan


if __name__ == "__main__":
    sys.exit(main())
