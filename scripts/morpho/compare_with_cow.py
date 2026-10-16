#!/usr/bin/env python
"""Compare nvitk stage-7 morphometrics against the cow_morpho reference, subject by subject.

cow_morpho (https://… /home/imarcoss/cow_morpho-main) is the pipeline the ``eicab_cw`` profile was
tuned with. This script runs it — in its own environment (VMTK 1.5.0 / VTK 9.2.6 / Python 3.10;
see cow_morpho's env_ymls) — and nvitk on the same eICAB CW masks, then compares every anatomic
centerline VTP both write (``LICA_M1.vtp``, ``LMCA_M2s.vtp``, …): length, inscribed (MIS) and
cross-section radius, curvature, and the stenosis / enlargement point fractions.

Typical use (bug-for-bug parity: VMTK backend + cow's siphon mask)::

    python scripts/morpho/compare_with_cow.py \\
        --subjects PESA17808400 PESA4485924 \\
        --cow-python /path/to/vmtk-env/bin/python --cow-repo /home/imarcoss/cow_morpho-main \\
        --out /tmp/cow_parity --legacy-siphon

Existing outputs are reused (``--reuse``). The summary is printed and written to
``<out>/comparison.csv`` (one row per subject × centerline) and ``<out>/summary.csv``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RESULTS = Path("/data_lab_MCC/imarcoss/LabMCC/RESULTS/QVTPy")

_COW_DRIVER = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    def main():
        cow_repo, seg, case_dir, mapping, workers = sys.argv[1:6]
        sys.path.insert(0, cow_repo)
        from utils import preprocess_taubin
        from pipeline import pipeline_parallel
        seg, case_dir = Path(seg), Path(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        out = case_dir / (seg.name.replace(".nii.gz", "") + "_taubin.nii.gz")
        if not out.exists():
            s, aff, hdr = preprocess_taubin.load_segmentation(str(seg))
            sm = preprocess_taubin.preprocess_segmentation(
                seg=s, keep_largest=False, taubin_iters=20, taubin_lambda=0.65, taubin_mu=-0.65)
            preprocess_taubin.save_segmentation(str(out), sm, aff, hdr)
        pipeline_parallel.run_case(seg_path=str(out), out_dir=str(case_dir.parent), mapping_json=mapping,
                                   case_out_dir_override=str(case_dir), n_workers=int(workers))

    if __name__ == "__main__":
        main()
    """
)


def _read_vtp(path: Path) -> dict[str, np.ndarray]:
    import vtk
    from vtk.util import numpy_support

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    out = {"points": numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(float)}
    pd_ = poly.GetPointData()
    for i in range(pd_.GetNumberOfArrays()):
        arr = pd_.GetArray(i)
        if arr is not None and arr.GetNumberOfComponents() == 1:
            out[pd_.GetArrayName(i)] = numpy_support.vtk_to_numpy(arr).astype(float)
    return out


def _vtp_metrics(path: Path) -> dict[str, float]:
    d = _read_vtp(path)
    pts = d["points"]
    n = len(pts)
    nan = np.full(n, np.nan)

    def med(name):
        v = d.get(name, nan)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else np.nan

    def frac(name):
        v = d.get(name, nan)
        v = v[np.isfinite(v)]
        return float(np.mean(v >= 0.5)) if v.size else np.nan

    curv = d.get("Curvature", nan)
    return {
        "n_points": n,
        "length_mm": float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) if n > 1 else np.nan,
        "mis_median_mm": med("MaximumInscribedSphereRadius"),
        "cs_median_mm": med("CrossSectionRadius"),
        "curvature_p95": float(np.nanpercentile(curv, 95)) if np.isfinite(curv).any() else np.nan,
        "siphon_fraction": frac("SiphonMask"),
        "stenosis_fraction": frac("StenosisBinary"),
        "enlargement_fraction": frac("EnlargementBinary"),
    }


def compare_case(cow_dir: Path, nv_dir: Path) -> pd.DataFrame:
    """One row per centerline VTP present in either output."""
    cow = {p.stem: p for p in (cow_dir / "centerlines").glob("*.vtp")}
    nv = {p.stem: p for p in (nv_dir / "centerlines").glob("*.vtp")}
    rows = []
    for name in sorted(set(cow) | set(nv)):
        row = {"centerline": name, "in_cow": name in cow, "in_nvitk": name in nv}
        a = _vtp_metrics(cow[name]) if name in cow else {}
        b = _vtp_metrics(nv[name]) if name in nv else {}
        for key in set(a) | set(b):
            row[f"{key}_cow"] = a.get(key, np.nan)
            row[f"{key}_nvitk"] = b.get(key, np.nan)
            va, vb = a.get(key, np.nan), b.get(key, np.nan)
            if key in ("length_mm", "mis_median_mm", "cs_median_mm") and np.isfinite(va) and va:
                row[f"{key}_diff_pct"] = 100.0 * (vb / va - 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def lesion_counts(case_dir: Path) -> dict[str, int]:
    """Stenosis / enlargement segment totals from ``00_Path_Summary``."""
    xlsx = case_dir / "case_metrics_donut_tree.xlsx"
    if not xlsx.is_file():
        return {}
    df = pd.read_excel(xlsx, sheet_name="00_Path_Summary")
    return {
        "stenosis_segments": int(df.get("stenosis_segments_n", pd.Series(dtype=float)).sum()),
        "enlargement_segments": int(df.get("enlargement_segments_n", pd.Series(dtype=float)).sum()),
        "rows": int(len(df)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", nargs="+", required=True)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS,
                    help="<root>/<subject>/eicab/TOF_eICAB_CW.nii.gz")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cow-python", type=Path, help="Python of an environment with cow_morpho's deps + VMTK")
    ap.add_argument("--cow-repo", type=Path, default=Path("/home/imarcoss/cow_morpho-main"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--legacy-siphon", action="store_true", help="nvitk with cow's siphon mask (bug parity)")
    ap.add_argument("--backend", default="vmtk", choices=("vmtk", "native", "raycast"))
    ap.add_argument("--reuse", action="store_true", help="skip runs whose workbook exists")
    args = ap.parse_args(argv)

    from nvitk.measure.morpho.topology_io import default_eicab_topology_path
    from nvitk.measure.morphometrics import morphometrics_config, run_morphometrics_case

    cfg = morphometrics_config("eicab_cw").with_params(
        SIPHON_LEGACY_MASK=bool(args.legacy_siphon), CENTERLINE_BACKEND=args.backend
    )
    args.out.mkdir(parents=True, exist_ok=True)
    driver = args.out / "_cow_driver.py"
    driver.write_text(_COW_DRIVER, encoding="utf-8")
    all_rows, summary = [], []
    for subject in args.subjects:
        seg = args.results_root / subject / "eicab" / "TOF_eICAB_CW.nii.gz"
        cow_dir, nv_dir = args.out / "cow" / subject, args.out / "nvitk" / subject
        if not (args.reuse and (cow_dir / "case_metrics_donut_tree.xlsx").is_file()):
            if args.cow_python is None:
                raise SystemExit("--cow-python is required to run cow_morpho (or use --reuse)")
            subprocess.run([str(args.cow_python), str(driver), str(args.cow_repo), str(seg), str(cow_dir),
                            str(default_eicab_topology_path()), str(args.workers)], check=True)
        if not (args.reuse and (nv_dir / "case_metrics_donut_tree.xlsx").is_file()):
            run_morphometrics_case(seg, nv_dir, config=cfg, n_workers=args.workers)
        df = compare_case(cow_dir, nv_dir)
        df.insert(0, "subject", subject)
        all_rows.append(df)
        both = df[df["in_cow"] & df["in_nvitk"]]
        summary.append({
            "subject": subject,
            "centerlines_both": int(len(both)),
            "only_cow": int((df["in_cow"] & ~df["in_nvitk"]).sum()),
            "only_nvitk": int((~df["in_cow"] & df["in_nvitk"]).sum()),
            "length_abs_diff_pct_median": float(np.nanmedian(np.abs(both.get("length_mm_diff_pct", np.nan)))),
            "mis_abs_diff_pct_median": float(np.nanmedian(np.abs(both.get("mis_median_mm_diff_pct", np.nan)))),
            "cs_abs_diff_pct_median": float(np.nanmedian(np.abs(both.get("cs_median_mm_diff_pct", np.nan)))),
            **{f"cow_{k}": v for k, v in lesion_counts(cow_dir).items()},
            **{f"nvitk_{k}": v for k, v in lesion_counts(nv_dir).items()},
        })
    comparison = pd.concat(all_rows, ignore_index=True)
    comparison.to_csv(args.out / "comparison.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(args.out / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
