"""Log per-subject ICA genus (β₁) from eICAB masks under a PESA* results tree.

No Otsu / repair / centerline correction — only the raw eICAB label mask for
LICA and RICA (labels 1 and 2). Rows with β₁ > 0 are printed in red.

Example::

    python tests/genus_report.py /path/to/WVI-BB/RESULTS/eicab_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.core.backend import set_global_backend
from nvitk.io.imageio import imread
from nvitk.morphology.centerline_siphon import compute_mask_genus
from nvitk.pipes.bbtpy.labels import EICAB_LICA, EICAB_RICA, bb_vessel_name
from nvitk.util.colors import bcolors

log = Logger()
set_global_backend("cpu")

ICA_IDS = (EICAB_LICA, EICAB_RICA)

EICAB_MASK_CANDIDATES = (
    "TOF_eICAB_CW.nii.gz",
    "TOF_eICAB_CW.nii",
    "r_TOF_eICAB_CW.nii.gz",
    "r_TOF_eICAB_CW.nii",
)


def _find_eicab_mask(subj_dir: Path) -> Path | None:
    for name in EICAB_MASK_CANDIDATES:
        path = subj_dir / name
        if path.is_file():
            return path
    return None


def log_ica_genus_report(
    data_root: Path | str,
    *,
    ica_ids: tuple[int, ...] = ICA_IDS,
) -> int:
    """Scan *data_root*/PESA*/ for eICAB masks and log ICA genus per subject.

    Returns the number of ICA rows with β₁ > 0 (suspect handles / donuts).
    """
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")

    subjects = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith("PESA")
    )
    if not subjects:
        log.warning(f"No PESA* folders under {root}")
        return 0

    log.step(f"ICA genus report — {root} ({len(subjects)} subjects)")

    header = (
        f"{'subject':<16} {'ICA':6} {'voxels':>8} {'β₀':>4} {'β₁':>4} "
        f"{'χ':>7} {'cycles':>7} {'status':>8}"
    )
    log.info(header)
    log.info("-" * len(header))

    n_suspect = 0
    for subj in subjects:
        mask_path = _find_eicab_mask(subj)
        if mask_path is None:
            log.warning(
                f"{subj.name}: no eICAB mask "
                f"(tried {', '.join(EICAB_MASK_CANDIDATES[:2])}, …)"
            )
            continue

        vol = to_numpy(imread(str(mask_path)).data)
        for lid in ica_ids:
            name = bb_vessel_name(int(lid))
            rep = compute_mask_genus(vol == int(lid), label_name=name)
            if rep.n_voxels == 0:
                status = "empty"
            elif rep.beta1 > 0:
                status = "SUSPECT"
                n_suspect += 1
            else:
                status = "OK"

            line = (
                f"{subj.name:<16} {name:6} {rep.n_voxels:8d} {rep.beta0:4d} "
                f"{rep.beta1:4d} {rep.euler_chi:7.1f} {rep.skeleton_cycles:7d} "
                f"{status:>8}"
            )
            if rep.beta1 > 0:
                line = f"{bcolors.FAIL}{line}{bcolors.ENDC}"
            log.info(line)

    log.step(
        f"done — {n_suspect} suspect ICA row(s) (β₁ > 0) "
        f"across {len(subjects)} subject(s)"
    )
    return n_suspect


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Log ICA genus (β₁) for each PESA subject under an eICAB results root.",
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Directory containing PESA*/TOF_eICAB_CW.nii.gz (or similar).",
    )
    args = parser.parse_args()
    log_ica_genus_report(args.data_root)


if __name__ == "__main__":
    main()
