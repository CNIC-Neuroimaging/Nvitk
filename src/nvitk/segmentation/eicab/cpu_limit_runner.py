"""Run eICAB ``express_cw.py`` after patching ``os.cpu_count`` / ``mp.cpu_count``.

Used when ``NVITK_CPU_LIMIT`` is set inside the eICAB Singularity container so VED
does not spawn one worker per host logical CPU (e.g. 48) during parallel batches.

Invoked as::

    python3 /nvitk/src/nvitk/segmentation/eicab/cpu_limit_runner.py \\
        -t /TOF.nii.gz -o /output -r 0.5 -d cpu -f
"""

import multiprocessing as mp
import os
import runpy
import sys

_EXPRESS_CW = "/vessel_segmentation_snaillab/scripts/express_cw.py"


def _apply_cpu_limit():
    """Cap BLAS/OpenMP thread counts from ``NVITK_CPU_LIMIT`` so eICAB stays within its SGE slot."""
    raw = os.environ.get("NVITK_CPU_LIMIT", "").strip()
    if not raw:
        return
    limit = max(1, int(raw))

    def _limited():
        """Patched ``os.cpu_count`` that reports the capped thread count to libraries."""
        return limit

    if hasattr(os, "cpu_count"):
        os.cpu_count = _limited
    mp.cpu_count = _limited


def main():
    """Entry point: apply the CPU cap, then exec the eICAB express-CoW script."""
    _apply_cpu_limit()
    script = os.environ.get("NVITK_EICAB_EXPRESS_CW", _EXPRESS_CW)
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
