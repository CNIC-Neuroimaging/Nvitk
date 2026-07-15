"""Run eICAB ``express_cw.py`` after patching ``os.cpu_count`` / ``mp.cpu_count``.

Used when ``NVITK_CPU_LIMIT`` is set inside the eICAB Singularity container so VED
does not spawn one worker per host logical CPU (e.g. 48) during parallel batches.

Invoked as::

    python /nvitk/src/nvitk/segmentation/eicab/cpu_limit_runner.py \\
        -t /TOF.nii.gz -o /output -r 0.5 -d cpu -f
"""

from __future__ import annotations

import multiprocessing as mp
import os
import runpy
import sys

_EXPRESS_CW = "/vessel_segmentation_snaillab/scripts/express_cw.py"


def _apply_cpu_limit() -> None:
    raw = os.environ.get("NVITK_CPU_LIMIT", "").strip()
    if not raw:
        return
    limit = max(1, int(raw))

    def _limited() -> int:
        return limit

    os.cpu_count = _limited  # type: ignore[method-assign]
    mp.cpu_count = _limited  # type: ignore[assignment]


def main() -> None:
    _apply_cpu_limit()
    script = os.environ.get("NVITK_EICAB_EXPRESS_CW", _EXPRESS_CW)
    sys.argv = [script, *sys.argv[1:]]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
