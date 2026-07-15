"""Patch ``cpu_count`` before eICAB's ``express_cw.py`` starts (via ``PYTHONPATH``).

Python imports this module automatically when the directory is on ``PYTHONPATH``
and ``NVITK_CPU_LIMIT`` is set in the environment.
"""

import multiprocessing as mp
import os


def _apply_cpu_limit():
    raw = os.environ.get("NVITK_CPU_LIMIT", "").strip()
    if not raw:
        return
    limit = max(1, int(raw))

    def _limited():
        return limit

    if hasattr(os, "cpu_count"):
        os.cpu_count = _limited
    mp.cpu_count = _limited


_apply_cpu_limit()
