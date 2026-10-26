"""Patch eICAB's ``express_cw.py`` at interpreter start-up (via ``PYTHONPATH``).

Python imports this module automatically when the directory is on ``PYTHONPATH``.
Each patch is switched on by its own environment variable:

- ``NVITK_CPU_LIMIT=<n>``: ``os.cpu_count`` / ``multiprocessing.cpu_count`` report *n*,
  so VED does not spawn one worker per host logical CPU during parallel batches.
- ``NVITK_EICAB_CW_ONLY=1``: stop eICAB once the Circle-of-Willis mask
  (``<prefix>_eICAB_CW.nii.gz``) and ``<prefix>_resampled.nii.gz`` are written.
  eICAB has no option for this: ``full_pipeline`` always ends with
  ``skip_preprocessing_pipeline(option=2)``, which computes the CW centerline and metrics and
  then the whole-brain (WB) flood fill. The hook replaces that call with a no-op, so the run
  ends normally right after the CW mask is saved.

Runs under the container's Python 3.7: keep the syntax compatible.
"""

import importlib.abc
import logging
import multiprocessing as mp
import os
import sys

_PIPELINE_MODULE = "vessel_segmentation.pipeline"


# CPU cap


def _apply_cpu_limit():
    """Import-time CPU cap: applied automatically via ``sitecustomize`` inside the container."""
    raw = os.environ.get("NVITK_CPU_LIMIT", "").strip()
    if not raw:
        return
    limit = max(1, int(raw))

    def _limited():
        """Patched ``os.cpu_count`` reporting the capped thread count."""
        return limit

    if hasattr(os, "cpu_count"):
        os.cpu_count = _limited
    mp.cpu_count = _limited


# CW-only run (skip the CW metrics and the whole-brain flood fill)


def _skip_after_cw(*args, **kwargs):
    """Stand-in for ``skip_preprocessing_pipeline``: the CW mask is already on disk."""
    msg = (
        "nvitk: eICAB CW-only run: CW mask and TOF_resampled are written; "
        "skipping the CW centerline/metrics and the whole-brain (WB) flood fill"
    )
    logging.info(msg)
    print(msg, flush=True)


class _CwOnlyFinder(importlib.abc.MetaPathFinder):
    """Meta-path finder that patches eICAB's pipeline module right after it executes."""

    def find_spec(self, fullname, path, target=None):
        """Find *fullname* with the other finders; wrap its loader when it is the pipeline."""
        if fullname != _PIPELINE_MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                continue
            exec_module = spec.loader.exec_module

            def _exec_and_patch(module, _exec=exec_module):
                """Run the module, then swap the post-CW step for the no-op."""
                _exec(module)
                module.skip_preprocessing_pipeline = _skip_after_cw

            spec.loader.exec_module = _exec_and_patch
            return spec
        return None


def _apply_cw_only():
    """Install :class:`_CwOnlyFinder` when ``NVITK_EICAB_CW_ONLY`` is set."""
    if os.environ.get("NVITK_EICAB_CW_ONLY", "").strip().lower() not in ("1", "true", "yes"):
        return
    sys.meta_path.insert(0, _CwOnlyFinder())
    print("nvitk: eICAB CW-only hook installed (NVITK_EICAB_CW_ONLY)", flush=True)


_apply_cpu_limit()
_apply_cw_only()
