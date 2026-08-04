"""
Thin wrapper around the ``TotalSegmentator`` command-line executable.

Intentionally does **not** import the ``totalsegmentator`` Python API; the CLI
is what we run inside Singularity on the cluster and in pip installs locally.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from nvitk.core.logger import Logger

from .class_maps import AVAILABLE_TASKS

log = Logger()


def _which_totalseg() -> str:
    """Resolve the ``TotalSegmentator`` executable on ``PATH`` or raise a clear error."""
    exe = shutil.which("TotalSegmentator")
    if exe is None:
        raise RuntimeError(
            "TotalSegmentator CLI not found on PATH. Install with "
            "`pip install totalsegmentator` or enter the pipelined Singularity container."
        )
    return exe


def run_totalsegmentator(
    input: str | Path,
    output: str | Path,
    task: str,
    *,
    device: str = "gpu",
    roi_subset: Iterable[str] | None = None,
    multilabel: bool = True,
    statistics: bool = True,
    fast: bool = False,
    preview: bool = False,
    model_dir: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Invoke the ``TotalSegmentator`` CLI on *input* and write to *output*.

    Parameters
    ----------
    input
        Single NIfTI file or a directory of NIfTI files.
    output
        Output directory. Created if absent.
    task
        One of :data:`AVAILABLE_TASKS` (passed via ``-ta``).
    device
        ``'gpu'`` or ``'cpu'`` (passed via ``--device``).
    roi_subset
        ROI whitelist appended via ``-rs`` (``None`` means all ROIs).
    multilabel
        Pass ``--ml`` (single multilabel NIfTI output). Default True.
    statistics
        Pass ``--statistics``. Default True.
    fast, preview
        Forwarded to ``--fast`` / ``--preview``.
    model_dir
        Optional path exported as ``TOTALSEG_HOME_DIR`` for the subprocess.
    extra_env
        Additional environment variables for the subprocess.

    Returns
    -------
    subprocess.CompletedProcess
        The completed process. Raises :class:`subprocess.CalledProcessError`
        when *check* is True and the CLI exits non-zero.
    """
    if task not in AVAILABLE_TASKS:
        raise ValueError(
            f"Unknown task '{task}'. Must be one of {AVAILABLE_TASKS}."
        )
    if device not in {"gpu", "cpu"}:
        raise ValueError(f"Unknown device '{device}'. Use 'gpu' or 'cpu'.")

    exe = _which_totalseg()

    input_p = Path(input)
    output_p = Path(output)

    cmd: list[str] = [exe, "-i", str(input_p), "-o", str(output_p), "-ta", str(task)]
    if multilabel:
        cmd.append("--ml")
    if statistics:
        cmd.append("--statistics")
    if fast:
        cmd.append("--fast")
    if preview:
        cmd.append("--preview")
    cmd.extend(["--device", device])
    if roi_subset:
        cmd.append("-rs")
        cmd.extend([str(r) for r in roi_subset])

    env = os.environ.copy()
    if model_dir is not None:
        log.info(f"Model directory: {model_dir}")
        env["TOTALSEG_HOME_DIR"] = str(model_dir)
    if extra_env:
        env.update(extra_env)

    log.info(f"Running TotalSegmentator: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=capture_output, text=True, env=env)


def run_totalsegmentator_batch(
    inputs_outputs: Sequence[tuple[str | Path, str | Path]],
    task: str,
    **kwargs,
) -> list[subprocess.CompletedProcess]:
    """
    Convenience: run :func:`run_totalsegmentator` over a list of (input, output) pairs.
    """
    results: list[subprocess.CompletedProcess] = []
    for inp, out in inputs_outputs:
        results.append(run_totalsegmentator(inp, out, task, **kwargs))
    return results


__all__ = ["run_totalsegmentator", "run_totalsegmentator_batch"]
