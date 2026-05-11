#!/usr/bin/env python3
"""In-container entry point for eICAB (used by SGE ``singularity exec`` jobs).

Paths are under the outer pipeline container mounts (``/nvitk/data/``,
``/nvitk/output/``, etc.). :func:`run_eicab` runs the inner eICAB ``singularity run``
with the same bind layout as legacy ``run_eicab_inference.sh`` (TOF bind, output,
``/tmp``, vasculature on ``PATH``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nvitk.core.logger import Logger

from .runner import run_eicab


def main(argv: list[str] | None = None) -> int:
    Logger(level="INFO")
    log = Logger()
    p = argparse.ArgumentParser(description="Run eICAB (inside cluster container).")
    p.add_argument("--input", type=Path, required=True, help="TOF/MRA NIfTI path (container).")
    p.add_argument("--output", type=Path, required=True, help="Output directory (container).")
    p.add_argument("--tmp", type=Path, required=True, help="Writable temp directory (container).")
    p.add_argument("--eicab-container", type=Path, required=True, help="Path to eICAB .sif (container).")
    p.add_argument("--resolution", type=float, default=0.5)
    p.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    p.add_argument("--simple-segmentation", action="store_true")
    p.add_argument("--attention", action="store_true")
    p.add_argument("--keep-aux-outputs", action="store_true")
    args = p.parse_args(argv)

    try:
        run_eicab(
            args.input,
            args.output,
            resolution=args.resolution,
            simple_segmentation=args.simple_segmentation,
            attention=args.attention,
            device=args.device,
            container=args.eicab_container,
            tmp_dir=args.tmp,
            keep_aux_outputs=args.keep_aux_outputs,
        )
    except Exception as exc:
        log.error("run_job failed: %s", exc)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
