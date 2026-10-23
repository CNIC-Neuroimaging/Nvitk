#!/usr/bin/env python3
"""In-container entry point for eICAB (used by SGE ``singularity exec`` jobs).

Paths are under the outer pipeline container mounts (``/nvitk/data/``,
``/nvitk/output/``, etc.). :func:`run_eicab` runs the inner eICAB ``singularity run``
with the TOF, output, ``/tmp`` and vasculature (on ``PATH``) binds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from nvitk.core.click_backend import apply_cli_backend
from nvitk.core.logger import Logger

from .runner import run_eicab


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for a single eICAB job (parse args, run, return an exit code)."""
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
    p.add_argument(
        "--post-process-eicab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="eICAB post-process after eICAB (re-segment, extend the trunks, siphon cuts).",
    )
    p.add_argument(
        "--eicab-pp-siphon",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the ICA siphon correction in the post-process.",
    )
    p.add_argument(
        "--eicab-pp-distal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Grow the ACAs / MCAs / PCAs / SCAs distally in the post-process.",
    )
    p.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="gpu",
        help="Array backend: cpu (NumPy) or gpu (CuPy).",
    )
    args = p.parse_args(argv)
    apply_cli_backend(args.backend)

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
        if args.post_process_eicab:
            from nvitk.pipes.qvtpy.util.eicab.eicab_postprocess import postprocess_eicab_directory

            postprocess_eicab_directory(args.output, siphon=args.eicab_pp_siphon, distal=args.eicab_pp_distal)
    except Exception as exc:
        log.error("run_job failed: %s", exc)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
