"""FireANTs registration backend (GPU).

FireANTs is a PyTorch-based registration library that ships its own CLI tool
`fireantsRegistration` with an ANTs-like interface.

This wrapper currently drives the FireANTs CLI so we can integrate it into
nvitk pipelines without hard-coding the internal Python API.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from nvitk.core.exceptions import BackendUnavailableError


@dataclass(frozen=True)
class FireAntsResult:
    """Outputs produced by `fireantsRegistration`."""

    output_prefix: Path
    warped_moving_path: Path


def _which(exe: str) -> str:
    """Resolve *exe* on ``PATH``, raising a FireANTs install hint if it is missing."""
    path = shutil.which(exe)
    if path is None:
        raise BackendUnavailableError(
            f"{exe!r} was not found on PATH. Install FireANTs with: pip install fireants"
        )
    return path


def fireants_register(
    *,
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    device: str = "cuda:0",
    verbose: bool = False,
) -> FireAntsResult:
    """Run a default multi-stage FireANTs registration (Rigid→Affine→SyN)."""
    exe = _which("fireantsRegistration")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / "fireantsTransform"
    warped = out_dir / "moving_warped.nii.gz"

    cmd: list[str] = [
        exe,
        "--output",
        f"{prefix},{warped}",
        "--device",
        str(device),
        "--transform",
        "Rigid[3e-2]",
        "--metric",
        f"MI[{fixed_path},{moving_path},gaussian,16]",
        "--convergence",
        "[100x50x25x10,1e-6,10]",
        "--shrink-factors",
        "8x4x2x1",
        "--transform",
        "Affine[3e-2]",
        "--metric",
        f"CC[{fixed_path},{moving_path},5]",
        "--convergence",
        "[100x50x25x10,1e-4,10]",
        "--shrink-factors",
        "8x4x2x1",
        "--transform",
        "SyN[0.2]",
        "--metric",
        f"MSE[{fixed_path},{moving_path}]",
        "--convergence",
        "[100x70x50x20,1e-4,10]",
        "--shrink-factors",
        "8x4x2x1",
    ]
    if verbose:
        cmd.append("--verbose")

    subprocess.run(cmd, check=True)
    if not warped.exists():
        raise RuntimeError(f"fireantsRegistration did not create warped output: {warped}")
    return FireAntsResult(output_prefix=prefix, warped_moving_path=warped)


def fireants_apply(
    *,
    fixed_path: Path,
    moving_path: Path,
    out_path: Path,
    transforms: list[Path],
) -> Path:
    """Apply FireANTs transforms if `fireantsApplyTransforms` is available."""
    exe = shutil.which("fireantsApplyTransforms")
    if exe is None:
        raise BackendUnavailableError(
            "fireantsApplyTransforms is not available in this FireANTs install. "
            "Install a version that provides it, or apply transforms via the FireANTs API."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        exe,
        "--fixed",
        str(fixed_path),
        "--moving",
        str(moving_path),
        "--output",
        str(out_path),
        "--transform",
        ",".join(str(p) for p in transforms),
    ]
    subprocess.run(cmd, check=True)
    return out_path


__all__ = ["FireAntsResult", "fireants_register", "fireants_apply"]

