"""Local eICAB inference via ``singularity run`` and optional output pruning."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from nvitk.core.logger import Logger

log = Logger()

# Circle-of-Willis multilabel (legacy naming).
_COW_RE = re.compile(r"eICAB_CW", re.IGNORECASE)
# Whole-brain style outputs (heuristic; extend if your build uses other stems).
_WB_RE = re.compile(
    r"(eICAB_(WB|WHOLE)|whole[_-]?brain|TOF[_-]?WB|_WB_)",
    re.IGNORECASE,
)
_RESAMPLED_RE = re.compile(r"resampled", re.IGNORECASE)
_NIFTI_SUFFIXES = (".nii.gz", ".nii")


def _is_nifti(p: Path) -> bool:
    return p.is_file() and (p.suffix == ".gz" and p.name.endswith(".nii.gz") or p.suffix == ".nii")


def segmentation_outputs_to_keep(out_dir: Path) -> list[Path]:
    """Pick CoW + whole-brain eICAB NIfTIs; ignore resampled / non-segmentation."""
    kept: list[Path] = []
    for p in sorted(out_dir.rglob("*")):
        if not _is_nifti(p):
            continue
        name = p.name
        if _RESAMPLED_RE.search(name):
            continue
        if _COW_RE.search(name):
            kept.append(p)
        elif _WB_RE.search(name):
            kept.append(p)
    return kept


def prune_eicab_outputs(
    out_dir: Path,
    *,
    keep_aux_outputs: bool,
    keep_paths: Iterable[Path] | None = None,
) -> None:
    """Drop auxiliary folders/files unless *keep_aux_outputs* is True."""
    if keep_aux_outputs:
        return
    keep_set = {p.resolve() for p in (keep_paths or ())}
    # Remove known legacy subdirs.
    for sub in ("original_space", "nn_space", "metric_space"):
        d = out_dir / sub
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
    # Remove non-kept NIfTIs and other artifacts.
    for p in list(out_dir.rglob("*")):
        if not p.exists():
            continue
        if p.is_dir():
            if p.name in {"original_space", "nn_space", "metric_space"}:
                continue
            # remove empty dirs later
            continue
        if p.suffix.lower() in {".csv", ".txt", ".log"}:
            p.unlink(missing_ok=True)
            continue
        if _is_nifti(p) and p.resolve() not in keep_set:
            p.unlink(missing_ok=True)
    # Clean up empty directories (except out_dir itself).
    for p in sorted(out_dir.rglob("*"), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            try:
                p.rmdir()
            except OSError:
                pass


def run_eicab(
    input_nii: str | Path,
    output_dir: str | Path,
    *,
    resolution: float = 0.625,
    simple_segmentation: bool = False,
    attention: bool = False,
    device: str = "cpu",
    container: str | Path,
    tmp_dir: str | Path,
    keep_aux_outputs: bool = False,
    vasculature_host_path: str | Path | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run eICAB via ``singularity run`` (same contract as legacy BioImaging).

    *vasculature_host_path* should be the **host** directory bind-mounted to
    ``/programs/Neuro/vasculature2`` (see ``run_eicab_inference.sh``). When
    omitted and the path does not exist, the vasculature bind is skipped (only
    sensible for debugging).
    """
    input_p = Path(input_nii).resolve()
    output_p = Path(output_dir).resolve()
    tmp_p = Path(tmp_dir).resolve()
    container_p = Path(container).resolve()

    if not container_p.is_file():
        raise FileNotFoundError(
            f"eICAB Singularity image not found: {container_p}. "
            "Set a valid path with --container or update "
            "`nvitk.segmentation.eicab.config.CONTAINER_PATH`."
        )
    if not input_p.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {input_p}")
    output_p.mkdir(parents=True, exist_ok=True)
    tmp_p.mkdir(parents=True, exist_ok=True)

    if input_p.name.endswith(".nii.gz") or input_p.suffix == ".gz":
        container_input = "/TOF.nii.gz"
    elif input_p.suffix == ".nii":
        container_input = "/TOF.nii"
    else:
        raise ValueError(f"Unsupported NIfTI path: {input_p}")

    container_path_env = (
        "/vessel_segmentation_snaillab:/programs/Neuro/vasculature2:$PATH"
    )
    cmd: list[str] = [
        "singularity",
        "run",
        "--cleanenv",
        "--env",
        f"PATH={container_path_env}",
    ]
    dev_l = device.lower()
    if dev_l in ("cuda", "gpu"):
        cmd.append("--nv")
        device_arg = "cuda"
    elif dev_l == "cpu":
        device_arg = "cpu"
    else:
        raise ValueError("device must be 'cpu', 'cuda', or 'gpu' (alias for cuda).")

    cmd.extend(
        [
            "--bind",
            f"{input_p}:{container_input}:ro",
            "--bind",
            f"{output_p}:/output",
            "--bind",
            f"{tmp_p}:/tmp",
        ]
    )
    vhp = Path(vasculature_host_path) if vasculature_host_path else None
    if vhp and vhp.is_dir():
        cmd.extend(["--bind", f"{vhp}:/programs/Neuro/vasculature2"])

    cmd.extend(
        [
            str(container_p),
            "-t",
            container_input,
            "-o",
            "/output",
            "-r",
            str(resolution),
            "-d",
            device_arg,
            "-f",
        ]
    )
    if simple_segmentation:
        cmd.append("-s")
    if attention:
        cmd.append("-a")

    log.info("Running eICAB: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=capture_output,
        text=True,
    )
    if proc.returncode != 0:
        log.error("eICAB failed rc=%s stderr=%s", proc.returncode, proc.stderr)
        if check:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=proc.stdout, stderr=proc.stderr
            )
        return proc

    if not keep_aux_outputs:
        keep = segmentation_outputs_to_keep(output_p)
        prune_eicab_outputs(output_p, keep_aux_outputs=False, keep_paths=keep)
        log.info(
            "eICAB finished; kept %d segmentation NIfTI(s): %s",
            len(keep),
            [k.name for k in keep],
        )
    return proc


__all__ = ["prune_eicab_outputs", "run_eicab", "segmentation_outputs_to_keep"]
