"""Local eICAB inference via ``singularity run`` and optional output pruning."""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from nvitk.core.logger import Logger

log = Logger()

_THREAD_LIMIT_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)

_CONTAINER_PATH_ENV = (
    "/vessel_segmentation_snaillab:/programs/Neuro/vasculature2:$PATH"
)
_EICAB_EXPRESS_HOME = "/vessel_segmentation_snaillab"
_NVITK_SRC_BIND = "/nvitk/src"
_EICAB_SH = f"{_EICAB_EXPRESS_HOME}/eICAB.sh"
_CPU_LIMIT_SITE = (
    f"{_NVITK_SRC_BIND}/nvitk/segmentation/eicab/cpu_limit_site"
)
_EICAB_PYTHONPATH = f"{_CPU_LIMIT_SITE}:{_EICAB_EXPRESS_HOME}"
# ComputeVED writes Scale_*/Ved_* into process CWD, then moves them. Parallel SGE
# jobs must not share a CWD (home / SGE_O_WORKDIR) or they steal each other's files.
# Cluster node-local scratch is /data_tmp (no $TMPDIR on this site).
_DEFAULT_METRIC_SCRATCH_ROOT = "/data_tmp"
# Force container CWD onto a path we also bind from node-local scratch.
_EICAB_CONTAINER_PWD = "/tmp/ved_cwd"


def _metric_scratch_job_dir_expr(root: str) -> str:
    """Shell path expression for this job's node-local scratch (expands at job runtime)."""
    r = (root or _DEFAULT_METRIC_SCRATCH_ROOT).rstrip("/") or _DEFAULT_METRIC_SCRATCH_ROOT
    # Only JOB_ID — SGE_TASK_ID is unset/“undefined” on non-array jobs here.
    return f"{r}/nvitk_eicab_${{JOB_ID:-$$}}"


def metric_scratch_bind_args(root: str | None = None) -> list[str]:
    """Singularity ``--bind`` args for VED metric_space + CWD on node-local scratch."""
    job_dir = _metric_scratch_job_dir_expr(root or _DEFAULT_METRIC_SCRATCH_ROOT)
    # Double-quoted so $JOB_ID expands at job runtime; not shlex.quote (would freeze it).
    return [
        "--bind",
        f'"{job_dir}/metric_space:/output/metric_space"',
        "--bind",
        f'"{job_dir}/cwd:{_EICAB_CONTAINER_PWD}"',
    ]


def metric_scratch_prep_shell(root: str | None = None) -> str:
    """Shell snippet: create node-local dirs for VED ``metric_space`` and CWD."""
    preferred = (root or _DEFAULT_METRIC_SCRATCH_ROOT).rstrip("/") or _DEFAULT_METRIC_SCRATCH_ROOT
    job_dir = _metric_scratch_job_dir_expr(preferred)
    return (
        f"METRIC_SCRATCH_ROOT={shlex.quote(preferred)} ; "
        'if [ ! -d "$METRIC_SCRATCH_ROOT" ] || [ ! -w "$METRIC_SCRATCH_ROOT" ]; then '
        'echo "ERROR: eICAB metric scratch root not writable: $METRIC_SCRATCH_ROOT" >&2; exit 1; fi ; '
        f'METRIC_SCRATCH="{job_dir}" && '
        'rm -rf "$METRIC_SCRATCH" && '
        'mkdir -p "$METRIC_SCRATCH/metric_space" "$METRIC_SCRATCH/cwd" && '
        'echo "eICAB node-local scratch: $METRIC_SCRATCH"'
    )


def metric_scratch_cleanup_shell(root: str | None = None) -> str:
    """Shell snippet: remove **only** this job's ``nvitk_eicab_<JOB_ID>`` dir.

    Guards:
    - rebuilds the path from scratch root + ``JOB_ID`` (no glob);
    - requires basename prefix ``nvitk_eicab_``;
    - refuses to delete the scratch root itself.
    """
    preferred = (root or _DEFAULT_METRIC_SCRATCH_ROOT).rstrip("/") or _DEFAULT_METRIC_SCRATCH_ROOT
    root_q = shlex.quote(preferred)
    return (
        f'_NVITK_SCRATCH_ROOT={root_q} ; '
        '_NVITK_SCRATCH="$_NVITK_SCRATCH_ROOT/nvitk_eicab_${JOB_ID:-$$}" ; '
        'case "$_NVITK_SCRATCH" in '
        '"$_NVITK_SCRATCH_ROOT"/nvitk_eicab_*) '
        'if [ -n "$_NVITK_SCRATCH" ] '
        '&& [ "$_NVITK_SCRATCH" != "$_NVITK_SCRATCH_ROOT" ] '
        '&& [ "$_NVITK_SCRATCH" != "$_NVITK_SCRATCH_ROOT/" ] '
        '&& [ -d "$_NVITK_SCRATCH" ]; then '
        'rm -rf -- "$_NVITK_SCRATCH" && '
        'echo "cleaned eICAB node-local scratch: $_NVITK_SCRATCH" ; '
        'fi ;; '
        '*) echo "skip scratch cleanup (unexpected path): $_NVITK_SCRATCH" >&2 ;; '
        'esac'
    )

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


def _stem_endswith_resampled(p: Path) -> bool:
    """True for ``<img>_resampled.nii`` / ``<img>_resampled.nii.gz`` style outputs."""
    name = p.name
    if name.lower().endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
    elif name.lower().endswith(".nii"):
        stem = name[: -len(".nii")]
    else:
        return False
    return stem.lower().endswith("_resampled")


def segmentation_outputs_to_keep(out_dir: Path) -> list[Path]:
    """Pick CoW, whole-brain, and ``*_resampled`` NIfTIs; drop other intermediates."""
    kept: list[Path] = []
    for p in sorted(out_dir.rglob("*")):
        if not _is_nifti(p):
            continue
        name = p.name
        if _stem_endswith_resampled(p):
            kept.append(p)
            continue
        if _RESAMPLED_RE.search(name):
            continue
        if _COW_RE.search(name):
            kept.append(p)
        elif _WB_RE.search(name):
            kept.append(p)
    return list(dict.fromkeys(kept))


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


def build_eicab_singularity_argv(
    input_nii: str | Path,
    output_dir: str | Path,
    *,
    tmp_dir: str | Path,
    container: str | Path,
    resolution: float = 0.625,
    simple_segmentation: bool = False,
    attention: bool = False,
    device: str = "cpu",
    vasculature_host_path: str | Path | None = None,
) -> list[str]:
    """Build ``singularity run`` argv for eICAB on the cluster host (host paths)."""
    input_p = Path(input_nii).resolve()
    output_p = Path(output_dir).resolve()
    tmp_p = Path(tmp_dir).resolve()
    container_p = Path(container).resolve()

    if input_p.name.endswith(".nii.gz") or input_p.suffix == ".gz":
        container_input = "/TOF.nii.gz"
    elif input_p.suffix == ".nii":
        container_input = "/TOF.nii"
    else:
        raise ValueError(f"Unsupported NIfTI path: {input_p}")

    cmd: list[str] = [
        "singularity",
        "run",
        "--cleanenv",
        "--env",
        f"PATH={_CONTAINER_PATH_ENV}",
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
            "--pwd",
            _EICAB_CONTAINER_PWD,
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
    return cmd


def _eicab_container_input_path(input_p: Path) -> str:
    if input_p.name.endswith(".nii.gz") or input_p.suffix == ".gz":
        return "/TOF.nii.gz"
    if input_p.suffix == ".nii":
        return "/TOF.nii"
    raise ValueError(f"Unsupported NIfTI path: {input_p}")


def _eicab_device_arg(device: str) -> tuple[str, bool]:
    dev_l = device.lower()
    if dev_l in ("cuda", "gpu"):
        return "cuda", True
    if dev_l == "cpu":
        return "cpu", False
    raise ValueError("device must be 'cpu', 'cuda', or 'gpu' (alias for cuda).")


def _eicab_cli_args(
    container_input: str,
    resolution: float,
    device_arg: str,
    *,
    simple_segmentation: bool,
    attention: bool,
) -> list[str]:
    args = [
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
    if simple_segmentation:
        args.append("-s")
    if attention:
        args.append("-a")
    return args


def build_eicab_singularity_shell_cmd(
    input_nii: str | Path,
    output_dir: str | Path,
    *,
    tmp_dir: str | Path,
    container: str | Path,
    resolution: float = 0.625,
    simple_segmentation: bool = False,
    attention: bool = False,
    device: str = "cpu",
    vasculature_host_path: str | Path | None = None,
    cpu_limit_shell_expr: str | None = None,
    nvitk_src_dir: str | Path | None = None,
    local_metric_scratch: bool = False,
    metric_scratch_root: str | None = None,
) -> str:
    """Shell command string for eICAB ``singularity run`` or ``exec``.

    When *cpu_limit_shell_expr* is set and *nvitk_src_dir* is provided, uses
    ``singularity exec`` with ``eICAB.sh`` and a ``sitecustomize`` hook on
    ``PYTHONPATH`` so VED respects ``NVITK_CPU_LIMIT`` (``multiprocessing.cpu_count``).

    Always passes ``--pwd /tmp/ved_cwd`` so ComputeVED's CWD-relative Scale_*
    globs stay isolated under the per-subject ``/tmp`` bind.
    """
    needs_custom = bool(cpu_limit_shell_expr) or local_metric_scratch or (
        nvitk_src_dir is not None
    )
    if not needs_custom and not cpu_limit_shell_expr:
        # Local / simple path: still use argv with --pwd.
        return shlex.join(
            build_eicab_singularity_argv(
                input_nii,
                output_dir,
                tmp_dir=tmp_dir,
                container=container,
                resolution=resolution,
                simple_segmentation=simple_segmentation,
                attention=attention,
                device=device,
                vasculature_host_path=vasculature_host_path,
            )
        )

    input_p = Path(input_nii).resolve()
    output_p = Path(output_dir).resolve()
    tmp_p = Path(tmp_dir).resolve()
    container_p = Path(container).resolve()
    container_input = _eicab_container_input_path(input_p)
    device_arg, use_nv = _eicab_device_arg(device)

    use_cpu_runner = bool(cpu_limit_shell_expr) and nvitk_src_dir is not None
    parts: list[str] = [
        "singularity",
        "exec" if use_cpu_runner else "run",
        "--cleanenv",
    ]
    parts.extend(["--env", shlex.quote(f"PATH={_CONTAINER_PATH_ENV}")])
    if cpu_limit_shell_expr:
        for var in _THREAD_LIMIT_VARS:
            parts.extend(["--env", f"{var}={cpu_limit_shell_expr}"])
    if use_cpu_runner:
        parts.extend(
            [
                "--env",
                f"EXPRESS_HOME={_EICAB_EXPRESS_HOME}",
                "--env",
                f"PYTHONPATH={_EICAB_PYTHONPATH}",
                "--env",
                f"NVITK_CPU_LIMIT={cpu_limit_shell_expr}",
            ]
        )
    if use_nv:
        parts.append("--nv")
    parts.extend(
        [
            "--pwd",
            _EICAB_CONTAINER_PWD,
            "--bind",
            shlex.quote(f"{input_p}:{container_input}:ro"),
            "--bind",
            shlex.quote(f"{output_p}:/output"),
            "--bind",
            shlex.quote(f"{tmp_p}:/tmp"),
        ]
    )
    vhp = Path(vasculature_host_path) if vasculature_host_path else None
    if vhp and vhp.is_dir():
        parts.extend(["--bind", shlex.quote(f"{vhp}:/programs/Neuro/vasculature2")])
    if local_metric_scratch:
        # After /output bind so this overlays NFS metric_space; explicit ${JOB_ID}
        # path (not $METRIC_SCRATCH) so the bind cannot silently fall back to NFS.
        parts.extend(metric_scratch_bind_args(metric_scratch_root))
    if use_cpu_runner:
        src_p = Path(nvitk_src_dir).resolve()
        parts.extend(
            ["--bind", shlex.quote(f"{src_p}:{_NVITK_SRC_BIND}")]
        )
    parts.append(shlex.quote(str(container_p)))
    if use_cpu_runner:
        parts.append(shlex.quote(_EICAB_SH))
    parts.extend(
        _eicab_cli_args(
            container_input,
            resolution,
            device_arg,
            simple_segmentation=simple_segmentation,
            attention=attention,
        )
    )
    return " ".join(parts)


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
    (tmp_p / "ved_cwd").mkdir(parents=True, exist_ok=True)
    metric = output_p / "metric_space"
    if metric.exists():
        shutil.rmtree(metric, ignore_errors=True)
    metric.mkdir(parents=True, exist_ok=True)

    cmd = build_eicab_singularity_argv(
        input_p,
        output_p,
        tmp_dir=tmp_p,
        container=container_p,
        resolution=resolution,
        simple_segmentation=simple_segmentation,
        attention=attention,
        device=device,
        vasculature_host_path=vasculature_host_path,
    )

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


__all__ = [
    "build_eicab_singularity_argv",
    "metric_scratch_bind_args",
    "metric_scratch_cleanup_shell",
    "metric_scratch_prep_shell",
    "prune_eicab_outputs",
    "run_eicab",
    "segmentation_outputs_to_keep",
]
