"""SGE + Singularity chain helpers for the PESA-Fat pipelines.

The helpers here generalise the ``qsub | singularity exec`` pattern used by
:mod:`nvitk.segmentation.total_segmentator.cluster` so every pipeline stage
(conversion, segmentation, post-processing, measurement) can be submitted as
a self-contained Singularity job on an SGE cluster.

Typical usage inside a pipeline master::

    from nvitk.pipes.pesa_fat.common.sge import (
        ClusterPaths, SgeResources, SingularityBinds, StageSpec, submit_chain,
    )

    paths = ClusterPaths(src=..., container=..., models=..., data_root=...,
                        output_root=..., log_dir=..., err_dir=...)
    stages = [
        StageSpec(job_name="ctpet_stage1_PESA001",
                  python_cmd="python -m nvitk.pipes.pesa_fat.ct_pet_v5.stage1_segment "
                             "--batch X --subject PESA001 ...",
                  resources=SgeResources(ngpu=1, h_vmem="50G")),
        StageSpec(job_name="ctpet_stage2_PESA001", python_cmd="...", resources=...),
        StageSpec(job_name="ctpet_stage3_PESA001", python_cmd="...", resources=...),
    ]
    jids = submit_chain(stages, paths, base_hold=stage0_jid)

Each stage's job is submitted with ``qsub -hold_jid <prev_jid>`` so SGE waits
for the previous stage of the same subject to finish before starting the next
one. Many subjects run their chains in parallel.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, TextIO


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SingularityBinds:
    """Container bind-points used by the PESA-Fat cluster pipelines."""

    src: str = "/PESAFat/src/"
    data: str = "/PESAFat/data/"
    output: str = "/PESAFat/output/"
    models: str = "/models/"


@dataclass
class SgeResources:
    """SGE submission resources."""

    project: str = "GPU"
    account: str = "Prod"
    ngpu: int = 1
    h_vmem: str = "50G"
    queue: str | None = None


@dataclass
class ClusterPaths:
    """Host-side paths that must exist before submission."""

    src: Path
    container: Path
    models: Path
    data_root: Path
    output_root: Path
    log_dir: Path
    err_dir: Path

    def ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.err_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class StageSpec:
    """A single SGE-submitted pipeline stage.

    ``python_cmd`` is the literal command run *inside* the Singularity
    container (e.g. ``python -m nvitk.pipes.pesa_fat.ct_pet_v5.stage2_postprocess
    --batch X --subject PESA001 ...``). Host paths referenced inside the command
    must fall within the container bind mounts defined by :class:`ClusterPaths`
    and :class:`SingularityBinds`.
    """

    job_name: str
    python_cmd: str
    resources: SgeResources = field(default_factory=SgeResources)
    binds: SingularityBinds = field(default_factory=SingularityBinds)
    extra_env: dict[str, str] = field(default_factory=dict)
    use_nv: bool = True


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------


def build_singularity_command(spec: StageSpec, paths: ClusterPaths) -> str:
    """Wrap ``spec.python_cmd`` in ``singularity exec`` with the standard binds."""
    env_exports = " ".join(
        f'export {k}="{v}" &&' for k, v in spec.extra_env.items()
    )
    inner = f"{env_exports} {spec.python_cmd}".strip()
    nv = "--nv " if spec.use_nv else ""
    cmd = (
        f"singularity exec {nv}"
        f"-B {paths.src}:{spec.binds.src} "
        f"-B {paths.data_root}:{spec.binds.data} "
        f"-B {paths.output_root}:{spec.binds.output} "
        f"-B {paths.models}:{spec.binds.models} "
        f"{paths.container} bash -c " + shlex.quote(inner)
    )
    return cmd


def build_qsub_command(
    spec: StageSpec,
    paths: ClusterPaths,
    *,
    hold_jid: str | Sequence[str] | None = None,
) -> list[str]:
    """Build the ``qsub`` argv for *spec*.

    If *hold_jid* is provided (either a single jid string or a sequence), the
    ``-hold_jid`` flag is appended so SGE will wait for those job(s) to
    terminate before starting this one.
    """
    log_file = paths.log_dir / f"{spec.job_name}.log"
    err_file = paths.err_dir / f"{spec.job_name}.err"

    argv = [
        "qsub",
        "-P", spec.resources.project,
        "-terse",
        "-N", spec.job_name,
        "-A", spec.resources.account,
        "-l", f"ngpu={spec.resources.ngpu}",
        "-l", f"h_vmem={spec.resources.h_vmem}",
        "-o", str(log_file),
        "-e", str(err_file),
    ]
    if spec.resources.queue:
        argv.extend(["-q", spec.resources.queue])

    if hold_jid:
        if isinstance(hold_jid, str):
            joined = hold_jid.strip()
        else:
            joined = ",".join(str(j).strip() for j in hold_jid if j)
        if joined:
            argv.extend(["-hold_jid", joined])

    return argv


# ---------------------------------------------------------------------------
# Bash emitter helpers
# ---------------------------------------------------------------------------


_SHELL_VAR_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _shell_var(job_name: str) -> str:
    """Turn a job name like ``PESAFat_ctpet-v5_stage1_PESA001`` into a safe
    bash variable suffix (letters / digits / underscores only)."""
    return _SHELL_VAR_SAFE.sub("_", job_name)


def _quote_qsub_arg(arg: str) -> str:
    """Shell-quote a single qsub argv token, but leave bare shell-variable
    references (``$jid_foo`` / ``${jid_foo}``) unquoted so bash expands them.
    Wrap those in double-quotes for hygiene against whitespace."""
    if arg.startswith("$"):
        return f'"{arg}"'
    return shlex.quote(arg)


def _emit_stage_block(
    emit: TextIO,
    spec: StageSpec,
    inner: str,
    qsub_argv: Sequence[str],
    hold_jid: str | Sequence[str] | None,
) -> str:
    """Write a bash block reproducing ``echo <inner> | qsub ...`` to *emit*.

    Returns the shell-variable reference (``$jid_<job>``) that subsequent
    chained stages should pass as their ``-hold_jid``.
    """
    var = _shell_var(spec.job_name)
    jid_var = f"jid_{var}"
    singcmd_var = f"singcmd_{var}"
    qsub_var = f"qsub_{var}"

    if hold_jid is None:
        hold_descr = "none"
    elif isinstance(hold_jid, str):
        hold_descr = hold_jid
    else:
        hold_descr = ",".join(str(h) for h in hold_jid if h) or "none"

    qsub_lines = "\n  ".join(_quote_qsub_arg(a) for a in qsub_argv)

    emit.write(
        f"# --- {spec.job_name} (hold: {hold_descr}) ---\n"
        f"read -r -d '' {singcmd_var} << 'SINGULARITY_EOF' || true\n"
        f"{inner}\n"
        f"SINGULARITY_EOF\n"
        f"\n"
        f"{qsub_var}=(\n"
        f"  {qsub_lines}\n"
        f")\n"
        f'{jid_var}=$(echo "${singcmd_var}" | "${{{qsub_var}[@]}}")\n'
        f'echo "{spec.job_name} -> ${jid_var}"\n'
        f"\n"
    )
    return f"${jid_var}"


def write_script_header(
    emit: TextIO,
    *,
    log_dir: Path,
    err_dir: Path,
    title: str,
) -> None:
    """Write the common preamble for an emitted submission script."""
    ts = datetime.now().isoformat(timespec="seconds")
    emit.write(
        "#!/usr/bin/env bash\n"
        f"# Auto-generated by nvitk-pesa-fat ({title}) on {ts}\n"
        "# Run this on the cluster login node:\n"
        "#     bash <this_file>\n"
        "# Only `bash`, `qsub` and `singularity` are required on the host.\n"
        "set -euo pipefail\n"
        "\n"
        f'log_dir={shlex.quote(str(log_dir))}\n'
        f'err_dir={shlex.quote(str(err_dir))}\n'
        'mkdir -p "$log_dir" "$err_dir"\n'
        "\n"
    )


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


def submit_stage(
    spec: StageSpec,
    paths: ClusterPaths,
    *,
    hold_jid: str | Sequence[str] | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
) -> str:
    """Submit *spec* to SGE by piping the Singularity command into ``qsub``.

    When *emit* is provided, the qsub+singularity pair is written as a bash
    block to that file object instead of being executed, and the returned
    value is a shell-variable reference (e.g. ``"$jid_<job_name>"``) that
    subsequent chained stages can use as their ``-hold_jid``.

    Otherwise, returns the SGE job id (or ``'DRY_RUN'`` when *dry_run*).
    """
    inner = build_singularity_command(spec, paths)
    qsub_argv = build_qsub_command(spec, paths, hold_jid=hold_jid)

    if emit is not None:
        return _emit_stage_block(emit, spec, inner, qsub_argv, hold_jid)

    if dry_run:
        return "DRY_RUN"

    paths.ensure_dirs()
    result = subprocess.run(
        qsub_argv,
        input=inner,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def submit_chain(
    stages: Iterable[StageSpec],
    paths: ClusterPaths,
    *,
    base_hold: str | Sequence[str] | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
) -> list[str]:
    """Submit a linear chain of *stages* for a single subject.

    Each stage is submitted with ``-hold_jid`` set to the previous stage's jid
    (or *base_hold* for the first stage). Returns the list of jids in order
    (or shell-variable references when *emit* is set).
    """
    jids: list[str] = []
    prev: str | Sequence[str] | None = base_hold
    for s in stages:
        jid = submit_stage(s, paths, hold_jid=prev, dry_run=dry_run, emit=emit)
        jids.append(jid)
        prev = jid
    return jids


__all__ = [
    "ClusterPaths",
    "SgeResources",
    "SingularityBinds",
    "StageSpec",
    "build_qsub_command",
    "build_singularity_command",
    "submit_chain",
    "submit_stage",
    "write_script_header",
]
