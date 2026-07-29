"""SGE + Singularity helpers shared across nvitk pipelines and segmentation CLIs.

Generalises the ``echo <singularity exec ...> | qsub ...`` pattern so each stage
(conversion, segmentation, post-processing, measurement) can be submitted as a
self-contained Singularity job on an SGE cluster.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, TextIO

# Longest prefix first so ``XSGPU`` matches ``XS``, not ``S``.
_VIRTUAL_GPU_PROJECT_PREFIXES: tuple[tuple[str, str], ...] = (
    ("XS", "xsgpu"),
    ("S", "sgpu"),
    ("L", "lgpu"),
)


def sge_virtual_gpu_resource_name(project: str | None) -> str | None:
    """
    Virtual-GPU ``qsub -l`` resource for *project*, or ``None`` for classic ``ngpu``.

    Projects whose ``-P`` name starts with ``L``, ``S``, or ``XS`` (e.g. ``LGPU``,
    ``SGPU``, ``XSGPU``) use ``-l lgpu=0``, ``-l sgpu=0``, or ``-l xsgpu=0`` instead of
    ``-l ngpu=…``. The original ``GPU`` / ``MCC_GPU`` projects keep ``-l ngpu``.
    """
    proj = str(project or "").strip().upper()
    for prefix, resource in _VIRTUAL_GPU_PROJECT_PREFIXES:
        if proj.startswith(prefix):
            return resource
    return None


def sge_project_uses_virtual_gpu_resource(project: str | None) -> bool:
    """True when ``qsub`` must use ``-l lgpu|sgpu|xsgpu=0`` instead of ``-l ngpu``."""
    return sge_virtual_gpu_resource_name(project) is not None


def sge_project_uses_xsgpu_resource(project: str | None) -> bool:
    """True when *project* is an XS* virtual-GPU queue (``-l xsgpu=0``)."""
    return sge_virtual_gpu_resource_name(project) == "xsgpu"


def sge_project_omits_ngpu_request(project: str | None) -> bool:
    """Alias for :func:`sge_project_uses_virtual_gpu_resource`."""
    return sge_project_uses_virtual_gpu_resource(project)


def qsub_l_resource_args(resources: SgeResources) -> list[str]:
    """``qsub`` ``-l`` option pairs derived from *resources*."""
    args: list[str] = []
    vgpu = sge_virtual_gpu_resource_name(resources.project)
    if vgpu is not None:
        args.extend(["-l", f"{vgpu}=0"])
    elif resources.ngpu:
        args.extend(["-l", f"ngpu={resources.ngpu}"])
    args.extend(["-l", f"h_vmem={resources.h_vmem}"])
    return args


@dataclass
class SingularityBinds:
    """Container bind-points used by cluster pipelines."""

    src: str = "/nvitk/src/"
    data: str = "/nvitk/data/"
    output: str = "/nvitk/output/"
    models: str = "/models/"


@dataclass
class SgeResources:
    """SGE submission resources."""

    project: str = "GPU"
    account: str = "Prod"
    ngpu: int = 1
    h_vmem: str = "50G"
    queue: str | None = None
    pe_smp: int | None = None


@dataclass
class ClusterPaths:
    """Host-side paths that must exist before submission."""

    src: Path
    container: Path
    models: Path | None
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
    container. Host paths referenced inside the command must fall within the
    bind mounts defined by :class:`ClusterPaths` and :class:`SingularityBinds`.
    """

    job_name: str
    python_cmd: str
    resources: SgeResources = field(default_factory=SgeResources)
    binds: SingularityBinds = field(default_factory=SingularityBinds)
    extra_env: dict[str, str] = field(default_factory=dict)
    use_nv: bool = True
    #: Additional ``singularity exec -B host:container`` pairs (host paths only).
    extra_host_binds: tuple[tuple[Path, str], ...] = field(default_factory=tuple)


def python_module_argv(module: str, *, python: str = "python") -> list[str]:
    """Argv prefix for in-container workers: ``python -m <module>``.

    Use with relative-import package modules (e.g. ``nvitk.pipes.qvtpy.stage0_convert``).
    Requires ``PYTHONPATH`` to include the tree that contains the ``nvitk`` package
    (typically ``/nvitk/src/`` via :class:`SingularityBinds`).
    """
    return [python, "-m", module]


def python_script_argv(
    script_path_in_container: str,
    *,
    python: str = "python",
) -> list[str]:
    """Argv prefix to run a ``.py`` file by absolute path inside the container.

    Prefer this over :func:`python_module_argv` when the cluster image does not ship
    optional subpackages (e.g. ``nvitk.gui.sge``) but the repo is bind-mounted at
    ``SingularityBinds.src``. ``PYTHONPATH`` must still include that src root.
    """
    path = str(script_path_in_container).strip()
    if not path.startswith("/"):
        raise ValueError(
            f"script_path_in_container must be absolute inside the container, got {path!r}"
        )
    return [python, path]


def gui_sge_worker_script_path(binds: SingularityBinds | None = None) -> str:
    """Absolute in-container path to :mod:`nvitk.gui.sge.worker` (bind-mounted source)."""
    root = (binds or SingularityBinds()).src.rstrip("/")
    return f"{root}/nvitk/gui/sge/worker.py"


def gui_sge_worker_argv(
    binds: SingularityBinds | None = None,
    *,
    python: str = "python",
) -> list[str]:
    """Argv prefix for the Napari GUI SGE headless worker."""
    return python_script_argv(gui_sge_worker_script_path(binds), python=python)


def build_singularity_command(spec: StageSpec, paths: ClusterPaths) -> str:
    """Wrap ``spec.python_cmd`` in ``singularity exec`` with the standard binds.

    Injects a small BLAS/OMP thread-cap preamble (``NSLOTS`` / ``NVITK_CPU_LIMIT``,
    capped at 64) before ``extra_env`` exports so OpenBLAS/MKL never spawn
    unbounded threads on fat SGE nodes (avoids known segfaults).
    """
    # Cap before Python starts: OpenBLAS reads env at library init.
    thread_preamble = (
        '_t="${NVITK_CPU_LIMIT:-${NSLOTS:-8}}"; '
        'case "$_t" in (*[!0-9]*|"") _t=8 ;; esac; '
        'if [ "$_t" -gt 64 ]; then _t=64; fi; '
        'export OPENBLAS_NUM_THREADS="$_t" OMP_NUM_THREADS="$_t" '
        'MKL_NUM_THREADS="$_t" NUMEXPR_NUM_THREADS="$_t" '
        'VECLIB_MAXIMUM_THREADS="$_t"; '
    )
    env_exports = " ".join(
        f'export {k}="{v}" &&' for k, v in spec.extra_env.items()
    )
    inner = f"{thread_preamble}{env_exports} {spec.python_cmd}".strip()
    nv = "--nv " if spec.use_nv else ""
    parts: list[str] = [
        f"singularity exec {nv}",
        f"-B {shlex.quote(str(paths.src))}:{shlex.quote(spec.binds.src)} ",
        f"-B {shlex.quote(str(paths.data_root))}:{shlex.quote(spec.binds.data)} ",
        f"-B {shlex.quote(str(paths.output_root))}:{shlex.quote(spec.binds.output)} ",
    ]
    if paths.models is not None:
        parts.append(
            f"-B {shlex.quote(str(paths.models))}:{shlex.quote(spec.binds.models)} "
        )
    for host, mnt in spec.extra_host_binds:
        parts.append(
            f"-B {shlex.quote(str(host))}:{shlex.quote(mnt)} "
        )
    parts.append(f"{shlex.quote(str(paths.container))} bash -c ")
    parts.append(shlex.quote(inner))
    return "".join(parts)


def build_qsub_command(
    spec: StageSpec,
    paths: ClusterPaths,
    *,
    hold_jid: str | Sequence[str] | None = None,
    array_tasks: int | None = None,
    task_concurrency: int | None = None,
) -> list[str]:
    """Build the ``qsub`` argv for *spec*.

    When *array_tasks* is set, emit ``-t 1-N`` (and optional ``-tc``) with
    per-task log/err paths using SGE's ``$TASK_ID``.
    """
    if array_tasks is not None:
        n = int(array_tasks)
        if n < 1:
            raise ValueError(f"array_tasks must be >= 1, got {array_tasks!r}")
        log_file = paths.log_dir / f"{spec.job_name}.$TASK_ID.log"
        err_file = paths.err_dir / f"{spec.job_name}.$TASK_ID.err"
    else:
        log_file = paths.log_dir / f"{spec.job_name}.log"
        err_file = paths.err_dir / f"{spec.job_name}.err"

    argv = [
        "qsub",
        "-P", spec.resources.project,
        "-terse",
        "-N", spec.job_name,
        "-A", spec.resources.account,
        *qsub_l_resource_args(spec.resources),
        "-o", str(log_file),
        "-e", str(err_file),
    ]
    if spec.resources.queue:
        argv.extend(["-q", spec.resources.queue])
    if spec.resources.pe_smp:
        argv.extend(["-pe", "smp", str(spec.resources.pe_smp)])
    if array_tasks is not None:
        argv.extend(["-t", f"1-{int(array_tasks)}"])
        if task_concurrency is not None:
            tc = int(task_concurrency)
            if tc < 1:
                raise ValueError(
                    f"task_concurrency must be >= 1, got {task_concurrency!r}"
                )
            argv.extend(["-tc", str(tc)])

    if hold_jid:
        if isinstance(hold_jid, str):
            joined = hold_jid.strip()
        else:
            joined = ",".join(str(j).strip() for j in hold_jid if j)
        if joined:
            argv.extend(["-hold_jid", joined])

    return argv


@dataclass(frozen=True)
class ArrayTaskSpec:
    """One task inside an SGE array job (``SGE_TASK_ID`` maps 1-based to order)."""

    stage_id: str
    shell_cmd: str


def build_array_worker_script(
    tasks: Sequence[ArrayTaskSpec],
    *,
    marker_dir: Path | str,
    marker_wait_timeout_sec: int = 1800,
) -> str:
    """Bash worker body for ``qsub -t``: dispatch by ``SGE_TASK_ID`` with done-markers.

    Task ``k`` waits for ``${JOB_ID}.(k-1).done`` before running (except task 1).
    With ``-tc 1`` the predecessor has already left the queue when task ``k``
    starts, so the wait is only for NFS/marker visibility — not the predecessor
    runtime. An ``EXIT`` trap always writes a marker when possible; ``SIGKILL``
    (OOM killer) cannot run the trap, so the wait has a hard timeout to avoid
    sleeping forever with empty logs.
    """
    if not tasks:
        raise ValueError("tasks must be non-empty")
    timeout = max(60, int(marker_wait_timeout_sec))
    lines: list[str] = [
        "#!/usr/bin/env bash",
        "set -uo pipefail",
        f"MARKER_DIR={shlex.quote(str(marker_dir))}",
        f"MARKER_WAIT_TIMEOUT={timeout}",
        "rc=1",
        'mkdir -p "$MARKER_DIR"',
        "_write_done_marker() {",
        '  if [[ -n "${JOB_ID:-}" && -n "${SGE_TASK_ID:-}" ]]; then',
        '    mkdir -p "$MARKER_DIR" 2>/dev/null || true',
        '    echo "${rc:-1}" > "$MARKER_DIR/${JOB_ID}.${SGE_TASK_ID}.done" || true',
        "  fi",
        "}",
        "trap '_write_done_marker' EXIT",
        'if [[ -z "${SGE_TASK_ID:-}" ]]; then',
        '  echo "SGE_TASK_ID is unset; this script must run as an SGE array task" >&2',
        "  rc=1",
        "  exit 1",
        "fi",
        'if [[ -z "${JOB_ID:-}" ]]; then',
        '  echo "JOB_ID is unset; this script must run under SGE" >&2',
        "  rc=1",
        "  exit 1",
        "fi",
        'if [[ "$SGE_TASK_ID" -gt 1 ]]; then',
        '  prev=$((SGE_TASK_ID - 1))',
        '  marker="$MARKER_DIR/${JOB_ID}.${prev}.done"',
        '  echo "[nvitk|SGE] task $SGE_TASK_ID waiting for predecessor marker $marker '
        '(timeout=${MARKER_WAIT_TIMEOUT}s)"',
        "  waited=0",
        '  while [[ ! -f "$marker" ]]; do',
        "    if (( waited >= MARKER_WAIT_TIMEOUT )); then",
        '      echo "[nvitk|SGE] ERROR: timed out after ${waited}s waiting for $marker" >&2',
        '      echo "[nvitk|SGE] Predecessor task $prev likely exited without a done-marker '
        '(SIGKILL/OOM or node failure). Aborting." >&2',
        "      rc=99",
        "      exit 99",
        "    fi",
        "    sleep 15",
        "    waited=$((waited + 15))",
        "    if (( waited % 60 == 0 )); then",
        '      echo "[nvitk|SGE] still waiting for $marker (${waited}s / '
        '${MARKER_WAIT_TIMEOUT}s)"',
        "    fi",
        "  done",
        '  pred_rc=$(cat "$marker" 2>/dev/null || echo "?")',
        '  echo "[nvitk|SGE] predecessor task $prev done (marker_rc=$pred_rc); '
        'starting task $SGE_TASK_ID"',
        "fi",
        'echo "[nvitk|SGE] running task $SGE_TASK_ID / '
        f'{len(tasks)}"',
        "case \"$SGE_TASK_ID\" in",
    ]
    for i, task in enumerate(tasks, start=1):
        # Single-quoted heredoc body: commands must not contain the delimiter.
        cmd = task.shell_cmd.rstrip("\n")
        lines.append(f"  {i})")
        lines.append(f"    # stage: {task.stage_id}")
        lines.append("    set +e")
        lines.append(f"    {cmd}")
        lines.append("    rc=$?")
        lines.append("    set -e")
        lines.append("    ;;")
    lines.extend(
        [
            "  *)",
            '    echo "Unexpected SGE_TASK_ID=$SGE_TASK_ID (expected 1-'
            f'{len(tasks)})" >&2',
            "    rc=1",
            "    ;;",
            "esac",
            'echo "[nvitk|SGE] task $SGE_TASK_ID finished rc=$rc"',
            "exit \"$rc\"",
            "",
        ]
    )
    return "\n".join(lines)


def emit_array_job_block(
    emit: TextIO,
    *,
    job_name: str,
    resources: SgeResources,
    paths: ClusterPaths,
    tasks: Sequence[ArrayTaskSpec],
    marker_dir: Path | str,
    task_concurrency: int = 1,
    use_nv: bool = True,
    hold_jid: str | Sequence[str] | None = None,
) -> str:
    """Emit one ``qsub -t 1-N -tc …`` block; return shell ``$jid_…`` reference.

    *hold_jid* is applied to the whole array (classic ``-hold_jid``), e.g. so a
    per-subject stage0 job completes before the stage1–3 array starts.
    """
    if not tasks:
        raise ValueError("tasks must be non-empty")
    n_tasks = len(tasks)
    stage_list = ",".join(t.stage_id for t in tasks)
    worker = build_array_worker_script(tasks, marker_dir=marker_dir)
    spec = StageSpec(
        job_name=job_name,
        python_cmd="",
        resources=resources,
        use_nv=use_nv,
    )
    qsub_argv = build_qsub_command(
        spec,
        paths,
        hold_jid=hold_jid,
        array_tasks=n_tasks,
        task_concurrency=task_concurrency,
    )
    emit_sge_submission_summary_to_terminal(
        spec,
        paths,
        hold_jid=hold_jid,
        qsub_argv=qsub_argv,
        singularity_cmd=(
            f"[array {n_tasks} tasks: {stage_list}; markers under {marker_dir}]"
        ),
    )

    var = _shell_var(job_name)
    jid_var = f"jid_{var}"
    workercmd_var = f"workercmd_{var}"
    qsub_var = f"qsub_{var}"
    qsub_lines = "\n  ".join(_quote_qsub_arg(a) for a in qsub_argv)
    hold_descr = _hold_jid_repr(hold_jid)

    emit.write(
        f"# --- {job_name} (array 1-{n_tasks}; stages: {stage_list}; "
        f"tc={task_concurrency}; hold: {hold_descr}) ---\n"
        f"mkdir -p {shlex.quote(str(marker_dir))}\n"
        f"read -r -d '' {workercmd_var} << 'ARRAY_WORKER_EOF' || true\n"
        f"{worker}"
        f"ARRAY_WORKER_EOF\n"
        f"\n"
        f"{qsub_var}=(\n"
        f"  {qsub_lines}\n"
        f")\n"
        f'{jid_var}=$(echo "${workercmd_var}" | "${{{qsub_var}[@]}}")\n'
        f'echo "{job_name} -> ${jid_var}"\n'
        f"\n"
    )
    return f"${jid_var}"


def submit_array_job(
    *,
    job_name: str,
    resources: SgeResources,
    paths: ClusterPaths,
    tasks: Sequence[ArrayTaskSpec],
    marker_dir: Path | str,
    task_concurrency: int = 1,
    use_nv: bool = True,
    hold_jid: str | Sequence[str] | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
) -> str:
    """Submit or emit one SGE array job; return jid (or ``$jid_…`` / ``DRY_RUN``)."""
    if emit is not None:
        return emit_array_job_block(
            emit,
            job_name=job_name,
            resources=resources,
            paths=paths,
            tasks=tasks,
            marker_dir=marker_dir,
            task_concurrency=task_concurrency,
            use_nv=use_nv,
            hold_jid=hold_jid,
        )

    if not tasks:
        raise ValueError("tasks must be non-empty")
    n_tasks = len(tasks)
    stage_list = ",".join(t.stage_id for t in tasks)
    worker = build_array_worker_script(tasks, marker_dir=marker_dir)
    spec = StageSpec(
        job_name=job_name,
        python_cmd="",
        resources=resources,
        use_nv=use_nv,
    )
    qsub_argv = build_qsub_command(
        spec,
        paths,
        hold_jid=hold_jid,
        array_tasks=n_tasks,
        task_concurrency=task_concurrency,
    )
    emit_sge_submission_summary_to_terminal(
        spec,
        paths,
        hold_jid=hold_jid,
        qsub_argv=qsub_argv,
        singularity_cmd=(
            f"[array {n_tasks} tasks: {stage_list}; markers under {marker_dir}]"
        ),
    )
    if dry_run:
        return "DRY_RUN"

    paths.ensure_dirs()
    Path(marker_dir).mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        qsub_argv,
        input=worker,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _hold_jid_repr(hold_jid: str | Sequence[str] | None) -> str:
    if hold_jid is None:
        return "none"
    if isinstance(hold_jid, str):
        return hold_jid.strip() or "none"
    joined = ",".join(str(h).strip() for h in hold_jid if h)
    return joined or "none"


def _sge_gpu_resource_log_line(resources: SgeResources) -> str:
    vgpu = sge_virtual_gpu_resource_name(resources.project)
    if vgpu is not None:
        return f"  sge_{vgpu} (-l):     0 (virtual GPU; no -l ngpu)"
    if resources.ngpu:
        return f"  sge_ngpu (-l):    {resources.ngpu}"
    return "  sge_ngpu (-l):    (omitted; CPU job)"


def format_sge_submission_summary(
    spec: StageSpec,
    paths: ClusterPaths,
    *,
    hold_jid: str | Sequence[str] | None,
    qsub_argv: Sequence[str],
    singularity_cmd: str,
    max_singularity_chars: int = 4000,
) -> str:
    """Human-readable lines for logging (qsub argv, resources, bind mounts)."""
    r = spec.resources
    lines: list[str] = [
        "[nvitk|SGE] stage submission",
        f"  job_name:        {spec.job_name}",
        f"  hold_jid:        {_hold_jid_repr(hold_jid)}",
        f"  sge_project (-P): {r.project}",
        f"  sge_account (-A): {r.account}",
        _sge_gpu_resource_log_line(r),
        f"  sge_h_vmem (-l):  {r.h_vmem}",
        f"  sge_queue (-q):   {r.queue if r.queue else '(default)'}",
        f"  sge_pe_smp (-pe): {r.pe_smp if r.pe_smp else '(omitted)'}",
        f"  use_nv (outer):   {spec.use_nv}",
        "  bind mounts (host -> container):",
        f"    src:          {paths.src} -> {spec.binds.src}",
        f"    data:         {paths.data_root} -> {spec.binds.data}",
        f"    output:       {paths.output_root} -> {spec.binds.output}",
    ]
    if paths.models is not None:
        lines.append(
            f"    models:       {paths.models} -> {spec.binds.models}"
        )
    else:
        lines.append("    models:       (no -B models bind)")
    for host, mnt in spec.extra_host_binds:
        lines.append(f"    extra:        {host} -> {mnt}")
    lines.extend(
        [
            f"  outer_container: {paths.container}",
            f"  log_file:        {paths.log_dir / f'{spec.job_name}.log'}",
            f"  err_file:        {paths.err_dir / f'{spec.job_name}.err'}",
        ]
    )
    if spec.extra_env:
        lines.append(f"  extra_env:       {spec.extra_env}")
    qsub_s = " ".join(shlex.quote(str(a)) for a in qsub_argv)
    lines.append(f"  qsub argv:       {qsub_s}")
    sing = singularity_cmd
    if len(sing) > max_singularity_chars:
        sing = sing[:max_singularity_chars] + "\n  ... [singularity command truncated] ..."
    lines.append("  singularity (piped to qsub stdin):")
    for part in sing.split("\n"):
        lines.append(f"    {part}")
    return "\n".join(lines)


def emit_sge_submission_summary_to_terminal(
    spec: StageSpec,
    paths: ClusterPaths,
    *,
    hold_jid: str | Sequence[str] | None,
    qsub_argv: Sequence[str],
    singularity_cmd: str,
) -> None:
    """Print :func:`format_sge_submission_summary` to stderr unless silenced."""
    quiet = os.environ.get("NVITK_QUIET_SGE_SUMMARY", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if quiet:
        return
    print(
        format_sge_submission_summary(
            spec,
            paths,
            hold_jid=hold_jid,
            qsub_argv=qsub_argv,
            singularity_cmd=singularity_cmd,
        ),
        file=sys.stderr,
        flush=True,
    )


_HEADER_ASSIGN_RE = re.compile(
    r"^(?P<key>log_dir|err_dir)=(?P<val>.+)$",
    re.MULTILINE,
)
_FIRST_QSUB_ARRAY_RE = re.compile(
    r"^qsub_\w+=\(\n(?P<body>.*?)^\)\s*$",
    re.MULTILINE | re.DOTALL,
)


def _bash_single_quoted_tokens(qbody: str) -> list[str]:
    tokens: list[str] = []
    for line in qbody.splitlines():
        s = line.strip()
        if len(s) >= 2 and s[0] == "'" and s.endswith("'"):
            tokens.append(s[1:-1])
    return tokens


def format_sge_driver_script_variables(
    script_text: str,
    script_path: Path | None = None,
) -> str:
    """Extract ``log_dir`` / ``err_dir`` / first ``qsub_*=(…)`` flags for terminal echo."""
    lines: list[str] = ["[nvitk|SGE] remote submission (variables)"]
    if script_path is not None:
        lines.append(f"  script_path={script_path}")
    n_stage = len(re.findall(r"^# --- .+ ---$", script_text, re.MULTILINE))
    if n_stage:
        lines.append(f"  num_stages={n_stage}")
    for m in _HEADER_ASSIGN_RE.finditer(script_text):
        raw = m.group("val").strip()
        try:
            (vv,) = shlex.split(raw, posix=True)
        except ValueError:
            vv = raw.strip().strip("'\"")
        lines.append(f"  {m.group('key')}={vv}")
    if n_stage > 1:
        lines.append(
            "  (multi-stage script; per-stage qsub/singularity details are in the file)"
        )
        return "\n".join(lines)
    qm = _FIRST_QSUB_ARRAY_RE.search(script_text)
    if not qm:
        lines.append("  (no qsub_*=(…) block found in script)")
        return "\n".join(lines)
    tokens = _bash_single_quoted_tokens(qm.group("body"))
    flag_to_key = {
        "-P": "qsub_project",
        "-A": "qsub_account",
        "-N": "qsub_job_name",
        "-o": "qsub_stdout",
        "-e": "qsub_stderr",
        "-q": "qsub_queue",
        "-hold_jid": "qsub_hold_jid",
    }
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in flag_to_key and i + 1 < len(tokens):
            lines.append(f"  {flag_to_key[t]}={tokens[i + 1]}")
            i += 2
        elif t == "-l" and i + 1 < len(tokens):
            lines.append(f"  qsub_resource={tokens[i + 1]}")
            i += 2
        elif t == "-terse":
            lines.append("  qsub_terse=true")
            i += 1
        elif t == "qsub":
            i += 1
        else:
            i += 1
    return "\n".join(lines)


_SHELL_VAR_SAFE = re.compile(r"[^A-Za-z0-9_]")


def _shell_var(job_name: str) -> str:
    return _SHELL_VAR_SAFE.sub("_", job_name)


def _quote_qsub_arg(arg: str) -> str:
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
    extra_dirs: Sequence[Path] | None = None,
) -> None:
    """Write the common preamble for an emitted submission script."""
    ts = datetime.now().isoformat(timespec="seconds")
    mkdir_dirs: list[Path] = []
    seen: set[Path] = set()
    for candidate in (log_dir, err_dir, *(extra_dirs or ())):
        path = Path(candidate)
        if path in seen:
            continue
        seen.add(path)
        mkdir_dirs.append(path)
    mkdir_line = " ".join(shlex.quote(str(d)) for d in mkdir_dirs)
    emit.write(
        "#!/usr/bin/env bash\n"
        f"# Auto-generated by nvitk ({title}) on {ts}\n"
        "# Run this on the cluster login node:\n"
        "#     bash <this_file>\n"
        "# Only `bash`, `qsub` and `singularity` are required on the host.\n"
        "set -euo pipefail\n"
        "\n"
        f'log_dir={shlex.quote(str(log_dir))}\n'
        f'err_dir={shlex.quote(str(err_dir))}\n'
        f"mkdir -p {mkdir_line}\n"
        "\n"
    )


def submit_stage(
    spec: StageSpec,
    paths: ClusterPaths,
    *,
    hold_jid: str | Sequence[str] | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
) -> str:
    """Submit *spec* to SGE by piping the Singularity command into ``qsub``."""
    inner = build_singularity_command(spec, paths)
    qsub_argv = build_qsub_command(spec, paths, hold_jid=hold_jid)
    emit_sge_submission_summary_to_terminal(
        spec,
        paths,
        hold_jid=hold_jid,
        qsub_argv=qsub_argv,
        singularity_cmd=inner,
    )

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


def submit_host_stage(
    spec: StageSpec,
    paths: ClusterPaths,
    host_shell_cmd: str,
    *,
    hold_jid: str | Sequence[str] | None = None,
    dry_run: bool = False,
    emit: TextIO | None = None,
) -> str:
    """Submit *host_shell_cmd* to SGE directly (no outer ``singularity exec`` wrapper).

    Use when the job must invoke ``singularity run`` on the cluster host itself
    (e.g. eICAB inference), avoiding nested Singularity inside a pipeline container.
    """
    qsub_argv = build_qsub_command(spec, paths, hold_jid=hold_jid)
    emit_sge_submission_summary_to_terminal(
        spec,
        paths,
        hold_jid=hold_jid,
        qsub_argv=qsub_argv,
        singularity_cmd=host_shell_cmd,
    )

    if emit is not None:
        return _emit_stage_block(emit, spec, host_shell_cmd, qsub_argv, hold_jid)

    if dry_run:
        return "DRY_RUN"

    paths.ensure_dirs()
    result = subprocess.run(
        qsub_argv,
        input=host_shell_cmd,
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
    """Submit a linear chain of *stages* for a single subject."""
    jids: list[str] = []
    prev: str | Sequence[str] | None = base_hold
    for s in stages:
        jid = submit_stage(s, paths, hold_jid=prev, dry_run=dry_run, emit=emit)
        jids.append(jid)
        prev = jid
    return jids


__all__ = [
    "ArrayTaskSpec",
    "ClusterPaths",
    "SgeResources",
    "SingularityBinds",
    "StageSpec",
    "build_array_worker_script",
    "build_qsub_command",
    "build_singularity_command",
    "emit_array_job_block",
    "submit_array_job",
    "python_module_argv",
    "python_script_argv",
    "gui_sge_worker_script_path",
    "gui_sge_worker_argv",
    "sge_virtual_gpu_resource_name",
    "sge_project_uses_virtual_gpu_resource",
    "sge_project_uses_xsgpu_resource",
    "sge_project_omits_ngpu_request",
    "qsub_l_resource_args",
    "emit_sge_submission_summary_to_terminal",
    "format_sge_submission_summary",
    "submit_chain",
    "submit_host_stage",
    "submit_stage",
    "format_sge_driver_script_variables",
    "write_script_header",
]
