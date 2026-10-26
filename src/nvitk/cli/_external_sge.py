"""``--submit sge`` for command groups with many path arguments (``nvitk-fsl``, ``nvitk-fastsurfer``).

The image-tool CLIs take one ``-i`` and one ``-o`` (:func:`nvitk.cli._common.run_sge`).
FSL and FastSurfer commands take several inputs, output basenames, folders; so
:func:`enable_sge` gives every subcommand of a group the same submission options:
``--submit local|sge``, ``--emit-script``, ``--direct-submit``, ``--no-remote``,
``--dry-run``, plus ``--sge-gpu/--sge-cpu``, and with ``--submit sge``:

1. rebuilds the subcommand's command line from the values click parsed;
2. binds every folder a path argument lives in (or the folder itself) into the nvitk
   container at ``/nvitk/ext/<n>``, and rewrites the paths to match: an output's folder
   is created first, so it can be bound;
3. writes the SGE script running ``python -m <module> <subcommand> ...`` in the container,
   and runs it over SSH (``NVITK_SGE_SSH_HOST`` / ``_USER`` / ``_PASSWORD``) unless
   ``--no-remote``, or ``qsub``\\ s it directly with ``--direct-submit`` (on the cluster).

As with every nvitk CLI submission, the paths must be ones the cluster sees too (shared
storage mounted at the same place, or a run from the cluster itself).
"""

from __future__ import annotations

import functools
import os
import shlex
from pathlib import Path
from typing import Any, Callable, Iterable

import click

from nvitk.core.logger import Logger

log = Logger()

#: Parameter names :func:`enable_sge` adds (and strips before the command runs).
SGE_PARAMS = ("submit", "emit_script", "direct_submit", "no_remote", "dry_run", "sge_gpu")

#: Container mount points for the bound folders.
EXT_ROOT = "/nvitk/ext"


class PathBinder:
    """Host folder -> container mount, for the paths of one command line."""

    def __init__(self) -> None:
        self.mounts: dict[Path, str] = {}

    def mount(self, folder: Path) -> str:
        folder = Path(folder).expanduser().absolute()
        if folder not in self.mounts:
            self.mounts[folder] = f"{EXT_ROOT}/{len(self.mounts)}"
        return self.mounts[folder]

    def map(self, value: Any) -> str:
        """The container path for host path *value* (an existing file or folder, or an output)."""
        path = Path(str(value)).expanduser().absolute()
        if path.is_dir():
            return self.mount(path)
        if not path.exists():
            # An output (or an FSL basename): its folder has to exist to be bound.
            path.parent.mkdir(parents=True, exist_ok=True)
        return f"{self.mount(path.parent)}/{path.name}"

    def binds(self) -> tuple[tuple[Path, str], ...]:
        return tuple(self.mounts.items())


def _is_path_type(param: click.Parameter) -> bool:
    return isinstance(param.type, click.Path) or getattr(param.type, "name", "") == "path"


def _render(param: click.Parameter, value: Any, binder: PathBinder) -> str:
    if value is None:
        return ""
    if _is_path_type(param):
        return binder.map(value)
    if param.type is click.UNPROCESSED or getattr(param.type, "name", "") == "text":
        # Free tokens (``nvitk-fsl run ...``, fslmaths operations): bind what is a path here.
        token = str(value)
        if not token.startswith("-") and os.sep in token and Path(token).expanduser().exists():
            return binder.map(token)
        return token
    return str(value)


def rebuild_argv(command: click.Command, values: dict[str, Any], binder: PathBinder) -> list[str]:
    """*command*'s arguments from parsed *values*, paths mapped through *binder*.

    Options first, then the positional arguments (a command with free trailing tokens (
    ``run``, ``maths``) takes them last).
    """
    options: list[str] = []
    arguments: list[str] = []
    for param in command.params:
        if param.name in SGE_PARAMS or param.name not in values:
            continue
        value = values[param.name]
        if isinstance(param, click.Argument):
            if value is None:
                continue
            items = list(value) if param.nargs != 1 else [value]
            arguments.extend(_render(param, v, binder) for v in items)
            continue
        assert isinstance(param, click.Option)
        name = param.opts[0]
        if param.is_flag:
            if param.secondary_opts:
                if value is not None:
                    options.append(name if value else param.secondary_opts[0])
            elif value:
                options.append(name)
            continue
        if value is None or value == ():
            continue
        entries = list(value) if param.multiple else [value]
        for entry in entries:
            parts = list(entry) if (param.nargs and param.nargs > 1) else [entry]
            options.append(name)
            options.extend(_render(param, part, binder) for part in parts)
    return options + arguments


def submit_external(
    *,
    module: str,
    subcommand: str,
    argv: list[str],
    binder: PathBinder,
    gpu: bool,
    title: str,
    emit_script: Path | None,
    direct_submit: bool,
    no_remote: bool,
    dry_run: bool,
) -> None:
    """Write (and run) the SGE script for one containerised ``python -m <module> ...`` call."""
    from nvitk.cli._sge import default_emit_path, emit_submit_script, submit_tool_job

    python_cmd = " ".join(shlex.quote(a) for a in ["python", "-m", module, subcommand, *argv])
    binds = binder.binds()
    if dry_run:
        click.echo(python_cmd)
        for host, mount in binds:
            click.echo(f"  -B {host}:{mount}")
        return
    host_root = binds[0][0] if binds else Path.cwd()
    job_name = f"{module.split('.')[1]}_{subcommand}"[:200]
    if direct_submit and not emit_script:
        submit_tool_job(job_name=job_name, python_cmd=python_cmd, data_root=host_root, output_root=host_root,
                        gpu=gpu, extra_host_binds=binds)
        return
    script = emit_script or default_emit_path(module.split(".")[1], subcommand)
    emit_submit_script(script_path=script, stages=[(job_name, python_cmd)], data_root=host_root,
                       output_root=host_root, gpu=gpu, extra_host_binds=binds, title=title)
    click.echo(f"Wrote SGE script: {script}")
    if direct_submit or no_remote:
        return
    from nvitk.cluster.remote_submit import run_sge_script_ssh
    from nvitk.cluster.remote_transfer import resolve_cluster_host

    host = os.environ.get("NVITK_SGE_SSH_HOST", "").strip()
    user = os.environ.get("NVITK_SGE_SSH_USER", "").strip()
    password = os.environ.get("NVITK_SGE_SSH_PASSWORD", "")
    if host and user and password:
        run_sge_script_ssh(resolve_cluster_host(host), user, password, script)
    else:
        log.warning(
            "Skipping remote SSH (set NVITK_SGE_SSH_HOST, NVITK_SGE_SSH_USER, "
            "NVITK_SGE_SSH_PASSWORD to auto-run on cluster)."
        )


def _sge_options(gpu_default: bool) -> list[click.Option]:
    return [
        click.Option(["--submit"], type=click.Choice(["local", "sge"], case_sensitive=False), default="local",
                     show_default=True, help="Run here, or as an SGE job in the nvitk container."),
        click.Option(["--emit-script", "emit_script"], type=click.Path(path_type=Path), default=None,
                     help="Where to write the SGE script (default: a timestamped staging path)."),
        click.Option(["--direct-submit"], is_flag=True, default=False,
                     help="qsub from this machine (when running on the cluster)."),
        click.Option(["--no-remote"], is_flag=True, default=False,
                     help="Only write the SGE script; do not run it over SSH."),
        click.Option(["--dry-run"], is_flag=True, default=False,
                     help="Print the container command and its binds, and stop."),
        click.Option(["--sge-gpu/--sge-cpu", "sge_gpu"], default=None,
                     help=f"Ask for a GPU node (default: {'yes' if gpu_default else 'no'} for this command)."),
    ]


def enable_sge(
    group: click.Group,
    *,
    module: str,
    title: str,
    gpu_commands: Iterable[str] = (),
    local_only: Iterable[str] = (),
    prepare: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    """Give every subcommand of *group* (but *local_only*) the SGE submission options.

    *prepare(subcommand, values)* may adjust the parsed values before a cluster run: fill in
    what only this machine knows (a license path from the local configuration), drop what
    the container brings itself.
    """
    gpu_set, skip = set(gpu_commands), set(local_only)
    for name, command in group.commands.items():
        if name in skip or command.callback is None:
            continue
        gpu_default = name in gpu_set
        command.params.extend(_sge_options(gpu_default))
        original = command.callback

        def callback(*args: Any, _cmd: click.Command = command, _orig: Callable = original,
                     _gpu: bool = gpu_default, **values: Any) -> Any:
            sge = {key: values.pop(key, None) for key in SGE_PARAMS}
            if str(sge["submit"] or "local").lower() == "local":
                return _orig(*args, **values)
            if prepare is not None:
                prepare(_cmd.name, values)
            binder = PathBinder()
            argv = rebuild_argv(_cmd, values, binder)
            submit_external(
                module=module, subcommand=_cmd.name, argv=argv, binder=binder,
                gpu=_gpu if sge["sge_gpu"] is None else bool(sge["sge_gpu"]), title=title,
                emit_script=sge["emit_script"], direct_submit=bool(sge["direct_submit"]),
                no_remote=bool(sge["no_remote"]), dry_run=bool(sge["dry_run"]),
            )
            return None

        command.callback = functools.update_wrapper(callback, original)


__all__ = ["EXT_ROOT", "PathBinder", "SGE_PARAMS", "enable_sge", "rebuild_argv", "submit_external"]
