"""Running FreeSurfer programs: :class:`FsCommand`.

A wrapper returns an :class:`FsCommand` (the argv, the outputs it will write and whether it is
long-running) without running anything, like :class:`nvitk.fsl.run.FslCommand`. The CLI calls
:meth:`FsCommand.run`; the GUI runs short commands the same way and long ones as background jobs.
Commands run in :func:`~nvitk.freesurfer.env.fs_env`. A license refusal becomes a
:class:`~nvitk.freesurfer.env.FreeSurferLicenseError` that says where to get one and where to put
it.
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from nvitk.core.logger import Logger

from . import env as fs_env_mod

log = Logger()


@dataclass
class FsCommand:
    """One FreeSurfer invocation (or a ``bash`` chain of them)."""

    argv: list[str]
    title: str
    #: Outputs by role (``surface``, ``volume``, ``defects`` ...).
    outputs: dict[str, Path] = field(default_factory=dict)
    #: Output roles that are label maps (opened as Labels layers).
    label_outputs: tuple[str, ...] = ()
    #: Output roles that are FreeSurfer surfaces (opened as Surface layers).
    surface_outputs: tuple[str, ...] = ()
    long: bool = False
    #: Programs that must exist in the installation (``None``: the argv's program; ``()``: none,
    #: for a Python step or a shell helper).
    requires: tuple[str, ...] | None = None
    cwd: Path | None = None
    subjects_dir: Path | None = None
    license_file: Path | None = None
    #: Run through ``bash -c`` (a chain of commands).
    shell: bool = False
    stdout: str = ""

    def command_line(self) -> str:
        return self.argv[0] if self.shell else " ".join(shlex.quote(a) for a in self.argv)

    def full_argv(self) -> list[str]:
        return ["bash", "-c", self.argv[0]] if self.shell else list(self.argv)

    def environment(self) -> dict[str, str]:
        needed = self.requires if self.requires is not None else (() if self.shell else (self.argv[0],))
        home = fs_env_mod.freesurfer_home()
        if home is None:
            if not needed:  # a Python step (FastSurfer's spectral sphere): no FreeSurfer needed
                import os

                return dict(os.environ)
            raise fs_env_mod.FreeSurferNotFound(fs_env_mod.not_found_message())
        absent = fs_env_mod.missing([n for n in needed if "/" not in n], home)
        if absent:
            raise fs_env_mod.FreeSurferNotFound(
                f"{self.title}: FreeSurfer at {home} has no {', '.join(absent)}."
            )
        return fs_env_mod.fs_env(home, subjects_dir=self.subjects_dir, license_file=self.license_file)

    def existing_outputs(self) -> dict[str, Path]:
        return {role: Path(p) for role, p in self.outputs.items() if Path(p).exists()}

    def run(self, on_line: Callable[[str], None] | None = None) -> dict[str, Path]:
        """Run in the foreground, streaming output; returns the outputs written. Raises
        :class:`~nvitk.freesurfer.env.FreeSurferLicenseError` on a license refusal and
        ``RuntimeError`` on any other failure."""
        environment = self.environment()
        for path in self.outputs.values():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        log.info(f"$ {self.command_line()}")
        proc = subprocess.Popen(
            self.full_argv(), env=environment, cwd=str(self.cwd) if self.cwd else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        lines: list[str] = []
        for raw in proc.stdout:
            text = raw.rstrip("\n")
            lines.append(text)
            (on_line or (lambda t: print(t, flush=True)))(text)
        code = proc.wait()
        self.stdout = "\n".join(lines)
        problem = fs_env_mod.license_problem(self.stdout)
        if problem:
            raise fs_env_mod.FreeSurferLicenseError(problem)
        if code != 0:
            tail = "\n".join(lines[-15:])
            raise RuntimeError(f"{self.title} failed (exit code {code}).\n{tail}")
        return self.existing_outputs()


def chain(commands: list[FsCommand], *, title: str, long: bool = True, cwd: Path | None = None) -> FsCommand:
    """One ``bash -c`` command running *commands* in order (``set -e``), with their outputs."""
    script = "set -e\n" + "\n".join(c.command_line() for c in commands)
    outputs: dict[str, Path] = {}
    requires: tuple[str, ...] = ()
    for c in commands:
        outputs.update(c.outputs)
        own = c.requires if c.requires is not None else (c.argv[0],)
        requires += tuple(r for r in own if "/" not in r and r not in requires)
    first = commands[0] if commands else None
    return FsCommand(
        [script], title, outputs=outputs, long=long, requires=requires, cwd=cwd, shell=True,
        label_outputs=tuple(r for c in commands for r in c.label_outputs),
        surface_outputs=tuple(r for c in commands for r in c.surface_outputs),
        subjects_dir=first.subjects_dir if first else None,
        license_file=first.license_file if first else None,
    )


__all__ = ["FsCommand", "chain"]
