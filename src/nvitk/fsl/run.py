"""Running FSL commands: :class:`FslCommand` and the helpers every wrapper shares.

A wrapper returns an :class:`FslCommand` (the argv, the outputs it will write and
whether it is long-running) without running anything. The CLI calls :meth:`run`; the
GUI runs short commands the same way and long ones as background jobs, then opens
:attr:`outputs`. Commands run in :func:`~nvitk.fsl.env.fsl_env` (``FSLDIR``, FSL's
command directory first on ``PATH``, ``FSLOUTPUTTYPE=NIFTI_GZ``).
"""

from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from nvitk.core.logger import Logger

from . import env as fsl_env_mod

log = Logger()


def nii(path: str | Path) -> Path:
    """*path* with the ``.nii.gz`` FSL writes (FSL takes, and reports, basenames)."""
    p = Path(path)
    name = p.name
    for suffix in (".nii.gz", ".nii"):
        if name.endswith(suffix):
            return p
    return p.with_name(name + ".nii.gz")


def base(path: str | Path) -> Path:
    """*path* without a NIfTI suffix (the basename FSL options take)."""
    p = Path(path)
    for suffix in (".nii.gz", ".nii"):
        if p.name.endswith(suffix):
            return p.with_name(p.name[: -len(suffix)])
    return p


def resolve_output(path: Path) -> Path | None:
    """*path*, or its ``.nii`` / ``.nii.gz`` sibling when FSL wrote the other one.

    A ``*`` in the name is a pattern (outputs whose name depends on an option): the
    first match wins.
    """
    path = Path(path)
    if "*" in path.name:
        matches = sorted(path.parent.glob(path.name)) or sorted(path.parent.glob(path.name.replace(".nii.gz", ".nii")))
        return matches[0] if matches else None
    if path.exists():
        return path
    if path.name.endswith(".nii.gz"):
        alt = path.with_name(path.name[:-3])
        return alt if alt.exists() else None
    if path.suffix == ".nii":
        alt = Path(f"{path}.gz")
        return alt if alt.exists() else None
    return None


@dataclass
class FslCommand:
    """One FSL invocation: what runs, what it writes, and how long it is expected to take."""

    argv: list[str]
    title: str
    #: Outputs by role (``brain``, ``mask``, ``seg``, ...); a directory for directory outputs.
    outputs: dict[str, Path] = field(default_factory=dict)
    #: Output roles that are label maps (opened as Labels layers).
    label_outputs: tuple[str, ...] = ()
    #: Minutes to hours: the GUI runs it as a background job.
    long: bool = False
    #: Commands that must exist in the FSL installation.
    requires: tuple[str, ...] = ()
    cwd: Path | None = None
    #: Run through ``bash -c`` (a chain of commands, e.g. TBSS's four steps).
    shell: bool = False
    #: What the command printed, after :meth:`run` (``fslstats``, ``cluster`` report there).
    stdout: str = ""

    def command_line(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)

    def full_argv(self) -> list[str]:
        return ["bash", "-c", self.argv[0]] if self.shell else list(self.argv)

    def environment(self) -> dict[str, str]:
        needed = self.requires or (() if self.shell else (self.argv[0],))
        return fsl_env_mod.fsl_env(fsl_env_mod.require_commands(*needed))

    def existing_outputs(self) -> dict[str, Path]:
        out: dict[str, Path] = {}
        for role, path in self.outputs.items():
            found = resolve_output(path)
            if found is not None:
                out[role] = found
        return out

    def run(self, on_line: Callable[[str], None] | None = None) -> dict[str, Path]:
        """Run in the foreground, streaming output; returns the outputs written. Raises on failure."""
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
        if code != 0:
            tail = "\n".join(lines[-15:])
            raise RuntimeError(f"{self.title} failed (exit code {code}).\n{tail}")
        self.stdout = "\n".join(lines)
        return self.existing_outputs()


def as_args(values: Sequence[object]) -> list[str]:
    return [str(v) for v in values]


def parse_floats(text: str) -> list[float]:
    out: list[float] = []
    for token in text.split():
        try:
            out.append(float(token))
        except ValueError:
            continue
    return out


__all__ = ["FslCommand", "as_args", "base", "nii", "parse_floats", "resolve_output"]
