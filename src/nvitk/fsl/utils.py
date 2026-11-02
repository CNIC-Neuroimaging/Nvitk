"""FSL utilities: fslmaths, fslstats, fslmeants, cluster, smoothest, and any other FSL command.
Each function returns an :class:`~nvitk.fsl.run.FslCommand`; the ``parse_*`` helpers read what
the text-producing ones printed."""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Sequence

from .run import FslCommand, nii, parse_floats

#: fslstats options offered by default, and what each prints.
STATS_OPTIONS: dict[str, tuple[str, ...]] = {
    "-R": ("min", "max"),
    "-r": ("robust min", "robust max"),
    "-M": ("mean (nonzero)",),
    "-S": ("std (nonzero)",),
    "-V": ("voxels (nonzero)", "volume mm³ (nonzero)"),
    "-E": ("entropy",),
    "-C": ("centre of gravity x mm", "y mm", "z mm"),
}


def maths(input: Path, operations: str | Sequence[str], output: Path, *, odt: str | None = None) -> FslCommand:
    """``fslmaths <input> <operations> <output>`` (e.g. ``"-thr 100 -bin"``, ``"-s 2"``)."""
    ops = shlex.split(operations) if isinstance(operations, str) else [str(o) for o in operations]
    if not ops:
        raise ValueError("Give fslmaths at least one operation.")
    argv = ["fslmaths", str(input), *ops, str(nii(output))]
    if odt:
        argv += ["-odt", odt]
    return FslCommand(argv, "fslmaths", {"image": nii(output)})


def stats(input: Path, options: Sequence[str] = ("-R", "-r", "-M", "-S", "-V"), *, mask: Path | None = None) -> FslCommand:
    """``fslstats`` (``-k mask`` restricts every following option to the mask)."""
    argv = ["fslstats", str(input)]
    if mask is not None:
        argv += ["-k", str(mask)]
    argv += list(options)
    return FslCommand(argv, "fslstats")


def parse_stats(stdout: str, options: Sequence[str]) -> dict[str, float]:
    """``{name: value}`` from fslstats output for *options* (in order)."""
    values = parse_floats(stdout.strip().splitlines()[-1] if stdout.strip() else "")
    names: list[str] = []
    for option in options:
        names += list(STATS_OPTIONS.get(option, (option,)))
    return {name: value for name, value in zip(names, values)}


def meants(input: Path, output: Path, *, mask: Path | None = None, coord: Sequence[float] | None = None,
           mm: bool = False) -> FslCommand:
    """Mean time series of a 4D image inside *mask*, or at voxel/mm *coord*."""
    argv = ["fslmeants", "-i", str(input), "-o", str(output)]
    if mask is not None:
        argv += ["-m", str(mask)]
    elif coord is not None:
        argv += ["-c", *[f"{float(v):g}" for v in coord]]
        if mm:
            argv.append("--usemm")
    else:
        raise ValueError("Give a mask or a coordinate.")
    return FslCommand(argv, "fslmeants", {"timeseries": Path(output)})


def cluster(
    input: Path,
    threshold: float,
    out_base: Path,
    *,
    connectivity: int = 26,
    min_extent: int = 0,
    mm: bool = True,
    minima: bool = False,
) -> FslCommand:
    """Connected clusters above *threshold*: a size-ordered index map and a table (stdout)."""
    base_name = str(out_base)
    argv = ["cluster", f"--in={input}", f"--thresh={threshold:g}", f"--oindex={nii(base_name + '_index')}",
            f"--osize={nii(base_name + '_size')}", f"--olmax={base_name}_lmax.txt",
            f"--connectivity={int(connectivity)}"]
    if min_extent:
        argv.append(f"--minextent={int(min_extent)}")
    if mm:
        argv.append("--mm")
    if minima:
        argv.append("--min")
    return FslCommand(argv, "cluster", {"index": nii(base_name + "_index"), "size": nii(base_name + "_size"),
                                         "lmax": Path(f"{base_name}_lmax.txt")}, label_outputs=("index",))


def parse_cluster_table(stdout: str) -> dict[str, dict[str, Any]]:
    """``{"cluster <n>": {column: value}}`` from cluster's printed table."""
    lines = [line for line in stdout.splitlines() if line.strip()]
    header_index = next((i for i, line in enumerate(lines) if line.startswith("Cluster Index")), None)
    if header_index is None:
        return {}
    columns = lines[header_index].split("\t")
    out: dict[str, dict[str, Any]] = {}
    for line in lines[header_index + 1:]:
        values = line.split("\t")
        if len(values) < 2:
            continue
        row: dict[str, Any] = {}
        for column, value in zip(columns[1:], values[1:]):
            try:
                row[column.strip()] = float(value)
            except ValueError:
                row[column.strip()] = value.strip()
        out[f"cluster {values[0].strip()}"] = row
    return out


def smoothest(*, zstat: Path | None = None, residuals: Path | None = None, dof: int | None = None,
              mask: Path) -> FslCommand:
    """Smoothness (DLH, VOLUME, RESELS, FWHM) of a z-stat or of residuals (with their *dof*)."""
    argv = ["smoothest", "-m", str(mask)]
    if zstat is not None:
        argv += ["-z", str(zstat)]
    elif residuals is not None and dof:
        argv += ["-r", str(residuals), "-d", str(int(dof))]
    else:
        raise ValueError("Give zstat, or residuals with dof.")
    return FslCommand(argv, "smoothest")


def parse_smoothest(stdout: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in ("DLH", "VOLUME", "RESELS"):
            out[parts[0]] = float(parts[1])
        elif len(parts) >= 4 and parts[0] == "FWHMmm":
            out.update({"FWHM x mm": float(parts[1]), "FWHM y mm": float(parts[2]), "FWHM z mm": float(parts[3])})
    return out


def command(command_line: str | Sequence[str], *, title: str = "FSL command", long: bool = False,
            outputs: dict[str, Path] | None = None) -> FslCommand:
    """Any FSL command, run in the FSL environment (the escape hatch for FEAT and everything else)."""
    argv = shlex.split(command_line) if isinstance(command_line, str) else [str(a) for a in command_line]
    if not argv:
        raise ValueError("Empty command.")
    return FslCommand(argv, title, dict(outputs or {}), long=long)


__all__ = [
    "STATS_OPTIONS",
    "cluster",
    "command",
    "maths",
    "meants",
    "parse_cluster_table",
    "parse_smoothest",
    "parse_stats",
    "smoothest",
    "stats",
]
