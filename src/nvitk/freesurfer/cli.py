"""``nvitk-freesurfer``: FreeSurfer's topology tools on any mask, on the installation nvitk finds.

Set-up::

    nvitk-freesurfer info                          # installation, version, license, programs
    nvitk-freesurfer use --home /opt/freesurfer --license ~/license.txt

Volumes (NIfTI or MGZ in, MGZ / NIfTI out by suffix): ``pretess``, ``mc`` (marching cubes →
FreeSurfer surface). Surfaces: ``main-component``, ``euler``, ``smooth``, ``inflate``, ``sphere``
(``mris_sphere -q``, or ``--spectral`` for FastSurfer's projection). The whole chain on a mask:
``fix-topology`` (``mris_fix_topology`` or ``--fixer mris_topo_fixer``; defect segmentation)::

    nvitk-freesurfer fix-topology ica_mask.nii.gz --intensity TOF.nii.gz --label 1 -o fix_run/

FreeSurfer programs need a license (free, https://surfer.nmr.mgh.harvard.edu/registration.html).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from nvitk.core.click_config import config_dir_click_option

from . import env
from . import topology as topo
from .run import FsCommand

_PATH = click.Path(path_type=Path)
_IN = click.Path(path_type=Path, exists=True)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@config_dir_click_option()
def main() -> None:
    """FreeSurfer topology tools (pretess, marching cubes, Euler number, spheres, topology fixers)."""


def _run(cmd: FsCommand, license_file: Path | None = None) -> FsCommand:
    if license_file is not None:
        cmd.license_file = license_file
    try:
        outputs = cmd.run()
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(str(exc)) from exc
    for role, path in outputs.items():
        click.echo(f"{role:12s} {path}")
    return cmd


_LICENSE = click.option("--license", "license_file", type=_IN, default=None,
                        help="FreeSurfer license file (default: $FS_LICENSE, external_tools.json, the installation).")


@main.command("info")
@click.option("--json", "as_json", is_flag=True)
def info_cmd(as_json: bool) -> None:
    """Which FreeSurfer nvitk uses, its version, its license and the topology programs present."""
    report = env.report()
    if as_json:
        click.echo(json.dumps(report, indent=2))
        return
    for key, value in report.items():
        click.echo(f"{key:18s} {json.dumps(value) if isinstance(value, dict) else value}")
    if report.get("freesurfer_home") and not report.get("license"):
        click.echo(f"\nNo license found: get one (free) at {env.LICENSE_URL}.")


@main.command("use")
@click.option("--home", type=_IN, default=None, help="FreeSurfer installation (FREESURFER_HOME).")
@click.option("--license", "license_file", type=_IN, default=None, help="License file.")
@click.option("--clear", is_flag=True, help="Forget both.")
def use_cmd(home: Path | None, license_file: Path | None, clear: bool) -> None:
    """Remember the installation / license (external_tools.json)."""
    from nvitk.core import external_tools

    if clear:
        click.echo(f"Cleared in {external_tools.remember(freesurfer_home=None, fs_license=None)}")
        return
    values: dict[str, Any] = {}
    if home is not None:
        values["freesurfer_home"] = str(home.resolve())
    if license_file is not None:
        values["fs_license"] = str(license_file.resolve())
    if not values:
        raise click.UsageError("Give --home and/or --license, or --clear.")
    click.echo(f"Saved in {external_tools.remember(**values)}")


def _as_mgz(path: Path, work: Path, *, label: int | None = None, value: int | None = None) -> Path:
    """*path* as an MGZ FreeSurfer reads, with *label* (0 = every nonzero voxel) set to *value*."""
    import nibabel as nib
    import numpy as np

    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    if label is not None:
        m = data != 0 if int(label) == 0 else data == int(label)
        data = m.astype(np.uint8) * int(value if value is not None else 1)
    return topo.write_volume(data, img.affine, work / (path.name.split(".")[0] + ".mgz"),
                             dtype=np.uint8 if label is not None else np.float32)


@main.command("pretess")
@click.argument("mask", type=_IN)
@click.option("--label", type=int, default=1, show_default=True, help="Label to repair (0 = every nonzero voxel).")
@click.option("--intensity", type=_IN, default=None, help="Image that decides which voxels to add (default: the mask).")
@click.option("-o", "--output", type=_PATH, required=True, help="Output volume (.mgz / .nii.gz).")
@_LICENSE
def pretess_cmd(mask: Path, label: int, intensity: Path | None, output: Path, license_file: Path | None) -> None:
    """mri_pretess: make *label*'s edge / corner-only contacts face contacts."""
    work = output.parent
    work.mkdir(parents=True, exist_ok=True)
    m = _as_mgz(mask, work, label=label, value=1)
    norm = _as_mgz(intensity, work) if intensity else m
    _run(topo.pretess(m, 1, norm, output), license_file)


@main.command("mc")
@click.argument("mask", type=_IN)
@click.option("--label", type=int, default=1, show_default=True, help="Label to tessellate (0 = every nonzero voxel).")
@click.option("--connectivity", type=click.IntRange(1, 4), default=1, show_default=True,
              help="1 = 6+, 2 = 18, 3 = 6, 4 = 26.")
@click.option("-o", "--output", type=_PATH, required=True, help="Output FreeSurfer surface.")
@_LICENSE
def mc_cmd(mask: Path, label: int, connectivity: int, output: Path, license_file: Path | None) -> None:
    """mri_mc: marching-cubes surface of a label."""
    output.parent.mkdir(parents=True, exist_ok=True)
    m = _as_mgz(mask, output.parent, label=label, value=1)
    _run(topo.marching_cubes(m, 1, output, connectivity=connectivity), license_file)


@main.command("main-component")
@click.argument("surface", type=_IN)
@click.option("-o", "--output", type=_PATH, required=True)
@_LICENSE
def main_component_cmd(surface: Path, output: Path, license_file: Path | None) -> None:
    """mris_extract_main_component: the largest connected piece of a surface."""
    _run(topo.main_component(surface, output), license_file)


@main.command("euler")
@click.argument("surface", type=_IN)
@click.option("--no-freesurfer", is_flag=True, help="Only nvitk's own count (no FreeSurfer, no license).")
@_LICENSE
def euler_cmd(surface: Path, no_freesurfer: bool, license_file: Path | None) -> None:
    """Euler number, holes (handles) and defect index of a surface."""
    v, f = topo.read_surface(surface)
    own = topo.surface_euler(v, f)
    click.echo(f"nvitk:      euler {own['euler']}, pieces {own['pieces']}, genus of the largest {own['genus_main']}")
    if not no_freesurfer:
        cmd = topo.euler_number(surface)
        if license_file is not None:
            cmd.license_file = license_file
        try:
            cmd.run(on_line=lambda _line: None)
        except Exception as exc:  # noqa: BLE001
            raise click.ClickException(str(exc)) from exc
        click.echo(f"FreeSurfer: {topo.parse_euler(cmd.stdout)}")


@main.command("smooth")
@click.argument("surface", type=_IN)
@click.option("-o", "--output", type=_PATH, required=True)
@click.option("-n", "--iterations", type=int, default=10, show_default=True)
@_LICENSE
def smooth_cmd(surface: Path, output: Path, iterations: int, license_file: Path | None) -> None:
    """mris_smooth -nw."""
    _run(topo.smooth(surface, output, iterations=iterations), license_file)


@main.command("inflate")
@click.argument("surface", type=_IN)
@click.option("-o", "--output", type=_PATH, required=True)
@_LICENSE
def inflate_cmd(surface: Path, output: Path, license_file: Path | None) -> None:
    """mris_inflate -no-save-sulc."""
    _run(topo.inflate(surface, output), license_file)


@main.command("sphere")
@click.argument("surface", type=_IN)
@click.option("-o", "--output", type=_PATH, required=True)
@click.option("--spectral", is_flag=True,
              help="FastSurfer's spectral projection (input: smoothwm; needs FastSurfer + lapy) instead of "
                   "mris_sphere -q (input: inflated).")
@_LICENSE
def sphere_cmd(surface: Path, output: Path, spectral: bool, license_file: Path | None) -> None:
    """A spherical map of a genus-0 surface."""
    _run(topo.spectral_sphere(surface, output) if spectral else topo.quick_sphere(surface, output), license_file)


@main.command("fix-topology")
@click.argument("mask", type=_IN)
@click.option("--intensity", type=_IN, default=None, help="The image the mask came from (TOF, T1) for the fixer's likelihood.")
@click.option("--label", type=int, default=1, show_default=True, help="Label to fix (0 = every nonzero voxel).")
@click.option("-o", "--output", "subjects_dir", type=_PATH, required=True, help="Subjects folder to build the run in.")
@click.option("--subject", default="nvitk_topology", show_default=True)
@click.option("--fixer", type=click.Choice(topo.FIXERS), default="mris_fix_topology", show_default=True)
@click.option("--sphere", type=click.Choice(topo.SPHERES), default="qsphere", show_default=True)
@click.option("--no-ga", is_flag=True, help="mris_fix_topology without the genetic search (-ga).")
@click.option("--no-defect-seg", is_flag=True, help="Skip defect-seg (the defects as a volume).")
@_LICENSE
def fix_topology_cmd(mask: Path, intensity: Path | None, label: int, subjects_dir: Path, subject: str, fixer: str,
                     sphere: str, no_ga: bool, no_defect_seg: bool, license_file: Path | None) -> None:
    """Tessellate a mask and make its surface genus 0 with FreeSurfer's topology fixer (recon-all order)."""
    import nibabel as nib
    import numpy as np

    img = nib.load(str(mask))
    data = np.asarray(img.dataobj)
    m = data != 0 if label == 0 else data == label
    inten = np.asarray(nib.load(str(intensity)).dataobj) if intensity else None
    topo.prepare_subject(m, img.affine, subjects_dir, subject, intensity=inten)
    cmd = topo.fix_topology(subjects_dir, subject, fixer=fixer, sphere=sphere, ga=not no_ga,
                            defect_seg=not no_defect_seg, license_file=license_file)
    _run(cmd)
    click.echo(json.dumps(topo.topology_report(subjects_dir, subject, fixer=fixer), indent=2))


if __name__ == "__main__":
    main()
