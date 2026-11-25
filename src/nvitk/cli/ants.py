"""ANTsPy / ANTsPyNet CLI: ``nvitk-ants``.

Registration (``register`` with ``--preset``, ``apply``), image tools (``denoise``,
``histmatch``, ``atropos``, ``thickness``, ``jlf``, ``template``), the ANTsPyNet models
(``antsxnet <tool> --input ...``) and ``tools`` to list them all.
"""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.click_backend import apply_cli_backend
from nvitk.core.logger import Logger
from nvitk.registration.ants import ants_apply, ants_register
from nvitk.registration.ants_extra import SYN_PRESETS
from nvitk.core.click_config import config_dir_click_option

log = Logger()


@click.group("nvitk-ants", context_settings={"help_option_names": ["-h", "--help"]})
@config_dir_click_option()
@click.option(
    "--backend",
    type=click.Choice(["cpu", "gpu"], case_sensitive=False),
    default="gpu",
    show_default=True,
    help="Array backend: cpu (NumPy) or gpu (CuPy).",
)
@click.pass_context
def main(ctx: click.Context, backend: str) -> None:
    """ANTsPy registration helpers."""
    apply_cli_backend(backend)
    ctx.ensure_object(dict)
    ctx.obj["backend"] = backend


@main.command("register")
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out-dir", type=click.Path(path_type=Path), required=True)
@click.option(
    "--type-of-transform",
    "type_of_transform",
    default="SyN",
    show_default=True,
    help="ANTsPy type_of_transform (e.g. Translation, Rigid, Affine, SyN, SyNCC, SyNRA).",
)
@click.option(
    "--preset",
    type=click.Choice(sorted(SYN_PRESETS)),
    default=None,
    help="Named preset (overrides --type-of-transform); see `nvitk-ants tools`.",
)
@click.option("--write-composite-transform", is_flag=True, default=False)
@click.option("--verbose", is_flag=True, default=False)
def register_cmd(
    moving_path: Path,
    fixed_path: Path,
    out_dir: Path,
    type_of_transform: str,
    preset: str | None,
    write_composite_transform: bool,
    verbose: bool,
) -> None:
    """Register MOVING->FIXED using ANTsPy."""
    if preset:
        type_of_transform = SYN_PRESETS[preset][0]
    res = ants_register(
        fixed_path=fixed_path,
        moving_path=moving_path,
        out_dir=out_dir,
        type_of_transform=type_of_transform,
        write_composite_transform=write_composite_transform,
        verbose=verbose,
    )
    log.info(f"ANTs warped: {res.warped_moving_path}")
    log.info(f"ANTs fwd: {list(map(str, res.fwd_transforms))}")


@main.command("apply")
@click.option("--moving", "moving_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--fixed", "fixed_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--transform",
    "transform_paths",
    multiple=True,
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Repeat for each transform file; order matters (ANTs transformlist).",
)
@click.option("--interp", "interpolator", default="linear", show_default=True)
@click.option("--verbose", is_flag=True, default=False)
def apply_cmd(
    moving_path: Path,
    fixed_path: Path,
    out_path: Path,
    transform_paths: tuple[Path, ...],
    interpolator: str,
    verbose: bool,
) -> None:
    """Apply ANTs transforms to map MOVING into FIXED space."""
    outp = ants_apply(
        fixed_path=fixed_path,
        moving_path=moving_path,
        out_path=out_path,
        transforms=list(transform_paths),
        interpolator=interpolator,
        verbose=verbose,
    )
    log.info(f"ANTs output: {outp}")


# Image tools
def _read(path: Path):
    from nvitk.io import imread

    return imread(path)


def _write(path: Path, like, data) -> None:
    from nvitk.io import imsave

    path.parent.mkdir(parents=True, exist_ok=True)
    imsave(path, like.with_data(data))
    log.info(f"wrote {path}")


_IN = click.Path(path_type=Path, exists=True, dir_okay=False)
_OUT = click.Path(path_type=Path, dir_okay=False)


@main.command("tools")
def tools_cmd() -> None:
    """List the ANTs / ANTsPyNet tools, presets and what they need."""
    from nvitk.registration.ants_extra import TOOLS as ANTS_TOOLS
    from nvitk.segmentation.antsxnet_tools import TOOLS as NET_TOOLS

    click.echo("ANTs (CPU, no downloads):")
    for info in ANTS_TOOLS.values():
        click.echo(f"  {info.id:<26} {info.label}: {info.description}  [inputs: {', '.join(info.inputs)}]")
    click.echo("\nANTsPyNet (models download to ~/.keras/ANTsXNet on first use):")
    for _fn, info in NET_TOOLS.values():
        click.echo(f"  {info.id:<26} {info.label}: {info.description}  [inputs: {', '.join(info.inputs)}]")
    click.echo("\nRegistration presets (register --preset):")
    for name, (tot, text) in SYN_PRESETS.items():
        click.echo(f"  {name:<18} {tot:<30} {text}")


@main.command("denoise")
@click.option("--input", "in_path", type=_IN, required=True)
@click.option("--out", "out_path", type=_OUT, required=True)
@click.option("--mask", "mask_path", type=_IN, default=None)
@click.option("--noise-model", type=click.Choice(["Rician", "Gaussian"]), default="Rician", show_default=True)
@click.option("--shrink", type=int, default=1, show_default=True)
def denoise_cmd(in_path: Path, out_path: Path, mask_path: Path | None, noise_model: str, shrink: int) -> None:
    """Non-local-means denoising."""
    from nvitk.registration.ants_extra import denoise

    image = _read(in_path)
    mask = _read(mask_path) if mask_path else None
    _write(out_path, image, denoise(image, mask=mask, noise_model=noise_model, shrink_factor=shrink))


@main.command("histmatch")
@click.option("--input", "in_path", type=_IN, required=True)
@click.option("--reference", "ref_path", type=_IN, required=True)
@click.option("--out", "out_path", type=_OUT, required=True)
@click.option("--bins", type=int, default=255, show_default=True)
@click.option("--points", type=int, default=64, show_default=True)
def histmatch_cmd(in_path: Path, ref_path: Path, out_path: Path, bins: int, points: int) -> None:
    """Match INPUT's histogram to REFERENCE's."""
    from nvitk.registration.ants_extra import histogram_match

    image = _read(in_path)
    _write(out_path, image, histogram_match(image, _read(ref_path), bins=bins, match_points=points))


@main.command("atropos")
@click.option("--input", "in_path", type=_IN, required=True)
@click.option("--mask", "mask_path", type=_IN, required=True)
@click.option("--out", "out_path", type=_OUT, required=True, help="Segmentation; probabilities go beside it.")
@click.option("--classes", type=int, default=3, show_default=True)
@click.option("--mrf", default="[0.2,1x1x1]", show_default=True)
def atropos_cmd(in_path: Path, mask_path: Path, out_path: Path, classes: int, mrf: str) -> None:
    """Atropos n-class segmentation inside a mask."""
    from nvitk.registration.ants_extra import atropos

    image = _read(in_path)
    result = atropos(image, _read(mask_path), n_classes=classes, mrf=mrf)
    _write(out_path, image, result.pop("segmentation").astype("int16"))
    stem = out_path.name.split(".nii")[0]
    for name, data in result.items():
        _write(out_path.with_name(f"{stem}_{name}.nii.gz"), image, data)


@main.command("thickness")
@click.option("--segmentation", "seg_path", type=_IN, required=True)
@click.option("--gm", "gm_path", type=_IN, required=True, help="Gray-matter probability.")
@click.option("--wm", "wm_path", type=_IN, required=True, help="White-matter probability.")
@click.option("--out", "out_path", type=_OUT, required=True)
@click.option("--gm-label", type=int, default=2, show_default=True)
@click.option("--wm-label", type=int, default=3, show_default=True)
@click.option("--iterations", type=int, default=45, show_default=True)
def thickness_cmd(seg_path: Path, gm_path: Path, wm_path: Path, out_path: Path, gm_label: int, wm_label: int, iterations: int) -> None:
    """DiReCT (KellyKapowski) cortical thickness."""
    from nvitk.registration.ants_extra import cortical_thickness_direct

    seg = _read(seg_path)
    out = cortical_thickness_direct(seg, _read(gm_path), _read(wm_path), iterations=iterations, gm_label=gm_label, wm_label=wm_label)
    _write(out_path, seg, out)


@main.command("jlf")
@click.option("--target", "target_path", type=_IN, required=True)
@click.option("--mask", "mask_path", type=_IN, required=True)
@click.option("--atlas", "atlas_paths", type=_IN, multiple=True, required=True, help="Registered atlas image (repeat).")
@click.option("--labels", "label_paths", type=_IN, multiple=True, required=True, help="Its label map (repeat, same order).")
@click.option("--out", "out_path", type=_OUT, required=True)
@click.option("--beta", type=float, default=4.0, show_default=True)
def jlf_cmd(target_path: Path, mask_path: Path, atlas_paths: tuple[Path, ...], label_paths: tuple[Path, ...], out_path: Path, beta: float) -> None:
    """Joint label fusion of atlases registered to TARGET."""
    from nvitk.registration.ants_extra import joint_label_fusion

    target = _read(target_path)
    result = joint_label_fusion(target, _read(mask_path), [_read(p) for p in atlas_paths], [_read(p) for p in label_paths], beta=beta)
    _write(out_path, target, result["segmentation"].astype("int32"))


@main.command("template")
@click.option("--image", "image_paths", type=_IN, multiple=True, required=True, help="Repeat for each image.")
@click.option("--out", "out_path", type=_OUT, required=True)
@click.option("--iterations", type=int, default=3, show_default=True)
@click.option("--type-of-transform", default="SyN", show_default=True)
def template_cmd(image_paths: tuple[Path, ...], out_path: Path, iterations: int, type_of_transform: str) -> None:
    """Build a population template from several images."""
    import nibabel as nib

    from nvitk.registration.ants_extra import build_template

    result = build_template([_read(p) for p in image_paths], iterations=iterations, type_of_transform=type_of_transform)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result.data.astype("float32"), result.affine_ras()), str(out_path))
    log.info(f"wrote {out_path}")


@main.command("antsxnet")
@click.argument("tool")
@click.option("--input", "in_paths", type=_IN, multiple=True, required=True, help="Inputs in the tool's order (see `tools`).")
@click.option("--out", "out_path", type=_OUT, required=True, help="Main output; other outputs go beside it.")
def antsxnet_cmd(tool: str, in_paths: tuple[Path, ...], out_path: Path) -> None:
    """Run an ANTsPyNet model: TOOL is an id from `nvitk-ants tools` (e.g. antsxnet_deep_atropos)."""
    import json

    from nvitk.segmentation.antsxnet_tools import TOOLS

    key = tool if tool in TOOLS else f"antsxnet_{tool}"
    if key not in TOOLS:
        raise click.ClickException(f"Unknown tool {tool!r}; see `nvitk-ants tools`.")
    fn, _info = TOOLS[key]
    images = [_read(p) for p in in_paths]
    result = fn(*images)
    stem = out_path.name.split(".nii")[0]
    primary = result.primary
    if primary is not None:
        _write(out_path, images[0], primary)
    for name, data in result.arrays.items():
        if data is primary:
            continue
        _write(out_path.with_name(f"{stem}_{name}.nii.gz"), images[0], data)
    if result.scalars or result.labels:
        side = out_path.with_name(f"{stem}.json")
        side.write_text(json.dumps({"scalars": result.scalars, "labels": result.labels}, indent=2), encoding="utf-8")
        log.info(f"wrote {side}")
    for name, value in result.scalars.items():
        click.echo(f"{name}: {value:.3f}")


__all__ = ["main"]

