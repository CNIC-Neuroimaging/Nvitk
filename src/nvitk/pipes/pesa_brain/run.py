"""PESA-Brain master CLI: dispatch black_blood or g_pet sub-pipelines."""

from __future__ import annotations

from pathlib import Path

import click

from nvitk.core.logger import Logger

log = Logger()


@click.command("nvitk-pesa-brain")
@click.option(
    "--pipeline",
    type=click.Choice(["black_blood", "g_pet"]),
    required=True,
)
@click.option("--subjects", default=None)
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--output-root", type=click.Path(path_type=Path), default=None)
@click.option("--eicab-results-root", type=click.Path(path_type=Path), default=None)
@click.option("--qvtpy-results-root", type=click.Path(path_type=Path), default=None)
@click.option("--vwi-bb-rel-path", default=None)
@click.option("--wvi-rel-path", default=None, hidden=True)
@click.option("--eicab-subdir", default=None)
@click.option("--stages", default="stage0_c,stage1,stage2")
@click.option("--with-download", is_flag=True, default=False)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--xnat-config", type=click.Path(path_type=Path), default=None)
@click.option("--server", type=str, default=None)
@click.option("--project", type=str, default=None)
@click.option("--user", type=str, default=None)
@click.option("--password", type=str, default=None)
@click.option("--dof", type=int, default=6)
@click.option("--cost", default="normmi")
@click.option(
    "--seg-strategy",
    type=click.Choice(["crop-resegment", "centerline-growth"]),
    default="crop-resegment",
)
@click.option(
    "--thr-algorithm",
    type=click.Choice(["otsu", "lsthr", "lthr"]),
    default="otsu",
)
@click.option("--crop-padding-bbox", type=int, default=3)
@click.option("--cl-barrier-radius", type=int, default=2)
@click.option("--min-component-frac", type=float, default=0.005)
@click.option("--rg-intensity-frac", type=float, default=0.45)
@click.option("--rg-barrier-radius", type=int, default=2)
@click.pass_context
def main(ctx: click.Context, pipeline: str, **kwargs: object) -> None:
    """Run a PESA-Brain sub-pipeline."""
    if pipeline == "g_pet":
        from nvitk.pipes.pesa_brain.g_pet import run as g_pet_run

        try:
            ctx.invoke(g_pet_run.main)
        except NotImplementedError:
            raise SystemExit(1) from None
        return

    from nvitk.pipes.pesa_brain.black_blood import run as bb_run

    ctx.invoke(
        bb_run.main,
        subjects=kwargs.get("subjects"),
        dicom_root=kwargs.get("dicom_root"),
        nifti_root=kwargs.get("nifti_root"),
        output_root=kwargs.get("output_root"),
        eicab_results_root=kwargs.get("eicab_results_root"),
        qvtpy_results_root=kwargs.get("qvtpy_results_root"),
        vwi_bb_rel_path=kwargs.get("vwi_bb_rel_path"),
        wvi_rel_path=kwargs.get("wvi_rel_path"),
        eicab_subdir=kwargs.get("eicab_subdir"),
        stages=kwargs.get("stages"),
        with_download=kwargs.get("with_download"),
        skip_existing=kwargs.get("skip_existing"),
        xnat_config=kwargs.get("xnat_config"),
        server=kwargs.get("server"),
        project=kwargs.get("project"),
        user=kwargs.get("user"),
        password=kwargs.get("password"),
        dof=kwargs.get("dof"),
        cost=kwargs.get("cost"),
        seg_strategy=kwargs.get("seg_strategy"),
        thr_algorithm=kwargs.get("thr_algorithm"),
        crop_padding_bbox=kwargs.get("crop_padding_bbox"),
        cl_barrier_radius=kwargs.get("cl_barrier_radius"),
        min_component_frac=kwargs.get("min_component_frac"),
        rg_intensity_frac=kwargs.get("rg_intensity_frac"),
        rg_barrier_radius=kwargs.get("rg_barrier_radius"),
    )


if __name__ == "__main__":
    main()
