"""
PESA-Brain master CLI (``nvitk-pesa-brain``).

Dispatches to sub-pipelines:
- ``black_blood`` → :mod:`nvitk.pipes.pesa_brain.black_blood.run` (or use ``nvitk-pesa-brain-bb``).
- ``g_pet`` → stub (``nvitk-pesa-brain-gpet``).

For black-blood options and stage details, prefer ``nvitk-pesa-brain-bb --help``.
"""

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
    help="Sub-pipeline: black_blood (VWI_BB + eICAB) or g_pet (not implemented).",
)
@click.option("--subjects", default=None, help="Comma-separated subject IDs.")
@click.option(
    "--subjects-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Subject list file for cohort / download.",
)
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--output-root", type=click.Path(path_type=Path), default=None)
@click.option("--eicab-results-root", type=click.Path(path_type=Path), default=None)
@click.option(
    "--qvtpy-results-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Alias for --eicab-results-root.",
)
@click.option("--vwi-bb-rel-path", default=None, help="Relative path to vwi_bb.nii.gz per subject.")
@click.option("--wvi-rel-path", default=None, hidden=True)
@click.option("--eicab-subdir", default=None)
@click.option(
    "--eicab-mask",
    type=click.Choice(["cw", "wb"]),
    default="cw",
    help="eICAB CW or WB multilabel (black_blood only).",
)
@click.option(
    "--stages",
    default="stage0_c,stage1,stage2",
    help="Black-blood stages (see nvitk-pesa-brain-bb --help).",
)
@click.option("--with-download", is_flag=True, default=False)
@click.option("--skip-existing", is_flag=True, default=False)
@click.option("--xnat-config", type=click.Path(path_type=Path), default=None)
@click.option("--server", type=str, default=None)
@click.option("--project", type=str, default=None)
@click.option("--user", type=str, default=None)
@click.option("--password", type=str, default=None)
@click.option("--netrc-file", type=click.Path(path_type=Path), default=None)
@click.option("--report", is_flag=True, default=False)
@click.option("--dof", type=int, default=6, help="FLIRT DOF (stage1).")
@click.option("--cost", default="normmi", help="FLIRT cost (stage1).")
@click.option(
    "--bbox-padding",
    type=int,
    default=8,
    help="Symmetric bbox padding around centerline (stage2).",
)
@click.option(
    "--eicab-prior-dilate",
    type=int,
    default=3,
    help="Dilate warped eICAB label for vessel corridor prior (stage2).",
)
@click.option("--centerline-max-dist", type=int, default=12, help="Centerline tube radius (stage2).")
@click.option(
    "--ica-centerline-max-dist",
    type=int,
    default=6,
    help="Tighter centerline tube for LICA/RICA (stage2).",
)
@click.option(
    "--lumen-intensity-frac",
    type=float,
    default=1.2,
    help="Hypointense lumen ceiling multiplier (non-ICA, stage2).",
)
@click.option(
    "--lumen-percentile",
    type=float,
    default=55.0,
    help="Local tube percentile cap (non-ICA, stage2).",
)
@click.option(
    "--ica-lumen-intensity-frac",
    type=float,
    default=1.05,
    help="Stricter ceiling multiplier for LICA/RICA (stage2).",
)
@click.option(
    "--ica-lumen-percentile",
    type=float,
    default=42.0,
    help="Stricter tube percentile for LICA/RICA (stage2).",
)
@click.option(
    "--min-component-frac",
    type=float,
    default=0.005,
    help="Drop small lumen islands after threshold (stage2).",
)
@click.option(
    "--cl-barrier-radius",
    type=int,
    default=2,
    help="Centerline barrier radius (stage2).",
)
@click.option(
    "--rg-intensity-frac",
    type=float,
    default=1.0,
    help="Hypointense RG intensity fraction after initial mask (stage2).",
)
@click.option(
    "--rg-barrier-radius",
    type=int,
    default=2,
    help="Segmentation barrier radius (stage2).",
)
@click.option(
    "--rg-constraint/--no-rg-constraint",
    default=True,
    help="Constrain RG to bbox and eICAB∩tube (stage2); --no-rg-constraint for free RG.",
)
@click.option(
    "--min-centerline-points",
    type=int,
    default=5,
    help="Min skeleton points per vessel label (stage2).",
)
@click.option(
    "--vwi-preprocess",
    type=click.Choice(["none", "median", "gaussian"]),
    default="median",
    help="vwi_bb smoothing before segmentation (stage2).",
)
@click.option("--vwi-median-size", type=int, default=3, help="Median kernel size (stage2).")
@click.option("--vwi-gaussian-sigma", type=float, default=0.8, help="Gaussian sigma (stage2).")
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
        subjects_file=kwargs.get("subjects_file"),
        dicom_root=kwargs.get("dicom_root"),
        nifti_root=kwargs.get("nifti_root"),
        output_root=kwargs.get("output_root"),
        eicab_results_root=kwargs.get("eicab_results_root"),
        qvtpy_results_root=kwargs.get("qvtpy_results_root"),
        vwi_bb_rel_path=kwargs.get("vwi_bb_rel_path"),
        wvi_rel_path=kwargs.get("wvi_rel_path"),
        eicab_subdir=kwargs.get("eicab_subdir"),
        eicab_mask=kwargs.get("eicab_mask"),
        stages=kwargs.get("stages"),
        with_download=kwargs.get("with_download"),
        skip_existing=kwargs.get("skip_existing"),
        xnat_config=kwargs.get("xnat_config"),
        server=kwargs.get("server"),
        project=kwargs.get("project"),
        user=kwargs.get("user"),
        password=kwargs.get("password"),
        netrc_file=kwargs.get("netrc_file"),
        report=kwargs.get("report"),
        dof=kwargs.get("dof"),
        cost=kwargs.get("cost"),
        bbox_padding=kwargs.get("bbox_padding"),
        eicab_prior_dilate=kwargs.get("eicab_prior_dilate"),
        centerline_max_dist=kwargs.get("centerline_max_dist"),
        ica_centerline_max_dist=kwargs.get("ica_centerline_max_dist"),
        lumen_intensity_frac=kwargs.get("lumen_intensity_frac"),
        lumen_percentile=kwargs.get("lumen_percentile"),
        ica_lumen_intensity_frac=kwargs.get("ica_lumen_intensity_frac"),
        ica_lumen_percentile=kwargs.get("ica_lumen_percentile"),
        min_component_frac=kwargs.get("min_component_frac"),
        cl_barrier_radius=kwargs.get("cl_barrier_radius"),
        rg_intensity_frac=kwargs.get("rg_intensity_frac"),
        rg_barrier_radius=kwargs.get("rg_barrier_radius"),
        rg_constraint=kwargs.get("rg_constraint"),
        min_centerline_points=kwargs.get("min_centerline_points"),
        vwi_preprocess=kwargs.get("vwi_preprocess"),
        vwi_median_size=kwargs.get("vwi_median_size"),
        vwi_gaussian_sigma=kwargs.get("vwi_gaussian_sigma"),
    )


if __name__ == "__main__":
    main()
