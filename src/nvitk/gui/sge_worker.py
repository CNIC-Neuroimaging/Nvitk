"""Headless container worker for GUI SGE jobs."""

from __future__ import annotations

import json
from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.types import Image


@click.command("sge-worker")
@click.option(
    "--job",
    "job_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to job.json inside the bind-mounted data directory.",
)
def main(job_path: Path) -> None:
    """Run a staged GUI tool inside Singularity (no Napari)."""
    log = Logger()
    spec = json.loads(job_path.read_text(encoding="utf-8"))
    data_dir = job_path.parent
    output_dir = Path("/nvitk/output")

    from nvitk.gui.sge_job import GuiSgeJob
    from nvitk.gui.tool_runner import run_gui_tool_headless

    job = GuiSgeJob.from_dict(spec)
    primary = imread(data_dir / job.input_name, backend="numpy")

    aux: dict[str, Image] = {}
    for aux_spec in job.aux_layers:
        aux[aux_spec.layer_name] = imread(data_dir / aux_spec.file, backend="numpy")

    log.info("Running GUI tool %s (job %s)", job.tool_id, job.job_id)
    out_data = run_gui_tool_headless(
        job.tool_id,
        primary=primary,
        aux=aux,
        target_mode=job.target_mode,
        label_ids=job.label_ids or None,
        params=job.params,
    )
    if out_data is None:
        raise click.ClickException(f"Tool {job.tool_id!r} produced no output array.")

    meta = dict(primary.metadata or {})
    meta.pop("source", None)
    out_img = Image(data=out_data, metadata=meta, axes=primary.axes, name=job.output_name)
    out_path = output_dir / job.output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imsave(out_path, out_img)
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
