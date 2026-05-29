"""Headless container worker for GUI SGE jobs."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import click

from nvitk.core.logger import Logger
from nvitk.io import imread, imsave
from nvitk.types import Image

DONE_FILE = ".done"


def _write_done_marker(
    output_dir: Path,
    *,
    job_id: str,
    exit_code: int,
    output_files: list[str],
    error = None,
) -> None:
    payload = {
        "job_id": job_id,
        "exit_code": int(exit_code),
        "output_files": list(output_files),
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "error": error,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / DONE_FILE).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


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

    from nvitk.gui.sge.job import GuiSgeJob
    from nvitk.gui.tools.runner import run_gui_tool_headless

    job = GuiSgeJob.from_dict(spec)
    exit_code = 0
    err_msg = None
    output_files = []

    try:
        primary = imread(data_dir / job.input_name, backend="numpy")

        aux = {}
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
        output_files = [job.output_name]
        log.info("Wrote %s", out_path)
    except Exception as exc:
        exit_code = 1
        err_msg = str(exc) or exc.__class__.__name__
        log.error("SGE worker failed: %s", err_msg)
        traceback.print_exc()
    finally:
        _write_done_marker(
            output_dir,
            job_id=job.job_id,
            exit_code=exit_code,
            output_files=output_files,
            error=err_msg,
        )

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
