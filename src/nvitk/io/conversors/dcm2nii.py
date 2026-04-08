from __future__ import annotations

from pathlib import Path

try:
    import click
except Exception:
    click = None


def _cli_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator

from nvitk.core.exceptions import BackendUnavailableError
from ._dicom_conversion import run_dicom2nifti

__all__ = [
    "dcm2nii",
]


def _is_nifti_file_path(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def dcm2nii(
    input_path: str,
    output_folder: str,
    *,
    custom_naming: str | None = None,
    force_ras: bool = False,
    process_rtstruct: bool = False,
    revert_scaling: bool = False,
    save_metadata: bool = False,
    additional_tags: list[str] | None = None,
    compress: bool = False,
    rescale_type: str = "DV",
    series_number: str | None = None,
    series_index: int | None = None,
    include_private_tags: bool = False,
    skip_existing: bool = False,
    tmp_dir: Path | None = None,
) -> str | list[str]:
    output_path = Path(output_folder)
    explicit_output_path = str(output_path) if _is_nifti_file_path(output_path) else None
    output_root = str(output_path.parent if explicit_output_path else output_path)

    outputs = run_dicom2nifti(
        input_path,
        output_root,
        custom_naming=custom_naming,
        force_ras=force_ras,
        process_rtstruct=process_rtstruct,
        revert_scaling=revert_scaling,
        save_metadata=save_metadata,
        additional_tags=additional_tags,
        compress=compress,
        rescale_type=rescale_type,
        series_number=series_number,
        series_index=series_index,
        include_private_tags=include_private_tags,
        skip_existing=skip_existing,
        tmp_dir=tmp_dir,
        explicit_output_path=explicit_output_path,
    )
    if explicit_output_path and len(outputs) == 1:
        return outputs[0]
    return outputs

@_click_command()
@_click_option("-i", "--input", "input_path", type=click.Path(exists=True, path_type=Path) if click is not None else None, required=True, help="Path to DICOM directory or file.")
@_click_option("-o", "--output", "output_path", type=click.Path(path_type=Path) if click is not None else None, required=True, help="Output directory, or .nii/.nii.gz for single-series explicit output.")
@_click_option("--naming", type=str, default=None, help='Custom naming with DICOM tags split by underscore (e.g. "AccessionNumber_Modality").')
@_click_option("--multifile", is_flag=True, help="Process each direct input subdirectory as a separate case.")
@_click_option("--force-ras", is_flag=True, help="Force canonical RAS orientation.")
@_click_option("--log-level", "--log_level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False) if click is not None else None, default="INFO", help="Logging level.")
@_click_option("--log-path", "--log_path", type=click.Path(path_type=Path) if click is not None else None, default=None, help="Log directory (reserved).")
@_click_option("--debug", is_flag=True, help="Raise full traceback on failure.")
@_click_option("--process-rtstruct", is_flag=True, help="Process RTStruct files into mask NIfTI outputs.")
@_click_option("--revert-scaling", is_flag=True, help="Revert scanner-applied scaling to raw counts.")
@_click_option("--save-metadata", is_flag=True, help="Save metadata as a JSON sidecar alongside NIfTI outputs.")
@_click_option("--additional-tags", type=str, default=None, help='Comma-separated extra metadata tags (e.g. "ProtocolName,SequenceName").')
@_click_option("--compress", is_flag=True, help="When output is a directory, write compressed .nii.gz files.")
@_click_option("--skip-existing", is_flag=True, help="Skip already-existing outputs.")
@_click_option("--rescale-type", type=click.Choice(["DV", "FP"], case_sensitive=False) if click is not None else None, default="DV", help="Rescale type for scaling conversion: DV or FP.")
@_click_option("--tmp-dir", type=click.Path(path_type=Path) if click is not None else None, default=None, help="Temporary directory for intermediate files.")
def main(
    input_path: Path,
    output_path: Path,
    naming: str | None,
    multifile: bool,
    force_ras: bool,
    log_level: str,
    log_path: Path | None,
    debug: bool,
    process_rtstruct: bool,
    revert_scaling: bool,
    save_metadata: bool,
    additional_tags: str | None,
    compress: bool,
    skip_existing: bool,
    rescale_type: str,
    tmp_dir: Path | None,
) -> None:
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')
    _ = (log_level, log_path)
    tags = [item.strip() for item in additional_tags.split(",") if item.strip()] if additional_tags else None

    try:
        if multifile:
            if not input_path.is_dir():
                raise click.ClickException("--multifile requires a directory input path.")
            output_path.mkdir(parents=True, exist_ok=True)
            patient_dirs = sorted([item for item in input_path.iterdir() if item.is_dir()])
            if not patient_dirs:
                raise click.ClickException(f"No subdirectories found under {input_path}")

            ok = 0
            skipped = 0
            failed = 0
            for patient_dir in patient_dirs:
                target = output_path / patient_dir.name
                try:
                    if skip_existing and target.exists():
                        has_files = any(item.is_file() for item in target.iterdir() if item.exists())
                        if has_files:
                            click.echo(f"Skipping existing patient folder: {patient_dir.name}")
                            skipped += 1
                            continue
                    dcm2nii(
                        str(patient_dir),
                        str(target),
                        custom_naming=naming,
                        force_ras=force_ras,
                        process_rtstruct=process_rtstruct,
                        revert_scaling=revert_scaling,
                        save_metadata=save_metadata,
                        additional_tags=tags,
                        compress=compress,
                        rescale_type=rescale_type,
                        skip_existing=skip_existing,
                        tmp_dir=tmp_dir,
                    )
                    ok += 1
                except Exception:
                    failed += 1
                    if debug:
                        raise
                    click.echo(f"Failed: {patient_dir}", err=True)
            click.echo(f"Completed: {ok}, skipped: {skipped}, failed: {failed}")
            return

        outputs = dcm2nii(
            str(input_path),
            str(output_path),
            custom_naming=naming,
            force_ras=force_ras,
            process_rtstruct=process_rtstruct,
            revert_scaling=revert_scaling,
            save_metadata=save_metadata,
            additional_tags=tags,
            compress=compress,
            rescale_type=rescale_type,
            skip_existing=skip_existing,
            tmp_dir=tmp_dir,
        )
        if isinstance(outputs, str):
            click.echo(outputs)
        else:
            for item in outputs:
                click.echo(item)
    except Exception as exc:
        if debug:
            raise
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
