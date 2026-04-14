"""Rasterize STL surfaces to labeled NIfTI volumes (VTK-based; CLI optional)."""

from __future__ import annotations

from pathlib import Path

try:
    import click
except Exception:
    click = None

from nvitk.core.exceptions import BackendUnavailableError, ValidationError

from ._poly_stencil import list_stl_files, multilabel_from_stls, stl_to_vtk_binary, write_nifti

__all__ = [
    "stl2nifti",
    "run_stl2nifti",
]


def _cli_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator


def _info(message: str) -> None:
    print(message)


def _warn(message: str) -> None:
    print(message)


def _err(message: str) -> None:
    print(message)


def _is_nifti_file_path(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _select_reference(
    candidates: list[Path],
    reference_rule: str | None,
    reference_rule_mode: str,
    patient_identifier: str,
) -> Path:
    if reference_rule:
        if reference_rule_mode == "startswith":
            filtered = [c for c in candidates if c.name.startswith(reference_rule)]
        elif reference_rule_mode == "endswith":
            filtered = [c for c in candidates if c.name.endswith(reference_rule)]
        else:
            filtered = [c for c in candidates if reference_rule in c.name]

        if filtered:
            return filtered[0]
        raise FileNotFoundError(
            f"No matching reference found for patient '{patient_identifier}' with rule '{reference_rule}'"
        )
    return candidates[0]


def _resolve_reference_path(
    reference: Path,
    reference_rule: str | None,
    reference_rule_mode: str,
    patient_identifier: str,
) -> Path:
    if reference.is_file():
        return reference

    if reference.is_dir():
        patient_dir = reference / patient_identifier
        if patient_dir.exists() and patient_dir.is_dir():
            candidates = sorted(p for p in patient_dir.rglob("*.nii*") if p.is_file())
            if candidates:
                return _select_reference(candidates, reference_rule, reference_rule_mode, patient_identifier)

        candidates = sorted(p for p in reference.rglob("*.nii*") if p.is_file())
        if candidates:
            return _select_reference(candidates, reference_rule, reference_rule_mode, patient_identifier)

    raise FileNotFoundError(f"No reference NIfTI found for patient '{patient_identifier}' in {reference}")


def run_stl2nifti(
    input_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    multifile: bool = False,
    multilabel: bool = False,
    overwrite: bool = False,
    log_level: str = "INFO",
    reference_rule: str | None = None,
    reference_rule_mode: str = "contains",
) -> list[str]:
    _ = log_level
    input_root = Path(input_path)
    reference_root = Path(reference_path)
    output_root = Path(output_path)

    reference_rule_mode = (reference_rule_mode or "contains").lower()
    if reference_rule_mode not in {"contains", "startswith", "endswith"}:
        _warn(
            f"Invalid reference_rule_mode '{reference_rule_mode}', expected one of: "
            "contains, startswith, endswith"
        )
        reference_rule_mode = "contains"

    if reference_rule is not None:
        reference_rule = reference_rule.strip() or None

    outputs: list[str] = []
    if multifile:
        if not input_root.is_dir():
            raise ValidationError("Multifile mode requires input to be a directory.")
        if not reference_root.is_dir():
            raise ValidationError("Multifile mode requires reference to be a directory matching the input structure.")

        patient_dirs = sorted(d for d in input_root.iterdir() if d.is_dir())
        if not patient_dirs:
            raise ValidationError(f"No patient directories found in {input_root}")

        successful = 0
        failed = 0
        _info(f"Found {len(patient_dirs)} patient directories to process")
        for idx, patient_dir in enumerate(patient_dirs, 1):
            patient_name = patient_dir.name
            _info(f"[info]Processing patient {idx}/{len(patient_dirs)}: {patient_name}[/info]")
            try:
                reference_patient_dir = reference_root / patient_name
                if not reference_patient_dir.exists() or not reference_patient_dir.is_dir():
                    raise FileNotFoundError(
                        f"Reference directory not found for patient '{patient_name}' in {reference_root}"
                    )

                reference_candidates = sorted(p for p in reference_patient_dir.rglob("*.nii*") if p.is_file())
                if not reference_candidates:
                    raise FileNotFoundError(
                        f"No reference NIfTI files found for patient '{patient_name}' in {reference_patient_dir}"
                    )

                patient_reference = _select_reference(
                    reference_candidates,
                    reference_rule,
                    reference_rule_mode,
                    patient_name,
                )
                _info(f"Using reference for {patient_name}: {patient_reference}")

                stl_files = list_stl_files(patient_dir)
                if not stl_files:
                    _warn(f"[warn]No STL files found in {patient_dir}, skipping[/warn]")
                    continue

                patient_output_dir = output_root / patient_name
                patient_output_dir.mkdir(parents=True, exist_ok=True)
                if multilabel:
                    out_file = patient_output_dir / f"{patient_name}_multilabel.nii.gz"
                    vtk_img, qform, qfac, _ = multilabel_from_stls(
                        stl_files,
                        patient_reference,
                        overwrite=overwrite,
                    )
                    write_nifti(out_file, vtk_img, qform, qfac)
                    outputs.append(str(out_file))
                    _info(f"[ok]Wrote multilabel NIfTI for {patient_name}: {out_file}[/ok]")
                else:
                    for stl_path in stl_files:
                        vtk_img, qform, qfac = stl_to_vtk_binary(stl_path, patient_reference)
                        out_file = patient_output_dir / f"{Path(stl_path).stem}.nii.gz"
                        write_nifti(out_file, vtk_img, qform, qfac)
                        outputs.append(str(out_file))
                    _info(f"[ok]Wrote {len(stl_files)} per-label NIfTI files for {patient_name}[/ok]")
                successful += 1
            except Exception as exc:
                failed += 1
                _err(f"[error]Error processing {patient_name}: {exc}[/error]")

        _info("[info]========================================[/info]")
        _info("[info]Multifile Processing Summary[/info]")
        _info("[info]========================================[/info]")
        _info(f"Total patients: {len(patient_dirs)}")
        _info(f"[ok]Successful: {successful}[/ok]")
        if failed > 0:
            _warn(f"[error]Failed: {failed}[/error]")
        if not outputs:
            raise ValidationError(f"No STL files were successfully converted from {input_root}")
        return outputs

    _info("[info]Starting STL to NIfTI conversion...[/info]")
    if input_root.is_file():
        reference_file = _resolve_reference_path(
            reference_root,
            reference_rule,
            reference_rule_mode,
            input_root.stem,
        )
        vtk_img, qform, qfac = stl_to_vtk_binary(input_root, reference_file)
        target = output_root / f"{input_root.stem}.nii.gz" if output_root.is_dir() else output_root
        write_nifti(target, vtk_img, qform, qfac)
        outputs.append(str(target))
        _info(f"[ok]Wrote NIfTI: {target}[/ok]")
        return outputs

    stl_files = list_stl_files(input_root)
    if not stl_files:
        raise ValidationError(f"No STL files found in {input_root}")

    reference_file = _resolve_reference_path(
        reference_root,
        reference_rule,
        reference_rule_mode,
        input_root.name,
    )
    if multilabel:
        if not _is_nifti_file_path(output_root):
            raise ValidationError("For multilabel mode, output must be a .nii or .nii.gz file path.")
        vtk_img, qform, qfac, _ = multilabel_from_stls(
            stl_files,
            reference_file,
            overwrite=overwrite,
        )
        write_nifti(output_root, vtk_img, qform, qfac)
        outputs.append(str(output_root))
        _info(f"[ok]Wrote multilabel NIfTI: {output_root}[/ok]")
        return outputs

    output_root.mkdir(parents=True, exist_ok=True)
    for stl_path in stl_files:
        vtk_img, qform, qfac = stl_to_vtk_binary(stl_path, reference_file)
        out_file = output_root / f"{Path(stl_path).stem}.nii.gz"
        write_nifti(out_file, vtk_img, qform, qfac)
        outputs.append(str(out_file))
    _info(f"[ok]Wrote {len(stl_files)} per-label NIfTI files to: {output_root}[/ok]")
    return outputs


def stl2nifti(
    input_path: str | Path,
    reference_path: str | Path,
    output_path: str | Path,
    *,
    multifile: bool = False,
    multilabel: bool = False,
    overwrite: bool = False,
    log_level: str = "INFO",
    reference_rule: str | None = None,
    reference_rule_mode: str = "contains",
) -> str | list[str]:
    outputs = run_stl2nifti(
        input_path,
        reference_path,
        output_path,
        multifile=multifile,
        multilabel=multilabel,
        overwrite=overwrite,
        log_level=log_level,
        reference_rule=reference_rule,
        reference_rule_mode=reference_rule_mode,
    )

    input_root = Path(input_path)
    if multifile:
        return outputs
    if input_root.is_file() or multilabel:
        return outputs[0]
    return outputs


@_click_command()
@_click_option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path) if click is not None else None,
    required=True,
    help="Path to one STL file or a directory containing STL files.",
)
@_click_option(
    "-r",
    "--reference",
    "reference_path",
    type=click.Path(exists=True, path_type=Path) if click is not None else None,
    required=True,
    help="Path to the reference NIfTI image, or a directory containing references.",
)
@_click_option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(path_type=Path) if click is not None else None,
    required=True,
    help="Output directory or file path.",
)
@_click_option(
    "--multifile",
    is_flag=True,
    help="Process multiple patient directories, each containing STL files.",
)
@_click_option(
    "--multilabel/--per_label",
    default=True,
    help="Export one multi-label NIfTI or one binary NIfTI per label.",
)
@_click_option(
    "--overwrite",
    is_flag=True,
    help="When creating multilabel outputs, allow later labels to overwrite earlier ones.",
)
@_click_option(
    "--log-level",
    "--log_level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False) if click is not None else None,
    default="INFO",
    help="Logging level.",
)
@_click_option(
    "--reference-rule",
    "--reference_rule",
    type=str,
    default=None,
    help="Optional text rule to select a reference file when multiple are found.",
)
@_click_option(
    "--reference-rule-mode",
    "--reference_rule_mode",
    type=click.Choice(["startswith", "endswith", "contains"], case_sensitive=False) if click is not None else None,
    default="contains",
    help="How to apply reference rule when selecting among multiple references.",
)
def main(
    input_path: Path,
    reference_path: Path,
    output_path: Path,
    multifile: bool,
    multilabel: bool,
    overwrite: bool,
    log_level: str,
    reference_rule: str | None,
    reference_rule_mode: str,
) -> None:
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')
    try:
        outputs = stl2nifti(
            input_path,
            reference_path,
            output_path,
            multifile=multifile,
            multilabel=multilabel,
            overwrite=overwrite,
            log_level=log_level,
            reference_rule=reference_rule,
            reference_rule_mode=reference_rule_mode.lower() if reference_rule_mode else "contains",
        )
        if isinstance(outputs, str):
            click.echo(outputs)
        else:
            for item in outputs:
                click.echo(item)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


if __name__ == "__main__":
    main()
