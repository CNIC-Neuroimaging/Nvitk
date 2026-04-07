from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from nvitk.core.exceptions import BackendUnavailableError, ValidationError

from ._dicom_rtstructs import integrate_rtstruct_processing, is_rtstruct_file
from ._dicom_zeiss import extract_zeiss_raw_oct, is_zeiss_raw_storage
from ..writers import write_nifti

try:
    import nibabel as nib
except Exception:
    nib = None

try:
    import pydicom
except Exception:
    pydicom = None

try:
    import dicom2nifti as _dicom2nifti
except Exception:
    _dicom2nifti = None

try:
    import click
except Exception:
    click = None


_DEFAULT_METADATA_KEYS = {
    "Modality",
    "(0008,0060)",
    "SeriesDescription",
    "(0008,103e)",
    "SeriesInstanceUID",
    "(0020,000e)",
    "SeriesNumber",
    "(0020,0011)",
    "StudyInstanceUID",
    "(0020,000d)",
    "AccessionNumber",
    "(0008,0050)",
    "PatientID",
    "(0010,0020)",
    "PixelSpacing",
    "(0028,0030)",
    "SliceThickness",
    "(0018,0050)",
    "SpacingBetweenSlices",
    "(0018,0088)",
    "ImagePositionPatient",
    "(0020,0032)",
    "ImageOrientationPatient",
    "(0020,0037)",
    "RescaleSlope",
    "(0028,1053)",
    "RescaleIntercept",
    "(0028,1052)",
    "ScaleSlope",
    "(2005,100e)",
    "RadiopharmaceuticalStartTime",
    "(0018,1072)",
    "RadionuclideHalfLife",
    "(0018,1075)",
    "RadionuclideTotalDose",
    "(0018,1074)",
    "DecayCorrection",
    "(0054,1102)",
    "DecayFactor",
    "(0054,1321)",
    "FrameReferenceTime",
    "(0054,1300)",
    "DoseUnits",
    "(0054,1004)",
    "Units",
    "(0054,1001)",
}


def _require_deps() -> None:
    if pydicom is None:
        raise BackendUnavailableError('pydicom is not installed. Please install it with "pip install pydicom".')
    if nib is None:
        raise BackendUnavailableError('nibabel is not installed. Please install it with "pip install nibabel".')


def _sanitize_filename(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(value))
    return safe.strip("_") or "series"


def _meta_get_case_insensitive(metadata: dict[str, Any], key: str) -> Any:
    if key in metadata:
        return metadata[key]
    lowered = key.lower()
    for k, value in metadata.items():
        if isinstance(k, str) and k.lower() == lowered:
            return value
    return None


def _resolve_naming(custom_naming: str | None, metadata: dict[str, Any], fallback: str) -> str:
    if not custom_naming:
        return fallback
    chunks: list[str] = []
    for token in (part.strip() for part in custom_naming.split("_")):
        if not token:
            continue
        value = _meta_get_case_insensitive(metadata, token)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            chunks.append(text)
    return "_".join(chunks) if chunks else fallback


def _is_nifti_file_path(path: Path) -> bool:
    lower = path.name.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz")


def _build_target_path(
    output_path: Path,
    *,
    series_idx: int,
    metadata: dict[str, Any],
    custom_naming: str | None,
    compress: bool,
) -> Path:
    if _is_nifti_file_path(output_path):
        return output_path

    output_path.mkdir(parents=True, exist_ok=True)
    uid = str(metadata.get("series_uid") or metadata.get("SeriesInstanceUID") or f"series_{series_idx:03d}")
    desc = str(metadata.get("series_description") or metadata.get("SeriesDescription") or "")
    fallback = f"{series_idx:03d}_{desc}_{uid}".strip("_")
    stem = _sanitize_filename(_resolve_naming(custom_naming, metadata, fallback))
    ext = ".nii.gz" if compress else ".nii"
    return output_path / f"{stem}{ext}"


def _to_python(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, (list, tuple)):
        return [_to_python(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "__float__"):
        try:
            return float(value)
        except Exception:
            pass
    if hasattr(value, "__int__"):
        try:
            return int(value)
        except Exception:
            pass
    return str(value)


def _dataset_to_metadata(ds: Any, *, include_private_tags: bool) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for element in ds:
        if element.keyword == "PixelData":
            continue
        if element.tag.is_private and not include_private_tags:
            continue
        key = element.keyword if element.keyword else str(element.tag)
        value = _to_python(element.value)
        metadata[key] = value
        metadata[str(element.tag)] = value
    return metadata


def _sort_slice_key(ds: Any, idx: int) -> tuple[int, float, int]:
    try:
        instance = int(ds.get("InstanceNumber", 10**9))
    except Exception:
        instance = 10**9
    z_value = 0.0
    try:
        ipp = ds.get("ImagePositionPatient")
        if ipp is not None and len(ipp) >= 3:
            z_value = float(ipp[2])
        elif ds.get("SliceLocation") is not None:
            z_value = float(ds.get("SliceLocation"))
    except Exception:
        z_value = 0.0
    return instance, z_value, idx


def _iter_input_files(input_path: str) -> list[Path]:
    src = Path(input_path)
    if src.is_file():
        return [src]
    return [item for item in sorted(src.rglob("*")) if item.is_file()]


def _collect_rescale_metadata(ds_list: list[Any], metadata: dict[str, Any]) -> None:
    slopes: list[float] = []
    intercepts: list[float] = []
    scale_slopes: list[float] = []
    for ds in ds_list:
        try:
            slopes.append(float(ds.get("RescaleSlope", 1.0)))
        except Exception:
            slopes.append(1.0)
        try:
            intercepts.append(float(ds.get("RescaleIntercept", 0.0)))
        except Exception:
            intercepts.append(0.0)
        try:
            scale_slopes.append(float(ds.get((0x2005, 0x100E), ds.get("ScaleSlope", 1.0))))
        except Exception:
            scale_slopes.append(1.0)

    if slopes:
        metadata["RescaleSlope"] = slopes[0]
        metadata["_rescale_slopes"] = slopes
    if intercepts:
        metadata["RescaleIntercept"] = intercepts[0]
        metadata["_rescale_intercepts"] = intercepts
    if scale_slopes:
        metadata["ScaleSlope"] = scale_slopes[0]
        metadata["_scale_slopes"] = scale_slopes


def _collect_series(input_path: str, *, include_private_tags: bool) -> tuple[list[tuple[str, list[Any], dict[str, Any]]], list[str]]:
    grouped: dict[str, list[Any]] = {}
    meta_map: dict[str, dict[str, Any]] = {}
    rtstruct_paths: list[str] = []

    for fp in _iter_input_files(input_path):
        try:
            ds = pydicom.dcmread(str(fp), force=True)
        except Exception:
            continue

        if is_rtstruct_file(ds):
            rtstruct_paths.append(str(fp))
            continue

        has_pixel = getattr(ds, "PixelData", None) is not None
        if not has_pixel and not is_zeiss_raw_storage(ds):
            continue

        uid = str(ds.get("SeriesInstanceUID", "NO_SERIES"))
        grouped.setdefault(uid, []).append(ds)
        if uid not in meta_map:
            md = _dataset_to_metadata(ds, include_private_tags=include_private_tags)
            md["series_uid"] = uid
            md["series_description"] = str(ds.get("SeriesDescription", ""))
            md["series_number"] = _to_python(ds.get("SeriesNumber"))
            md["Modality"] = str(ds.get("Modality", md.get("Modality", "")))
            md["(0008,0060)"] = md["Modality"]
            meta_map[uid] = md

    out: list[tuple[str, list[Any], dict[str, Any]]] = []
    for uid, ds_list in grouped.items():
        ordered = [ds for _, ds in sorted(enumerate(ds_list), key=lambda pair: _sort_slice_key(pair[1], pair[0]))]
        md = meta_map.get(uid, {})
        _collect_rescale_metadata(ordered, md)
        out.append((uid, ordered, md))

    def _series_sort_key(item: tuple[str, list[Any], dict[str, Any]]) -> tuple[int, str]:
        _, _, md = item
        try:
            number = int(md.get("series_number", 10**9))
        except Exception:
            number = 10**9
        return number, str(md.get("series_uid", ""))

    return sorted(out, key=_series_sort_key), rtstruct_paths


def _build_affine_from_dicom(ds_list: list[Any]) -> np.ndarray:
    first = ds_list[0]
    iop = first.get("ImageOrientationPatient")
    ipp = first.get("ImagePositionPatient")
    pixel_spacing = first.get("PixelSpacing")
    if iop is None or ipp is None or pixel_spacing is None:
        return np.eye(4, dtype=float)

    try:
        row_cos = np.asarray([float(v) for v in iop[:3]], dtype=float)
        col_cos = np.asarray([float(v) for v in iop[3:6]], dtype=float)
        row_spacing = float(pixel_spacing[0])
        col_spacing = float(pixel_spacing[1])
        origin = np.asarray([float(v) for v in ipp[:3]], dtype=float)
    except Exception:
        return np.eye(4, dtype=float)

    z_spacing = None
    try:
        z_spacing = float(first.get("SpacingBetweenSlices", first.get("SliceThickness")))
    except Exception:
        z_spacing = None
    if z_spacing is None and len(ds_list) > 1:
        try:
            ipp0 = ds_list[0].get("ImagePositionPatient")
            ipp1 = ds_list[1].get("ImagePositionPatient")
            if ipp0 is not None and ipp1 is not None:
                z_spacing = abs(float(ipp1[2]) - float(ipp0[2]))
        except Exception:
            z_spacing = None
    if z_spacing is None:
        z_spacing = 1.0

    slice_cos = np.cross(row_cos, col_cos)
    affine = np.eye(4, dtype=float)
    affine[:3, 0] = col_cos * col_spacing
    affine[:3, 1] = row_cos * row_spacing
    affine[:3, 2] = slice_cos * float(z_spacing)
    affine[:3, 3] = origin
    return affine


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return str(value)


def _select_metadata_extension(metadata: dict[str, Any], additional_tags: list[str] | None) -> dict[str, Any]:
    keep = set(_DEFAULT_METADATA_KEYS)
    if additional_tags:
        keep.update(tag for tag in additional_tags if tag)
    out: dict[str, Any] = {}
    for key in keep:
        if key in metadata:
            out[key] = metadata[key]
    return out


def _apply_metadata_extension(image: Any, metadata: dict[str, Any], additional_tags: list[str] | None) -> Any:
    payload = _select_metadata_extension(metadata, additional_tags)
    if not payload:
        return image
    try:
        encoded = json.dumps(_jsonable(payload), ensure_ascii=True).encode("utf-8")
        image.header.extensions.append(nib.nifti1.Nifti1Extension(16, encoded))
    except Exception:
        pass
    return image


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _rescale_image(image: Any, metadata: dict[str, Any], *, revert_scaling: bool, rescale_type: str) -> Any:
    mode = (rescale_type or "DV").strip().upper()
    if not revert_scaling and mode != "FP":
        return image

    data = image.get_fdata().astype(np.float32, copy=False)
    slopes = metadata.get("_rescale_slopes")
    intercepts = metadata.get("_rescale_intercepts")
    scale_slopes = metadata.get("_scale_slopes")

    if isinstance(slopes, list) and slopes and data.ndim >= 3 and len(slopes) == data.shape[-1]:
        for z in range(data.shape[-1]):
            s = _to_float(slopes[z], 1.0)
            i = _to_float(intercepts[z] if isinstance(intercepts, list) and z < len(intercepts) else 0.0, 0.0)
            ss = _to_float(scale_slopes[z] if isinstance(scale_slopes, list) and z < len(scale_slopes) else 1.0, 1.0)
            if s == 0:
                s = 1.0
            if ss == 0:
                ss = 1.0
            if revert_scaling:
                data[..., z] = (data[..., z] - i) / s
            elif mode == "FP":
                data[..., z] = data[..., z] / (s * ss)
    else:
        s = _to_float(metadata.get("RescaleSlope", metadata.get("(0028,1053)", 1.0)), 1.0)
        i = _to_float(metadata.get("RescaleIntercept", metadata.get("(0028,1052)", 0.0)), 0.0)
        ss = _to_float(metadata.get("ScaleSlope", metadata.get("(2005,100e)", 1.0)), 1.0)
        if s == 0:
            s = 1.0
        if ss == 0:
            ss = 1.0
        if revert_scaling:
            data = (data - i) / s
        elif mode == "FP":
            data = data / (s * ss)

    return nib.Nifti1Image(data, image.affine, image.header)


def _convert_series_to_nifti(ds_list: list[Any], *, force_ras: bool) -> Any:
    if _dicom2nifti is not None:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tf:
            tmp = Path(tf.name)
        try:
            _dicom2nifti.convert_dicom.dicom_array_to_nifti(ds_list, str(tmp), reorient_nifti=bool(force_ras))
            return nib.load(str(tmp))
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    arrays = [np.asarray(ds.pixel_array) for ds in ds_list]
    if not arrays:
        raise ValidationError("Series has no readable pixel arrays.")

    if len(arrays) == 1:
        arr = arrays[0]
        if arr.ndim == 2:
            arr = arr.T
    else:
        arr = np.stack(arrays, axis=0)  # Z,Y,X
        if arr.ndim == 3:
            arr = arr.transpose(2, 1, 0)  # X,Y,Z
        elif arr.ndim == 4:
            arr = arr.transpose(3, 2, 1, 0)  # X,Y,C,Z

    image = nib.Nifti1Image(arr, _build_affine_from_dicom(ds_list))
    if force_ras:
        image = nib.as_closest_canonical(image)
    return image


def _prepare_write_metadata(metadata: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("axes", "shape", "affine", "x_res", "y_res", "z_res", "t_res", "temporal_resolution", "spacing"):
        if key in metadata:
            out[key] = metadata[key]
    if extra:
        out.update(extra)
    return out


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
    series_uid: str | None = None,
    series_index: int | None = None,
    include_private_tags: bool = False,
    skip_existing: bool = False,
) -> str | list[str]:
    _require_deps()
    series_all, _rtstruct_paths = _collect_series(input_path, include_private_tags=include_private_tags)
    if not series_all:
        raise ValidationError(f"No valid DICOM image series found at: {input_path}")

    if series_uid is not None:
        series_all = [item for item in series_all if item[0] == series_uid]
        if not series_all:
            raise ValidationError(f"series_uid={series_uid!r} not found.")
    elif series_index is not None:
        if not (0 <= series_index < len(series_all)):
            raise ValidationError(f"series_index={series_index} out of range for {len(series_all)} series.")
        series_all = [series_all[series_index]]

    output_path = Path(output_folder)
    output_is_file = _is_nifti_file_path(output_path)
    if len(series_all) > 1 and output_is_file:
        raise ValidationError("Input contains multiple series. Use output directory or select one series.")

    written_paths: list[str] = []
    for idx, (_uid, ds_list, metadata) in enumerate(series_all):
        target = _build_target_path(
            output_path,
            series_idx=idx,
            metadata=metadata,
            custom_naming=custom_naming,
            compress=compress,
        )
        if skip_existing and target.exists():
            written_paths.append(str(target))
            continue

        first = ds_list[0]
        if is_zeiss_raw_storage(first):
            volume, affine, zmeta = extract_zeiss_raw_oct(ds_list, metadata)
            if revert_scaling:
                s = _to_float(metadata.get("RescaleSlope", 1.0), 1.0) or 1.0
                i = _to_float(metadata.get("RescaleIntercept", 0.0), 0.0)
                volume = (volume.astype(np.float32, copy=False) - i) / s
            zeiss_md = dict(metadata)
            zeiss_md.update(zmeta)
            zeiss_md["affine"] = affine
            zeiss_md["axes"] = "XYZ"
            zeiss_md["shape"] = tuple(volume.shape)
            write_nifti(
                str(target),
                volume,
                metadata=_prepare_write_metadata(
                    zeiss_md,
                    _select_metadata_extension(metadata, additional_tags) if save_metadata else None,
                ),
                save_metadata_extension=save_metadata,
            )
            written_paths.append(str(target))
            continue

        image = _convert_series_to_nifti(ds_list, force_ras=force_ras)
        image = _rescale_image(image, metadata, revert_scaling=revert_scaling, rescale_type=rescale_type)
        if save_metadata:
            image = _apply_metadata_extension(image, metadata, additional_tags)
        target.parent.mkdir(parents=True, exist_ok=True)
        nib.save(image, str(target))
        written_paths.append(str(target))

    if process_rtstruct:
        rt_output_root = str(output_path.parent if output_is_file else output_path)
        integrate_rtstruct_processing(input_path, rt_output_root)

    if output_is_file and len(written_paths) == 1:
        return written_paths[0]
    return written_paths


@click.command()
@click.option("-i", "--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True, help="Path to DICOM directory or file.")
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), required=True, help="Output directory, or .nii/.nii.gz for single-series explicit output.")
@click.option("--naming", type=str, default=None, help='Custom naming with DICOM tags split by underscore (e.g. "AccessionNumber_Modality").')
@click.option("--multifile", is_flag=True, help="Process each direct input subdirectory as a separate case.")
@click.option("--force-ras", is_flag=True, help="Force canonical RAS orientation.")
@click.option("--log-level", "--log_level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False), default="INFO", help="Logging level.")
@click.option("--log-path", "--log_path", type=click.Path(path_type=Path), default=None, help="Log directory (reserved).")
@click.option("--debug", is_flag=True, help="Raise full traceback on failure.")
@click.option("--dry-run", "--dry_run", is_flag=True, help="Preview operations without writing files.")
@click.option("--process-rtstruct", is_flag=True, help="Process RTStruct files into mask NIfTI outputs.")
@click.option("--revert-scaling", is_flag=True, help="Revert scanner-applied scaling to raw counts.")
@click.option("--save-metadata", is_flag=True, help="Store selected DICOM metadata in NIfTI extension.")
@click.option("--additional-tags", type=str, default=None, help='Comma-separated extra metadata tags (e.g. "ProtocolName,SequenceName").')
@click.option("--compress", is_flag=True, help="When output is a directory, write compressed .nii.gz files.")
@click.option("--skip-existing", is_flag=True, help="Skip already-existing output files.")
@click.option("--rescale-type", type=click.Choice(["DV", "FP"], case_sensitive=False), default="DV", help="Rescale type for scaling conversion: DV or FP.")
def main(
    input_path: Path,
    output_path: Path,
    naming: str | None,
    multifile: bool,
    force_ras: bool,
    log_level: str,
    log_path: Path | None,
    debug: bool,
    dry_run: bool,
    process_rtstruct: bool,
    revert_scaling: bool,
    save_metadata: bool,
    additional_tags: str | None,
    compress: bool,
    skip_existing: bool,
    rescale_type: str,
) -> None:
    _ = (log_level, log_path)
    tags = [item.strip() for item in additional_tags.split(",") if item.strip()] if additional_tags else None

    if dry_run:
        click.echo("DRY RUN")
        click.echo(f"input={input_path}")
        click.echo(f"output={output_path}")
        click.echo(f"multifile={multifile}")
        return

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
                    before = set(target.rglob("*.nii*")) if target.exists() else set()
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
                    )
                    after = set(target.rglob("*.nii*")) if target.exists() else set()
                    if skip_existing and after == before:
                        skipped += 1
                    else:
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

