"""Phase-contrast / cine MRI-style tabular pipelines to 3D/4D volumes (CLI optional).

4DFlow derivatives follow the QVTplus ``loadNII`` / ``calc_angio`` convention:
time-mean angiography, velocity magnitude, complex-difference contrast, optional
spatial polynomial background correction on velocity (see ``_phase2volume_bg``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import click
except Exception:
    click = None

from nvitk.core.array import as_backend_array
from nvitk.core.backend import setup
from nvitk.core.exceptions import BackendUnavailableError
from nvitk.core.logger import Logger
from .._common import default_nifti_axes
from ..imageio import imread, imsave

setup(globals())

from ._phase2volume_bg import (
    fit_polynomial_background_3vector,
    subtract_mean_background_from_temporal,
)

log = Logger()


def _cli_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


_click_command = click.command if click is not None else _cli_decorator
_click_option = click.option if click is not None else _cli_decorator


@dataclass(frozen=True)
class PhaseInputs:
    patient_dir: Path
    flow_dir: Path
    ap_dir: Path
    rl_dir: Path
    fh_dir: Path
    angio_path: Path
    ap_phase_path: Path
    rl_phase_path: Path
    fh_phase_path: Path


def _first_match(directory: Path, *patterns: str) -> Path | None:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_direction_dirs(flow_dir: Path) -> tuple[Path, Path, Path]:
    ap_dir = rl_dir = fh_dir = None
    for subdir in sorted(flow_dir.iterdir()):
        if not subdir.is_dir():
            continue
        upper = subdir.name.upper()
        if "AP" in upper and ap_dir is None:
            ap_dir = subdir
        elif "RL" in upper and rl_dir is None:
            rl_dir = subdir
        elif "FH" in upper and fh_dir is None:
            fh_dir = subdir

    if ap_dir is None or rl_dir is None or fh_dir is None:
        missing = []
        if ap_dir is None:
            missing.append("AP")
        if rl_dir is None:
            missing.append("RL")
        if fh_dir is None:
            missing.append("FH")
        raise FileNotFoundError(f"Missing 4DFlow direction folders: {', '.join(missing)}")

    return ap_dir, rl_dir, fh_dir


def discover_phase_inputs(patient_dir: str | Path) -> PhaseInputs:
    patient_path = Path(patient_dir)
    flow_dir = patient_path / "4DFlow"
    if not flow_dir.exists():
        raise FileNotFoundError(f"4DFlow folder not found in {patient_path}")

    ap_dir, rl_dir, fh_dir = _find_direction_dirs(flow_dir)
    angio_path = _first_match(ap_dir, "*_m.nii.gz", "*_m.nii")
    ap_phase_path = _first_match(ap_dir, "*_ph.nii.gz", "*_ph.nii")
    rl_phase_path = _first_match(rl_dir, "*_ph.nii.gz", "*_ph.nii")
    fh_phase_path = _first_match(fh_dir, "*_ph.nii.gz", "*_ph.nii")

    if angio_path is None:
        raise FileNotFoundError(f"No angiography magnitude file found in {ap_dir}")
    if ap_phase_path is None or rl_phase_path is None or fh_phase_path is None:
        raise FileNotFoundError("Missing one or more AP/RL/FH phase volumes.")

    return PhaseInputs(
        patient_dir=patient_path,
        flow_dir=flow_dir,
        ap_dir=ap_dir,
        rl_dir=rl_dir,
        fh_dir=fh_dir,
        angio_path=angio_path,
        ap_phase_path=ap_phase_path,
        rl_phase_path=rl_phase_path,
        fh_phase_path=fh_phase_path,
    )


def _normalize_venc_to_mm_s(raw: float, *, source_hint: str = "") -> float:
    """Interpret scalar as mm/s; values that look like cm/s are scaled by 10."""
    v = float(raw)
    if v <= 0:
        raise ValueError(f"non-positive VENC {v!r} {source_hint}")
    # Heuristic: typical clinical VENC in mm/s is hundreds; cm/s is tens.
    if v < 150.0 and "mm" not in source_hint.lower():
        return v * 10.0
    return v


def _venc_from_json_payload(payload: dict[str, Any]) -> float | None:
    for field_name in ("VelocityEncoding", "PhaseEncodingVelocity"):
        if field_name in payload:
            # Legacy nvitk sidecars: store cm/s–scale; multiply to mm/s (unchanged).
            return float(payload[field_name]) * 10.0
    if "VENC" in payload:
        return _normalize_venc_to_mm_s(float(payload["VENC"]), source_hint="VENC")
    return None


def _venc_from_json_dir(ap_dir: Path) -> tuple[float | None, str | None]:
    for json_path in sorted(ap_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        v = _venc_from_json_payload(payload)
        if v is not None:
            return v, f"json:{json_path.name}"
    return None, None


def _venc_from_nifti_metadata(meta: dict[str, Any] | None) -> tuple[float | None, str | None]:
    if not meta:
        return None, None
    candidates = ("VENC", "VelocityEncoding", "PhaseEncodingVelocity", "venc", "velocity_encoding")
    for key in candidates:
        raw = meta.get(key)
        if raw is None:
            raw = meta.get(key.lower()) if key != key.lower() else None
        if raw is None:
            continue
        try:
            v = _normalize_venc_to_mm_s(float(raw), source_hint=key)
            return v, f"nifti_meta:{key}"
        except (TypeError, ValueError):
            continue
    return None, None


def _venc_from_single_dicom(path: Path) -> float | None:
    try:
        import pydicom
    except Exception:
        return None
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None
    # Standard: Velocity Encoding Maximum Value (0018,9217), mm/s in MR often as FD.
    elem = ds.get((0x0018, 0x9217), None)
    if elem is not None and elem.value is not None:
        try:
            return _normalize_venc_to_mm_s(float(elem.value), source_hint="DICOM_0018_9217")
        except (TypeError, ValueError):
            pass
    # GE-style private (see MATLAB loadDCM comments); optional.
    elem_ge = ds.get((0x0019, 0x10CC), None) or ds.get((0x0019, 0x10CE), None)
    if elem_ge is not None and elem_ge.value is not None:
        try:
            return _normalize_venc_to_mm_s(float(elem_ge.value), source_hint="DICOM_private_GE")
        except (TypeError, ValueError):
            pass
    return None


def _venc_from_dicom_directory(dicom_dir: Path, *, max_files: int = 100) -> tuple[float | None, str | None]:
    if not dicom_dir.is_dir():
        return None, None
    n = 0
    for p in sorted(dicom_dir.rglob("*.dcm")):
        if not p.is_file():
            continue
        v = _venc_from_single_dicom(p)
        if v is not None:
            return v, f"dicom:{p.relative_to(dicom_dir)}"
        n += 1
        if n >= max_files:
            break
    return None, None


def resolve_venc_mm_s(
    *,
    ap_dir: Path,
    default_mm_s: float = 700.0,
    ap_phase_metadata: dict[str, Any] | None = None,
    magnitude_metadata: dict[str, Any] | None = None,
    dicom_search_dir: Path | None = None,
    log_context: str = "",
) -> tuple[float, str]:
    """Resolve VENC in mm/s: JSON sidecars, then NIfTI metadata, then DICOM, then *default_mm_s*.

    Returns ``(venc_mm_s, source_tag)``. When falling back to default, logs a warning
    (MATLAB QVTplus often used fixed 700 mm/s when tags were absent).
    """
    v, src = _venc_from_json_dir(ap_dir)
    if v is not None:
        return v, src

    for meta, label in (
        (ap_phase_metadata, "ap_phase"),
        (magnitude_metadata, "magnitude"),
    ):
        v2, src2 = _venc_from_nifti_metadata(meta)
        if v2 is not None:
            return v2, f"{label}:{src2}"

    if dicom_search_dir is not None:
        v3, src3 = _venc_from_dicom_directory(dicom_search_dir)
        if v3 is not None:
            return v3, src3

    ctx = f" ({log_context})" if log_context else ""
    log.warning(
        "VENC not found in JSON, NIfTI metadata, or DICOM under search paths%s; "
        "using default %.1f mm/s (MATLAB QVTplus-style fallback).",
        ctx,
        float(default_mm_s),
    )
    return float(default_mm_s), "default"


def _calc_angio(angio_mag, v_mag, venc: float):
    """QVTplus-style complex-difference magnitude: MAG * sin(pi/2 * min(|V|, VENC) / VENC)."""
    vm = np.clip(as_backend_array(v_mag).astype(np.float64), 0.0, float(venc))
    return as_backend_array(angio_mag).astype(np.float64) * np.sin((np.pi / 2.0 * vm) / float(venc))


def compute_phase_derivatives(
    angio_data,
    ap_phase,
    rl_phase,
    fh_phase,
    *,
    venc: float,
    background_phase_correction: bool = True,
    bg_poly_order: int = 2,
    bg_static_percentile: float = 25.0,
) -> dict[str, Any]:
    """Compute angiography, complex-difference, and velocity magnitude derivatives (QVTplus-aligned).

    Velocity in mm/s: ``vx = -RL_phase * 10``, ``vy = -AP_phase * 10``, ``vz = FH_phase * 10``.

    When ``background_phase_correction`` is True (default), fits spatial polynomials
    (order ``bg_poly_order``, default 2 per MATLAB ``loadNII`` ``fit_order``) on the
    temporal-mean field using low-speed voxels (``bg_static_percentile``), subtracts
    from each frame, then recomputes magnitudes and CD. MATLAB's cd_thresh/noise_thresh
    mask is not replicated; see ``_phase2volume_bg``.

    Arrays follow :func:`~nvitk.core.array.as_backend_array` (NumPy or CuPy per
    :mod:`nvitk.core.backend`).
    """
    angio_data = as_backend_array(angio_data)
    ap_phase = as_backend_array(ap_phase)
    rl_phase = as_backend_array(rl_phase)
    fh_phase = as_backend_array(fh_phase)

    if not (ap_phase.shape == rl_phase.shape == fh_phase.shape):
        raise ValueError("AP, RL, and FH phase volumes must share the same shape.")

    if angio_data.shape[:3] != ap_phase.shape[:3]:
        raise ValueError("Angiography and phase volumes must match in spatial dimensions.")

    angio_tr = angio_data
    angio_mean = np.mean(angio_data, axis=-1) if angio_data.ndim == 4 else as_backend_array(angio_data)

    vx = -rl_phase * 10.0 # R (Left 2 Right) -> RL (Right 2 Left)
    vy = -ap_phase * 10.0 # A (Posterior 2 Anterior) -> AP (Anterior 2 Posterior)
    vz =  fh_phase * 10.0 # S (Inferior 2 Superior) = FH (Feet 2 Head)

    if background_phase_correction and vx.ndim == 4:
        vx_m = np.mean(vx, axis=-1)
        vy_m = np.mean(vy, axis=-1)
        vz_m = np.mean(vz, axis=-1)
        bg_x, bg_y, bg_z = fit_polynomial_background_3vector(
            vx_m,
            vy_m,
            vz_m,
            spatial_order=int(bg_poly_order),
            static_percentile=float(bg_static_percentile),
        )
        vx, vy, vz = subtract_mean_background_from_temporal(vx, vy, vz, bg_x, bg_y, bg_z)
    elif background_phase_correction and vx.ndim == 3:
        bg_x, bg_y, bg_z = fit_polynomial_background_3vector(
            vx,
            vy,
            vz,
            spatial_order=int(bg_poly_order),
            static_percentile=float(bg_static_percentile),
        )
        vx, vy, vz = vx - bg_x, vy - bg_y, vz - bg_z

    if vx.ndim == 4:
        v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)
        v_mag_capped = np.clip(v_mag, 0, venc)
        if angio_tr.ndim == 3:
            angio_tr = np.broadcast_to(np.expand_dims(angio_tr, axis=-1), v_mag_capped.shape)
        elif angio_tr.shape[3] != v_mag_capped.shape[3]:
            min_t = min(angio_tr.shape[3], v_mag_capped.shape[3])
            angio_tr = angio_tr[..., :min_t]
            v_mag_capped = v_mag_capped[..., :min_t]
            v_mag = v_mag[..., :min_t]
        cd = _calc_angio(angio_tr, v_mag_capped, venc)
        v_mag_mean = np.mean(v_mag, axis=-1)
        cd_mean = _calc_angio(angio_mean, np.clip(v_mag_mean, 0, venc), venc)
        v_mean_stack = np.stack([np.mean(v, axis=-1), np.mean(vy, axis=-1), np.mean(vz, axis=-1)], axis=-1)
    else:
        v_mag = np.sqrt(vx * vx + vy * vy + vz * vz)
        v_mag_capped = np.clip(v_mag, 0, venc)
        if angio_tr.ndim == 4:
            angio_tr = np.mean(angio_tr, axis=-1)
        cd = _calc_angio(angio_tr, v_mag_capped, venc)
        v_mag_mean = v_mag
        cd_mean = cd
        v_mean_stack = np.stack([vx, vy, vz], axis=-1)

    return {
        "Angiography_4D": angio_data if angio_data.ndim == 4 else None,
        "Angiography_3D": angio_mean.astype(np.float32, copy=False),
        "ComplexDifference_4D": cd if cd.ndim == 4 else None,
        "VelocityMagnitude_4D": v_mag if v_mag.ndim == 4 else None,
        "ComplexDifference_3D": cd_mean.astype(np.float32, copy=False),
        "VelocityMagnitude_3D": v_mag_mean.astype(np.float32, copy=False),
        "VelocityMeanComponents": v_mean_stack.astype(np.float32, copy=False),
    }


def _write_outputs(flow_dir: Path, outputs: dict[str, np.ndarray | None], metadata: dict[str, Any]) -> list[Path]:
    written: list[Path] = []
    for name, array in outputs.items():
        if array is None:
            continue
        output_path = flow_dir / f"{name}.nii.gz"
        output_metadata = dict(metadata)
        output_metadata["axes"] = default_nifti_axes(array.ndim)
        output_metadata["shape"] = tuple(array.shape)
        if array.ndim < 4:
            output_metadata.pop("t_res", None)
            output_metadata.pop("temporal_resolution", None)
        imsave(output_path, array, metadata=output_metadata)
        written.append(output_path)
    return written


def utc_now() -> str:
    from nvitk.db.storage import utc_now_iso

    return utc_now_iso()


def process_patient(
    patient_dir: str | Path,
    *,
    venc: float = 700.0,
    dry_run: bool = False,
    subject_uid: str | None = None,
    pipeline_id: str = "1.0.0",
    background_phase_correction: bool = True,
    bg_poly_order: int = 2,
    bg_static_percentile: float = 25.0,
    dicom_search_dir: Path | None = None,
) -> list[Path]:
    inputs = discover_phase_inputs(patient_dir)
    ctx = subject_uid or inputs.patient_dir.name
    if dry_run:
        return [
            inputs.angio_path,
            inputs.ap_phase_path,
            inputs.rl_phase_path,
            inputs.fh_phase_path,
        ]

    angio_image = imread(inputs.angio_path)
    ap_image = imread(inputs.ap_phase_path)
    rl_image = imread(inputs.rl_phase_path)
    fh_image = imread(inputs.fh_phase_path)

    actual_venc, venc_src = resolve_venc_mm_s(
        ap_dir=inputs.ap_dir,
        default_mm_s=venc,
        ap_phase_metadata=dict(ap_image.metadata or {}),
        magnitude_metadata=dict(angio_image.metadata or {}),
        dicom_search_dir=dicom_search_dir,
        log_context=ctx,
    )
    if venc_src != "default":
        log.info("VENC=%.1f mm/s from %s (subject=%s)", actual_venc, venc_src, ctx)

    outputs = compute_phase_derivatives(
        angio_image.data,
        ap_image.data,
        rl_image.data,
        fh_image.data,
        venc=actual_venc,
        background_phase_correction=background_phase_correction,
        bg_poly_order=bg_poly_order,
        bg_static_percentile=bg_static_percentile,
    )
    written = _write_outputs(inputs.flow_dir, outputs, dict(angio_image.metadata or {}))
    return written


def phase2volume(
    input_path: str | Path,
    *,
    multifile: bool = False,
    venc: float = 700.0,
    dry_run: bool = False,
    pipeline_id: str = "1.0.0",
    background_phase_correction: bool = True,
    bg_poly_order: int = 2,
    bg_static_percentile: float = 25.0,
    dicom_search_dir: Path | None = None,
) -> list[Path]:
    try:
        source = Path(input_path)
        if multifile:
            written: list[Path] = []
            for patient_dir in sorted(item for item in source.iterdir() if item.is_dir()):
                written.extend(
                    process_patient(
                        patient_dir,
                        venc=venc,
                        dry_run=dry_run,
                        subject_uid=patient_dir.name,
                        pipeline_id=pipeline_id,
                        background_phase_correction=background_phase_correction,
                        bg_poly_order=bg_poly_order,
                        bg_static_percentile=bg_static_percentile,
                        dicom_search_dir=dicom_search_dir,
                    )
                )
            return written

        return process_patient(
            source,
            venc=venc,
            dry_run=dry_run,
            subject_uid=source.name,
            pipeline_id=pipeline_id,
            background_phase_correction=background_phase_correction,
            bg_poly_order=bg_poly_order,
            bg_static_percentile=bg_static_percentile,
            dicom_search_dir=dicom_search_dir,
        )
    except Exception as exc:
        log.exception(exc)
        raise exc


@_click_command()
@_click_option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path) if click is not None else None,
    required=True,
    help="Patient directory or directory containing multiple patient folders.",
)
@_click_option("--multifile", is_flag=True, help="Process each direct child directory as a separate patient.")
@_click_option("--venc", type=float, default=700.0, show_default=True, help="Fallback VENC in mm/s when metadata is missing.")
@_click_option(
    "--dicom-dir",
    "dicom_search_dir",
    type=click.Path(exists=True, path_type=Path) if click is not None else None,
    default=None,
    help="Optional subject DICOM folder to read VENC (0018,9217) when JSON/NIfTI lack it.",
)
@_click_option(
    "--background-phase-correction/--no-background-phase-correction",
    is_flag=True,
    default=True,
    show_default=True,
    help="Polynomial spatial background on mean velocity; subtract from each frame (QVTplus-style).",
)
@_click_option("--bg-poly-order", type=int, default=2, show_default=True)
@_click_option("--bg-static-percentile", type=float, default=25.0, show_default=True)
@_click_option("--dry-run", is_flag=True, help="Only report the discovered inputs.")
@_click_option(
    "--pipeline-id",
    "pipeline_id_opt",
    type=str,
    default=None,
    help="Pipeline identifier stored with registered derivative assets (default: 1.0.0).",
)
@_click_option(
    "--pipeline-version",
    "pipeline_version_legacy",
    type=str,
    default=None,
    hidden=True,
    help="Deprecated; use --pipeline-id.",
)
def main(
    input_path: Path,
    multifile: bool,
    venc: float,
    dicom_search_dir: Path | None,
    background_phase_correction: bool,
    bg_poly_order: int,
    bg_static_percentile: float,
    dry_run: bool,
    pipeline_id_opt: str | None,
    pipeline_version_legacy: str | None,
) -> None:
    if click is None:
        raise BackendUnavailableError('click is not installed. Please install it with "pip install click".')

    pipeline_id = (pipeline_id_opt or pipeline_version_legacy or "1.0.0").strip()
    outputs = phase2volume(
        input_path,
        multifile=multifile,
        venc=venc,
        dry_run=dry_run,
        pipeline_id=pipeline_id,
        background_phase_correction=background_phase_correction,
        bg_poly_order=bg_poly_order,
        bg_static_percentile=bg_static_percentile,
        dicom_search_dir=dicom_search_dir,
    )
    for output in outputs:
        click.echo(str(output))


if __name__ == "__main__":
    main()
