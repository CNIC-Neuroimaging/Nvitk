"""Phase-contrast / cine MRI-style tabular pipelines to 3D/4D volumes (CLI optional)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import click
except Exception:
    click = None

from nvitk.core.exceptions import BackendUnavailableError
from nvitk.core.array import as_backend_array
from .._common import default_nifti_axes
from ..imageio import imread, imsave
from nvitk.core.logger import Logger

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


def read_venc_from_metadata(ap_dir: Path, default_venc: float) -> float:
    json_paths = sorted(ap_dir.glob("*.json"))
    for json_path in json_paths:
        try:
            with json_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            continue

        for field_name in ("VelocityEncoding", "PhaseEncodingVelocity"):
            if field_name in payload:
                return float(payload[field_name]) * 10.0
        if "VENC" in payload:
            return float(payload["VENC"])
    return float(default_venc)


def compute_phase_derivatives(
    angio_data: np.ndarray,
    ap_phase: np.ndarray,
    rl_phase: np.ndarray,
    fh_phase: np.ndarray,
    *,
    venc: float,
) -> dict[str, np.ndarray]:
    if not (ap_phase.shape == rl_phase.shape == fh_phase.shape):
        raise ValueError("AP, RL, and FH phase volumes must share the same shape.")

    if angio_data.shape[:3] != ap_phase.shape[:3]:
        raise ValueError("Angiography and phase volumes must match in spatial dimensions.")

    angio_tr = angio_data
    angio_mean = np.mean(angio_data, axis=-1) if angio_data.ndim == 4 else angio_data

    vx = -rl_phase * 10.0
    vy = -ap_phase * 10.0
    vz = fh_phase * 10.0

    if vx.ndim == 4:
        velocity = np.stack([vx, vy, vz], axis=-2)
        v_mag = np.sqrt(np.sum(velocity**2, axis=3))
    else:
        velocity = np.stack([vx, vy, vz], axis=-1)
        v_mag = np.sqrt(np.sum(velocity**2, axis=3))

    v_mag_capped = np.clip(v_mag, 0, venc)

    if v_mag_capped.ndim == 4:
        if angio_tr.ndim == 3:
            angio_tr = np.broadcast_to(np.expand_dims(angio_tr, axis=-1), v_mag_capped.shape)
        elif angio_tr.shape[3] != v_mag_capped.shape[3]:
            min_t = min(angio_tr.shape[3], v_mag_capped.shape[3])
            angio_tr = angio_tr[..., :min_t]
            v_mag_capped = v_mag_capped[..., :min_t]
            v_mag = v_mag[..., :min_t]
        cd = angio_tr * np.sin((np.pi / 2.0 * v_mag_capped) / venc)
        v_mag_mean = np.mean(v_mag, axis=-1)
        v_mag_mean_capped = np.clip(v_mag_mean, 0, venc)
        cd_mean = angio_mean * np.sin((np.pi / 2.0 * v_mag_mean_capped) / venc)
    else:
        if angio_tr.ndim == 4:
            angio_tr = np.mean(angio_tr, axis=-1)
        cd = angio_tr * np.sin((np.pi / 2.0 * v_mag_capped) / venc)
        v_mag_mean = v_mag
        cd_mean = cd

    return {
        "Angiography_4D": angio_tr if angio_tr.ndim == 4 else None,
        "Angiography_3D": angio_mean,
        "ComplexDifference_4D": cd if cd.ndim == 4 else None,
        "VelocityMagnitude_4D": v_mag if v_mag.ndim == 4 else None,
        "ComplexDifference_3D": cd_mean,
        "VelocityMagnitude_3D": v_mag_mean,
    }


def _write_outputs(flow_dir: Path, outputs: dict[str, np.ndarray], metadata: dict[str, Any]) -> list[Path]:
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
) -> list[Path]:
    inputs = discover_phase_inputs(patient_dir)
    actual_venc = read_venc_from_metadata(inputs.ap_dir, venc)
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

    outputs = compute_phase_derivatives(
        angio_image.data,
        ap_image.data,
        rl_image.data,
        fh_image.data,
        venc=actual_venc,
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
) -> list[Path]:
    from nvitk import using
    with using('cpu'):
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
                        )
                    )
                return written

            return process_patient(
                source,
                venc=venc,
                dry_run=dry_run,
                subject_uid=source.name,
                pipeline_id=pipeline_id,
            )
        except Exception as exc:
            log.exception(exc)
            raise exc


@_click_command()
@_click_option("-i", "--input", "input_path", type=click.Path(exists=True, path_type=Path) if click is not None else None, required=True, help="Patient directory or directory containing multiple patient folders.")
@_click_option("--multifile", is_flag=True, help="Process each direct child directory as a separate patient.")
@_click_option("--venc", type=float, default=700.0, show_default=True, help="Fallback VENC in mm/s when metadata is missing.")
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
    )
    for output in outputs:
        click.echo(str(output))


if __name__ == "__main__":
    main()
