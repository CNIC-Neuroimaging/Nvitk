"""
Standardized Uptake Value (SUV) utilities for PET.

Port of [BioImaging _pet.py](/home/imarcoss/BioImaging/src/imaging/measure/_pet.py)
with the following fixes:

- Drops the stray ``from curses import raw`` import.
- Makes the Philips SUV scale factor path consistent between :func:`suv_image`
  and :func:`suv_stats` (the legacy ``compute_pet_metrics`` hard-coded
  ``philips_factor = None``).
- Uses :class:`~nvitk.types.Image` as the public input type.

Backend policy
--------------
Arithmetic on voxel data (``activity * factor``, per-slice rescaling,
statistics) stays on the active backend. ``to_numpy`` is only invoked when
emitting Python floats at the final stat step.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple, Union

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup
from nvitk.types import Image

from ._common import bool_mask, ensure_same_shape, resolve_array

setup(globals())


SERIES_TIME_TAGS = ["SeriesTime", "(0008,0031)"]
SERIES_DATE_TAGS = ["SeriesDate", "(0008,0021)"]
PATIENT_WEIGHT_TAGS = ["PatientWeight", "(0010,1030)"]
PATIENT_SEX_TAGS = ["PatientSex", "(0010,0040)"]
PATIENT_SIZE_TAGS = ["PatientSize", "(0010,1020)"]
RADIOPHARM_START_TIME_TAGS = ["RadiopharmaceuticalStartTime", "(0018,1072)"]
RADIONUCLIDE_HALF_LIFE_TAGS = ["RadionuclideHalfLife", "(0018,1075)"]
RADIONUCLIDE_TOTAL_DOSE_TAGS = ["RadionuclideTotalDose", "(0018,1074)"]
DECAY_CORRECTION_TAGS = ["DecayCorrection", "(0054,1102)"]
DECAY_FACTOR_TAGS = ["DecayFactor", "(0054,1321)"]
FRAME_REFERENCE_TIME_TAGS = ["FrameReferenceTime", "(0054,1300)"]
CORRECTED_IMAGE_TAGS = ["CorrectedImage", "(0028,0x0051)"]
PRIVATE_CREATOR_TAGS = ["Private Creator", "(7053,1000)"]
PHILIPS_SUV_FACTOR_TAGS = ["[SUV Scale Factor]", "(7053,1000)"]
DOSE_UNITS_TAGS = ["DoseUnits", "(0054,1004)"]
UNITS_TAGS = ["Units", "(0054,1001)"]
RESCALE_SLOPE_TAGS = ["RescaleSlope", "slope", "(0028,1053)"]
RESCALE_INTERCEPT_TAGS = ["RescaleIntercept", "intercept", "(0028,1052)"]

SUV_KIND_ALIASES = {
    "bw": "SUVbw",
    "SUVbw": "SUVbw",
    "lbm": "SUVlbm",
    "SUVlbm": "SUVlbm",
    "bsa": "SUVbsa",
    "SUVbsa": "SUVbsa",
    "ibw": "SUVibw",
    "SUVibw": "SUVibw",
}


def _normalize_kind(kind: str) -> str:
    try:
        return SUV_KIND_ALIASES[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown SUV kind '{kind}'. Accepts bw/lbm/bsa/ibw or SUVbw/SUVlbm/SUVbsa/SUVibw."
        ) from exc


def _get_dicom_value(
    metadata: Dict[str, Any],
    keys: Union[str, Tuple, List[Union[str, Tuple]]],
    default: Any = None,
    required: bool = False,
    name: str = "DICOM tag",
) -> Any:
    if not isinstance(keys, list):
        keys = [keys]
    for key in keys:
        try:
            if metadata.get(key) is not None:
                value = metadata[key]
                if hasattr(value, "value"):
                    return value.value
                return value
        except (TypeError, AttributeError):
            continue
    if required:
        raise ValueError(f"Required {name} not found in metadata. Looked for keys: {keys}")
    return default


def _activity2bq(dose: float, units: str) -> float:
    u = str(units).upper()
    if "MBQ" in u:
        return dose * 1e6
    if "KBQ" in u:
        return dose * 1e3
    if "BQ" in u:
        return dose
    if "MCI" in u:
        return dose * 3.7e7
    if "UCI" in u:
        return dose * 3.7e4
    if "KCI" in u:
        return dose * 3.7e13
    if "CI" in u:
        return dose * 3.7e10
    warnings.warn(f"Unknown dose units '{units}'. Assuming Bq.", UserWarning)
    return dose


def _parse_datetime(value: Any, series_date: str | None = None) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    if isinstance(value, str):
        date_str = series_date or datetime.now().strftime("%Y%m%d")
        for fmt in (
            "%Y-%m-%dT%H:%M:%S",
            "%Y%m%d%H%M%S",
            "%Y%m%d %H:%M:%S",
            f"{date_str}%H%M%S.%f",
            f"{date_str}%H%M%S",
            f"{date_str}%H:%M:%S.%f",
            f"{date_str}%H:%M:%S",
        ):
            try:
                if "%" not in fmt[:8]:
                    return datetime.strptime(value, fmt[len(date_str):])
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
    raise ValueError(f"Unable to parse datetime from {value!r}")


def _suv_factor(metadata: Dict[str, Any], kind: str = "SUVbw") -> float:
    half_life = float(_get_dicom_value(metadata, RADIONUCLIDE_HALF_LIFE_TAGS, required=True))
    total_dose = float(_get_dicom_value(metadata, RADIONUCLIDE_TOTAL_DOSE_TAGS, required=True))
    dose_units = _get_dicom_value(metadata, DOSE_UNITS_TAGS, "Bq")
    total_dose_bq = _activity2bq(total_dose, dose_units)

    series_date = _get_dicom_value(metadata, SERIES_DATE_TAGS, required=True)
    series_time = _get_dicom_value(metadata, SERIES_TIME_TAGS, required=True)
    inj_time_str = _get_dicom_value(metadata, RADIOPHARM_START_TIME_TAGS, required=True)

    acq_time = _parse_datetime(series_time, series_date=series_date)
    inj_time = _parse_datetime(inj_time_str, series_date=series_date)
    if acq_time < inj_time:
        warnings.warn(
            f"Acquisition time ({acq_time}) is before injection time ({inj_time}). "
            "Assuming they are on the same day."
        )

    dt_seconds = (acq_time - inj_time).total_seconds()
    if dt_seconds < 0:
        raise ValueError(f"Negative decay time ({dt_seconds}s). Check acquisition and injection times.")

    decayed_dose_bq = total_dose_bq * (2.0 ** (-dt_seconds / half_life))
    if decayed_dose_bq == 0:
        raise ValueError("Decayed dose is zero, cannot compute SUV factor.")

    weight_kg = float(
        _get_dicom_value(metadata, PATIENT_WEIGHT_TAGS, required=True, name="PatientWeight")
    )

    if kind == "SUVbw":
        normalization = weight_kg * 1000.0
    elif kind in ("SUVlbm", "SUVbsa", "SUVibw"):
        height_m = float(
            _get_dicom_value(metadata, PATIENT_SIZE_TAGS, required=True, name="PatientSize")
        )
        height_cm = height_m * 100.0
        if kind == "SUVbsa":
            bsa_m2 = 0.007184 * (weight_kg ** 0.425) * (height_cm ** 0.725)
            normalization = bsa_m2 * 10000.0
        else:
            sex = _get_dicom_value(
                metadata, PATIENT_SEX_TAGS, required=True, name="PatientSex"
            ).upper()
            if sex not in ("M", "F"):
                raise ValueError(f"PatientSex must be 'M' or 'F' for {kind}, got {sex!r}.")
            if kind == "SUVlbm":
                if sex == "M":
                    lbm_kg = 1.10 * weight_kg - 128 * ((weight_kg / height_cm) ** 2)
                else:
                    lbm_kg = 1.07 * weight_kg - 148 * ((weight_kg / height_cm) ** 2)
                normalization = lbm_kg * 1000.0
            else:  # SUVibw
                if sex == "M":
                    ibw_kg = 48.0 + 1.06 * (height_cm - 152)
                else:
                    ibw_kg = 45.5 + 0.91 * (height_cm - 152)
                if ibw_kg > weight_kg:
                    ibw_kg = weight_kg
                normalization = ibw_kg * 1000.0
    else:
        raise ValueError(f"Unknown SUV kind '{kind}'.")

    return float(normalization / decayed_dose_bq)


def _apply_per_slice_rescale(raw: Any, metadata: Dict[str, Any], z_dim: int = 2) -> Any:
    slopes = metadata.get("RescaleSlope", [])
    intercepts = metadata.get("RescaleIntercept", [])
    if not slopes or not intercepts:
        slope = float(_get_dicom_value(metadata, RESCALE_SLOPE_TAGS, 1.0))
        intercept = float(_get_dicom_value(metadata, RESCALE_INTERCEPT_TAGS, 0.0))
        return (raw - intercept) / slope
    if len(slopes) != len(intercepts):
        warnings.warn(
            f"Mismatch rescale lengths ({len(slopes)} vs {len(intercepts)}). Returning raw."
        )
        return raw
    if raw.ndim != 3:
        slope = float(_get_dicom_value(metadata, RESCALE_SLOPE_TAGS, 1.0))
        intercept = float(_get_dicom_value(metadata, RESCALE_INTERCEPT_TAGS, 0.0))
        return (raw - intercept) / slope
    n_slices = raw.shape[z_dim]
    if n_slices != len(slopes):
        warnings.warn(f"Image has {n_slices} slices but {len(slopes)} slopes. Returning raw.")
        return raw
    out = raw.copy()
    for z in range(n_slices):
        sl = float(slopes[z])
        it = float(intercepts[z])
        if z_dim == 0:
            out[z, :, :] = (raw[z, :, :] - it) / sl
        elif z_dim == 2:
            out[:, :, z] = (raw[:, :, z] - it) / sl
        else:
            out[:, z, :] = (raw[:, z, :] - it) / sl
    return out


def suv_image(
    pet: Image | Any,
    metadata: Dict[str, Any] | None = None,
    *,
    kind: str = "bw",
    philips: bool = False,
    revert_scaling: bool = False,
) -> Image | Any:
    """
    Convert a raw PET image to an SUV image.

    Parameters
    ----------
    pet
        PET image. If an :class:`Image`, *metadata* defaults to ``pet.metadata``.
    metadata
        DICOM-style metadata. Required if *pet* is a bare array.
    kind
        ``bw``, ``lbm``, ``bsa``, or ``ibw`` (also accepts legacy ``SUVbw``...).
    philips
        If True (default) and the Philips ``(7053,1000)`` SUV factor is present,
        use that as a direct multiplier. Set to False to always compute from
        half-life/weight.
    revert_scaling
        If True, undo scanner-applied ``RescaleSlope/Intercept`` per slice before
        applying the SUV factor.

    Returns
    -------
    Image | ndarray
        SUV image. Matches the input type (``Image`` in -> ``Image`` out).
    """
    kind = _normalize_kind(kind)
    if metadata is None:
        if isinstance(pet, Image):
            metadata = pet.metadata or {}
        else:
            raise ValueError("metadata is required when pet is a bare array.")

    raw = as_backend_array(resolve_array(pet))
    activity = _apply_per_slice_rescale(raw, metadata) if revert_scaling else raw

    if philips:
        philips_factor = _get_dicom_value(metadata, PHILIPS_SUV_FACTOR_TAGS)
        if philips_factor is not None:
            if kind != "SUVbw":
                warnings.warn(
                    f"Philips SUV factor is only valid for SUVbw; requested {kind}. "
                    "Ignoring Philips factor and computing from DICOM tags.",
                    UserWarning,
                )
            else:
                out = activity * float(philips_factor)
                return pet.with_data(out) if isinstance(pet, Image) else out

    factor = _suv_factor(metadata, kind)
    out = activity * factor
    return pet.with_data(out) if isinstance(pet, Image) else out


def _f(x: Any) -> float:
    """Final scalar -> Python float (host hop)."""
    arr = to_numpy(x)
    return float(arr) if arr.ndim == 0 else float(arr.item())


def _stats_from_values(values: Any, kind: str, stats: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    requested = [s.lower() for s in stats]
    if "mean" in requested:
        out[f"{kind}_mean"] = _f(np.mean(values))
    if "median" in requested:
        out[f"{kind}_median"] = _f(np.median(values))
    if "max" in requested:
        out[f"{kind}_max"] = _f(np.max(values))
    if "min" in requested:
        out[f"{kind}_min"] = _f(np.min(values))
    if "std" in requested:
        out[f"{kind}_std"] = _f(np.std(values))
    if "p95" in requested:
        out[f"{kind}_95percentile"] = _f(np.percentile(values, 95))
    if "p5" in requested:
        out[f"{kind}_5percentile"] = _f(np.percentile(values, 5))
    if "sum" in requested:
        out[f"{kind}_sum"] = _f(np.sum(values))
    return out


def suv_stats(
    pet: Image | Any,
    mask: Image | Any,
    metadata: Dict[str, Any] | None = None,
    *,
    kinds: Iterable[str] = ("bw",),
    stats: Iterable[str] = ("mean", "median", "max", "min", "std", "p95", "p5"),
    philips: bool = False,
    revert_scaling: bool = False,
) -> Dict[str, float]:
    """
    Compute SUV statistics restricted to ``mask > 0``.

    Parameters
    ----------
    pet
        PET image (raw scanner counts unless *revert_scaling* is used).
    mask
        Mask image/array with same shape as *pet*.
    metadata
        DICOM-style metadata. Defaults to ``pet.metadata`` when available.
    kinds
        Iterable of SUV kinds (``bw``/``lbm``/``bsa``/``ibw``).
    stats
        Per-kind stats to compute (same names as :func:`nvitk.measure.masked_stats`).
    philips
        Enable the Philips ``(7053,1000)`` short-circuit for SUVbw.
    revert_scaling
        If True, undo per-slice rescaling before applying the SUV factor.

    Returns
    -------
    dict[str, float]
        ``{'SUVbw_mean': ..., 'SUVbw_max': ..., 'SUVlbm_mean': ...}``.
    """
    ensure_same_shape(pet, mask)
    if metadata is None:
        if isinstance(pet, Image):
            metadata = pet.metadata or {}
        else:
            raise ValueError("metadata is required when pet is a bare array.")

    m = bool_mask(mask)
    if not bool(m.any()):
        raise ValueError("Segmentation mask is empty.")

    raw = as_backend_array(resolve_array(pet))
    activity = _apply_per_slice_rescale(raw, metadata) if revert_scaling else raw

    normalized_kinds = [_normalize_kind(k) for k in kinds]
    all_metrics: Dict[str, float] = {}

    philips_factor = _get_dicom_value(metadata, PHILIPS_SUV_FACTOR_TAGS) if philips else None

    if philips_factor is not None:
        if "SUVbw" in normalized_kinds:
            suv = activity * float(philips_factor)
            all_metrics.update(_stats_from_values(suv[m], "SUVbw", stats))
        other_kinds = [k for k in normalized_kinds if k != "SUVbw"]
        if other_kinds:
            warnings.warn(
                f"Philips SUV factor only yields SUVbw; {other_kinds} will be "
                "computed from half-life/weight instead.",
                UserWarning,
            )
            normalized_kinds = other_kinds
        else:
            normalized_kinds = []

    for kind in normalized_kinds:
        try:
            factor = _suv_factor(metadata, kind)
            suv = activity * factor
            all_metrics.update(_stats_from_values(suv[m], kind, stats))
        except Exception as exc:
            warnings.warn(f"Could not compute SUV metrics for {kind}: {exc}", UserWarning)

    if not all_metrics:
        raise ValueError("Could not compute any SUV metric. Check metadata and warnings.")

    return all_metrics


__all__ = ["suv_image", "suv_stats"]
