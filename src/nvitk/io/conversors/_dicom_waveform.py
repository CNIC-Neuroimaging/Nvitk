"""
DICOM waveforms (ECG) — export as signals, not as images.

Description
-----------
A gated cardiac CT ships the patient's ECG as a *General ECG Waveform* object
(SOP 1.2.840.10008.5.1.4.1.1.9.1.2; Philips: series ``For Series: 401 - 40997``). Treated as an
image it became an 8x8 NIfTI of the vendor's icon. What it actually holds is the trace the
scanner gated on — one lead, 250 Hz, a few seconds either side of the acquisition — which is
how a heart rate, an arrhythmia, or a premature beat during the scan is checked afterwards.

:func:`export_waveforms` writes, per waveform object:

- ``<name>.csv`` — ``time_s`` plus one column per channel (scaled by the channel sensitivity
  when the file records one, raw ADC units otherwise);
- ``<name>.json`` — sampling frequency, channel labels/units, the absolute start time (so the
  trace can be aligned with the CT's acquisition time), detected R-peak times, RR intervals and
  the median heart rate.

R-peaks come from a deliberately simple detector (polarity by the larger excursion, peaks at
least 0.33 s apart above 50 % of the robust amplitude). It is a QC aid, not a diagnostic one.

I/O: pydicom datasets in, CSV + JSON out. NumPy/SciPy on the host.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from nvitk.core.logger import Logger

log = Logger()

#: SOP Class UID prefix shared by every DICOM waveform IOD (ECG, haemodynamic, audio, …).
WAVEFORM_SOP_PREFIX = "1.2.840.10008.5.1.4.1.1.9."


def is_waveform(ds: Any) -> bool:
    """True for any DICOM waveform object."""
    sop = str(getattr(ds, "SOPClassUID", "") or "")
    return sop.startswith(WAVEFORM_SOP_PREFIX) or "WaveformSequence" in ds


def _start_time(ds: Any, group: Any) -> str | None:
    """ISO start time of a multiplex group (acquisition date-time + group offset)."""
    text = str(ds.get("AcquisitionDateTime", "") or "").strip()
    if len(text) < 14:
        date = str(ds.get("AcquisitionDate", ds.get("ContentDate", "")) or "")
        tm = str(ds.get("AcquisitionTime", ds.get("ContentTime", "")) or "")
        text = (date + tm) if len(date) == 8 and len(tm) >= 6 else ""
    if len(text) < 14:
        return None
    try:
        base = datetime.strptime(text[:14], "%Y%m%d%H%M%S")
        offset_ms = float(group.get("MultiplexGroupTimeOffset", 0) or 0)
        start = datetime.fromtimestamp(base.timestamp() + offset_ms / 1000.0)
        return start.isoformat(timespec="milliseconds")
    except (ValueError, OverflowError, OSError):
        return None


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Sample indices of R-peaks in a single ECG lead (simple, polarity-aware)."""
    from scipy.signal import find_peaks

    x = np.asarray(signal, dtype=float)
    x = x - np.median(x)
    # QRS points whichever way the lead was placed; follow the larger excursion.
    if abs(np.percentile(x, 0.5)) > abs(np.percentile(x, 99.5)):
        x = -x
    amplitude = np.percentile(x, 99.5)
    if amplitude <= 0:
        return np.zeros(0, dtype=int)
    peaks, _ = find_peaks(x, height=0.5 * amplitude, distance=max(int(0.33 * fs), 1))
    return peaks


def waveform_channels(ds: Any, index: int = 0) -> tuple[np.ndarray, float, list[str], list[str]]:
    """``(samples x channels array, sampling Hz, labels, units)`` for multiplex group *index*."""
    group = ds.WaveformSequence[index]
    fs = float(group.SamplingFrequency)
    data = np.asarray(ds.waveform_array(index), dtype=float)
    labels: list[str] = []
    units: list[str] = []
    for i, ch in enumerate(group.ChannelDefinitionSequence):
        label = str(ch.get("ChannelLabel") or "")
        if not label and "ChannelSourceSequence" in ch:
            label = str(ch.ChannelSourceSequence[0].get("CodeMeaning", "") or "")
        labels.append(label or f"channel_{i + 1}")
        unit = ""
        if "ChannelSensitivityUnitsSequence" in ch and ch.get("ChannelSensitivity") not in (None, ""):
            unit = str(ch.ChannelSensitivityUnitsSequence[0].get("CodeValue", "") or "")
        units.append(unit or "raw")
    return data, fs, labels, units


def export_waveforms(ds_list: Sequence[Any], output_base: str) -> list[str]:
    """
    Write every multiplex group of every waveform object in *ds_list* as CSV + JSON.

    Parameters
    ----------
    ds_list
        Waveform datasets of one series.
    output_base
        Path without extension; suffixes ``_<n>`` are added when there is more than one group.

    Returns
    -------
    list of str
        The CSV files written (each has a JSON beside it).
    """
    written: list[str] = []
    groups = [(ds, i) for ds in ds_list for i in range(len(ds.get("WaveformSequence", [])))]
    for n, (ds, index) in enumerate(groups):
        data, fs, labels, units = waveform_channels(ds, index)
        base = output_base if len(groups) == 1 else f"{output_base}_{n + 1}"
        times = np.arange(data.shape[0], dtype=float) / fs
        # ---- CSV: time plus one column per channel ---------------------------
        header = "time_s," + ",".join(
            f"{lab}_{unit}" if unit != "raw" else lab for lab, unit in zip(labels, units)
        )
        table = np.column_stack([times, data])
        np.savetxt(f"{base}.csv", table, delimiter=",", header=header, comments="", fmt="%.6g")
        # ---- JSON: acquisition context and a heart-rate summary --------------
        group = ds.WaveformSequence[index]
        info: dict[str, Any] = {
            "sop_class": str(ds.SOPClassUID),
            "series_number": str(ds.get("SeriesNumber", "")),
            "series_description": str(ds.get("SeriesDescription", "")),
            "sampling_frequency_hz": fs,
            "n_samples": int(data.shape[0]),
            "duration_s": float(data.shape[0] / fs),
            "channels": [{"label": lab, "units": unit} for lab, unit in zip(labels, units)],
            "start_time": _start_time(ds, group),
            "multiplex_group_time_offset_ms": float(group.get("MultiplexGroupTimeOffset", 0) or 0),
        }
        lead = data[:, 0]
        peaks = detect_r_peaks(lead, fs)
        if peaks.size >= 2:
            rr = np.diff(peaks) / fs
            info.update(
                {
                    "r_peak_times_s": [round(float(p / fs), 4) for p in peaks],
                    "rr_intervals_s": [round(float(v), 4) for v in rr],
                    "heart_rate_bpm_median": round(float(60.0 / np.median(rr)), 2),
                    "heart_rate_bpm_range": [round(float(60.0 / rr.max()), 1), round(float(60.0 / rr.min()), 1)],
                    "heart_rate_method": "R-peaks: polarity by larger excursion, >=0.33 s apart, >50% of robust amplitude",
                }
            )
        with open(f"{base}.json", "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2)
        written.append(f"{base}.csv")
        hr = info.get("heart_rate_bpm_median")
        log.info(
            "ECG waveform → %s (%d samples @ %g Hz, %.1f s%s)",
            os.path.basename(f"{base}.csv"), data.shape[0], fs, data.shape[0] / fs,
            f", HR ≈ {hr:g} bpm" if hr else "",
        )
    return written


__all__ = ["WAVEFORM_SOP_PREFIX", "detect_r_peaks", "export_waveforms", "is_waveform", "waveform_channels"]
