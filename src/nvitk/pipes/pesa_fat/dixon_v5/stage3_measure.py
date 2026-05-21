"""Dixon v5 stage 3 (per-subject): volume, FF, T2*, R2* and WF measurements.

Port of :code:`BioImaging/src/pesa_fat/dixon/3_measure_vol_int.py`.

For a PESA* subject with stage-2 outputs (``HEAD``, ``THORAX``, ``LEGS``),
computes per label and per region:

* ``VOL`` - volume in cc (voxel count * prod(spacing) / 1000)
* ``FF``  - mean of the Dixon fat-fraction map inside the mask
* ``T2``  - mean of the T2* map inside the mask
* ``R2``  - mean of the reciprocal T2* map (``1/T2`` where T2 > 0)
* ``WF``  - for liver only: ``(liver_water_mean / pvm_water_mean) * 100``.
            The reference is the mean Dixon water signal pooled over the
            bilateral paravertebral muscle (``T_PVM_L`` + ``T_PVM_R``) in
            the THORAX region. See ``cfg.WF_REFERENCE_LABELS``.

Writes a single-row Excel file per subject under
``RESULTS/<batch>/res_measure_dixon/per_subject/<SUBJECT>.xlsx``; the batch
master aggregates these into ``<batch>_SummaryCodebook.xlsx``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pandas as pd

from nvitk.core.array import to_numpy, as_backend_array
from nvitk.core.click_backend import backend_click_option, set_default_backend
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.io import imread
from nvitk.measure import volume_cc
from nvitk.pipes.pesa_fat.common.paths import (
    BatchLayout,
    layout,
    resolve_nii_optional,
)
from nvitk.pipes.pesa_fat.dixon_v5 import config as cfg
from nvitk.pipes.pesa_fat.dixon_v5.labels import (
    HEAD_LABELS,
    LEGS_LABELS,
    THORAX_LABELS,
)
from nvitk.segmentation.labels import get_label
from nvitk.types import Image

setup(globals())

log = Logger()


# ---------------------------------------------------------------------------
# Measurement plan (single source of truth in cfg.MEASURE_SPECS)
# ---------------------------------------------------------------------------

SPECS = cfg.MEASURE_SPECS


def column_order() -> list[str]:
    cols = ["pesa_id"]
    for spec in SPECS:
        for metric in spec.metrics:
            cols.append(f"{spec.prefix}_{metric}")
    return cols


_REGION_LABELS: dict[str, dict[str, int]] = {
    "HEAD": HEAD_LABELS,
    "THORAX": THORAX_LABELS,
    "LEGS": LEGS_LABELS,
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _imread_opt(parent: Path, stem: str) -> Image | None:
    path = resolve_nii_optional(parent, stem)
    if path is None:
        return None
    return imread(str(path), axes="XYZ")


def _load_maps(subject_nifti_dir: Path, region: str) -> dict[str, Image | None]:
    out: dict[str, Image | None] = {}
    for suffix, key in (
        ("FAT_FRACTION", "FF"),
        ("T2STAR", "T2"),
        ("WATER", "WATER"),
    ):
        out[key] = _imread_opt(subject_nifti_dir, f"{cfg.INPUT_PREFIX}_{region}_{suffix}")
    return out


def _mean_under_mask(values: Any, mask: Any) -> float:
    vals_np = to_numpy(values)
    mask_np = to_numpy(mask) > 0
    if not mask_np.any():
        return float("nan")
    sel = vals_np[mask_np]
    if sel.size == 0:
        return float("nan")
    return float(np.mean(sel))


def _binary_mask(label_img: Image, label_ids: tuple[int, ...]) -> Image:
    if len(label_ids) == 1:
        return get_label(label_img, label_ids[0], missing="empty")
    acc = get_label(label_img, label_ids[0], missing="empty").data.copy()
    for lid in label_ids[1:]:
        extra = get_label(label_img, lid, missing="empty").data
        acc[extra > 0] = 1
    return label_img.with_data(acc.astype("uint8"))


def _nslices_axial_xyz(mask: Image) -> int:
    return int(np.any(mask.data > 0, axis=(0, 1)).sum())

# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------


def _wf_reference(
    region_masks: dict[str, Image | None],
    region_waters: dict[str, Image | None],
) -> float | None:
    """Pooled mean of the water-map signal over ``cfg.WF_REFERENCE_LABELS``.

    All voxels from every referenced (region, label) are concatenated into a
    single pool, and the arithmetic mean of that pool is returned. This
    matches the definition
    ``WF_ref = mean(water[ union over PVM_L, PVM_R ])``
    rather than averaging per-mask means, so unbalanced mask sizes stay
    properly weighted.
    """
    total_sum = 0.0
    total_count = 0
    for region, _mask_file, label_key in cfg.WF_REFERENCE_LABELS:
        mask_img = region_masks.get(region)
        water_img = region_waters.get(region)
        if mask_img is None or water_img is None:
            continue
        label_id = _REGION_LABELS[region][label_key]
        m_np = (as_backend_array(get_label(mask_img, label_id, missing="empty").data) > 0).astype(bool)
        if not m_np.any():
            continue
        vals = as_backend_array(water_img.data)[m_np]
        total_sum += float(np.sum(vals))
        total_count += int(vals.size)

    if total_count == 0:
        return None
    return total_sum / total_count


def process_subject(
    subject_nifti_dir: Path,
    subject_stage2_dir: Path,
) -> dict[str, Any]:
    """Compute all stage-3 columns for one subject."""
    mask_head = _imread_opt(subject_stage2_dir, "HEAD")
    mask_thorax = _imread_opt(subject_stage2_dir, "THORAX")
    mask_legs = _imread_opt(subject_stage2_dir, "LEGS")

    maps = {
        "HEAD": _load_maps(subject_nifti_dir, "HEAD"),
        "THORAX": _load_maps(subject_nifti_dir, "THORAX"),
        "LEGS": _load_maps(subject_nifti_dir, "LEGS"),
    }

    region_masks: dict[str, Image | None] = {
        "HEAD": mask_head,
        "THORAX": mask_thorax,
        "LEGS": mask_legs,
    }

    wf_ref = _wf_reference(
        region_masks=region_masks,
        region_waters={
            region: maps[region].get("WATER") for region in maps
        },
    )

    row: dict[str, Any] = {}
    for spec in SPECS:
        label_img = region_masks.get(spec.region)
        if label_img is None:
            for metric in spec.metrics:
                row[f"{spec.prefix}_{metric}"] = None
            continue

        try:
            bm = _binary_mask(label_img, spec.label_ids)

            if "VOL" in spec.metrics:
                row[f"{spec.prefix}_VOL"] = (
                    volume_cc(bm) if bool(bm.data.any()) else 0.0
                )

            ff_map = maps[spec.region].get("FF")
            t2_map = maps[spec.region].get("T2")
            water_map = maps[spec.region].get("WATER")

            if "FF" in spec.metrics and ff_map is not None:
                row[f"{spec.prefix}_FF"] = _mean_under_mask(ff_map.data, bm.data)
            elif "FF" in spec.metrics:
                row[f"{spec.prefix}_FF"] = None

            if "T2" in spec.metrics and t2_map is not None:
                row[f"{spec.prefix}_T2"] = _mean_under_mask(t2_map.data, bm.data)
            elif "T2" in spec.metrics:
                row[f"{spec.prefix}_T2"] = None

            if "R2" in spec.metrics and t2_map is not None:
                t2_np = to_numpy(t2_map.data).astype("float32")
                r2_np = np.zeros_like(t2_np, dtype="float32")
                pos = t2_np > 0
                r2_np[pos] = 1.0 / t2_np[pos] * 1000.0 # convert to Hz
                row[f"{spec.prefix}_R2"] = _mean_under_mask(r2_np, bm.data)
            elif "R2" in spec.metrics:
                row[f"{spec.prefix}_R2"] = None

            if "WF" in spec.metrics:
                if water_map is None or wf_ref is None or wf_ref == 0:
                    row[f"{spec.prefix}_WF"] = None
                else:
                    tissue_water = _mean_under_mask(water_map.data, bm.data)
                    if np.isnan(tissue_water):
                        row[f"{spec.prefix}_WF"] = None
                    else:
                        row[f"{spec.prefix}_WF"] = float((tissue_water / wf_ref) * 100.0)

            if "NSlices" in spec.metrics:
                row[f"{spec.prefix}_NSlices"] = _nslices_axial_xyz(bm)
        except Exception as exc:
            log.error(f"spec {spec.prefix} failed: {exc}")
            for metric in spec.metrics:
                row[f"{spec.prefix}_{metric}"] = None

    return row


def run_subject(
    subject: str,
    lay: BatchLayout,
    *,
    backend: str = "cupy",
    output: Path | None = None,
) -> Path:
    """Run stage 3 for a single subject and write a one-row Excel file."""
    try:
        set_default_backend(backend, allow_fallback=True)
    except Exception as exc:
        log.warning(f"Backend '{backend}' unavailable, falling back: {exc}")

    stage2_dir = lay.results_dir / cfg.STAGE2_DIR / subject
    if not stage2_dir.exists():
        raise FileNotFoundError(f"Expected stage-2 outputs under {stage2_dir}")

    nifti_dir = lay.subject_nifti_dir(subject)
    out_root = lay.results_dir / cfg.STAGE3_DIR / "per_subject"
    out_root.mkdir(parents=True, exist_ok=True)
    output = output or (out_root / f"{subject}.xlsx")

    row = process_subject(nifti_dir, stage2_dir)
    row["pesa_id"] = subject

    df = pd.DataFrame([row], columns=column_order())
    df.to_excel(output, index=False)
    log.info(f"[{subject}] wrote {output}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("dixon-v5-stage3")
@backend_click_option()
@click.option("--batch", required=True)
@click.option("--subject", required=True)
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--log-level", default="INFO")
def main(
    batch: str,
    subject: str,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
    backend: str,
    output: Path | None,
    log_level: str,
) -> None:
    """Dixon v5 stage 3 worker (single subject)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())
    lay = layout(
        batch,
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        results_root=results_root,
    )
    run_subject(subject, lay, backend=backend, output=output)


if __name__ == "__main__":
    main()
