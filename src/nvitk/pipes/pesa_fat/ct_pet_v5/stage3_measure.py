"""CT-PET v5 stage 3 (per-subject): SUV and volume measurements.

Port of :code:`BioImaging/src/pesa_fat/ct_pet/3_measure_SUV.py`.

For a PESA* subject with stage-2 outputs (``MO``, ``FAT``, ``BODY``,
``ORGANS``, ``MUSCLES``) the stage resamples the PET volume onto each mask
grid (via :class:`nvitk.measure.Measurer` alignment) and records:

* SUV stats (bw, max/mean/p95) for L3, L4, L3+L4 (MO), fat depots, organs
  and hemisphere-split muscles.
* Volumes (cc) for fat, body, and per-hemisphere muscles.

The CLI writes a per-subject Excel row as a single-row file; the batch
master (``run_batch.py``) concatenates all subjects' rows into
``<batch>_SummaryCodebook.xlsx``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pandas as pd

from nvitk.core.click_backend import backend_click_option, set_default_backend
from nvitk.core.backend import setup
from nvitk.core.logger import Logger
from nvitk.io import imread
from nvitk.measure import Measurer
from nvitk.pipes.pesa_fat.common.db_publish import maybe_publish_stage3_on_sge
from nvitk.pipes.pesa_fat.common.paths import BatchLayout, layout, resolve_nii
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as cfg
from nvitk.segmentation.labels import get_label
from nvitk.types import Image

setup(globals())

log = Logger()


# ---------------------------------------------------------------------------
# Measurement plan
# ---------------------------------------------------------------------------

SUV_SPECS = cfg.SUV_SPECS
VOL_SPECS = cfg.VOL_SPECS
_SUV_STATS = cfg.SUV_STATS


def column_order() -> list[str]:
    cols = ["pesa_id"]
    for spec in SUV_SPECS:
        for suffix, _ in _SUV_STATS:
            cols.append(f"{spec.column_prefix}_{suffix}")
    for spec in VOL_SPECS:
        cols.append(spec.column)
        cols.append(spec.column.replace("_VOL", "_NSlices"))
    return cols

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _nslices_axial_xyz(mask: Image) -> int:
    return int(np.any(mask.data > 0, axis=(0, 1)).sum())

def _build_binary_mask(label_img: Image, label_ids: tuple[int, ...]) -> Image:
    if len(label_ids) == 1:
        return get_label(label_img, label_ids[0], missing="empty")
    first = get_label(label_img, label_ids[0], missing="empty").data.copy()
    for lid in label_ids[1:]:
        extra = get_label(label_img, lid, missing="empty").data
        first[extra > 0] = 1
    return label_img.with_data(first.astype("uint8"))


def _load_mask(subject_stage2_dir: Path, filename: str) -> Image:
    stem = filename
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return imread(str(resolve_nii(subject_stage2_dir, stem)), axes="XYZ")


def _load_pet(subject_nifti_dir: Path) -> Image:
    return imread(str(resolve_nii(subject_nifti_dir, cfg.PET_STEM)), axes="XYZ")

# ---------------------------------------------------------------------------
# Per-subject processing
# ---------------------------------------------------------------------------

def process_subject(
    pet: Image,
    stage2_dir: Path,
    *,
    suv_kind: str = "bw",
    philips: bool = False,
) -> dict[str, Any]:
    """Return ``{column: value, ...}`` for one subject."""
    cache: dict[str, Image] = {}

    def _mask_file(name: str) -> Image:
        if name not in cache:
            cache[name] = _load_mask(stage2_dir, name)
        return cache[name]

    row: dict[str, Any] = {}

    for spec in SUV_SPECS:
        prefix = spec.column_prefix
        try:
            label_img = _mask_file(spec.mask_file)
            binmask = _build_binary_mask(label_img, spec.label_ids)
            if not bool(binmask.data.any()):
                for suffix, _ in _SUV_STATS:
                    row[f"{prefix}_{suffix}"] = None
                continue
            m = Measurer(pet, binmask)
            if pet.data.shape != binmask.data.shape:
                m = m.align("mask_to_raw")
            metrics = m.suv(
                kinds=(suv_kind,),
                stats=tuple(s for _, s in _SUV_STATS),
                philips=philips,
                revert_scaling=False,
            )
            suv_prefix = f"SUV{suv_kind}" if not suv_kind.startswith("SUV") else suv_kind
            for suffix, stat in _SUV_STATS:
                key = f"{suv_prefix}_{stat}" if stat != "p95" and stat != "p99" else f"{suv_prefix}_95percentile" if stat == "p95" else f"{suv_prefix}_99percentile"
                row[f"{prefix}_{suffix}"] = metrics.get(key)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            log.error(f"SUV[{prefix}] failed: {exc}")
            for suffix, _ in _SUV_STATS:
                row[f"{prefix}_{suffix}"] = None

    for spec in VOL_SPECS:
        try:
            label_img = _mask_file(spec.mask_file)
            binmask = _build_binary_mask(label_img, spec.label_ids)
            row[spec.column] = Measurer(pet, binmask).volume()["volume_cc"]
            row[spec.column.replace("_VOL", "_NSlices")] = _nslices_axial_xyz(binmask)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            log.error(f"VOL[{spec.column}] failed: {exc}")
            row[spec.column] = None
            row[spec.column.replace("_VOL", "_NSlices")] = None
    return row


def run_subject(
    subject: str,
    lay: BatchLayout,
    *,
    backend: str = "cupy",
    suv_kind: str = "bw",
    philips: bool = False,
    output: Path | None = None,
) -> Path:
    """Run stage 3 for a single subject and write a one-row Excel file."""
    try:
        set_default_backend(backend, allow_fallback=True)
    except Exception as exc:
        log.warning(f"Backend '{backend}' unavailable, falling back: {exc}")

    stage2_dir = lay.results_dir / cfg.STAGE2_DIR / subject / "CT"
    if not stage2_dir.exists():
        raise FileNotFoundError(f"Expected stage-2 outputs under {stage2_dir}")

    nifti_dir = lay.subject_nifti_dir(subject)
    out_root = lay.results_dir / cfg.STAGE3_DIR / "per_subject"
    out_root.mkdir(parents=True, exist_ok=True)
    output = output or (out_root / f"{subject}.xlsx")

    pet = _load_pet(nifti_dir)
    row = process_subject(pet, stage2_dir, suv_kind=suv_kind, philips=philips)
    row["pesa_id"] = subject

    df = pd.DataFrame([row], columns=column_order())
    df.to_excel(output, index=False)
    log.info(f"[{subject}] wrote {output}")
    maybe_publish_stage3_on_sge(
        subject_uid=subject,
        excel_path=output,
        pipeline=cfg.PIPELINE_NAME,
        batch=lay.batch,
    )
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command("ctpet-v5-stage3")
@backend_click_option()
@click.option("--batch", required=True)
@click.option("--subject", required=True)
@click.option("--dicom-root", type=click.Path(path_type=Path), default=None)
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option("--suv-kind", default="bw", show_default=True)
@click.option("--no-philips", is_flag=True, default=True, help="Disable Philips SUV short-circuit.")
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Excel output path (default RESULTS/<batch>/res_measure_suv/per_subject/<SUBJECT>.xlsx).",
)
@click.option("--log-level", default="INFO")
def main(
    batch: str,
    subject: str,
    dicom_root: Path | None,
    nifti_root: Path | None,
    results_root: Path | None,
    backend: str,
    suv_kind: str,
    output: Path | None,
    log_level: str,
    no_philips: bool = True,
) -> None:
    """CT-PET v5 stage 3 worker (single subject)."""
    Logger(level=log_level.upper())
    log.set_level(log_level.upper())
    lay = layout(
        batch,
        dicom_root=dicom_root,
        nifti_root=nifti_root,
        results_root=results_root,
    )
    run_subject(
        subject,
        lay,
        backend=backend,
        suv_kind=suv_kind,
        philips=not no_philips,
        output=output,
    )


if __name__ == "__main__":
    main()
