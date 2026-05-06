"""PESA-Fat hotspot viewer CLI.

This script loads the appropriate image + postprocessed mask for a given
stage-3 *measure* name and launches the 3D hotspot viewer.

Measure naming
--------------
`--measure` uses the same column naming as stage-3 outputs.

CT-PET examples:
- `HIGADO_SUVMAX`
- `GRASA_V_SUVmean`

DIXON examples:
- `DIXON_KIDNEY_R_FF`
- `DIXON_LIVER_T2`

Notes
-----
- CT-PET requires raw PET (`PT.nii[.gz]`) and computes SUV via
  :func:`nvitk.measure.suv.suv_image` (assumes metadata is present).
- Dixon uses the raw voxelwise Dixon maps as the \"PET\" intensity image.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import click
import numpy as np

from nvitk.core.logger import Logger
from nvitk.core.exceptions import ValidationError
from nvitk.io import imread
from nvitk.measure.suv import suv_image
from nvitk.pipes.pesa_fat.common.paths import (
    DEFAULT_NIFTI_ROOT,
    DEFAULT_RESULTS_ROOT,
    BatchLayout,
    layout,
    resolve_nii,
)
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg
from nvitk.transform.resampling import resample_mask_to_pet
from nvitk.types import Image
from nvitk.viz import HotspotMode, show_suv_hotspots


log = Logger()


def _stem_from_mask_file(filename: str) -> str:
    stem = filename
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def _detect_batches_for_subject(
    results_root: Path,
    nifti_root: Path,
    subject: str,
    *,
    need: str,  # "ctpet" | "dixon"
) -> list[str]:
    """Return batch names under results_root that contain this subject."""
    out: list[str] = []
    for batch_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        batch = batch_dir.name
        if need == "ctpet":
            stage2 = batch_dir / ct_cfg.STAGE2_DIR / subject / "CT"
        elif need == "dixon":
            stage2 = batch_dir / dx_cfg.STAGE2_DIR / subject
        else:
            raise ValueError(need)

        if not stage2.exists():
            continue
        subj_nifti = nifti_root / batch / subject
        if not subj_nifti.exists():
            continue
        out.append(batch)
    return out


@dataclass(frozen=True)
class CtPetResolved:
    spec: ct_cfg.SuvSpec
    stat_suffix: str  # e.g. SUVMAX, SUVmean...


@dataclass(frozen=True)
class DixonResolved:
    spec: dx_cfg.DixonMeasureSpec
    metric: str  # FF | T2 | R2


def _resolve_measure_ctpet(measure: str) -> CtPetResolved | None:
    suffixes = [suf for suf, _ in ct_cfg.SUV_STATS]
    for suf in suffixes:
        tail = f"_{suf}"
        if not measure.endswith(tail):
            continue
        prefix = measure[: -len(tail)]
        for spec in ct_cfg.SUV_SPECS:
            if spec.column_prefix == prefix:
                return CtPetResolved(spec=spec, stat_suffix=suf)
        raise ValidationError(
            f"Unknown CT-PET SUV measure prefix {prefix!r}. "
            "Use --list-measures to see valid options."
        )
    return None


def _resolve_measure_dixon(measure: str) -> DixonResolved | None:
    if "_" not in measure:
        return None
    prefix, metric = measure.rsplit("_", 1)
    metric = metric.strip()
    if metric not in ("FF", "T2", "R2"):
        return None

    # Find the spec
    for spec in dx_cfg.MEASURE_SPECS:
        if spec.prefix == prefix:
            if metric not in spec.metrics:
                raise ValidationError(
                    f"Measure {measure!r} exists but metric {metric!r} is not computed for it."
                )
            return DixonResolved(spec=spec, metric=metric)

    raise ValidationError(
        f"Unknown DIXON measure prefix {prefix!r}. Use --list-measures to see valid options."
    )


def _list_valid_measures() -> list[str]:
    out: list[str] = []
    # CT-PET SUV measures
    for spec in ct_cfg.SUV_SPECS:
        for suf, _stat in ct_cfg.SUV_STATS:
            out.append(f"{spec.column_prefix}_{suf}")
    # Dixon voxelwise measures (FF/T2/R2 only)
    for spec in dx_cfg.MEASURE_SPECS:
        for metric in ("FF", "T2", "R2"):
            if metric in spec.metrics:
                out.append(f"{spec.prefix}_{metric}")
    return sorted(set(out))


def _load_ctpet_inputs(lay: BatchLayout, subject: str, resolved: CtPetResolved) -> tuple[Image, Image, Sequence[int]]:
    subj_nifti = lay.subject_nifti_dir(subject)
    pet = imread(str(resolve_nii(subj_nifti, ct_cfg.PET_STEM)), axes="XYZ")
    suv = suv_image(pet, pet.metadata)

    stage2_dir = lay.results_dir / ct_cfg.STAGE2_DIR / subject / "CT"
    mask = imread(str(resolve_nii(stage2_dir, _stem_from_mask_file(resolved.spec.mask_file))), axes="XYZ")
    mask_r = resample_mask_to_pet(mask, pet)
    return suv, mask_r, resolved.spec.label_ids


def _load_dixon_inputs(lay: BatchLayout, subject: str, resolved: DixonResolved) -> tuple[Image, Image, Sequence[int]]:
    subj_nifti = lay.subject_nifti_dir(subject)
    region = resolved.spec.region

    if resolved.metric == "FF":
        stem = f"{dx_cfg.INPUT_PREFIX}_{region}_FAT_FRACTION"
        img = imread(str(resolve_nii(subj_nifti, stem)), axes="XYZ")
    elif resolved.metric == "T2":
        stem = f"{dx_cfg.INPUT_PREFIX}_{region}_T2STAR"
        img = imread(str(resolve_nii(subj_nifti, stem)), axes="XYZ")
    elif resolved.metric == "R2":
        stem = f"{dx_cfg.INPUT_PREFIX}_{region}_T2STAR"
        t2 = imread(str(resolve_nii(subj_nifti, stem)), axes="XYZ")
        t2_np = np.asarray(t2.data, dtype=np.float32)
        r2_np = np.zeros_like(t2_np, dtype=np.float32)
        pos = t2_np > 0
        r2_np[pos] = 1.0 / t2_np[pos] * 1000.0  # Hz, matches dixon stage3
        img = t2.with_data(r2_np)
    else:
        raise ValidationError(f"Unsupported dixon metric {resolved.metric!r}.")

    stage2_dir = lay.results_dir / dx_cfg.STAGE2_DIR / subject
    mask = imread(str(resolve_nii(stage2_dir, _stem_from_mask_file(resolved.spec.mask_file))), axes="XYZ")

    if img.data.shape != mask.data.shape:
        raise ValidationError(
            f"Dixon map and mask shapes differ ({img.data.shape} vs {mask.data.shape}). "
            "If this happens in your data, we can add an affine-based resample."
        )

    return img, mask, resolved.spec.label_ids


@click.command("nvitk-pesa-fat-hotspot")
@click.option("--subject", required=False, help="Subject id (e.g. PESA123).")
@click.option("--measure", required=False, help="Stage-3 measure id (use --list-measures).")
@click.option("--batch", default=None, help="Batch name (optional; auto-detect if omitted).")
@click.option("--nifti-root", type=click.Path(path_type=Path), default=None)
@click.option("--results-root", type=click.Path(path_type=Path), default=None)
@click.option("--list-measures", is_flag=True, default=False, help="Print valid measures and exit.")
@click.option("--hotspot", type=click.Choice(["top_percent", "top_k", "threshold"]), default="top_percent", show_default=True)
@click.option("--top-percent", type=float, default=0.1, show_default=True)
@click.option("--top-k", type=int, default=None)
@click.option("--suv-threshold", type=float, default=None)
@click.option("--max-points", type=int, default=50_000, show_default=True)
@click.option("--mask-iso", type=float, default=0.5, show_default=True)
@click.option("--mask-opacity", type=float, default=0.25, show_default=True)
@click.option("--mask-smooth", is_flag=True, default=False)
@click.option("--point-size", type=float, default=6.0, show_default=True)
@click.option("--cmap", default="turbo", show_default=True)
@click.option("--notebook/--no-notebook", default=False, show_default=True)
@click.option("--no-show", is_flag=True, default=False, help="Do not open a render window (for smoke tests).")
def main(
    subject: str,
    measure: str,
    batch: str | None,
    nifti_root: Path | None,
    results_root: Path | None,
    list_measures: bool,
    hotspot: HotspotMode,
    top_percent: float,
    top_k: int | None,
    suv_threshold: float | None,
    max_points: int,
    mask_iso: float,
    mask_opacity: float,
    mask_smooth: bool,
    point_size: float,
    cmap: str,
    notebook: bool,
    no_show: bool,
) -> None:
    Logger()

    if list_measures:
        for m in _list_valid_measures():
            click.echo(m)
        return

    assert subject is not None, "Missing option '--subject'"
    assert measure is not None, "Missing option '--measure'"

    # Reject non-voxelwise measures early
    if measure.endswith("_VOL") or measure.endswith("_NSlices") or measure.endswith("_WF"):
        raise click.ClickException(
            f"Measure {measure!r} is not supported for hotspot visualization (not voxelwise)."
        )

    ct = _resolve_measure_ctpet(measure)
    dx = None if ct is not None else _resolve_measure_dixon(measure)
    if ct is None and dx is None:
        raise click.ClickException(
            f"Could not parse measure {measure!r}. Use --list-measures to see valid options."
        )

    results_root_eff = Path(results_root) if results_root else DEFAULT_RESULTS_ROOT
    nifti_root_eff = Path(nifti_root) if nifti_root else DEFAULT_NIFTI_ROOT

    need = "ctpet" if ct is not None else "dixon"

    if batch is None:
        batches = _detect_batches_for_subject(results_root_eff, nifti_root_eff, subject, need=need)
        if not batches:
            raise click.ClickException(
                f"Subject {subject} not found under any batch in results_root={results_root_eff}."
            )
        if len(batches) > 1:
            raise click.ClickException(
                f"--batch omitted but subject {subject} was found in multiple batches: {batches}. "
                "Please provide --batch."
            )
        batch = batches[0]
        log.info(f"Auto-detected batch={batch} for subject={subject}")

    lay = layout(batch, nifti_root=nifti_root_eff, results_root=results_root_eff)

    if ct is not None:
        img, mask_img, label_ids = _load_ctpet_inputs(lay, subject, ct)
        title = f"CT-PET {measure} | {subject} | {batch}"
    else:
        assert dx is not None
        img, mask_img, label_ids = _load_dixon_inputs(lay, subject, dx)
        title = f"DIXON {measure} | {subject} | {batch}"

    show_suv_hotspots(
        img,
        mask_img,
        label_ids=label_ids,
        hotspot=hotspot,
        top_percent=top_percent,
        top_k=top_k,
        suv_threshold=suv_threshold,
        max_points=max_points,
        mask_iso=mask_iso,
        mask_opacity=mask_opacity,
        mask_smooth=mask_smooth,
        point_size=point_size,
        cmap=cmap,
        notebook=notebook,
        show=not no_show,
        title=title,
    )


__all__ = ["main"]

