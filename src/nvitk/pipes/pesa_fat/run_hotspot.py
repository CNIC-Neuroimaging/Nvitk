"""PESA-Fat hotspot viewer CLI.

This script loads the appropriate image + postprocessed mask for a given
stage-3 *measure* name and launches the 3D hotspot viewer.

Measure naming
--------------
Hotspot visualization operates on **voxelwise images** (SUV volume / Dixon maps),
so for CT-PET the hotspot map does not depend on SUVMAX vs SUVmean vs percentiles.

This CLI therefore uses the naming scheme:

- CT-PET: `<ROI>_SUV` (e.g. `HIGADO_SUV`, `GRASA_V_SUV`, `L4_SUV`)
- DIXON: `<ROI>_<METRIC>` where `METRIC in {FF,T2,R2}` (e.g. `KIDNEY_R_FF`, `LIVER_T2`)

For backwards compatibility, CT-PET stage-3 style names like `HIGADO_SUVMAX`
are accepted and treated as `HIGADO_SUV`.

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

from nvitk.core.array import as_backend_array, to_numpy
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
    resolve_nii_optional,
)
from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg
from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg
from nvitk.transform.resampling import resample_mask_to_pet
from nvitk.types import Image
from nvitk.viz import HotspotMode, show_hotspots
from nvitk.segmentation.total_segmentator.class_maps import get_class_id


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
    metric: str  # always "SUV" for voxelwise hotspot visualization


@dataclass(frozen=True)
class DixonResolved:
    spec: dx_cfg.DixonMeasureSpec
    metric: str  # FF | T2 | R2


def _resolve_measure_ctpet(measure: str) -> CtPetResolved | None:
    # Preferred: <ROI>_SUV
    if measure.endswith("_SUV"):
        prefix = measure[: -len("_SUV")]
        for spec in ct_cfg.SUV_SPECS:
            if spec.column_prefix == prefix:
                return CtPetResolved(spec=spec, metric="SUV")
        raise ValidationError(
            f"Unknown CT-PET ROI {prefix!r}. Use --list-measures to see valid options."
        )

    # Backward compatible: <ROI>_<SUVSTAT> e.g. HIGADO_SUVMAX, GRASA_V_SUVmean...
    suffixes = [suf for suf, _ in ct_cfg.SUV_STATS]
    for suf in suffixes:
        tail = f"_{suf}"
        if not measure.endswith(tail):
            continue
        prefix = measure[: -len(tail)]
        for spec in ct_cfg.SUV_SPECS:
            if spec.column_prefix == prefix:
                return CtPetResolved(spec=spec, metric="SUV")
        raise ValidationError(
            f"Unknown CT-PET ROI {prefix!r}. Use --list-measures to see valid options."
        )

    return None


def _resolve_measure_dixon(measure: str) -> DixonResolved | None:
    if "_" not in measure:
        return None
    prefix, metric = measure.rsplit("_", 1)
    metric = metric.strip()
    if metric not in ("FF", "T2", "R2"):
        return None

    # Find the spec (accept both with and without leading "DIXON_")
    candidates = {prefix}
    if prefix.startswith("DIXON_"):
        candidates.add(prefix[len("DIXON_") :])
    else:
        candidates.add("DIXON_" + prefix)

    for spec in dx_cfg.MEASURE_SPECS:
        if spec.prefix in candidates:
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
    # CT-PET voxelwise measure is always SUV
    for spec in ct_cfg.SUV_SPECS:
        out.append(f"{spec.column_prefix}_SUV")
    # Dixon voxelwise measures (FF/T2/R2 only)
    for spec in dx_cfg.MEASURE_SPECS:
        for metric in ("FF", "T2", "R2"):
            if metric in spec.metrics:
                # Prefer without the "DIXON_" prefix for user-facing ROI ids.
                roi = spec.prefix[len("DIXON_") :] if spec.prefix.startswith("DIXON_") else spec.prefix
                out.append(f"{roi}_{metric}")
    return sorted(set(out))

def _require_pyvista():
    try:
        import pyvista as pv  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Extra mask overlays require 'pyvista'. Install it with: pip install pyvista"
        ) from exc
    return pv


def _surface_from_binary(pv, binary: np.ndarray):
    binary_u8 = (binary > 0).astype(np.uint8, copy=False)
    grid = pv.ImageData(
        dimensions=binary_u8.shape,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data["m"] = binary_u8.flatten(order="F")
    return grid.contour([0.5], scalars="m")


def _parse_extra_masks(value: str | None) -> list[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def _ctpet_load_extra_mask_on_pet_grid(
    lay: BatchLayout,
    subject: str,
    name: str,
    pet: Image,
) -> Image | None:
    """
    Load an extra mask for CT-PET and resample it onto the PET grid.

    - For `ureter`, load stage-2 `_URETER.nii[.gz]` if present.
    - Otherwise load from stage-1 task outputs (`total`, `body`, `tissue_types`, `thigh_shoulder_muscles`)
      and extract the TotalSegmentator class by name.
    """
    stage2_dir = lay.results_dir / ct_cfg.STAGE2_DIR / subject / "CT"
    if name == "ureter":
        ureter_path = resolve_nii_optional(stage2_dir, "_URETER")
        if ureter_path is None:
            raise ValidationError(
                f"Requested extra mask 'ureter' but {_stem_from_mask_file('_URETER.nii.gz')!r} not found under {stage2_dir}."
            )
        ureter = imread(str(ureter_path), axes="XYZ")
        return resample_mask_to_pet(ureter, pet)

    stage1_dir = lay.results_dir / ct_cfg.STAGE1_DIR / subject / "CT"
    tasks = ("total", "body", "tissue_types", "thigh_shoulder_muscles")
    last_err: Exception | None = None
    for task in tasks:
        try:
            cid = get_class_id(name, task)
        except Exception as exc:
            last_err = exc
            continue
        seg = imread(str(resolve_nii(stage1_dir, task)), axes="XYZ")
        seg_np = to_numpy(seg.data)
        bin_np = (seg_np == int(cid)).astype(np.uint8)
        if not bool(np.any(bin_np)):
            log.warning(f"Extra mask {name!r} found in task {task!r} but is empty; skipping.")
            return None
        bin_img = seg.with_data(as_backend_array(bin_np))
        return resample_mask_to_pet(bin_img, pet)

    raise ValidationError(
        f"Could not resolve extra mask {name!r} from CT-PET stage1 tasks {tasks}. "
        f"Last error: {last_err}"
    )


def _dixon_load_extra_mask_on_map_grid(
    lay: BatchLayout,
    subject: str,
    region: str,
    name: str,
) -> Image | None:
    """
    Load an extra mask for Dixon from stage-1 task outputs for the given region.
    """
    stage1_dir = lay.results_dir / dx_cfg.STAGE1_DIR / subject / f"{dx_cfg.INPUT_PREFIX}_{region}"
    tasks = ("total_mr", "body_mr", "vertebrae_mr", "thigh_shoulder_muscles_mr")
    last_err: Exception | None = None
    for task in tasks:
        try:
            cid = get_class_id(name, task)
        except Exception as exc:
            last_err = exc
            continue
        seg = imread(str(resolve_nii(stage1_dir, task)), axes="XYZ")
        seg_np = to_numpy(seg.data)
        bin_np = (seg_np == int(cid)).astype(np.uint8)
        if not bool(np.any(bin_np)):
            log.warning(f"Extra mask {name!r} found in task {task!r} but is empty; skipping.")
            return None
        return seg.with_data(as_backend_array(bin_np))

    raise ValidationError(
        f"Could not resolve extra mask {name!r} from Dixon stage1 tasks {tasks} in region {region}. "
        f"Last error: {last_err}"
    )


def _load_ctpet_inputs(
    lay: BatchLayout, subject: str, resolved: CtPetResolved
) -> tuple[Image, Image, Sequence[int], Image]:
    subj_nifti = lay.subject_nifti_dir(subject)
    pet = imread(str(resolve_nii(subj_nifti, ct_cfg.PET_STEM)), axes="XYZ")
    suv = suv_image(pet, pet.metadata)

    stage2_dir = lay.results_dir / ct_cfg.STAGE2_DIR / subject / "CT"
    mask = imread(str(resolve_nii(stage2_dir, _stem_from_mask_file(resolved.spec.mask_file))), axes="XYZ")
    mask_r = resample_mask_to_pet(mask, pet)
    return suv, mask_r, resolved.spec.label_ids, pet


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
        t2_np = as_backend_array(t2.data)
        r2_np = np.zeros_like(t2_np, dtype=np.float32)
        pos = t2_np > 0
        r2_np[pos] = 1.0 / t2_np[pos] * 1000.0  # Hz
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
@click.option("--measure", required=False, help="Measure id (use --list-measures).")
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
@click.option(
    "--extra-masks",
    default=None,
    help="Comma-separated extra masks to overlay as surfaces (TS class names; special: 'ureter').",
)
@click.option("--extra-mask-opacity", type=float, default=0.15, show_default=True)
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
    extra_masks: str | None,
    extra_mask_opacity: float,
) -> None:
    Logger()

    if list_measures:
        for m in _list_valid_measures():
            click.echo(m)
        return

    assert subject is not None, "Missing option '--subject'"
    assert measure is not None, "Missing option '--measure'"

    # Reject non-voxelwise measures early (keep the old checks too)
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

    extra_list = _parse_extra_masks(extra_masks)

    if ct is not None:
        img, mask_img, label_ids, pet = _load_ctpet_inputs(lay, subject, ct)
        title = f"CT-PET {measure} | {subject} | {batch}"
        sb_title = "SUV"
        extra_imgs: list[tuple[str, Image]] = []
        for nm in extra_list:
            m = _ctpet_load_extra_mask_on_pet_grid(lay, subject, nm, pet)
            if m is not None:
                extra_imgs.append((nm, m))
    else:
        assert dx is not None
        img, mask_img, label_ids = _load_dixon_inputs(lay, subject, dx)
        title = f"DIXON {measure} | {subject} | {batch}"
        sb_title = dx.metric
        extra_imgs = []
        for nm in extra_list:
            if nm == "ureter":
                raise ValidationError("Extra mask 'ureter' is only supported for CT-PET.")
            m = _dixon_load_extra_mask_on_map_grid(lay, subject, dx.spec.region, nm)
            if m is not None:
                extra_imgs.append((nm, m))

    # Always build plotter with show=False so we can optionally overlay extra surfaces.
    pl = show_hotspots(
        img,
        mask_img,
        label_ids=label_ids,
        hotspot=hotspot,
        top_percent=top_percent,
        top_k=top_k,
        threshold=suv_threshold,
        max_points=max_points,
        mask_iso=mask_iso,
        mask_opacity=mask_opacity,
        mask_smooth=mask_smooth,
        point_size=point_size,
        cmap=cmap,
        notebook=notebook,
        show=False,
        title=title,
        scalar_bar_title=sb_title,
    )

    if extra_imgs:
        pv = _require_pyvista()
        colors = ["#00A6FB", "#F7B801", "#F18701", "#7D53DE", "#2ECC71", "#E74C3C"]
        for i, (nm, mimg) in enumerate(extra_imgs):
            surf = _surface_from_binary(pv, to_numpy(mimg.data))
            pl.add_mesh(
                surf,
                color=colors[i % len(colors)],
                opacity=float(extra_mask_opacity),
                show_scalar_bar=False,
            )
            pl.add_text(f"+ {nm}", position="lower_left", font_size=10)

    if not no_show:
        pl.show()


__all__ = ["main"]

