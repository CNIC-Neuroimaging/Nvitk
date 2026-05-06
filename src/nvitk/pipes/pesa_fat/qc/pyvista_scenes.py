"""PyVista multi-panel mask overview scenes and HTML export."""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io import imread
from nvitk.pipes.pesa_fat.common.paths import BatchLayout, resolve_nii, resolve_nii_optional

log = Logger()


def _require_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "QC mask scenes require pyvista. Install with: pip install pyvista trame"
        ) from exc
    return pv


def _distinct_colors(n: int, *, sat: float = 0.65, val: float = 0.95) -> list[tuple[float, float, float]]:
    if n <= 0:
        return []
    return [colorsys.hsv_to_rgb(i / max(n, 1), sat, val) for i in range(n)]


def _add_label_volume_surfaces(
    pl,
    pv,
    arr: np.ndarray,
    *,
    mask_opacity: float = 0.35,
) -> None:
    """Add one translucent surface per nonzero integer label in *arr*."""
    u = np.unique(arr.astype(np.int32, copy=False))
    u = u[u > 0]
    if u.size == 0:
        pl.add_text("empty", font_size=10)
        return
    colors = _distinct_colors(int(u.size))
    for i, lid in enumerate(u):
        bin_u8 = (arr == lid).astype(np.uint8, copy=False)
        grid = pv.ImageData(
            dimensions=bin_u8.shape,
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        grid.point_data["m"] = bin_u8.flatten(order="F")
        surf = grid.contour([0.5], scalars="m")
        if surf.n_points == 0:
            continue
        pl.add_mesh(
            surf,
            color=colors[i % len(colors)],
            opacity=float(mask_opacity),
            show_scalar_bar=False,
        )


def export_ctpet_mask_strip_html(
    lay: BatchLayout,
    subject: str,
    out_html: Path,
    *,
    notebook: bool = True,
) -> Path | None:
    """Five panels: FAT, MO, MUSCLES, ORGANS (stage2) + total (stage1)."""
    from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg

    pv = _require_pyvista()
    stage2 = lay.results_dir / ct_cfg.STAGE2_DIR / subject / "CT"
    stage1 = lay.results_dir / ct_cfg.STAGE1_DIR / subject / "CT"
    panels: list[tuple[str, Path | None]] = [
        ("FAT", resolve_nii_optional(stage2, "FAT")),
        ("MO", resolve_nii_optional(stage2, "MO")),
        ("MUSCLES", resolve_nii_optional(stage2, "MUSCLES")),
        ("ORGANS", resolve_nii_optional(stage2, "ORGANS")),
        ("total (TS)", resolve_nii_optional(stage1, "total")),
    ]
    if all(p is None for _, p in panels):
        log.warning("[%s] CT-PET mask QC: no mask files found", subject)
        return None

    pl = pv.Plotter(shape=(1, 5), notebook=notebook)
    for j, (title, path) in enumerate(panels):
        pl.subplot(0, j)
        pl.add_text(title, font_size=9, position="upper_edge")
        if path is None:
            pl.add_text("missing", font_size=10)
            continue
        img = imread(str(path), axes="XYZ")
        arr = to_numpy(img.data)
        _add_label_volume_surfaces(pl, pv, arr)
        pl.view_isometric()

    out_html.parent.mkdir(parents=True, exist_ok=True)
    pl.export_html(str(out_html))
    return out_html


def export_dixon_mask_strip_html(
    lay: BatchLayout,
    subject: str,
    out_html: Path,
    *,
    notebook: bool = True,
) -> Path | None:
    """Four panels: HEAD, THORAX, LEGS (stage2) + vertebrae_mr (THORAX stage1)."""
    from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg

    pv = _require_pyvista()
    stage2 = lay.results_dir / dx_cfg.STAGE2_DIR / subject
    stage1_th = lay.results_dir / dx_cfg.STAGE1_DIR / subject / f"{dx_cfg.INPUT_PREFIX}_THORAX"
    panels: list[tuple[str, Path | None]] = [
        ("HEAD", resolve_nii_optional(stage2, "HEAD")),
        ("THORAX", resolve_nii_optional(stage2, "THORAX")),
        ("LEGS", resolve_nii_optional(stage2, "LEGS")),
        ("vertebrae_mr", resolve_nii_optional(stage1_th, "vertebrae_mr")),
    ]
    if all(p is None for _, p in panels):
        log.warning("[%s] Dixon mask QC: no mask files found", subject)
        return None

    pl = pv.Plotter(shape=(1, 4), notebook=notebook)
    for j, (title, path) in enumerate(panels):
        pl.subplot(0, j)
        pl.add_text(title, font_size=9, position="upper_edge")
        if path is None:
            pl.add_text("missing", font_size=10)
            continue
        img = imread(str(path), axes="XYZ")
        arr = to_numpy(img.data)
        _add_label_volume_surfaces(pl, pv, arr)
        pl.view_isometric()

    out_html.parent.mkdir(parents=True, exist_ok=True)
    pl.export_html(str(out_html))
    return out_html


__all__ = [
    "export_ctpet_mask_strip_html",
    "export_dixon_mask_strip_html",
]
