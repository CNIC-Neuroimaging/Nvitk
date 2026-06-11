"""PyVista 3D mask overview scenes and HTML export."""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.logger import Logger
from nvitk.io import imread
from nvitk.pipes.pesa_fat.common.paths import BatchLayout, resolve_nii_optional
from nvitk.segmentation.total_segmentator.class_maps import get_class_map

log = Logger()


def _require_pyvista():
    from nvitk.pipes.pesa_fat.qc.headless import configure_headless_viz, export_plotter_html

    configure_headless_viz()
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


def _qualitative_color_pool() -> list[tuple[float, float, float]]:
    """Build a pool of perceptually distinct RGB colors (0–1)."""
    pool: list[tuple[float, float, float]] = []
    try:
        import matplotlib.cm as cm

        for name in ("tab20", "tab20b", "Set1", "Set2", "Dark2", "Paired"):
            cmap = cm.get_cmap(name)
            n = int(getattr(cmap, "N", 20))
            for j in range(n):
                rgba = cmap(j / max(n - 1, 1))
                rgb = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
                if rgb not in pool:
                    pool.append(rgb)
    except Exception:
        pass
    if pool:
        return pool
    golden = 0.618033988749895
    for i in range(40):
        hue = (i * golden) % 1.0
        sat = 0.95 if i % 2 == 0 else 0.78
        val = 0.95 if (i // 2) % 2 == 0 else 0.68
        pool.append(colorsys.hsv_to_rgb(hue, sat, val))
    return pool


def _high_contrast_colors(n: int) -> list[tuple[float, float, float]]:
    """Pick *n* colors with large pairwise separation (for vertebra label IDs)."""
    if n <= 0:
        return []
    pool = _qualitative_color_pool()
    if n == 1:
        return [pool[0]]
    if n >= len(pool):
        return [pool[i % len(pool)] for i in range(n)]
    # Prime stride spreads picks across the pool (adjacent vertebrae → distant hues).
    stride = 7
    while stride % len(pool) == 0 or len({(i * stride) % len(pool) for i in range(n)}) < n:
        stride += 2
    return [pool[(i * stride) % len(pool)] for i in range(n)]


def _spacing_origin_direction(metadata: dict | None) -> tuple[tuple[float, float, float], tuple[float, float, float], np.ndarray | None]:
    md = metadata or {}
    sp = md.get("spacing")
    if sp is None:
        sp = (md.get("x_res", 1.0), md.get("y_res", 1.0), md.get("z_res", 1.0))
    spacing = tuple(float(v) for v in sp)
    aff = md.get("affine")
    if aff is None:
        return spacing, (0.0, 0.0, 0.0), None
    a = np.asarray(aff, dtype=float)
    if a.shape != (4, 4):
        return spacing, (0.0, 0.0, 0.0), None
    origin = (float(a[0, 3]), float(a[1, 3]), float(a[2, 3]))
    # direction matrix should be unit vectors in world space
    d = a[:3, :3].copy()
    # avoid divide-by-zero
    spv = np.array([spacing[0], spacing[1], spacing[2]], dtype=float)
    spv[spv == 0] = 1.0
    direction = d / spv.reshape(1, 3)
    return spacing, origin, direction


def _grid_from_binary(pv, bin_u8: np.ndarray, *, spacing: tuple[float, float, float], origin: tuple[float, float, float], direction: np.ndarray | None):
    grid = pv.ImageData(
        dimensions=bin_u8.shape,
        spacing=spacing,
        origin=origin,
    )
    if direction is not None:
        try:
            grid.direction_matrix = direction
        except Exception:
            try:
                grid.SetDirectionMatrix(direction.ravel(order="F"))
            except Exception:
                pass
    grid.point_data["m"] = bin_u8.flatten(order="F")
    return grid


def _add_label_volume_surfaces(
    pl,
    pv,
    arr: np.ndarray,
    *,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: np.ndarray | None = None,
    mask_opacity: float = 0.35,
) -> None:
    """Add one translucent surface per nonzero integer label in *arr* (world-aligned)."""
    u = np.unique(arr.astype(np.int32, copy=False))
    u = u[u > 0]
    if u.size == 0:
        pl.add_text("empty", font_size=10)
        return
    colors = _distinct_colors(int(u.size))
    for i, lid in enumerate(u):
        bin_u8 = (arr == lid).astype(np.uint8, copy=False)
        grid = _grid_from_binary(pv, bin_u8, spacing=spacing, origin=origin, direction=direction)
        surf = grid.contour([0.5], scalars="m")
        if surf.n_points == 0:
            continue
        pl.add_mesh(
            surf,
            color=colors[i % len(colors)],
            opacity=float(mask_opacity),
            show_scalar_bar=False,
        )


def _add_colored_vertebrae_surfaces(
    pl,
    pv,
    seg: np.ndarray,
    *,
    spacing,
    origin,
    direction,
    task: str = "total",
    opacity: float = 0.45,
) -> None:
    """Render each vertebra label with a high-contrast distinct color."""
    seg_i = np.asarray(seg, dtype=np.int32)
    cmap = get_class_map(task)
    vert_items = [(int(i), str(nm)) for i, nm in cmap.items() if str(nm).startswith("vertebrae_")]
    if not vert_items:
        vbin = (seg_i > 0).astype(np.uint8)
        _add_label_volume_surfaces(
            pl, pv, vbin, spacing=spacing, origin=origin, direction=direction, mask_opacity=opacity
        )
        return

    present = [
        int(lid)
        for lid, _nm in sorted(vert_items, key=lambda t: t[0])
        if bool(np.any(seg_i == int(lid)))
    ]
    if not present:
        pl.add_text("empty", font_size=10)
        return

    colors = _high_contrast_colors(len(present))
    for idx, lid in enumerate(present):
        bin_u8 = (seg_i == lid).astype(np.uint8, copy=False)
        grid = _grid_from_binary(pv, bin_u8, spacing=spacing, origin=origin, direction=direction)
        surf = grid.contour([0.5], scalars="m")
        if surf.n_points == 0:
            continue
        pl.add_mesh(
            surf,
            color=colors[idx],
            opacity=opacity,
            show_scalar_bar=False,
        )


def export_ctpet_overview_html(
    lay: BatchLayout,
    subject: str,
    out_html: Path,
    *,
    notebook: bool = True,
) -> Path | None:
    """CT-PET overview: FAT+vertebrae, and MO+ORGANS+MUSCLES (2 views)."""
    from nvitk.pipes.pesa_fat.ct_pet_v5 import config as ct_cfg

    pv = _require_pyvista()
    stage2 = lay.results_dir / ct_cfg.STAGE2_DIR / subject / "CT"
    stage1 = lay.results_dir / ct_cfg.STAGE1_DIR / subject / "CT"
    fat_p = resolve_nii_optional(stage2, "FAT")
    mo_p = resolve_nii_optional(stage2, "MO")
    muscles_p = resolve_nii_optional(stage2, "MUSCLES")
    organs_p = resolve_nii_optional(stage2, "ORGANS")
    total_p = resolve_nii_optional(stage1, "total")

    if all(p is None for p in (fat_p, mo_p, muscles_p, organs_p, total_p)):
        log.warning("[%s] CT-PET mask QC: no mask files found", subject)
        return None

    pl = pv.Plotter(shape=(1, 3), notebook=notebook)

    # View A: FAT + vertebrae
    pl.subplot(0, 0)
    pl.add_text("FAT + vertebrae", font_size=10, position="upper_edge")
    if fat_p is not None:
        img = imread(str(fat_p), axes="XYZ")
        spacing, origin, direction = _spacing_origin_direction(img.metadata)
        _add_label_volume_surfaces(pl, pv, to_numpy(img.data), spacing=spacing, origin=origin, direction=direction, mask_opacity=0.35)
    if total_p is not None:
        img = imread(str(total_p), axes="XYZ")
        spacing, origin, direction = _spacing_origin_direction(img.metadata)
        _add_colored_vertebrae_surfaces(
            pl,
            pv,
            to_numpy(img.data),
            spacing=spacing,
            origin=origin,
            direction=direction,
            task="total",
            opacity=0.35,
        )
    pl.view_isometric()

    # View B: MO + ORGANS + MUSCLES
    pl.subplot(0, 1)
    pl.add_text("MO + ORGANS + MUSCLES", font_size=10, position="upper_edge")
    for pth, op in ((mo_p, 0.35), (organs_p, 0.25), (muscles_p, 0.25)):
        if pth is None:
            continue
        img = imread(str(pth), axes="XYZ")
        spacing, origin, direction = _spacing_origin_direction(img.metadata)
        _add_label_volume_surfaces(pl, pv, to_numpy(img.data), spacing=spacing, origin=origin, direction=direction, mask_opacity=op)
    pl.view_isometric()

    # View C: raw vertebrae from TotalSegmentator 'total' (available labels only)
    pl.subplot(0, 2)
    pl.add_text("Vertebrae (raw total)", font_size=10, position="upper_edge")
    if total_p is not None:
        img = imread(str(total_p), axes="XYZ")
        spacing, origin, direction = _spacing_origin_direction(img.metadata)
        _add_colored_vertebrae_surfaces(
            pl,
            pv,
            to_numpy(img.data),
            spacing=spacing,
            origin=origin,
            direction=direction,
            task="total",
            opacity=0.45,
        )
    pl.view_isometric()

    ok = export_plotter_html(
        pl,
        out_html,
        fallback_message=f"CT-PET mask overview export failed for {subject}.",
    )
    return out_html if ok else None


def export_dixon_overview_html(
    lay: BatchLayout,
    subject: str,
    out_html: Path,
    *,
    notebook: bool = True,
) -> Path | None:
    """Dixon overview: aligned blocks (HEAD+THORAX+LEGS) and vertebrae (2 views)."""
    from nvitk.pipes.pesa_fat.dixon_v5 import config as dx_cfg

    pv = _require_pyvista()
    stage2 = lay.results_dir / dx_cfg.STAGE2_DIR / subject
    stage1_th = lay.results_dir / dx_cfg.STAGE1_DIR / subject / f"{dx_cfg.INPUT_PREFIX}_THORAX"
    head_p = resolve_nii_optional(stage2, "HEAD")
    thorax_p = resolve_nii_optional(stage2, "THORAX")
    legs_p = resolve_nii_optional(stage2, "LEGS")
    vert_p = resolve_nii_optional(stage1_th, "vertebrae_mr")
    if all(p is None for p in (head_p, thorax_p, legs_p, vert_p)):
        log.warning("[%s] Dixon mask QC: no mask files found", subject)
        return None

    pl = pv.Plotter(shape=(1, 2), notebook=notebook)

    # View A: HEAD+THORAX+LEGS aligned
    pl.subplot(0, 0)
    pl.add_text("HEAD + THORAX + LEGS", font_size=10, position="upper_edge")
    for pth, op in ((head_p, 0.25), (thorax_p, 0.25), (legs_p, 0.25)):
        if pth is None:
            continue
        img = imread(str(pth), axes="XYZ")
        spacing, origin, direction = _spacing_origin_direction(img.metadata)
        _add_label_volume_surfaces(pl, pv, to_numpy(img.data), spacing=spacing, origin=origin, direction=direction, mask_opacity=op)
    pl.view_isometric()

    # View B: vertebrae only
    pl.subplot(0, 1)
    pl.add_text("Vertebrae", font_size=10, position="upper_edge")
    if vert_p is not None:
        img = imread(str(vert_p), axes="XYZ")
        spacing, origin, direction = _spacing_origin_direction(img.metadata)
        _add_colored_vertebrae_surfaces(
            pl,
            pv,
            to_numpy(img.data),
            spacing=spacing,
            origin=origin,
            direction=direction,
            task="vertebrae_mr",
            opacity=0.45,
        )
    pl.view_isometric()

    ok = export_plotter_html(
        pl,
        out_html,
        fallback_message=f"Dixon mask overview export failed for {subject}.",
    )
    return out_html if ok else None


__all__ = [
    "export_ctpet_overview_html",
    "export_dixon_overview_html",
]
