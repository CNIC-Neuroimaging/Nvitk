"""
suv SUV hotspot visualization (3D).

The main entry point is :func:`show_suv_hotspots_3d`, which renders:

- A segmentation ROI (binary or multi-label) as a translucent surface.
- The highest SUV voxels *inside* that ROI as colored points (\"hotspots\").

Example
-------

```python
from nvitk.io import imread
from nvitk.viz import show_suv_hotspots_3d

suv = imread(\"/path/PT_SUV.nii.gz\", axes=\"XYZ\")        # already SUV
mask = imread(\"/path/FAT.nii.gz\", axes=\"XYZ\")          # labels or binary

show_suv_hotspots_3d(
    suv,
    mask,
    label_ids=(1,),          # e.g. visceral fat label
    hotspot=\"top_percent\",
    top_percent=0.1,         # top 0.1% SUVs inside ROI
    max_points=20000,
)
```

Quick CLI smoke test
--------------------

```bash
python -c 'from nvitk.io import imread; from nvitk.viz import show_suv_hotspots_3d; \
suv=imread(\"PT_SUV.nii.gz\", axes=\"XYZ\"); m=imread(\"MASK.nii.gz\", axes=\"XYZ\"); \
show_suv_hotspots_3d(suv, m, hotspot=\"top_percent\", top_percent=0.1, max_points=20000)'
```
"""

from __future__ import annotations

from typing import Any, Iterable, Literal, Sequence

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import ValidationError
from nvitk.types import Image

HotspotMode = Literal["top_percent", "top_k", "threshold"]


def _require_pyvista() -> Any:
    try:
        import pyvista as pv  
    except ImportError as exc:
        raise ImportError(
            "show_suv_hotspots requires the optional dependency 'pyvista' (VTK). "
            "Install it with: pip install pyvista"
        ) from exc
    return pv


def _as_numpy_3d(x: Image | np.ndarray, *, name: str) -> np.ndarray:
    arr = to_numpy(x.data) if isinstance(x, Image) else np.asarray(x)
    if arr.ndim != 3:
        raise ValidationError(f"{name} must be a 3D array; got shape {arr.shape}.")
    return arr


def _roi_mask(mask_arr: np.ndarray, label_ids: Sequence[int] | None) -> np.ndarray:
    if label_ids is None:
        return mask_arr > 0
    if len(label_ids) == 0:
        raise ValidationError("label_ids cannot be empty (pass None for all nonzero labels).")
    return np.isin(mask_arr, np.asarray(label_ids, dtype=np.int64))


def _select_hotspots(
    suv_arr: np.ndarray,
    roi: np.ndarray,
    *,
    hotspot: HotspotMode,
    top_percent: float,
    top_k: int | None,
    suv_threshold: float | None,
) -> np.ndarray:
    """Return boolean mask of selected hotspot voxels."""
    if not bool(np.any(roi)):
        raise ValidationError("ROI mask is empty (no voxels selected).")

    vals = suv_arr[roi].astype(np.float32, copy=False)
    if vals.size == 0:
        raise ValidationError("ROI selection produced no voxels.")

    if hotspot == "top_percent":
        if not (0 < float(top_percent) <= 100):
            raise ValidationError("top_percent must be in (0, 100].")
        thr = float(np.percentile(vals, 100.0 - float(top_percent)))
        return roi & (suv_arr >= thr)

    if hotspot == "top_k":
        if top_k is None:
            raise ValidationError("hotspot='top_k' requires top_k.")
        k = int(top_k)
        if k <= 0:
            raise ValidationError("top_k must be a positive integer.")
        # threshold at kth largest value within ROI
        k = min(k, int(vals.size))
        # np.partition gives kth smallest; convert to kth largest
        kth = float(np.partition(vals, vals.size - k)[vals.size - k])
        return roi & (suv_arr >= kth)

    if hotspot == "threshold":
        if suv_threshold is None:
            raise ValidationError("hotspot='threshold' requires suv_threshold.")
        thr = float(suv_threshold)
        return roi & (suv_arr >= thr)

    raise ValidationError(f"Unknown hotspot mode {hotspot!r}.")


def show_suv_hotspots(
    suv: Image | np.ndarray,
    mask: Image | np.ndarray,
    *,
    label_ids: Sequence[int] | None = None,
    hotspot: HotspotMode = "top_percent",
    top_percent: float = 0.1,
    top_k: int | None = None,
    suv_threshold: float | None = None,
    max_points: int = 50_000,
    mask_iso: float = 0.5,
    mask_opacity: float = 0.25,
    mask_smooth: bool = False,
    point_size: float = 6.0,
    cmap: str = "turbo",
    notebook: bool = False,
    show: bool = True,
    title: str | None = None,
) -> Any:
    """
    Render SUV hotspots inside a segmentation mask.

    Parameters
    ----------
    suv
        suv in **SUV units**, 3D. Either :class:`~nvitk.types.Image` or a NumPy array.
    mask
        Segmentation mask, 3D, same grid as *suv*. Binary or multi-label.
    label_ids
        Optional subset of labels (for multi-label masks). None means \"all nonzero\".
    hotspot
        Hotspot selection mode.
    top_percent
        For ``hotspot='top_percent'``: keep voxels within the top *top_percent* SUVs inside ROI.
        Example: ``0.1`` means top 0.1% (very sparse hotspots).
    top_k
        For ``hotspot='top_k'``: keep the top K voxels by SUV inside ROI.
    suv_threshold
        For ``hotspot='threshold'``: keep voxels with SUV >= this value inside ROI.
    max_points
        Cap number of rendered hotspot voxels (largest SUVs retained).
    mask_iso
        Isosurface threshold for the ROI volume (binary ROI -> 0.5 is typical).
    mask_opacity
        ROI surface opacity.
    mask_smooth
        If True, apply light smoothing to the ROI surface mesh.
    point_size
        Hotspot point size (rendered as spheres).
    cmap
        Colormap for SUV scalars.
    notebook
        If True, configure PyVista plotter for notebooks.
    show
        If True, immediately show the interactive window.
    title
        Optional plot title.

    Returns
    -------
    plotter
        A PyVista plotter instance.
    """
    pv = _require_pyvista()

    suv_arr = _as_numpy_3d(suv, name="suv")
    mask_arr = _as_numpy_3d(mask, name="mask")
    if suv_arr.shape != mask_arr.shape:
        raise ValidationError(
            f"suv and mask must have the same shape; got {suv_arr.shape} vs {mask_arr.shape}."
        )

    roi = _roi_mask(mask_arr, label_ids)
    hot = _select_hotspots(
        suv_arr,
        roi,
        hotspot=hotspot,
        top_percent=top_percent,
        top_k=top_k,
        suv_threshold=suv_threshold,
    )

    if not bool(np.any(hot)):
        raise ValidationError("Hotspot selection is empty (no voxels match criteria).")

    # Convert hotspot voxels to a capped point cloud in voxel coordinates.
    ijk = np.argwhere(hot)  # (N, 3) in (i, j, k) == (x, y, z) for axes='XYZ'
    suv_vals = suv_arr[hot].astype(np.float32, copy=False)

    if int(max_points) <= 0:
        raise ValidationError("max_points must be a positive integer.")
    if ijk.shape[0] > int(max_points):
        # Keep highest SUV points
        keep = np.argpartition(suv_vals, -int(max_points))[-int(max_points) :]
        ijk = ijk[keep]
        suv_vals = suv_vals[keep]

    # ROI surface from binary ROI volume.
    roi_u8 = roi.astype(np.uint8, copy=False)
    # PyVista's uniform-volume class is `ImageData` (preferred across versions).
    grid = pv.ImageData(
        dimensions=roi_u8.shape,
        spacing=(1.0, 1.0, 1.0),
        origin=(0.0, 0.0, 0.0),
    )
    grid.point_data["roi"] = roi_u8.flatten(order="F")
    surf = grid.contour([float(mask_iso)], scalars="roi")
    if mask_smooth:
        try:
            surf = surf.smooth(n_iter=20, relaxation_factor=0.1)
        except Exception:
            pass

    points = ijk.astype(np.float32, copy=False)
    cloud = pv.PolyData(points)
    cloud.point_data["SUV"] = suv_vals

    pl = pv.Plotter(notebook=notebook)
    pl.enable_depth_peeling()
    if title:
        pl.add_text(str(title), position="upper_left", font_size=12)

    pl.add_mesh(surf, color="white", opacity=float(mask_opacity), show_scalar_bar=False)
    pl.add_mesh(
        cloud,
        scalars="SUV",
        cmap=cmap,
        point_size=float(point_size),
        render_points_as_spheres=True,
        scalar_bar_args={"title": "SUV"},
    )
    pl.view_isometric()

    if show:
        pl.show()
    return pl


__all__ = ["show_suv_hotspots", "HotspotMode"]

