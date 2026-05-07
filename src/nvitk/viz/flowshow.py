"""4DFlow visualization helpers (PyVista + ipywidgets).

This is an initial interactive viewer for 4DFlow phase volumes:
- render vessel masks (multi-label supported) as surfaces
- show velocity glyphs inside mask region(s) with configurable **vector styling**
- optionally show simple streamlines (seeded in-label)
- **animate** over cardiac phases with optional precomputed glyph sample sites

By default **all nonzero labels** are shown together (distinct colors). Pass
``show_all_labels=False`` to focus on one label at a time (legacy behavior).

The velocity field ``(vx, vy, vz)`` is **precomputed once** from AP/RL/FH for all
time indices ``(X, Y, Z, T, 3)``. When ``precompute_glyph_indices=True`` (default),
fixed voxel indices for arrows are also cached so scrubbing / animation only
re-samples velocities at those sites (fast).

The velocity sign conventions follow `nvitk.io.conversors.phase2volume`:
    vx = -RL * 10
    vy = -AP * 10
    vz =  FH * 10
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence
import traceback

import numpy as np

from nvitk.core.array import to_numpy
from nvitk.core.exceptions import ValidationError
from nvitk.morphology import compute_centerlines
from nvitk.transform import oblique_slice
from nvitk.types import Image
from nvitk.core.logger import Logger

# Distinct colors for vessel labels (matplotlib tab10).
_TAB10_HEX: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)

VectorColorMode = Literal["label", "speed", "fixed"]


log = Logger()

@dataclass
class FlowshowVectorOptions:
    """How velocity vectors (glyphs) and streamlines are drawn."""

    color_mode: VectorColorMode = "label"
    """``label``: color by vessel id (multi-label). ``speed``: |v| colormap. ``fixed``: single color."""

    fixed_color: str = "#00A6FB"
    """Used when ``color_mode == \"fixed\"`` (single-label view default accent)."""

    scale_by_magnitude: bool = False
    """If True, arrow length scales with |v| (normalized per frame unless clim fixed)."""

    scale_factor: float = 0.35
    """PyVista glyph scale multiplier when ``scale_by_magnitude`` is True."""

    speed_cmap: str = "turbo"
    """Colormap for ``color_mode == \"speed\"``."""

    speed_clim: tuple[float, float] | None = None
    """Fixed |v| range for speed colors and optional scale normalization; if None, estimated once from data."""

    glyph_opacity: float = 0.88

    streamline_radius: float = 0.12
    streamline_max_time: float = 35.0
    streamline_n_seeds: int = 64
    streamline_opacity: float = 0.55
    streamline_fixed_color: str = "#F7B801"


@dataclass
class FlowshowAnimationOptions:
    """Time animation; glyph voxel sites can be precomputed for cheap updates."""

    precompute_glyph_indices: bool = True
    """Cache subsampled voxel coordinates per label so each frame only indexes ``vel[*, t]``."""

    auto_play: bool = False
    """Desktop: start timer immediately. Notebook: start ``Play`` widget if possible."""

    animation_fps: float = 8.0
    loop: bool = True
    """If False, animation stops at last time frame (desktop timer)."""


@dataclass
class _GlyphCacheAll:
    """Precomputed coords + color index per label group."""

    parts: list[tuple[np.ndarray, int]] = field(default_factory=list)


def _require_pyvista() -> Any:
    try:
        import pyvista as pv  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "flowshow requires 'pyvista'. Install it with: pip install pyvista"
        ) from exc
    return pv


def _unwrap_vtk_interactor(plotter: Any) -> Any | None:
    """Return a vtkRenderWindowInteractor (or compatible) with timer + observer APIs."""
    seen: set[int] = set()
    candidates: list[Any] = []

    def _add(obj: Any) -> None:
        if obj is None:
            return
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        candidates.append(obj)

    rw = getattr(plotter, "render_window", None)
    if rw is not None:
        gi = getattr(rw, "GetInteractor", None)
        if callable(gi):
            try:
                _add(gi())
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass
    _add(getattr(plotter, "iren", None))
    ir = getattr(plotter, "iren", None)
    if ir is not None:
        _add(getattr(ir, "interactor", None))
        _add(getattr(ir, "_interactor", None))
        _add(getattr(ir, "_iren", None))

    for c in candidates:
        has_timer = hasattr(c, "CreateRepeatingTimer") or hasattr(c, "create_repeating_timer")
        has_obs = hasattr(c, "add_observer") or hasattr(c, "AddObserver")
        if has_timer and has_obs:
            return c
    for c in candidates:
        if hasattr(c, "CreateRepeatingTimer") or hasattr(c, "create_repeating_timer"):
            return c
    return candidates[0] if candidates else None


def _install_repeating_timer_observer(vtk_iren: Any, timer_ms: int, callback: Callable[..., None]) -> bool:
    """Return True if a repeating timer + TimerEvent observer were registered."""
    import vtk as vtk_mod  # type: ignore

    created = False
    for name in ("CreateRepeatingTimer", "create_repeating_timer"):
        if hasattr(vtk_iren, name):
            getattr(vtk_iren, name)(int(timer_ms))
            created = True
            break
    if not created:
        return False
    add_obs = getattr(vtk_iren, "add_observer", None) or getattr(vtk_iren, "AddObserver", None)
    if add_obs is None:
        return False
    add_obs(vtk_mod.vtkCommand.TimerEvent, callback)
    return True


def _require_widgets() -> Any:
    try:
        import ipywidgets as widgets  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "flowshow interactive controls require 'ipywidgets'. "
            "Install it with: pip install ipywidgets"
        ) from exc
    return widgets


def _as_4d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim == 3:
        return arr[..., None]
    if arr.ndim == 4:
        return arr
    raise ValidationError(f"{name} must be 3D or 4D; got shape {arr.shape}.")


def _velocity_from_phases(ap: np.ndarray, rl: np.ndarray, fh: np.ndarray) -> np.ndarray:
    ap4 = _as_4d(ap, "ap_phase")
    rl4 = _as_4d(rl, "rl_phase")
    fh4 = _as_4d(fh, "fh_phase")
    if not (ap4.shape == rl4.shape == fh4.shape):
        raise ValidationError("AP, RL, and FH phase volumes must have identical shapes.")
    vx = (-rl4 * 10.0).astype(np.float32, copy=False)
    vy = (-ap4 * 10.0).astype(np.float32, copy=False)
    vz = (fh4 * 10.0).astype(np.float32, copy=False)
    return np.stack([vx, vy, vz], axis=-1)


def _mask_surface(pv: Any, roi: np.ndarray) -> Any | None:
    try:
        grid = pv.ImageData(dimensions=roi.shape, spacing=(1, 1, 1), origin=(0, 0, 0))
        grid.point_data["roi"] = roi.astype(np.uint8).flatten(order="F")
        surf = grid.contour([0.5], scalars="roi")
        if surf.n_points == 0:
            return None
        return surf
    except Exception as e:
        log.error(traceback.format_exc())
        log.exception(e)
        return None


def _vtk_rgb(hex_color: str) -> tuple[float, float, float]:
    hc = hex_color.lstrip("#")
    if len(hc) != 6:
        return (1.0, 1.0, 1.0)
    r = int(hc[0:2], 16) / 255.0
    g = int(hc[2:4], 16) / 255.0
    b = int(hc[4:6], 16) / 255.0
    return (r, g, b)


class _VtkGlyphPipeline:
    """Fast glyph pipeline: fixed points, update vectors per timepoint.

    We use VTK directly so animation only updates arrays and renders; we do NOT
    rebuild mask surfaces or recreate glyph meshes each frame.
    """

    def __init__(
        self,
        *,
        coords: np.ndarray,  # (N,3) int64
        color: str,
        vel: np.ndarray,  # (X,Y,Z,T,3) float32
        vec: FlowshowVectorOptions,
        speed_clim_eff: tuple[float, float],
        tt0: int,
    ) -> None:
        import vtk  # type: ignore
        from vtk.util import numpy_support  # type: ignore

        self.coords = coords.astype(np.int64, copy=False)
        self.color = color
        self.vel = vel
        self.vec = vec
        self.speed_clim_eff = speed_clim_eff

        pts = vtk.vtkPoints()
        pts.SetData(numpy_support.numpy_to_vtk(self.coords.astype(np.float32), deep=True))

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        self._pd = pd

        self._vec_arr = numpy_support.numpy_to_vtk(
            np.zeros((self.coords.shape[0], 3), dtype=np.float32),
            deep=True,
        )
        self._vec_arr.SetName("v")
        pd.GetPointData().SetVectors(self._vec_arr)

        self._mag_arr = numpy_support.numpy_to_vtk(
            np.zeros((self.coords.shape[0],), dtype=np.float32),
            deep=True,
        )
        self._mag_arr.SetName("mag")
        pd.GetPointData().AddArray(self._mag_arr)

        arrow = vtk.vtkArrowSource()

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(pd)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.OrientOn()
        glyph.SetVectorModeToUseVector()
        glyph.SetScaleModeToDataScalingOff()
        glyph.SetScaleFactor(1.0)
        self._glyph = glyph

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.ScalarVisibilityOff()
        self._mapper = mapper

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(_vtk_rgb(color))
        actor.GetProperty().SetOpacity(float(vec.glyph_opacity))
        self.actor = actor

        self.set_scale_by_magnitude(vec.scale_by_magnitude, vec.scale_factor)
        self.set_color_by_speed(vec.scale_by_magnitude)
        self.update_time(tt0)

    def set_opacity(self, opacity: float) -> None:
        self.actor.GetProperty().SetOpacity(float(opacity))

    def set_color_by_speed(self, enabled: bool) -> None:
        """When enabled, color glyphs by |v| using the 'mag' array."""
        import vtk  # type: ignore

        if enabled:
            lo, hi = self.speed_clim_eff
            # Prefer PyVista LUT generation (supports named cmaps like 'turbo').
            lut = None
            try:
                import pyvista as pv  # type: ignore

                lt = pv.LookupTable(cmap=str(getattr(self.vec, "speed_cmap", "turbo")), n_values=256)
                # Some PyVista versions expose to_vtk(); otherwise it is already a vtkLookupTable.
                lut = lt.to_vtk() if hasattr(lt, "to_vtk") else lt
            except Exception:
                lut = None

            self._mapper.ScalarVisibilityOn()
            try:
                self._mapper.SetScalarModeToUsePointFieldData()
            except Exception:
                pass
            try:
                self._mapper.SelectColorArray("mag")
            except Exception:
                # VTK fallback
                try:
                    self._mapper.SetArrayName("mag")
                except Exception:
                    pass
            if lut is not None:
                try:
                    self._mapper.SetLookupTable(lut)
                except Exception:
                    pass
            try:
                self._mapper.SetScalarRange(float(lo), float(hi))
            except Exception:
                pass
            # Actor property color should not override scalar coloring.
            try:
                self.actor.GetProperty().SetColor(1.0, 1.0, 1.0)
            except Exception:
                pass
        else:
            self._mapper.ScalarVisibilityOff()
            try:
                self.actor.GetProperty().SetColor(_vtk_rgb(self.color))
            except Exception:
                pass

    def set_scale_by_magnitude(self, enabled: bool, scale_factor: float) -> None:
        import vtk  # type: ignore

        if enabled:
            self._glyph.SetScaleModeToScaleByScalar()
            self._glyph.SetInputArrayToProcess(
                0,
                0,
                0,
                vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                "mag_scale",
            )
            self._glyph.SetScaleFactor(float(scale_factor))
        else:
            self._glyph.SetScaleModeToDataScalingOff()
            self._glyph.SetScaleFactor(1.0)

    def update_time(self, tt: int) -> None:
        from vtk.util import numpy_support  # type: ignore

        v = self.vel[
            self.coords[:, 0],
            self.coords[:, 1],
            self.coords[:, 2],
            int(tt),
            :,
        ].astype(np.float32, copy=False)
        mag = np.linalg.norm(v, axis=1).astype(np.float32, copy=False)

        numpy_support.vtk_to_numpy(self._vec_arr)[:] = v
        numpy_support.vtk_to_numpy(self._mag_arr)[:] = mag

        if self.vec.scale_by_magnitude:
            lo, hi = self.speed_clim_eff
            denom = max(float(hi - lo), 1e-6)
            mag_n = (mag - float(lo)) / denom
            mag_n = np.clip(mag_n, 0.05, 1.0).astype(np.float32, copy=False)

            arr = self._pd.GetPointData().GetArray("mag_scale")
            if arr is None:
                arr = numpy_support.numpy_to_vtk(mag_n, deep=True)
                arr.SetName("mag_scale")
                self._pd.GetPointData().AddArray(arr)
            else:
                numpy_support.vtk_to_numpy(arr)[:] = mag_n

        self._pd.Modified()
        self._glyph.Modified()


def _neighbors26(p: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    x, y, z = p
    out: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                out.append((x + dx, y + dy, z + dz))
    return out


def _centerline_longest_path(coords_zyx: np.ndarray) -> np.ndarray:
    """Order skeleton voxels into an approximate centerline using the graph diameter.

    Input coords are (N,3) in voxel (x,y,z) order (same as np.argwhere output here).
    """
    if coords_zyx.shape[0] <= 2:
        return coords_zyx.astype(np.float32, copy=False)

    nodes = [tuple(int(v) for v in row) for row in coords_zyx]
    node_set = set(nodes)
    adj: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
    deg: dict[tuple[int, int, int], int] = {}
    for n in nodes:
        nbrs = [m for m in _neighbors26(n) if m in node_set]
        adj[n] = nbrs
        deg[n] = len(nbrs)

    endpoints = [n for n, d in deg.items() if d <= 1]
    start = endpoints[0] if endpoints else nodes[0]

    def _bfs(src: tuple[int, int, int]) -> tuple[dict[tuple[int, int, int], int], dict[tuple[int, int, int], tuple[int, int, int] | None]]:
        from collections import deque

        dist: dict[tuple[int, int, int], int] = {src: 0}
        parent: dict[tuple[int, int, int], tuple[int, int, int] | None] = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj.get(u, []):
                if v in dist:
                    continue
                dist[v] = dist[u] + 1
                parent[v] = u
                q.append(v)
        return dist, parent

    d1, _ = _bfs(start)
    a = max(d1, key=d1.get)
    d2, parent = _bfs(a)
    b = max(d2, key=d2.get)

    # Reconstruct path b -> a
    path: list[tuple[int, int, int]] = []
    cur: tuple[int, int, int] | None = b
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur, None)
    path.reverse()
    return np.asarray(path, dtype=np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 1e-6:
        return np.zeros_like(v)
    return v / n


def _tangent_from_centerline(points: np.ndarray, idx: int, *, window: int = 5) -> np.ndarray:
    """Compute tangent at points[idx] using neighbors (window=3 or 5)."""
    w = int(window)
    if w not in (3, 5):
        w = 5
    k = 1 if w == 3 else 2
    n = points.shape[0]
    a = max(0, idx - k)
    b = min(n - 1, idx + k)
    if b == a:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return _unit(points[b] - points[a])


def _frame_from_tangent(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two in-plane unit vectors (u, v) orthogonal to tangent t."""
    t = _unit(t)
    # pick a stable axis least aligned with t
    axes = np.eye(3, dtype=np.float32)
    dots = np.abs(axes @ t)
    a = axes[int(np.argmin(dots))]
    u = _unit(np.cross(t, a))
    v = _unit(np.cross(t, u))
    return u, v


def _oblique_slice(
    vol: np.ndarray,
    *,
    center: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    radius_vox: float,
    res: int,
    order: int,
) -> np.ndarray:
    # Keep a local wrapper so existing call sites stay readable.
    sl = oblique_slice(
        vol,
        center_xyz=center,
        u_xyz=u,
        v_xyz=v,
        radius_vox=radius_vox,
        res=res,
        order=order,
    )
    return np.asarray(to_numpy(sl), dtype=np.float32)

def _coord_blocks_for_speed_clim(
    mask: np.ndarray,
    labels: Sequence[int],
    stride: int,
    max_glyphs: int,
    stream_seed: int | None,
    cache_all: _GlyphCacheAll | None,
    *,
    max_labels_sample: int = 8,
) -> list[np.ndarray]:
    """Voxel index sets used to estimate |v| percentiles when ``speed_clim`` is automatic."""
    if cache_all is not None:
        blocks = [c for c, _ in cache_all.parts if c.size > 0]
        if blocks:
            return blocks
    st = max(int(stride), 1)
    cap = min(int(max_glyphs), 4096)
    out: list[np.ndarray] = []
    for lbl in list(labels)[:max_labels_sample]:
        cc = _precompute_coords_single_label(mask, int(lbl), st, cap, stream_seed)
        if cc.size > 0:
            out.append(cc)
    if not out:
        raw = np.argwhere(mask > 0)[::st][:2048]
        if raw.size > 0:
            out.append(raw.astype(np.int64, copy=False))
    return out


def _estimate_speed_clim(
    vel: np.ndarray,
    coord_blocks: list[np.ndarray],
    *,
    n_samples: int = 12_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Sample |v| over time and cached coords for stable speed color limits."""
    _, _, _, nt, _ = vel.shape
    rng = np.random.default_rng(seed)
    mags: list[float] = []
    nonempty = [c for c in coord_blocks if c.shape[0] > 0]
    if not nonempty:
        return (0.0, 1.0)
    for _ in range(n_samples):
        tt = int(rng.integers(0, nt))
        blk = nonempty[int(rng.integers(0, len(nonempty)))]
        j = int(rng.integers(0, blk.shape[0]))
        x, y, z = int(blk[j, 0]), int(blk[j, 1]), int(blk[j, 2])
        mags.append(float(np.linalg.norm(vel[x, y, z, tt, :])))
    if not mags:
        return (0.0, 1.0)
    lo, hi = np.percentile(mags, [2.0, 98.0])
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1e-3
    return (lo, hi)


def _precompute_coords_all_labels(
    mask: np.ndarray,
    labels: Sequence[int],
    stride: int,
    max_glyphs: int,
    stream_seed: int | None,
) -> _GlyphCacheAll:
    st = max(int(stride), 1)
    nlab = len(labels)
    per_label = max(max_glyphs // max(nlab, 1), 32)
    parts: list[tuple[np.ndarray, int]] = []
    for i, lbl in enumerate(labels):
        roi = mask == int(lbl)
        coords = np.argwhere(roi)
        if coords.size == 0:
            continue
        coords_s = coords[::st]
        if coords_s.shape[0] > per_label:
            rng_i = np.random.default_rng((stream_seed if stream_seed is not None else 0) + i)
            pick = rng_i.choice(coords_s.shape[0], size=per_label, replace=False)
            coords_s = coords_s[pick]
        parts.append((coords_s.astype(np.int64, copy=False), i))
    return _GlyphCacheAll(parts=parts)


def _precompute_coords_single_label(
    mask: np.ndarray,
    lbl: int,
    stride: int,
    max_glyphs: int,
    stream_seed: int | None,
) -> np.ndarray:
    st = max(int(stride), 1)
    roi = mask == int(lbl)
    coords = np.argwhere(roi)
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.int64)
    coords = coords[::st]
    if coords.shape[0] > max_glyphs:
        rng = np.random.default_rng(stream_seed)
        pick = rng.choice(coords.shape[0], size=max_glyphs, replace=False)
        coords = coords[pick]
    return coords.astype(np.int64, copy=False)


def _build_glyphs_mesh(
    pv: Any,
    coords: np.ndarray,
    vel: np.ndarray,
    tt: int,
    vec: FlowshowVectorOptions,
    *,
    label_color: str | None,
    speed_clim_eff: tuple[float, float],
) -> tuple[Any, dict[str, Any]] | None:
    """Return (glyph_mesh, kwargs) for ``plotter.add_mesh``."""
    if coords.shape[0] == 0:
        return None
    v = vel[coords[:, 0], coords[:, 1], coords[:, 2], tt, :].astype(np.float32, copy=False)
    mag = np.linalg.norm(v, axis=1).astype(np.float32)
    pts = pv.PolyData(coords.astype(np.float32))
    pts["v"] = v
    pts["mag"] = mag

    scale_field: str | bool = False
    fac = 1.0
    if vec.scale_by_magnitude:
        lo, hi = speed_clim_eff
        denom = max(hi - lo, 1e-6)
        mag_n = ((mag - lo) / denom).astype(np.float32)
        mag_n = np.clip(mag_n, 0.05, 1.0)
        pts["mag_scale"] = mag_n
        scale_field = "mag_scale"
        fac = vec.scale_factor

    # PyVista: `clamp` was added in newer versions; omit for compatibility.
    glyphs = pts.glyph(orient="v", scale=scale_field, factor=fac)

    if vec.color_mode == "speed":
        kw: dict[str, Any] = {
            "scalars": "mag",
            "cmap": vec.speed_cmap,
            "clim": speed_clim_eff,
            "opacity": vec.glyph_opacity,
            "show_scalar_bar": True,
            "scalar_bar_args": {"vertical": True, "title": "|v|"},
        }
    elif vec.color_mode == "fixed":
        kw = {
            "color": vec.fixed_color,
            "opacity": vec.glyph_opacity,
            "show_scalar_bar": False,
        }
    else:
        c = label_color or _TAB10_HEX[0]
        kw = {"color": c, "opacity": vec.glyph_opacity, "show_scalar_bar": False}
    return glyphs, kw


def _add_glyph_actor(plotter: Any, built: tuple[Any, dict[str, Any]] | None) -> Any | None:
    if built is None:
        return None
    mesh, kw = built
    return plotter.add_mesh(mesh, **kw)


def _add_flow_scene_single(
    pv: Any,
    plotter: Any,
    *,
    lbl: int,
    tt: int,
    show_mask: bool,
    show_glyphs: bool,
    show_stream: bool,
    vel: np.ndarray,
    mask: np.ndarray,
    stride: int,
    max_glyphs: int,
    vec: FlowshowVectorOptions,
    speed_clim_eff: tuple[float, float],
    stream_seed: int | None,
    cache_coords: np.ndarray | None,
    use_cache: bool,
) -> list[Any]:
    actors: list[Any] = []
    x, y, z, nt, _ = vel.shape
    roi = mask == int(lbl)

    if show_mask:
        surf = _mask_surface(pv, roi)
        if surf is not None:
            actors.append(plotter.add_mesh(surf, color="white", opacity=0.25, smooth_shading=True))

    if show_glyphs:
        coords = cache_coords if (use_cache and cache_coords is not None) else _precompute_coords_single_label(
            mask, lbl, stride, max_glyphs, stream_seed
        )
        hex_col = None if vec.color_mode == "speed" else (
            vec.fixed_color if vec.color_mode == "fixed" else _TAB10_HEX[0]
        )
        built = _build_glyphs_mesh(pv, coords, vel, tt, vec, label_color=hex_col, speed_clim_eff=speed_clim_eff)
        act = _add_glyph_actor(plotter, built)
        if act is not None:
            actors.append(act)

    if show_stream:
        vec_f = vel[..., tt, :]
        grid = pv.ImageData(dimensions=(x, y, z), spacing=(1, 1, 1), origin=(0, 0, 0))
        grid.point_data["v"] = vec_f.reshape(-1, 3, order="F")
        coords = np.argwhere(roi)
        nseed = min(vec.streamline_n_seeds, coords.shape[0]) if coords.shape[0] > 0 else 0
        if nseed > 0:
            rng = np.random.default_rng(stream_seed)
            pick = rng.choice(coords.shape[0], size=nseed, replace=False)
            seed_cloud = pv.PolyData(coords[pick].astype(np.float32))
            try:
                stream = grid.streamlines_from_source(
                    seed_cloud,
                    vectors="v",
                    max_length=vec.streamline_max_time,
                    integration_direction="both",
                )
                if stream.n_points > 0:
                    tube = stream.tube(radius=vec.streamline_radius)
                    actors.append(
                        plotter.add_mesh(
                            tube,
                            color=vec.streamline_fixed_color,
                            opacity=vec.streamline_opacity,
                        )
                    )
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass

    actors.append(plotter.add_text(f"Label={lbl}  T={tt}", position="upper_left", font_size=12))
    return actors


def _add_flow_scene_all_labels(
    pv: Any,
    plotter: Any,
    *,
    labels: Sequence[int],
    tt: int,
    show_mask: bool,
    show_glyphs: bool,
    show_stream: bool,
    vel: np.ndarray,
    mask: np.ndarray,
    stride: int,
    max_glyphs: int,
    vec: FlowshowVectorOptions,
    speed_clim_eff: tuple[float, float],
    stream_seed: int | None,
    cache: _GlyphCacheAll | None,
    use_cache: bool,
) -> list[Any]:
    actors: list[Any] = []
    x, y, z, nt, _ = vel.shape
    labels = list(labels)
    nlab = len(labels)

    if show_mask:
        for i, lbl in enumerate(labels):
            roi = mask == int(lbl)
            if not np.any(roi):
                continue
            surf = _mask_surface(pv, roi)
            if surf is None:
                continue
            color = _TAB10_HEX[i % len(_TAB10_HEX)]
            actors.append(
                plotter.add_mesh(
                    surf,
                    color=color,
                    opacity=0.34,
                    smooth_shading=True,
                    show_scalar_bar=False,
                )
            )

    if show_glyphs and nlab > 0:
        eff_cache = cache if (use_cache and cache is not None) else _precompute_coords_all_labels(
            mask, labels, stride, max_glyphs, stream_seed
        )
        for coords, idx in eff_cache.parts:
            hex_col = _TAB10_HEX[idx % len(_TAB10_HEX)]
            lc = hex_col if vec.color_mode == "label" else None
            built = _build_glyphs_mesh(pv, coords, vel, tt, vec, label_color=lc, speed_clim_eff=speed_clim_eff)
            act = _add_glyph_actor(plotter, built)
            if act is not None:
                actors.append(act)

    if show_stream:
        roi_u = mask > 0
        vec_f = vel[..., tt, :]
        grid = pv.ImageData(dimensions=(x, y, z), spacing=(1, 1, 1), origin=(0, 0, 0))
        grid.point_data["v"] = vec_f.reshape(-1, 3, order="F")
        coords = np.argwhere(roi_u)
        nseed = min(vec.streamline_n_seeds, coords.shape[0]) if coords.shape[0] > 0 else 0
        if nseed > 0:
            rng = np.random.default_rng(stream_seed)
            pick = rng.choice(coords.shape[0], size=nseed, replace=False)
            seed_cloud = pv.PolyData(coords[pick].astype(np.float32))
            try:
                stream = grid.streamlines_from_source(
                    seed_cloud,
                    vectors="v",
                    max_length=vec.streamline_max_time,
                    integration_direction="both",
                )
                if stream.n_points > 0:
                    tube = stream.tube(radius=vec.streamline_radius)
                    actors.append(
                        plotter.add_mesh(
                            tube,
                            color=vec.streamline_fixed_color,
                            opacity=vec.streamline_opacity,
                        )
                    )
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass

    actors.append(
        plotter.add_text(
            f"Labels={nlab} (all)  T={tt}",
            position="upper_left",
            font_size=12,
        )
    )
    return actors


def _flowshow_notebook(
    *,
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    mask: np.ndarray,
    centerline_mask: Image | np.ndarray | None,
    stride: int,
    timepoint: int,
    show: bool,
    show_all_labels: bool,
    max_glyphs: int,
    depth_peeling: bool,
    vec: FlowshowVectorOptions,
    anim: FlowshowAnimationOptions,
    stream_seed: int | None,
) -> Any:
    _ = centerline_mask
    pv = _require_pyvista()
    widgets = _require_widgets()

    vel = _velocity_from_phases(ap, rl, fh)
    _, _, _, t, _ = vel.shape

    labels = sorted(int(v) for v in np.unique(mask) if int(v) != 0)
    if not labels:
        raise ValidationError("vessel_mask has no nonzero labels.")

    # Centerlines (computed once; rendering/picking handled elsewhere).
    cl_np = None
    if centerline_mask is not None:
        if isinstance(centerline_mask, Image):
            cl_np = to_numpy(centerline_mask.data)
        else:
            cl_np = np.asarray(centerline_mask)
    centerlines = compute_centerlines(mask, centerline_mask=cl_np, labels=labels)

    cache_all = (
        _precompute_coords_all_labels(mask, labels, stride, max_glyphs, stream_seed)
        if anim.precompute_glyph_indices and show_all_labels
        else None
    )
    coord_blocks = _coord_blocks_for_speed_clim(
        mask, labels, stride, max_glyphs, stream_seed, cache_all
    )
    if vec.speed_clim is not None:
        speed_clim_eff = vec.speed_clim
    else:
        speed_clim_eff = _estimate_speed_clim(vel, coord_blocks)

    w_label = widgets.Dropdown(options=labels, value=labels[0], description="Label")
    w_label.layout.display = "none" if show_all_labels else "flex"

    w_t = widgets.IntSlider(value=int(np.clip(timepoint, 0, t - 1)), min=0, max=t - 1, step=1, description="T")
    w_show_mask = widgets.Checkbox(value=True, description="Mask surface")
    w_show_glyphs = widgets.Checkbox(value=True, description="Velocity glyphs")
    w_show_stream = widgets.Checkbox(value=False, description="Streamlines")
    interval_ms = max(int(round(1000.0 / max(anim.animation_fps, 0.25))), 30)
    w_play = widgets.Play(
        interval=interval_ms,
        value=w_t.value,
        min=0,
        max=t - 1,
        step=1,
        description="Play",
    )
    widgets.jslink((w_play, "value"), (w_t, "value"))

    plotter = pv.Plotter(notebook=True)
    if depth_peeling:
        plotter.enable_depth_peeling()

    def _clear_dynamic():
        plotter.clear()

    def _single_cache_for(lbl: int) -> np.ndarray:
        if anim.precompute_glyph_indices:
            return _precompute_coords_single_label(mask, int(lbl), stride, max_glyphs, stream_seed)
        return np.empty((0, 3), dtype=np.int64)

    def _render():
        _clear_dynamic()
        lbl_now = int(w_label.value)
        sc = _single_cache_for(lbl_now) if (not show_all_labels and anim.precompute_glyph_indices) else None
        if show_all_labels:
            _add_flow_scene_all_labels(
                pv,
                plotter,
                labels=labels,
                tt=int(w_t.value),
                show_mask=bool(w_show_mask.value),
                show_glyphs=bool(w_show_glyphs.value),
                show_stream=bool(w_show_stream.value),
                vel=vel,
                mask=mask,
                stride=stride,
                max_glyphs=max_glyphs,
                vec=vec,
                speed_clim_eff=speed_clim_eff,
                stream_seed=stream_seed,
                cache=cache_all,
                use_cache=anim.precompute_glyph_indices,
            )
        else:
            _add_flow_scene_single(
                pv,
                plotter,
                lbl=lbl_now,
                tt=int(w_t.value),
                show_mask=bool(w_show_mask.value),
                show_glyphs=bool(w_show_glyphs.value),
                show_stream=bool(w_show_stream.value),
                vel=vel,
                mask=mask,
                stride=stride,
                max_glyphs=max_glyphs,
                vec=vec,
                speed_clim_eff=speed_clim_eff,
                stream_seed=stream_seed,
                cache_coords=sc,
                use_cache=anim.precompute_glyph_indices,
            )
        if show:
            plotter.show(auto_close=False)

    for w in (w_label, w_t, w_show_mask, w_show_glyphs, w_show_stream):
        w.observe(lambda _ch: _render(), names="value")
    _render()

    if anim.auto_play and anim.loop:
        w_play.play = True

    ui = widgets.VBox(
        [
            widgets.HBox([w_label, w_t, w_play]),
            widgets.HBox([w_show_mask, w_show_glyphs, w_show_stream]),
        ]
    )
    return widgets.VBox([ui])


def _flowshow_desktop(
    *,
    ap: np.ndarray,
    rl: np.ndarray,
    fh: np.ndarray,
    mask: np.ndarray,
    centerline_mask: Image | np.ndarray | None,
    stride: int,
    timepoint: int,
    show: bool,
    show_all_labels: bool,
    max_glyphs: int,
    depth_peeling: bool,
    vec: FlowshowVectorOptions,
    anim: FlowshowAnimationOptions,
    stream_seed: int | None,
    dt_seconds: float | None,
    cross_section_volumes: dict[str, np.ndarray] | None,
    centerline_window: int,
    cross_section_radius_vox: float,
    cross_section_res: int,
    show_gradient: bool,
) -> Any:
    pv = _require_pyvista()

    vel = _velocity_from_phases(ap, rl, fh)
    _, _, _, t, _ = vel.shape

    labels = sorted(int(v) for v in np.unique(mask) if int(v) != 0)
    if not labels:
        raise ValidationError("vessel_mask has no nonzero labels.")

    # Centerlines: provided mask wins, else skeletonize from vessel mask.
    cl_np = None
    if centerline_mask is not None:
        if isinstance(centerline_mask, Image):
            cl_np = to_numpy(centerline_mask.data)
        else:
            cl_np = np.asarray(centerline_mask)
    centerlines = compute_centerlines(mask, centerline_mask=cl_np, labels=labels)

    cache_all = (
        _precompute_coords_all_labels(mask, labels, stride, max_glyphs, stream_seed)
        if anim.precompute_glyph_indices and show_all_labels
        else None
    )
    coord_blocks = _coord_blocks_for_speed_clim(
        mask, labels, stride, max_glyphs, stream_seed, cache_all
    )
    if vec.speed_clim is not None:
        speed_clim_eff = vec.speed_clim
    else:
        speed_clim_eff = _estimate_speed_clim(vel, coord_blocks)

    state: dict[str, Any] = {
        "label_idx": 0,
        "tt": int(np.clip(timepoint, 0, t - 1)),
        "mask": True,
        "glyphs": True,
        "centerlines": True,
        "stream": False,
        "pathlines": False,
        "camera_ready": False,
        "playing": bool(anim.auto_play),
    }

    # 1 row × 2 columns:
    # - (0,0): 3D viewer
    # - (0,1): cross-section panel (populated in later todos)
    plotter = pv.Plotter(shape=(1, 2), notebook=False)
    if depth_peeling:
        plotter.enable_depth_peeling()

    # Force an ~80/20 viewport split (left=3D, right=2D).
    # PyVista's `shape` subplots are equal by default; we override renderer viewports.
    try:
        r0 = plotter.renderers[0]
        r1 = plotter.renderers[1]
        r0.SetViewport(0.0, 0.0, 0.80, 1.0)
        r1.SetViewport(0.80, 0.0, 1.00, 1.0)
    except Exception as e:
        log.error(traceback.format_exc())
        log.exception(e)
        pass

    # ---------------------------------------------------------------------
    # Interactor strategy:
    # Restrict interactions in the right cross-section viewport to wheel-zoom only.
    # Left viewport keeps normal 3D trackball camera controls.
    # ---------------------------------------------------------------------
    try:
        import vtk  # type: ignore

        vtk_iren = _unwrap_vtk_interactor(plotter)
        if vtk_iren is not None and hasattr(vtk_iren, "SetInteractorStyle"):

            class _RightPanelWheelOnlyStyle(vtk.vtkInteractorStyleTrackballCamera):  # type: ignore[misc]
                def __init__(self, *, left_renderer: Any, right_renderer: Any) -> None:
                    super().__init__()
                    self._left_renderer = left_renderer
                    self._right_renderer = right_renderer
                    self._img_style = vtk.vtkInteractorStyleImage()

                def _in_right_viewport(self) -> bool:
                    try:
                        x, y = self.GetInteractor().GetEventPosition()
                        rw = self.GetInteractor().GetRenderWindow()
                        w, h = rw.GetSize()
                    except Exception:
                        return False
                    if w <= 0 or h <= 0:
                        return False
                    xn = float(x) / float(w)
                    yn = float(y) / float(h)
                    return (0.80 <= xn <= 1.0) and (0.0 <= yn <= 1.0)

                # Disable drag/click interactions on right panel.
                def OnLeftButtonDown(self) -> None:  # noqa: N802
                    if self._in_right_viewport():
                        return
                    super().OnLeftButtonDown()

                def OnLeftButtonUp(self) -> None:  # noqa: N802
                    if self._in_right_viewport():
                        return
                    super().OnLeftButtonUp()

                def OnMiddleButtonDown(self) -> None:  # noqa: N802
                    if self._in_right_viewport():
                        return
                    super().OnMiddleButtonDown()

                def OnMiddleButtonUp(self) -> None:  # noqa: N802
                    if self._in_right_viewport():
                        return
                    super().OnMiddleButtonUp()

                def OnRightButtonDown(self) -> None:  # noqa: N802
                    if self._in_right_viewport():
                        return
                    super().OnRightButtonDown()

                def OnRightButtonUp(self) -> None:  # noqa: N802
                    if self._in_right_viewport():
                        return
                    super().OnRightButtonUp()

                def OnMouseMove(self) -> None:  # noqa: N802
                    if self._in_right_viewport():
                        return
                    super().OnMouseMove()

                # Allow wheel zoom in right panel using an Image-style zoom handler.
                def OnMouseWheelForward(self) -> None:  # noqa: N802
                    if self._in_right_viewport():
                        try:
                            self._img_style.SetInteractor(self.GetInteractor())
                            self._img_style.SetCurrentRenderer(self._right_renderer)
                            self._img_style.OnMouseWheelForward()
                            return
                        except Exception:
                            pass
                    super().OnMouseWheelForward()

                def OnMouseWheelBackward(self) -> None:  # noqa: N802
                    if self._in_right_viewport():
                        try:
                            self._img_style.SetInteractor(self.GetInteractor())
                            self._img_style.SetCurrentRenderer(self._right_renderer)
                            self._img_style.OnMouseWheelBackward()
                            return
                        except Exception:
                            pass
                    super().OnMouseWheelBackward()

            vtk_iren.SetInteractorStyle(_RightPanelWheelOnlyStyle(left_renderer=plotter.renderers[0], right_renderer=plotter.renderers[1]))
    except Exception as e:
        log.error(traceback.format_exc())
        log.exception(e)
        pass

    # Right panel placeholder (so the layout is stable and readable).
    plotter.subplot(0, 1)
    plotter.add_text(
        "Cross-section\n(click vessel/centerline to populate)",
        position="upper_left",
        font_size=12,
    )
    try:
        plotter.view_xy()
        plotter.camera.parallel_projection = True
    except Exception as e:
        log.error(traceback.format_exc())
        log.exception(e)
        pass

    # Switch back to 3D panel for the main scene.
    plotter.subplot(0, 0)

    # Centerline rendering (thin colored polylines). Picking will be added in a later todo.
    centerline_actors: list[Any] = []
    for i, lbl in enumerate(labels):
        pts = centerlines.get(int(lbl))
        if pts is None:
            continue
        pts_arr = np.asarray(pts, dtype=np.float32)
        # Normalize centerlines to a stable ndarray shape so later picking/rendering
        # never accidentally hits a plain Python list.
        centerlines[int(lbl)] = pts_arr
        if pts_arr.ndim != 2 or pts_arr.shape[1] != 3 or pts_arr.shape[0] < 2:
            continue
        try:
            pl = pv.lines_from_points(pts_arr, close=False)
            color = _TAB10_HEX[i % len(_TAB10_HEX)]
            centerline_actors.append(plotter.add_mesh(pl, color=color, line_width=3, opacity=0.9))
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass

    def _set_centerlines_visible(on: bool) -> None:
        for a in centerline_actors:
            try:
                a.SetVisibility(bool(on))
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass

    # Picking state: selected centerline point (label + index).
    selection: dict[str, Any] = {"label": None, "index": None, "point": None}
    pick_marker = None

    def _set_right_panel_text(text: str) -> None:
        try:
            plotter.subplot(0, 1)
            # Never call plotter.clear(): it clears the whole plotter and can wipe the 3D renderer.
            _clear_right_planes()
            plotter.add_text(text, position="upper_left", font_size=12)
            plotter.view_xy()
            plotter.camera.parallel_projection = True
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass
        finally:
            try:
                plotter.subplot(0, 0)
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass

    right_plane_actors: list[Any] = []
    # Cross-section panel shows up to 3 slices (CD/Angio/VelMag) simultaneously.
    overlay_actors: list[Any] = []

    def _clear_right_planes() -> None:
        nonlocal right_plane_actors
        for a in right_plane_actors:
            try:
                # Right panel may contain either PyVista actors or raw vtkProp (vtkImageActor).
                try:
                    ren = plotter.renderers[1]
                    rm = getattr(ren, "RemoveActor", None)
                    if callable(rm):
                        rm(a)  # type: ignore[arg-type]
                        continue
                except Exception:
                    log.exception(e)
                    pass
                plotter.remove_actor(a)
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass
        right_plane_actors = []

    def _render_cross_sections() -> None:
        """Render CD/Angio/VelMag oblique slices in the right subplot."""
        if selection["point"] is None or selection["label"] is None or selection["index"] is None:
            _set_right_panel_text("Cross-section\n(click vessel/centerline to populate)")
            return
        if not cross_section_volumes:
            _set_right_panel_text("Cross-section volumes not provided.\n(Will be loaded by CLI options.)")
            return
        lbl = int(selection["label"])
        idx = int(selection["index"])
        pts = centerlines.get(lbl)
        if pts is None:
            _set_right_panel_text(f"Cross-section unavailable:\nno centerline for label={lbl}")
            return
        pts_arr = np.asarray(pts, dtype=np.float32)
        if pts_arr.ndim != 2 or pts_arr.shape[1] != 3 or pts_arr.shape[0] < 2:
            _set_right_panel_text(f"Cross-section unavailable:\ninvalid centerline for label={lbl}")
            return
        if idx < 0 or idx >= int(pts_arr.shape[0]):
            _set_right_panel_text(f"Cross-section unavailable:\ncenterline index out of range ({idx})")
            return
        center = pts_arr[idx].astype(np.float32, copy=False)
        tvec = _tangent_from_centerline(pts_arr, idx, window=centerline_window)
        # flip tangent using local velocity direction (timepoint-specific)
        cx, cy, cz = int(round(float(center[0]))), int(round(float(center[1]))), int(round(float(center[2])))
        cx = int(np.clip(cx, 0, vel.shape[0] - 1))
        cy = int(np.clip(cy, 0, vel.shape[1] - 1))
        cz = int(np.clip(cz, 0, vel.shape[2] - 1))
        tt_now = int(np.clip(int(state["tt"]), 0, vel.shape[3] - 1))
        vloc = vel[cx, cy, cz, tt_now, :].astype(np.float32, copy=False)
        if float(np.dot(vloc, tvec)) < 0:
            tvec = -tvec
        u, v = _frame_from_tangent(tvec)

        # 3D overlays (red): plane + flow-direction arrow. Keep them updated.
        try:
            plotter.subplot(0, 0)
            for a in overlay_actors:
                try:
                    plotter.remove_actor(a)
                except Exception as e:
                    log.error(traceback.format_exc())
                    log.exception(e)
                    pass
            overlay_actors.clear()

            plane3d = pv.Plane(
                center=tuple(float(x) for x in center),
                direction=tuple(float(x) for x in tvec),
                i_size=2.0 * float(cross_section_radius_vox),
                j_size=2.0 * float(cross_section_radius_vox),
                i_resolution=1,
                j_resolution=1,
            )
            overlay_actors.append(
                plotter.add_mesh(
                    plane3d,
                    color="red",
                    opacity=0.25,
                    show_scalar_bar=False,
                    style="wireframe",
                    line_width=2,
                )
            )
            arr = pv.Arrow(
                start=tuple(float(x) for x in center),
                direction=tuple(float(x) for x in tvec),
                scale=float(cross_section_radius_vox) * 0.75,
            )
            overlay_actors.append(plotter.add_mesh(arr, color="red", opacity=0.9, show_scalar_bar=False))
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass

        # Build selected slice + overlay mask slice (small panel).
        mask_bin = (mask == lbl).astype(np.uint8, copy=False)
        msl = _oblique_slice(mask_bin, center=center, u=u, v=v, radius_vox=cross_section_radius_vox, res=cross_section_res, order=0)
        msl = (msl > 0.5).astype(np.float32)

        plotter.subplot(0, 1)
        _clear_right_planes()
        plotter.camera.parallel_projection = True

        # Strategy change: render 2D slices using VTK image actors instead of textured planes.
        # This is more reliable across VTK/PyVista backends and avoids camera/plane issues.
        preferred = ("ComplexDifference_3D", "Angiography_3D", "VelocityMagnitude_3D")
        avail = [k for k in preferred if k in cross_section_volumes]
        if not avail:
            _set_right_panel_text("Cross-section volumes missing.\nExpected CD/Angio/VelMag.")
            return

        try:
            import vtk  # type: ignore
            from vtk.util import numpy_support  # type: ignore

            def _vtk_image_from_2d(img2d: np.ndarray) -> Any:
                arr = np.asarray(img2d, dtype=np.float32)
                # VTK expects x-fastest; keep a consistent orientation for display.
                flat = arr.T.reshape(-1, order="C").astype(np.float32, copy=False)
                img = vtk.vtkImageData()
                img.SetDimensions(int(arr.shape[1]), int(arr.shape[0]), 1)
                img.AllocateScalars(vtk.VTK_FLOAT, 1)
                vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
                img.GetPointData().SetScalars(vtk_arr)
                return img

            def _make_lut_gray(lo: float, hi: float) -> Any:
                lut = vtk.vtkLookupTable()
                lut.SetNumberOfTableValues(256)
                lut.SetRange(float(lo), float(hi))
                lut.SetRampToLinear()
                lut.Build()
                for i in range(256):
                    g = float(i) / 255.0
                    lut.SetTableValue(i, g, g, g, 1.0)
                return lut

            def _make_lut_mask() -> Any:
                lut = vtk.vtkLookupTable()
                lut.SetNumberOfTableValues(2)
                lut.SetRange(0.0, 1.0)
                lut.Build()
                lut.SetTableValue(0, 0.0, 0.0, 0.0, 0.0)   # transparent
                lut.SetTableValue(1, 1.0, 0.0, 0.0, 0.30)  # red overlay
                return lut

            # Pre-build mask overlay actor once (reused per slice with same msl).
            mask_img = _vtk_image_from_2d(msl)
            mask_lut = _make_lut_mask()

            def _add_image_pair(img2d: np.ndarray, yoff: float) -> None:
                base_img = _vtk_image_from_2d(img2d)
                # intensity windowing
                vmin = float(np.percentile(img2d, 2))
                vmax = float(np.percentile(img2d, 98))
                if vmax <= vmin:
                    vmax = vmin + 1e-6
                gray_lut = _make_lut_gray(vmin, vmax)

                map_base = vtk.vtkImageMapToColors()
                map_base.SetInputData(base_img)
                map_base.SetLookupTable(gray_lut)
                map_base.Update()

                act_base = vtk.vtkImageActor()
                act_base.GetMapper().SetInputConnection(map_base.GetOutputPort())
                act_base.SetPosition(0.0, float(yoff), 0.0)

                map_mask = vtk.vtkImageMapToColors()
                map_mask.SetInputData(mask_img)
                map_mask.SetLookupTable(mask_lut)
                map_mask.Update()

                act_mask = vtk.vtkImageActor()
                act_mask.GetMapper().SetInputConnection(map_mask.GetOutputPort())
                act_mask.SetPosition(0.0, float(yoff), 0.0)

                # Add to renderer 1 directly to avoid subplot confusion.
                ren = plotter.renderers[1]
                ren.AddActor(act_base)
                ren.AddActor(act_mask)
                right_plane_actors.extend([act_base, act_mask])

            dy_pix = float(cross_section_res) * 1.10
            y_offsets = [dy_pix, 0.0, -dy_pix]
            for kk, key in enumerate(avail[:3]):
                sl = _oblique_slice(
                    cross_section_volumes[key],
                    center=center,
                    u=u,
                    v=v,
                    radius_vox=cross_section_radius_vox,
                    res=cross_section_res,
                    order=1,
                )
                _add_image_pair(sl, float(y_offsets[kk]))
                try:
                    plotter.add_text(str(key), position=(8, 18 + 14 * kk), font_size=10)
                except Exception as e:
                    log.error(traceback.format_exc())
                    log.exception(e)
                    pass
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            # If VTK image actors fail (rare), fall back to the textured-plane method.
            dy = 2.2 * float(cross_section_radius_vox)
            y_offsets = [dy, 0.0, -dy]
            for kk, key in enumerate(avail[:3]):
                sl = _oblique_slice(
                    cross_section_volumes[key],
                    center=center,
                    u=u,
                    v=v,
                    radius_vox=cross_section_radius_vox,
                    res=cross_section_res,
                    order=1,
                )
                yoff = float(y_offsets[kk])
                plane = pv.Plane(
                    center=(0.0, yoff, 0.0),
                    direction=(0.0, 0.0, 1.0),
                    i_size=2.0 * float(cross_section_radius_vox),
                    j_size=2.0 * float(cross_section_radius_vox),
                    i_resolution=int(cross_section_res - 1),
                    j_resolution=int(cross_section_res - 1),
                )
                plane.point_data["img"] = sl.T.flatten(order="C").astype(np.float32, copy=False)
                a_img = plotter.add_mesh(plane, scalars="img", cmap="gray", opacity=1.0, show_scalar_bar=False)
                plane2 = plane.copy(deep=True)
                plane2.point_data["m"] = msl.T.flatten(order="C").astype(np.float32, copy=False)
                a_m = plotter.add_mesh(plane2, scalars="m", cmap="Reds", opacity=0.25, show_scalar_bar=False)
                right_plane_actors.extend([a_img, a_m])

        plotter.view_xy()
        try:
            plotter.reset_camera()
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass
        try:
            plotter.camera.zoom(1.0)
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass

        # back to 3D
        plotter.subplot(0, 0)
        try:
            plotter.render()
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass

    def _select_nearest_centerline(picked_xyz: Any) -> None:
        nonlocal pick_marker
        arr = np.asarray(picked_xyz, dtype=np.float32)
        # Some pickers/callbacks return a list of points (N,3) instead of a single (3,) point.
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 1:
            arr = arr[0]
        p = arr.reshape(1, 3)
        best: tuple[float, int, int] | None = None  # (d2, label, idx)
        for lbl, pts in centerlines.items():
            if pts is None:
                continue
            pts_arr = np.asarray(pts, dtype=np.float32)
            if pts_arr.ndim != 2 or pts_arr.shape[0] == 0 or pts_arr.shape[1] != 3:
                continue
            d2 = np.sum((pts_arr - p) ** 2, axis=1)
            j = int(np.argmin(d2))
            v = float(d2[j])
            if best is None or v < best[0]:
                best = (v, int(lbl), j)
        if best is None:
            return
        _, lbl, j = best
        pt = np.asarray(centerlines[lbl], dtype=np.float32)[j]
        selection["label"] = lbl
        selection["index"] = j
        selection["point"] = pt

        # Update marker (small red sphere).
        try:
            if pick_marker is not None:
                plotter.remove_actor(pick_marker)
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass
        try:
            sph = pv.Sphere(radius=1.5, center=tuple(float(x) for x in pt))
            pick_marker = plotter.add_mesh(sph, color="red", opacity=0.9)
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pick_marker = None

        _render_cross_sections()
        plotter.render()

        # Debug hint in the HUD (helps confirm picking is firing).
        try:
            plotter.subplot(0, 0)
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass

    # Enable point picking (click in 3D view). We map picks to the nearest centerline point.
    try:
        def _on_pick(*args: Any) -> None:
            # PyVista callback signatures vary:
            # - callback(point)
            # - callback(picker, event)
            if not args:
                return
            point = None
            if len(args) == 1:
                point = args[0]
            else:
                picker = args[0]
                point = getattr(picker, "pick_position", None)
                if point is None:
                    gp = getattr(picker, "GetPickPosition", None)
                    point = gp() if callable(gp) else None
            if point is None:
                return
            _select_nearest_centerline(point)

        # IMPORTANT: keep picking in the 3D renderer only (the right panel is passive).
        try:
            plotter.enable_point_picking(
                callback=_on_pick,
                show_message=True,
                font_size=10,
                color="red",
                point_size=10,
                use_picker=True,
            )
        except TypeError:
            plotter.enable_point_picking(
                callback=_on_pick,
                show_message=True,
                font_size=10,
                color="red",
                point_size=10,
            )
    except Exception:
        import traceback
        log.error(traceback.format_exc())
        # Older PyVista / VTK builds may not support this API; we’ll still work without picking.
        pass

    # Robust picking fallback (VTK): observe left-clicks and run a renderer-0 cell pick.
    # This avoids silent failures across PyVista/VTK backends where enable_point_picking
    # might not fire or might not return a usable point.
    try:
        import vtk  # type: ignore

        vtk_iren = _unwrap_vtk_interactor(plotter)
        if vtk_iren is not None:
            picker = vtk.vtkCellPicker()
            try:
                picker.SetTolerance(0.0025)
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass

            def _is_in_left_viewport(xy: tuple[int, int]) -> bool:
                x, y = int(xy[0]), int(xy[1])
                try:
                    rw = vtk_iren.GetRenderWindow()
                    w, h = rw.GetSize()
                except Exception:
                    try:
                        w, h = int(plotter.window_size[0]), int(plotter.window_size[1])
                    except Exception:
                        w, h = 1, 1
                if w <= 0 or h <= 0:
                    return True
                xn = float(x) / float(w)
                yn = float(y) / float(h)
                # renderer 0 viewport is [0, 0, 0.80, 1]
                return (0.0 <= xn <= 0.80) and (0.0 <= yn <= 1.0)

            def _vtk_pick_cb(_obj: Any, _evt: Any) -> None:
                try:
                    x, y = vtk_iren.GetEventPosition()
                except Exception:
                    return
                if not _is_in_left_viewport((x, y)):
                    # Ignore clicks on the right (cross-section) panel.
                    try:
                        st = vtk_iren.GetInteractorStyle()
                        if st is not None:
                            st.OnLeftButtonDown()
                    except Exception as e:
                        log.error(traceback.format_exc())
                        log.exception(e)
                        pass
                    return
                try:
                    ok = int(picker.Pick(float(x), float(y), 0.0, plotter.renderers[0]))  # type: ignore[arg-type]
                except Exception as e:
                    log.error(traceback.format_exc())
                    log.exception(e)
                    ok = 0
                if ok:
                    try:
                        pt = picker.GetPickPosition()
                        if pt is not None:
                            _select_nearest_centerline(np.asarray(pt, dtype=np.float32))
                    except Exception as e:
                        import traceback
                        log.error(traceback.format_exc())
                        log.exception(e)
                        pass
                # Preserve normal camera interaction.
                try:
                    st = vtk_iren.GetInteractorStyle()
                    if st is not None:
                        st.OnLeftButtonDown()
                except Exception as e:
                    log.error(traceback.format_exc())
                    log.exception(e)
                    pass

            vtk_iren.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, _vtk_pick_cb)
    except Exception as e:
        log.error(traceback.format_exc())
        log.exception(e)
        pass

    # Make the 2D panel passive: lock camera and suppress interaction hints.
    # (VTK interactor styles are global; this is best-effort without breaking 3D controls.)
    try:
        plotter.subplot(0, 1)
        plotter.view_xy()
        plotter.camera.parallel_projection = True
    except Exception as e:
        log.error(traceback.format_exc())
        log.exception(e)
        pass
    finally:
        try:
            plotter.subplot(0, 0)
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass

    # ---------------------------------------------------------------------
    # Fast desktop rendering:
    # - mask surfaces are static (3D mask), so we build them once
    # - glyphs are VTK pipelines with fixed point sets; per-frame update only
    #   writes new vectors/scalars and renders (no rebuild)
    # ---------------------------------------------------------------------

    # Mask surfaces: cache PolyData so we can re-color without recomputing contours.
    mask_surfaces: list[tuple[int, Any, str]] = []  # (label, surface, base_color)
    mask_actors: list[Any] = []
    if show_all_labels:
        for i, lbl in enumerate(labels):
            roi = mask == int(lbl)
            if not np.any(roi):
                continue
            surf = _mask_surface(pv, roi)
            if surf is None:
                continue
            color = _TAB10_HEX[i % len(_TAB10_HEX)]
            mask_surfaces.append((int(lbl), surf, color))
            mask_actors.append(
                plotter.add_mesh(
                    surf,
                    color=color,
                    opacity=0.34,
                    smooth_shading=True,
                    show_scalar_bar=False,
                )
            )
    else:
        # single-label mode: surface for current label (recomputed only when label changes)
        roi = mask == int(labels[int(state["label_idx"])])
        surf = _mask_surface(pv, roi)
        if surf is not None:
            mask_surfaces.append((int(labels[int(state["label_idx"])]), surf, "white"))
            mask_actors.append(plotter.add_mesh(surf, color="white", opacity=0.25, smooth_shading=True))

    glyph_pipes: list[_VtkGlyphPipeline] = []
    glyph_handles: list[Any] = []
    stream_actors: list[Any] = []
    stream_cache: dict[tuple[int, int, float, float, int | None], Any] = {}
    stream_cache_order: list[tuple[int, int, float, float, int | None]] = []
    stream_last_key: dict[str, Any] = {"key": None}
    stream_seeds_cache: dict[tuple[int, int | None], Any] = {}
    stream_cache_max = 8

    def _clear_streamlines() -> None:
        for a in stream_actors:
            try:
                plotter.remove_actor(a)
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass
        stream_actors.clear()

    def _get_stream_seed_cloud(*, nseed: int, seed: int | None) -> Any | None:
        """Stable seed cloud so animated streamlines don't jump between frames."""
        if nseed <= 0:
            return None
        key = (int(nseed), int(seed) if seed is not None else None)
        if key in stream_seeds_cache:
            return stream_seeds_cache[key]
        roi_u = (mask > 0)
        coords = np.argwhere(roi_u)
        nseed_eff = min(int(nseed), int(coords.shape[0])) if coords.shape[0] > 0 else 0
        if nseed_eff <= 0:
            return None
        rng = np.random.default_rng(seed)
        pick = rng.choice(coords.shape[0], size=nseed_eff, replace=False)
        seed_cloud = pv.PolyData(coords[pick].astype(np.float32))
        stream_seeds_cache[key] = seed_cloud
        return seed_cloud

    def _cache_put_stream(key: tuple[int, int, float, float, int | None], tube: Any) -> None:
        stream_cache[key] = tube
        stream_cache_order.append(key)
        # simple FIFO cap
        if len(stream_cache_order) > int(stream_cache_max):
            old = stream_cache_order.pop(0)
            stream_cache.pop(old, None)

    def _build_streamlines(tt0: int) -> None:
        _clear_streamlines()
        if not state["stream"]:
            return
        try:
            # Always render streamlines in the left 3D panel.
            try:
                plotter.subplot(0, 0)
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass
            x, y, z, _, _ = vel.shape
            vec_f = vel[..., int(tt0), :]
            grid = pv.ImageData(dimensions=(x, y, z), spacing=(1, 1, 1), origin=(0, 0, 0))
            grid.point_data["v"] = vec_f.reshape(-1, 3, order="F")
            key = (
                int(tt0),
                int(vec.streamline_n_seeds),
                float(vec.streamline_max_time),
                float(vec.streamline_radius),
                int(stream_seed) if stream_seed is not None else None,
            )
            if key in stream_cache:
                tube = stream_cache[key]
            else:
                seed_cloud = _get_stream_seed_cloud(nseed=int(vec.streamline_n_seeds), seed=stream_seed)
                if seed_cloud is None:
                    return
                stream = grid.streamlines_from_source(
                    seed_cloud,
                    vectors="v",
                    max_length=vec.streamline_max_time,
                    integration_direction="both",
                )
                if stream.n_points <= 0:
                    return
                tube = stream.tube(radius=vec.streamline_radius)
                _cache_put_stream(key, tube)
            stream_last_key["key"] = key
            stream_actors.append(
                plotter.add_mesh(
                    tube,
                    color=vec.streamline_fixed_color,
                    opacity=vec.streamline_opacity,
                    pickable=False,
                )
            )
            # Ensure the checkbox state matches actual visibility after creation.
            state["stream"] = True
        except Exception as e:
            import traceback
            log.error(traceback.format_exc())
            log.exception(e)
            pass

    def _clear_glyphs() -> None:
        for h in glyph_handles:
            try:
                plotter.remove_actor(h)
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass
        glyph_handles.clear()
        glyph_pipes.clear()

    def _build_glyphs(tt0: int) -> None:
        _clear_glyphs()
        if not state["glyphs"]:
            return

        if show_all_labels:
            eff_cache = cache_all if (anim.precompute_glyph_indices and cache_all is not None) else _precompute_coords_all_labels(
                mask, labels, stride, max_glyphs, stream_seed
            )
            for coords, idx in eff_cache.parts:
                if coords.size == 0:
                    continue
                color = _TAB10_HEX[idx % len(_TAB10_HEX)]
                gp = _VtkGlyphPipeline(
                    coords=coords,
                    color=color,
                    vel=vel,
                    vec=vec,
                    speed_clim_eff=speed_clim_eff,
                    tt0=tt0,
                )
                glyph_pipes.append(gp)
                plotter.add_actor(gp.actor)
                glyph_handles.append(gp.actor)
        else:
            lbl = labels[int(state["label_idx"])]
            coords = _precompute_coords_single_label(mask, int(lbl), stride, max_glyphs, stream_seed)
            gp = _VtkGlyphPipeline(
                coords=coords,
                color=_TAB10_HEX[0],
                vel=vel,
                vec=vec,
                speed_clim_eff=speed_clim_eff,
                tt0=tt0,
            )
            glyph_pipes.append(gp)
            plotter.add_actor(gp.actor)
            glyph_handles.append(gp.actor)

    hud = plotter.add_text("", position="upper_left", font_size=12)

    def _set_mask_visible(on: bool) -> None:
        for a in mask_actors:
            try:
                a.SetVisibility(bool(on))
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass

    # Field overlays: (a) surface coloring (legacy) and (b) interior point cloud.
    mask_field = {"speed": False, "radial": False}
    interior_field = {
        "enabled": bool(show_gradient),
        "mode": "speed*radial",
        "n_points": 80_000,
        "opacity": 0.55,
        "point_size": 6.0,
    }
    interior_actor = {"actor": None}

    def _clear_interior() -> None:
        a = interior_actor.get("actor")
        if a is None:
            return
        try:
            plotter.remove_actor(a)
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass
        interior_actor["actor"] = None

    def _render_interior_field() -> None:
        """Render a dense point cloud inside vessels with a scalar field."""
        _clear_interior()
        if not interior_field["enabled"]:
            return
        tt_now = int(state["tt"])
        speed = np.linalg.norm(vel[..., tt_now, :], axis=-1).astype(np.float32, copy=False)

        pts_all = np.argwhere(mask > 0)
        if pts_all.shape[0] == 0:
            return
        rng = np.random.default_rng(0)
        n = int(min(int(interior_field["n_points"]), pts_all.shape[0]))
        sel = pts_all[rng.choice(pts_all.shape[0], size=n, replace=False)]

        scal = speed[sel[:, 0], sel[:, 1], sel[:, 2]].astype(np.float32, copy=False)

        if "radial" in str(interior_field["mode"]):
            try:
                from scipy.ndimage import distance_transform_edt  # type: ignore
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                distance_transform_edt = None
            if distance_transform_edt is not None:
                # radial normalized per-label
                radial = np.zeros_like(scal)
                for lbl in np.unique(mask[sel[:, 0], sel[:, 1], sel[:, 2]]):
                    lbl = int(lbl)
                    if lbl == 0:
                        continue
                    roi = (mask == lbl)
                    dist = distance_transform_edt(roi).astype(np.float32, copy=False)
                    dmax = float(np.percentile(dist[roi], 99)) if np.any(roi) else 1.0
                    dmax = max(dmax, 1e-6)
                    idx = (mask[sel[:, 0], sel[:, 1], sel[:, 2]] == lbl)
                    radial[idx] = dist[sel[idx, 0], sel[idx, 1], sel[idx, 2]] / dmax
                radial = np.clip(radial, 0.0, 1.0)
                if str(interior_field["mode"]) == "radial":
                    scal = radial
                else:
                    scal = scal * radial

        # Contrast-stretch for visibility.
        lo = float(np.percentile(scal, 5))
        hi = float(np.percentile(scal, 95))
        if hi <= lo:
            hi = lo + 1e-6
        scal_n = np.clip((scal - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)

        poly = pv.PolyData(sel.astype(np.float32, copy=False))
        poly.point_data["field"] = scal_n
        interior_actor["actor"] = plotter.add_mesh(
            poly,
            scalars="field",
            cmap="turbo",
            opacity=float(interior_field["opacity"]),
            point_size=float(interior_field["point_size"]),
            render_points_as_spheres=True,
            show_scalar_bar=False,
        )

    def _rebuild_mask_actors() -> None:
        # Remove current actors
        for a in mask_actors:
            try:
                plotter.remove_actor(a)
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass
        mask_actors.clear()

        tt_now = int(state["tt"])
        speed = np.linalg.norm(vel[..., tt_now, :], axis=-1).astype(np.float32, copy=False)
        # speed dynamic range (for speed-only and speed*radial)
        speed_clim = vec.speed_clim if vec.speed_clim is not None else (
            float(np.percentile(speed, 2)),
            float(np.percentile(speed, 98)),
        )
        if speed_clim[1] <= speed_clim[0]:
            speed_clim = (speed_clim[0], speed_clim[0] + 1e-3)

        for lbl, surf, base_color in mask_surfaces:
            if not (mask_field["speed"] or mask_field["radial"]):
                mask_actors.append(
                    plotter.add_mesh(
                        surf,
                        color=base_color,
                        opacity=0.34 if show_all_labels else 0.25,
                        smooth_shading=True,
                        show_scalar_bar=False,
                    )
                )
                continue

            pts = np.asarray(surf.points, dtype=np.float32)
            ijk = np.round(pts).astype(int)
            ijk[:, 0] = np.clip(ijk[:, 0], 0, speed.shape[0] - 1)
            ijk[:, 1] = np.clip(ijk[:, 1], 0, speed.shape[1] - 1)
            ijk[:, 2] = np.clip(ijk[:, 2], 0, speed.shape[2] - 1)
            scal = np.ones((ijk.shape[0],), dtype=np.float32)
            title = ""
            clim = None
            cmap = vec.speed_cmap
            if mask_field["speed"]:
                scal = scal * speed[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                title = "|v|"
                clim = speed_clim
            if mask_field["radial"]:
                try:
                    from scipy.ndimage import distance_transform_edt  # type: ignore
                except Exception as e:
                    log.error(traceback.format_exc())
                    log.exception(e)
                    distance_transform_edt = None
                if distance_transform_edt is not None:
                    roi = (mask == int(lbl))
                    dist = distance_transform_edt(roi).astype(np.float32, copy=False)
                    d = dist[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                    dmax = float(np.percentile(dist[roi], 99)) if np.any(roi) else 1.0
                    dmax = max(dmax, 1e-6)
                    d = (d / dmax).astype(np.float32, copy=False)
                    scal = scal * d
                    title = (title + "*r").strip("*") if title else "r"
                    # for radial-only, keep a stable 0..1 range
                    if not mask_field["speed"]:
                        clim = (0.0, 1.0)
                        cmap = "viridis"
            surf.point_data["field"] = scal
            mask_actors.append(
                plotter.add_mesh(
                    surf,
                    scalars="field",
                    cmap=cmap,
                    clim=clim,
                    opacity=0.34 if show_all_labels else 0.25,
                    smooth_shading=True,
                    show_scalar_bar=True,
                    scalar_bar_args={"title": title or "field"},
                )
            )

        # Interior point cloud is independent from surface actors.
        _render_interior_field()

    def _update_frame() -> None:
        tt_now = int(state["tt"])
        if state["glyphs"] and not glyph_pipes:
            _build_glyphs(tt_now)
        if (not state["glyphs"]) and glyph_pipes:
            _build_glyphs(tt_now)
        if state["glyphs"]:
            for gp in glyph_pipes:
                gp.vec = vec
                gp.speed_clim_eff = speed_clim_eff
                gp.set_opacity(vec.glyph_opacity)
                gp.set_scale_by_magnitude(vec.scale_by_magnitude, vec.scale_factor)
                gp.update_time(tt_now)
        if state["stream"]:
            # Rebuild if missing OR if time/config changed.
            want_key = (
                int(tt_now),
                int(vec.streamline_n_seeds),
                float(vec.streamline_max_time),
                float(vec.streamline_radius),
                int(stream_seed) if stream_seed is not None else None,
            )
            if (not stream_actors) or (stream_last_key.get("key") != want_key):
                _build_streamlines(tt_now)
        else:
            if stream_actors:
                _clear_streamlines()

        # Pathlines: lazily compute once per config and render when toggled on.
        if state.get("pathlines", False):
            dt_eff = float(dt_seconds) if dt_seconds is not None else 1.0
            want_pkey = (
                int(tt_now),
                int(vec.streamline_n_seeds),
                float(dt_eff),
                float(vec.streamline_max_time),
                int(stream_seed) if stream_seed is not None else None,
            )
            if (not path_actors) or (path_last_key.get("key") != want_pkey):
                _clear_pathlines()
                if want_pkey in path_cache:
                    tube = path_cache[want_pkey]
                else:
                    tube = _build_pathlines(int(tt_now), dt=float(dt_eff))
                    if tube is not None:
                        path_cache[want_pkey] = tube
                if tube is not None:
                    try:
                        plotter.subplot(0, 0)
                    except Exception:
                        pass
                    path_actors.append(
                        plotter.add_mesh(
                            tube,
                            color="#D81B60",
                            opacity=0.65,
                            pickable=False,
                        )
                    )
                path_last_key["key"] = want_pkey
        else:
            if path_actors:
                _clear_pathlines()
        hud_txt = f"T={tt_now} | " + ("all labels" if show_all_labels else f"label={labels[int(state['label_idx'])]}")
        try:
            if hasattr(hud, "SetText"):
                hud.SetText(0, hud_txt)  # type: ignore[attr-defined]
            elif hasattr(hud, "SetInput"):
                hud.SetInput(hud_txt)  # type: ignore[attr-defined]
        except Exception:
            pass
        if not state["camera_ready"]:
            plotter.view_isometric()
            state["camera_ready"] = True
        plotter.render()

    def _on_time(value: float):
        state["tt"] = int(round(value))
        if mask_field["speed"] or mask_field["radial"]:
            _rebuild_mask_actors()
            _set_mask_visible(bool(state["mask"]))
        else:
            _render_interior_field()
        _update_frame()

    def _on_label(value: float):
        state["label_idx"] = int(round(value))
        state["label_idx"] = int(np.clip(state["label_idx"], 0, len(labels) - 1))
        # rebuild glyphs (and surface in single-label mode)
        if not show_all_labels:
            # replace mask surface
            for a in mask_actors:
                try:
                    plotter.remove_actor(a)
                except Exception as e:
                    log.error(traceback.format_exc())
                    log.exception(e)
                    pass
            mask_actors.clear()
            mask_surfaces.clear()
            roi = mask == int(labels[int(state["label_idx"])])
            surf = _mask_surface(pv, roi)
            if surf is not None:
                mask_surfaces.append((int(labels[int(state["label_idx"])]), surf, "white"))
                mask_actors.append(plotter.add_mesh(surf, color="white", opacity=0.25, smooth_shading=True))
        _build_glyphs(int(state["tt"]))
        _update_frame()

    # ── Widgets layout (avoid overlaps): keep time controls on the left,
    # vector controls on the right.
    # Replace sliders with numeric input boxes (VTK text box widgets).
    try:
        import vtk  # type: ignore
        # Some VTK builds do not ship interaction widgets (vtkTextBoxWidget).
        # If missing, disable numeric text boxes gracefully.
        if not hasattr(vtk, "vtkTextBoxWidget"):
            vtk = None
    except Exception:
        vtk = None

    # Cross-sections are displayed as a 3-slice panel (no mode selector needed).

    # Use a fixed left-middle control strip; avoids resize jitter across VTK backends.
    y0 = 520

    def _add_num_box(
        *,
        label: str,
        x: int,
        y: int,
        w: int,
        h: int,
        initial: str,
        on_commit: Callable[[str], None],
        vtk_iren: Any,
    ) -> Any | None:
        if vtk is None:
            return None
        try:
            box = vtk.vtkTextBoxWidget()
            rep = vtk.vtkTextBoxRepresentation()
            rep.GetPositionCoordinate().SetCoordinateSystemToDisplay()
            rep.GetPositionCoordinate().SetValue(float(x), float(y))
            rep.GetPosition2Coordinate().SetCoordinateSystemToDisplay()
            rep.GetPosition2Coordinate().SetValue(float(w), float(h))
            box.SetRepresentation(rep)
            box.SetInteractor(vtk_iren)
            try:
                box.SetCurrentRenderer(plotter.renderers[0])
            except Exception as e:
                log.error(traceback.format_exc())
                log.exception(e)
                pass
            box.SelectableOn()
            box.SetText(initial)

            # Label text (static).
            plotter.add_text(label, position=(x, y + h + 2), font_size=9)

            def _cb(_obj: Any, _evt: Any) -> None:
                try:
                    txt = str(box.GetText())
                except Exception as e:
                    log.error(traceback.format_exc())
                    log.exception(e)
                    txt = initial
                on_commit(txt)

            box.AddObserver(vtk.vtkCommand.EndInteractionEvent, _cb)
            box.On()
            return box
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            return None

    # Keep refs so widgets are not garbage-collected.
    ui_boxes: list[Any] = []
    ui_box_built = [False]

    # Numeric text box widgets require a live VTK interactor; on many backends it
    # only exists after the first render/show. We queue the boxes and build them
    # once `_unwrap_vtk_interactor` succeeds.
    pending_boxes: list[dict[str, Any]] = []

    def _queue_num_box(
        *,
        label: str,
        x: int,
        y: int,
        w: int,
        h: int,
        initial: str,
        on_commit: Callable[[str], None],
    ) -> None:
        pending_boxes.append(
            {
                "label": label,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "initial": str(initial),
                "on_commit": on_commit,
            }
        )

    def _build_queued_num_boxes() -> None:
        if ui_box_built[0]:
            return
        if vtk is None:
            ui_box_built[0] = True
            return
        vtk_iren = _unwrap_vtk_interactor(plotter)
        if vtk_iren is None:
            return
        for spec in pending_boxes:
            b = _add_num_box(
                label=spec["label"],
                x=spec["x"],
                y=spec["y"],
                w=spec["w"],
                h=spec["h"],
                initial=spec["initial"],
                on_commit=spec["on_commit"],
                vtk_iren=vtk_iren,
            )
            if b is not None:
                ui_boxes.append(b)
        ui_box_built[0] = True

    # Place boxes in the control strip (queued; built after first render).
    y_box_top = int(y0)
    _queue_num_box(
        label="Time (0..T-1)",
        x=18,
        y=y_box_top,
        w=140,
        h=20,
        initial=str(int(state["tt"])),
        on_commit=lambda s: _on_time(float(int(np.clip(int(float(s)), 0, t - 1)))) if s.strip() else None,
    )
    if (not show_all_labels) and len(labels) > 1:
        _queue_num_box(
            label="Label idx",
            x=18,
            y=y_box_top - 34,
            w=140,
            h=20,
            initial=str(int(state["label_idx"])),
            on_commit=lambda s: _on_label(float(int(np.clip(int(float(s)), 0, len(labels) - 1)))) if s.strip() else None,
        )
    # Note: previously there was a numeric selector for which volume to show;
    # the panel now displays all available volumes simultaneously.

    def _cb_mask(val: bool):
        state["mask"] = val
        _set_mask_visible(bool(val))
        _update_frame()

    def _cb_glyphs(val: bool):
        state["glyphs"] = val
        _build_glyphs(int(state["tt"]))
        _update_frame()

    def _cb_stream(val: bool):
        state["stream"] = val
        _build_streamlines(int(state["tt"]))
        _update_frame()

    # Pathlines (true particle trajectories through time-varying field).
    path_actors: list[Any] = []
    path_cache: dict[tuple[int, int, float, float, int | None], Any] = {}
    path_last_key: dict[str, Any] = {"key": None}

    def _clear_pathlines() -> None:
        for a in path_actors:
            try:
                plotter.remove_actor(a)
            except Exception:
                pass
        path_actors.clear()

    def _sample_vel_trilinear(frame: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Trilinear sample of a (X,Y,Z,3) frame at fractional voxel position."""
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        nx, ny, nz, _ = frame.shape
        # clamp inside valid range for interpolation
        x = float(np.clip(x, 0.0, nx - 1.001))
        y = float(np.clip(y, 0.0, ny - 1.001))
        z = float(np.clip(z, 0.0, nz - 1.001))
        x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        x1, y1, z1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1), min(z0 + 1, nz - 1)
        fx, fy, fz = x - x0, y - y0, z - z0

        c000 = frame[x0, y0, z0, :]
        c100 = frame[x1, y0, z0, :]
        c010 = frame[x0, y1, z0, :]
        c110 = frame[x1, y1, z0, :]
        c001 = frame[x0, y0, z1, :]
        c101 = frame[x1, y0, z1, :]
        c011 = frame[x0, y1, z1, :]
        c111 = frame[x1, y1, z1, :]

        c00 = c000 * (1 - fx) + c100 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        c = c0 * (1 - fz) + c1 * fz
        return np.asarray(c, dtype=np.float32)

    def _build_pathlines(tt_start: int, *, dt: float) -> Any | None:
        """Integrate pathlines forward through time-varying vel field."""
        seed_cloud = _get_stream_seed_cloud(nseed=int(vec.streamline_n_seeds), seed=stream_seed)
        if seed_cloud is None:
            return None
        seeds = np.asarray(seed_cloud.points, dtype=np.float32)
        if seeds.ndim != 2 or seeds.shape[0] == 0:
            return None
        _, _, _, nt, _ = vel.shape
        if nt <= 1:
            return None
        dt_eff = float(dt) if float(dt) > 0 else 1.0
        # Use streamline_max_time as a time horizon in seconds for pathlines.
        n_steps = int(max(1, min(nt - 1, np.ceil(float(vec.streamline_max_time) / dt_eff))))

        pts_all: list[np.ndarray] = []
        lines: list[int] = []
        idx0 = 0

        for s in seeds:
            p = s.astype(np.float32, copy=True)
            seg: list[np.ndarray] = [p.copy()]
            tt = int(np.clip(tt_start, 0, nt - 1))
            for k in range(n_steps):
                if tt >= nt:
                    break
                # Sample velocity at current time frame, at current position.
                v0 = _sample_vel_trilinear(vel[..., tt, :], p)
                # Heun / RK2: predictor then corrector
                p1 = p + v0 * dt_eff
                tt1 = min(tt + 1, nt - 1)
                v1 = _sample_vel_trilinear(vel[..., tt1, :], p1)
                v = 0.5 * (v0 + v1)
                p = p + v * dt_eff
                seg.append(p.copy())
                tt = tt1
            seg_arr = np.vstack(seg).astype(np.float32, copy=False)
            pts_all.append(seg_arr)
            n = int(seg_arr.shape[0])
            lines.extend([n] + list(range(idx0, idx0 + n)))
            idx0 += n

        if not pts_all:
            return None
        points = np.vstack(pts_all).astype(np.float32, copy=False)
        if points.shape[0] < 2:
            return None
        try:
            poly = pv.PolyData(points)
            poly.lines = np.asarray(lines, dtype=np.int64)
            tube = poly.tube(radius=vec.streamline_radius)
            return tube
        except Exception:
            return None

    def _cb_pathlines(val: bool):
        state["pathlines"] = bool(val)
        # Lazy build handled in _update_frame.
        _update_frame()

    def _cb_centerlines(val: bool):
        state["centerlines"] = bool(val)
        _set_centerlines_visible(bool(val))
        _update_frame()

    # Place buttons in the same left-middle control strip (no resize forcing).
    def _win_h() -> int:
        try:
            rw = getattr(plotter, "render_window", None)
            if rw is not None:
                gs = getattr(rw, "GetSize", None)
                if callable(gs):
                    _w, _h = gs()
                    return int(_h)
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass
        try:
            return int(plotter.window_size[1])
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            return 800

    def _set_btn_pos(btn: Any, x: int, y: int) -> None:
        try:
            rep = btn.GetRepresentation()
            # Use display coordinates (pixels) to avoid normalized viewport confusion.
            pc = getattr(rep, "GetPositionCoordinate", None)
            if callable(pc):
                c = pc()
                try:
                    c.SetCoordinateSystemToDisplay()
                except Exception as e:
                    log.error(traceback.format_exc())
                    log.exception(e)
                    pass
                try:
                    c.SetValue(float(x), float(y))
                    return
                except Exception as e:
                    log.error(traceback.format_exc())
                    log.exception(e)
                    pass
            rep.SetPosition(float(x), float(y))
        except Exception as e:
            log.error(traceback.format_exc())
            log.exception(e)
            pass

    w_mask = plotter.add_checkbox_button_widget(_cb_mask, value=True, position=(18, y0 + 120), size=18, border_size=1)
    w_glyphs = plotter.add_checkbox_button_widget(_cb_glyphs, value=True, position=(18, y0 + 94), size=18, border_size=1)
    w_centerlines = plotter.add_checkbox_button_widget(_cb_centerlines, value=True, position=(18, y0 + 68), size=18, border_size=1)
    w_stream = plotter.add_checkbox_button_widget(_cb_stream, value=False, position=(18, y0 + 42), size=18, border_size=1)
    w_pathlines = plotter.add_checkbox_button_widget(_cb_pathlines, value=False, position=(18, y0 + 16), size=18, border_size=1)
    t_mask = plotter.add_text("Show mask", position=(42, y0 + 118), font_size=9)
    t_glyphs = plotter.add_text("Show vectors", position=(42, y0 + 92), font_size=9)
    t_centerlines = plotter.add_text("Show centerlines", position=(42, y0 + 66), font_size=9)
    t_stream = plotter.add_text("Show streamlines", position=(42, y0 + 40), font_size=9)
    t_pathlines = plotter.add_text("Show pathlines", position=(42, y0 + 14), font_size=9)
    plotter.add_text(
        "Mask / Vectors / Centerlines / Stream / Pathlines | Space: play/pause",
        position="lower_left",
        font_size=9,
    )

    # Live vector tuning widgets (desktop only).
    def _on_opacity(v: float) -> None:
        vec.glyph_opacity = float(v)
        for gp in glyph_pipes:
            gp.set_opacity(vec.glyph_opacity)
        plotter.render()

    def _on_scale_factor(v: float) -> None:
        vec.scale_factor = float(v)
        for gp in glyph_pipes:
            gp.set_scale_by_magnitude(vec.scale_by_magnitude, vec.scale_factor)
        plotter.render()

    def _on_scale_mag(v: bool) -> None:
        vec.scale_by_magnitude = bool(v)
        for gp in glyph_pipes:
            gp.set_scale_by_magnitude(vec.scale_by_magnitude, vec.scale_factor)
            gp.set_color_by_speed(vec.scale_by_magnitude)
        plotter.render()

    # Numeric inputs for vector tuning (no sliders).
    _queue_num_box(
        label="Glyph opacity (0.05..1)",
        x=18,
        y=y_box_top - 110,
        w=140,
        h=20,
        initial=str(float(vec.glyph_opacity)),
        on_commit=lambda s: _on_opacity(float(np.clip(float(s), 0.05, 1.0))) if s.strip() else None,
    )
    _queue_num_box(
        label="Scale factor (0.05..2)",
        x=18,
        y=y_box_top - 144,
        w=140,
        h=20,
        initial=str(float(vec.scale_factor)),
        on_commit=lambda s: _on_scale_factor(float(np.clip(float(s), 0.05, 2.0))) if s.strip() else None,
    )
    # Keep vector-scale toggle away from the main visibility strip.
    w_scale_mag = plotter.add_checkbox_button_widget(
        _on_scale_mag,
        value=bool(vec.scale_by_magnitude),
        position=(18, y0 - 62),
        size=18,
        border_size=1,
    )
    t_scale_mag = plotter.add_text("Scale arrows by |v|", position=(42, y0 - 64), font_size=9)

    # Mask field modes.
    def _on_mask_speed(val: bool) -> None:
        mask_field["speed"] = bool(val)
        # If interior points are enabled, these toggles should update the point scalars.
        if interior_field["enabled"]:
            if mask_field["speed"] and mask_field["radial"]:
                interior_field["mode"] = "speed*radial"
            elif mask_field["speed"]:
                interior_field["mode"] = "speed"
            elif mask_field["radial"]:
                interior_field["mode"] = "radial"
            else:
                interior_field["mode"] = "speed*radial"
            _render_interior_field()
        else:
            _rebuild_mask_actors()
            _set_mask_visible(bool(state["mask"]))
        plotter.render()

    def _on_mask_radial(val: bool) -> None:
        mask_field["radial"] = bool(val)
        if interior_field["enabled"]:
            if mask_field["speed"] and mask_field["radial"]:
                interior_field["mode"] = "speed*radial"
            elif mask_field["speed"]:
                interior_field["mode"] = "speed"
            elif mask_field["radial"]:
                interior_field["mode"] = "radial"
            else:
                interior_field["mode"] = "speed*radial"
            _render_interior_field()
        else:
            _rebuild_mask_actors()
            _set_mask_visible(bool(state["mask"]))
        plotter.render()

    # Field toggles (move down to avoid overlap with pathlines toggle).
    w_field_speed = plotter.add_checkbox_button_widget(
        _on_mask_speed,
        value=False,
        position=(18, y0 - 88),
        size=18,
        border_size=1,
    )
    t_field_speed = plotter.add_text("Field: speed |v|", position=(42, y0 - 90), font_size=9)
    w_field_radial = plotter.add_checkbox_button_widget(
        _on_mask_radial,
        value=False,
        position=(18, y0 - 114),
        size=18,
        border_size=1,
    )
    t_field_radial = plotter.add_text("Field: radial", position=(42, y0 - 116), font_size=9)

    # Interior field (dense point cloud).
    def _on_interior(val: bool) -> None:
        interior_field["enabled"] = bool(val)
        _render_interior_field()
        plotter.render()

    w_interior = plotter.add_checkbox_button_widget(
        _on_interior,
        value=bool(interior_field["enabled"]),
        position=(18, y0 - 140),
        size=18,
        border_size=1,
    )
    t_interior = plotter.add_text("Show interior points", position=(42, y0 - 142), font_size=9)

    def _toggle_play():
        state["playing"] = not state["playing"]

    plotter.add_key_event("space", _toggle_play)

    timer_ms = max(int(round(1000.0 / max(anim.animation_fps, 0.25))), 30)

    def _on_timer(_obj: Any, _event: Any) -> None:
        if not state["playing"]:
            return
        if anim.loop:
            state["tt"] = (state["tt"] + 1) % t
        else:
            state["tt"] = min(state["tt"] + 1, t - 1)
            if state["tt"] >= t - 1:
                state["playing"] = False
        _update_frame()

    timer_installed = [False]
    timer_warned = [False]

    def _try_install_animation_timer() -> bool:
        if timer_installed[0]:
            return True
        vtk_iren = _unwrap_vtk_interactor(plotter)
        if vtk_iren is None:
            return False
        ok = _install_repeating_timer_observer(vtk_iren, timer_ms, _on_timer)
        if ok:
            timer_installed[0] = True
        return ok

    def _on_first_render(*_args: Any) -> None:
        # Build numeric input widgets once the interactor exists.
        try:
            _build_queued_num_boxes()
        except Exception:
            pass
        if timer_installed[0]:
            return
        if _try_install_animation_timer():
            return
        if not timer_warned[0]:
            timer_warned[0] = True
            warnings.warn(
                "Could not install VTK repeating timer (PyVista/VTK build). "
                "Space may not advance time automatically; use the Time slider.",
                stacklevel=2,
            )

    if hasattr(plotter, "add_on_render_callback"):
        plotter.add_on_render_callback(_on_first_render)
    elif not _try_install_animation_timer():
        if not timer_warned[0]:
            timer_warned[0] = True
            warnings.warn(
                "No add_on_render_callback on Plotter and timer install failed before show(). "
                "Use the Time slider; try upgrading PyVista/VTK if auto-play is required.",
                stacklevel=2,
            )

    _set_mask_visible(True)
    _build_glyphs(int(state["tt"]))
    _update_frame()
    if show:
        plotter.show()
    return plotter


def flowshow(
    ap_phase: Image | np.ndarray,
    rl_phase: Image | np.ndarray,
    fh_phase: Image | np.ndarray,
    vessel_mask: Image | np.ndarray,
    *,
    centerline_mask: Image | np.ndarray | None = None,
    stride: int = 4,
    timepoint: int = 0,
    notebook: bool = True,
    show: bool = True,
    show_all_labels: bool = True,
    max_glyphs: int = 50_000,
    depth_peeling: bool = False,
    vector: FlowshowVectorOptions | None = None,
    animation: FlowshowAnimationOptions | None = None,
    stream_seed: int | None = 42,
    dt_seconds: float | None = None,
    cross_section_volumes: dict[str, Image | np.ndarray] | None = None,
    centerline_window: int = 5,
    cross_section_radius_vox: float = 12.0,
    cross_section_res: int = 112,
    show_gradient: bool = False,
) -> Any:
    """Interactive 4DFlow viewer.

    Parameters
    ----------
    vector
        Glyph / streamline styling (:class:`FlowshowVectorOptions`).
    animation
        Time animation and glyph index precomputation (:class:`FlowshowAnimationOptions`).
    stream_seed
        RNG seed for subsampling and streamlines.
    dt_seconds
        Temporal spacing (seconds) between frames for pathlines. If None and `ap_phase`
        is an `Image`, we try `ap_phase.temporal_resolution`.
    """
    vec_eff = vector or FlowshowVectorOptions()
    anim_eff = animation or FlowshowAnimationOptions()

    ap = to_numpy(ap_phase.data) if isinstance(ap_phase, Image) else np.asarray(ap_phase)
    rl = to_numpy(rl_phase.data) if isinstance(rl_phase, Image) else np.asarray(rl_phase)
    fh = to_numpy(fh_phase.data) if isinstance(fh_phase, Image) else np.asarray(fh_phase)
    mask = to_numpy(vessel_mask.data) if isinstance(vessel_mask, Image) else np.asarray(vessel_mask)
    mask = mask.astype(np.int32, copy=False)

    vel = _velocity_from_phases(ap, rl, fh)
    x, y, z, _, _ = vel.shape

    dt_eff = dt_seconds
    if dt_eff is None and isinstance(ap_phase, Image):
        try:
            dt_eff = ap_phase.temporal_resolution
        except Exception:
            dt_eff = None

    if mask.ndim != 3 or mask.shape != (x, y, z):
        raise ValidationError(f"vessel_mask must be 3D and match spatial dims {(x,y,z)}; got {mask.shape}.")

    cs_np: dict[str, np.ndarray] | None = None
    if cross_section_volumes:
        cs_np = {}
        for k, v in cross_section_volumes.items():
            if isinstance(v, Image):
                cs_np[k] = to_numpy(v.data)
            else:
                cs_np[k] = np.asarray(v)
        # best-effort sanity check: must match spatial dims
        for k, arr in cs_np.items():
            if tuple(arr.shape[:3]) != (x, y, z):
                raise ValidationError(
                    f"cross_section_volumes[{k!r}] shape {arr.shape} does not match phase spatial dims {(x, y, z)}."
                )

    if notebook:
        return _flowshow_notebook(
            ap=ap,
            rl=rl,
            fh=fh,
            mask=mask,
            centerline_mask=centerline_mask,
            stride=stride,
            timepoint=timepoint,
            show=show,
            show_all_labels=show_all_labels,
            max_glyphs=max_glyphs,
            depth_peeling=depth_peeling,
            vec=vec_eff,
            anim=anim_eff,
            stream_seed=stream_seed,
        )
    return _flowshow_desktop(
        ap=ap,
        rl=rl,
        fh=fh,
        mask=mask,
        centerline_mask=centerline_mask,
        stride=stride,
        timepoint=timepoint,
        show=show,
        show_all_labels=show_all_labels,
        max_glyphs=max_glyphs,
        depth_peeling=depth_peeling,
        vec=vec_eff,
        anim=anim_eff,
        stream_seed=stream_seed,
        dt_seconds=dt_eff,
        cross_section_volumes=cs_np,
        centerline_window=centerline_window,
        cross_section_radius_vox=cross_section_radius_vox,
        cross_section_res=cross_section_res,
        show_gradient=show_gradient,
    )


__all__ = [
    "flowshow",
    "FlowshowVectorOptions",
    "FlowshowAnimationOptions",
    "VectorColorMode",
]
