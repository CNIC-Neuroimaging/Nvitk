"""Editing label layers: the operations behind the Labeling tab and the layer list's label menu.

Every edit on a Napari ``Labels`` layer is written through Napari's own history
(:meth:`Labels.data_setitem` inside :meth:`Labels.block_history`), so one tool
application is one Ctrl+Z, exactly like a brush stroke. An Image layer used as a
mask has no history: its edits are written straight to the data (and to the
unfiltered copy the live label filter keeps, :func:`~nvitk.gui.labels.visibility.label_source_data`).

The region operations (grow, shrink, smooth, components, threshold…) run on the
active backend — CuPy when the GUI's GPU switch is on — and the result is brought
back to the host only to be written into the layer, which Napari holds in NumPy.
The magic wand grows its region with ``ndi.label`` on the backend too; only the
polygon rasteriser (skimage) runs under ``using("cpu")``.

Nothing here touches Qt: the panels call these and announce the change themselves.
"""

from __future__ import annotations

import math
import weakref
from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.gui.labels.catalog import custom_label_names, set_label_name
from nvitk.gui.labels.visibility import invalidate_label_ids, label_source_data, layer_label_ids

setup(globals())

#: Where an edit applies: the slice on screen, or the whole volume (the current
#: frame of a 3D+t layer).
SCOPES = ("slice", "volume")


# ──────────────────────────────────────────────────────────────────────────────
# Layers
# ──────────────────────────────────────────────────────────────────────────────


def is_labels(layer: Any) -> bool:
    """True for a Napari ``Labels`` layer (editable with history)."""
    return type(layer).__name__ == "Labels"


def editable_data(layer: Any) -> Any:
    """*layer*'s voxels as a writable NumPy array, materialising a lazy one once.

    Napari keeps label data on the host; a dask or zarr array cannot take the
    fancy-indexed writes an edit makes, so it is read into memory first.
    """
    data = layer.data
    if isinstance(data, (list, tuple)):
        raise ValueError(f"“{layer.name}” is multiscale; edit a single-resolution copy of it.")
    host = to_numpy(data)
    if host is not data:
        layer.data = host
    return layer.data


def announce_history_loads(layer: Any) -> None:
    """Have Napari's Undo / Redo on a Labels layer emit its ``paint`` event.

    Napari rewrites the voxels and refreshes the canvas but announces nothing, so
    whatever follows edits through ``paint`` — the cached label ids, the layer
    list's label lines, the orthogonal views — kept showing the state from before
    the Undo. The event carries no change list (``value=[]``).
    """
    if not is_labels(layer) or getattr(layer, "_nvitk_history_announced", False):
        return
    cls = type(layer)
    ref = weakref.ref(layer)

    def _load_history(before: Any, after: Any, undoing: bool = True) -> None:
        target = ref()
        if target is None:
            return
        cls._load_history(target, before, after, undoing=undoing)
        target.events.paint(value=[])

    try:
        layer._load_history = _load_history
        layer._nvitk_history_announced = True
    except Exception:  # noqa: BLE001 — a layer that rejects attributes keeps Napari's
        pass


def next_free_label(layer: Any) -> int:
    """One more than the largest label id in *layer* (1 on an empty layer)."""
    ids = layer_label_ids(layer, max_labels=1_000_000)
    return (max(ids) + 1) if ids else 1


def voxel_volume_mm3(layer: Any) -> float:
    """Volume of one voxel of *layer*, from its spacing (1 when unknown)."""
    from nvitk.gui.core.spatial import layer_spacing

    spacing = layer_spacing(layer)
    if not spacing:
        return 1.0
    vol = 1.0
    for value in list(spacing)[: min(3, layer.data.ndim)]:
        vol *= abs(float(value)) or 1.0
    return vol


# ──────────────────────────────────────────────────────────────────────────────
# Where an edit applies
# ──────────────────────────────────────────────────────────────────────────────


def time_axis(layer: Any) -> int | None:
    """The time axis of a 3D+t layer; ``None`` for 2D and 3D layers."""
    if layer.data.ndim < 4:
        return None
    from nvitk.gui.core.spatial import _time_axis_index

    return _time_axis_index(layer)


def current_point(viewer: Any, layer: Any) -> tuple[int, ...]:
    """The viewer's current position as a voxel index into *layer*."""
    shape = layer.data.shape
    try:
        position = layer.world_to_data(viewer.dims.point)
    except Exception:  # noqa: BLE001 — fall back to the middle of the volume
        position = [s / 2.0 for s in shape]
    index = [int(round(float(v))) for v in list(position)[-len(shape):]]
    return tuple(min(max(v, 0), n - 1) for v, n in zip(index, shape))


def event_voxel(layer: Any, event: Any) -> tuple[int, ...] | None:
    """The voxel of *layer* under a mouse *event*; ``None`` outside the volume."""
    try:
        position = layer.world_to_data(event.position)
    except Exception:  # noqa: BLE001
        return None
    shape = layer.data.shape
    index = tuple(int(round(float(v))) for v in list(position)[-len(shape):])
    if any(v < 0 or v >= n for v, n in zip(index, shape)):
        return None
    return index


def ray_through(layer: Any, position: Sequence[float], view_direction: Sequence[float]) -> tuple[Any, Any] | None:
    """Where a 3D click's ray enters and leaves *layer*'s data box, as data
    coordinates (``(start, end)``, *layer*'s ndim each); ``None`` when it misses."""
    import numpy

    dims = list(getattr(layer._slice_input, "displayed", ()))
    if len(dims) != 3 or view_direction is None:
        return None
    try:
        start, end = layer.get_ray_intersections(
            numpy.asarray(position, dtype=float), numpy.asarray(view_direction, dtype=float), dims, world=True)
    except Exception:  # noqa: BLE001
        return None
    if start is None or end is None:
        return None
    return numpy.asarray(start, dtype=float), numpy.asarray(end, dtype=float)


def pick_visible_voxel(image: Any, position: Sequence[float], view_direction: Sequence[float]) -> tuple[int, ...] | None:
    """The voxel of the Image layer *image* a click on the 3D canvas lands on —
    what the rendering shows there.

    The ray through the click is sampled voxel by voxel and the voxel kept is,
    by the layer's rendering: the brightest for ``mip`` (the darkest for
    ``minip``), the first at or above the iso threshold for ``iso``, the plane's
    for a plane depiction, otherwise the first above the middle of the display
    window (the first structure that shows), the brightest when none is.
    ``None`` when the ray misses the volume.
    """
    ray = ray_through(image, position, view_direction)
    if ray is None:
        return None
    with using("cpu"):
        start, end = ray
        shape = np.asarray(image.data.shape)
        if str(getattr(image, "depiction", "volume")) == "plane":
            dims = list(image._slice_input.displayed)
            plane = getattr(image, "plane", None)
            if plane is not None:
                normal = np.asarray(plane.normal, dtype=float)
                origin = np.asarray(plane.position, dtype=float)
                direction = (end - start)[dims]
                denom = float(direction @ normal)
                if abs(denom) > 1e-9:
                    t = float((origin - start[dims]) @ normal) / denom
                    hit = start + np.clip(t, 0.0, 1.0) * (end - start)
                    voxel = np.clip(np.round(hit), 0, shape - 1).astype(int)
                    return tuple(int(v) for v in voxel)
        length = float(np.linalg.norm(end - start))
        n = max(int(math.ceil(length)) + 1, 2)
        coords = np.clip(np.round(np.linspace(start, end, n)), 0, shape - 1).astype(int)
        values = to_numpy(image.data[tuple(coords.T)]).astype(float)
        finite = np.isfinite(values)
        if not finite.any():
            return None
        values = np.where(finite, values, -np.inf)
        rendering = str(getattr(image, "rendering", "mip"))
        k = None
        if rendering == "minip":
            k = int(np.argmin(np.where(finite, values, np.inf)))
        elif rendering == "iso":
            above = np.flatnonzero(values >= float(getattr(image, "iso_threshold", 0.0)))
            k = int(above[0]) if above.size else None
        elif rendering != "mip":
            lo, hi = (float(v) for v in getattr(image, "contrast_limits", (values.min(), values.max())))
            above = np.flatnonzero(values >= lo + 0.5 * (hi - lo))
            k = int(above[0]) if above.size else None
        if k is None:
            k = int(np.argmax(values))
        return tuple(int(v) for v in coords[k])


def displayed_axes(viewer: Any, layer: Any) -> list[int]:
    """*layer*'s axes on screen (two in a 2D view, three in 3D)."""
    from nvitk.gui.core.label_pick import _dims_displayed

    shown = _dims_displayed(layer, viewer)
    if shown is None:
        nd = layer.data.ndim
        return list(range(max(nd - 2, 0), nd))
    return [int(d) for d in shown]


def is_3d_view(viewer: Any) -> bool:
    """True when the viewer renders in 3D."""
    return int(getattr(getattr(viewer, "dims", None), "ndisplay", 2)) == 3


def normal_axis(viewer: Any, layer: Any) -> int | None:
    """The axis across the slices on screen (``None`` for 2D layers or a 3D view)."""
    if is_3d_view(viewer) or layer.data.ndim < 3:
        return None
    shown = set(displayed_axes(viewer, layer))
    t_ax = time_axis(layer)
    rest = [d for d in range(layer.data.ndim) if d not in shown and d != t_ax]
    return rest[-1] if rest else None


def region(viewer: Any, layer: Any, scope: str) -> tuple[Any, ...]:
    """Index of *layer*'s data an edit in *scope* works on.

    ``"slice"``: the plane on screen (a 2D view is required); ``"volume"``: the
    whole volume — for a 3D+t layer, the frame on screen.
    """
    data = layer.data
    nd = data.ndim
    point = current_point(viewer, layer)
    if scope == "slice":
        if nd <= 2:
            return (slice(None),) * nd
        if is_3d_view(viewer):
            raise ValueError("Slice tools work on the slice on screen: switch the viewer to 2D.")
        keep = set(displayed_axes(viewer, layer))
    else:
        if nd <= 3:
            return (slice(None),) * nd
        t_ax = time_axis(layer)
        keep = {d for d in range(nd) if d != t_ax}
    return tuple(slice(None) if d in keep else int(point[d]) for d in range(nd))


def _global_indices(index: Sequence[Any], local: Sequence[Any]) -> tuple[Any, ...]:
    """Indices into the whole array from *local* indices into ``data[index]``.

    Host index arithmetic for a write into Napari's data: call under ``using("cpu")``.
    """
    out = []
    it = iter(local)
    size = local[0].shape if len(local) else (0,)
    for ix in index:
        if isinstance(ix, slice):
            out.append(next(it) + int(ix.start or 0))
        else:
            out.append(np.full(size, int(ix), dtype=np.intp))
    return tuple(out)


# ──────────────────────────────────────────────────────────────────────────────
# Writing
# ──────────────────────────────────────────────────────────────────────────────


def write_region(layer: Any, index: Sequence[Any], new: Any) -> int:
    """Write *new* into ``layer.data[index]``; the number of voxels changed.

    Only the voxels that differ are written, as one undo step on a Labels layer.
    """
    data = editable_data(layer)
    index = tuple(index)
    old = data[index]
    # Napari's data is NumPy: the edited region comes back to the host here.
    new_host = to_numpy(new).astype(old.dtype, copy=False)
    with using("cpu"):
        changed = new_host != old
        values = new_host[changed]
        if values.size == 0:
            return 0
        indices = _global_indices(index, np.nonzero(changed))
    if is_labels(layer):
        with layer.block_history():
            layer.data_setitem(indices, values)
    else:
        _write_image_mask(layer, indices, values)
    invalidate_label_ids(layer)
    return int(values.size)


def _write_image_mask(layer: Any, indices: tuple[Any, ...], values: Any) -> None:
    """Write into an Image mask and into the unfiltered copy its label filter keeps."""
    source = label_source_data(layer)
    if source is not layer.data and getattr(source, "shape", None) == layer.data.shape:
        source[indices] = values
    data = layer.data
    data[indices] = values
    layer.data = data


def delete_label(layer: Any, label_id: int) -> int:
    """Clear every voxel of *label_id* in *layer* (one undo step on a Labels layer).

    A name given to it by hand is kept: Ctrl+Z brings the voxels back, and a name
    for an id no longer present shows nowhere. Returns the number of voxels cleared.
    """
    lid = int(label_id)
    if is_labels(layer):
        data = editable_data(layer)
        with using("cpu"):
            indices = np.nonzero(data == lid)
        count = int(indices[0].size)
        if count:
            with layer.block_history():
                layer.data_setitem(indices, 0)
    else:
        source = label_source_data(layer)
        with using("cpu"):
            hit = source == lid
            count = int(np.count_nonzero(hit))
            if count:
                source[hit] = 0
                if source is not layer.data:
                    shown = layer.data
                    shown[shown == lid] = 0
                    layer.data = shown
                else:
                    layer.data = source
    invalidate_label_ids(layer)
    return count


def replace_label(layer: Any, source_id: int, target_id: int) -> int:
    """Give every voxel of *source_id* the id *target_id* (merge, or change an id)."""
    data = editable_data(layer)
    with using("cpu"):
        indices = np.nonzero(data == int(source_id))
    count = int(indices[0].size)
    if not count:
        return 0
    if is_labels(layer):
        with layer.block_history():
            layer.data_setitem(indices, int(target_id))
    else:
        _write_image_mask(layer, indices, int(target_id))
    names = custom_label_names(layer)
    if int(source_id) in names and int(target_id) not in names:
        set_label_name(layer, int(target_id), names[int(source_id)])
    set_label_name(layer, int(source_id), None)
    invalidate_label_ids(layer)
    return count


def relabel_consecutive(layer: Any) -> dict[int, int]:
    """Renumber *layer*'s labels 1…N in order; the ``{old: new}`` map applied."""
    ids = layer_label_ids(layer, max_labels=1_000_000)
    mapping = {old: new for new, old in enumerate(ids, start=1) if old != new}
    if not mapping:
        return {}
    data = editable_data(layer)
    work = as_backend_array(data)
    lut = np.arange(int(max(ids)) + 1, dtype=data.dtype)
    for old, new in mapping.items():
        lut[old] = new
    write_region(layer, (slice(None),) * data.ndim, lut[work])
    names = custom_label_names(layer)
    for old, new in sorted(mapping.items()):
        set_label_name(layer, new, names.get(old))
    for old in mapping:
        if old not in mapping.values():
            set_label_name(layer, old, None)
    return mapping


# ──────────────────────────────────────────────────────────────────────────────
# Region operations (backend arrays in, backend arrays out)
# ──────────────────────────────────────────────────────────────────────────────


def writable(work: Any, label_id: int, *, preserve: bool) -> Any:
    """Where *label_id* may be written: background and itself when other labels
    are preserved, everywhere otherwise."""
    work = as_backend_array(work)
    if not preserve:
        return np.ones(work.shape, dtype=bool)
    return (work == 0) | (work == int(label_id))


def _ball(radius: int, ndim: int) -> Any:
    """Round structuring element of *radius* voxels."""
    r = max(int(radius), 1)
    axis = np.arange(-r, r + 1)
    grids = np.meshgrid(*([axis] * ndim), indexing="ij")
    dist2 = sum(g * g for g in grids)
    return dist2 <= r * r


def _full(ndim: int) -> Any:
    """Full connectivity (26 neighbours in 3D)."""
    return ndi.generate_binary_structure(ndim, ndim)


def grow(work: Any, label_id: int, radius: int, allowed: Any) -> Any:
    """Dilate *label_id* by *radius* voxels where *allowed*."""
    work = as_backend_array(work)
    mask = work == int(label_id)
    if not bool(mask.any()):
        return work
    grown = ndi.binary_dilation(mask, structure=_ball(radius, work.ndim))
    out = work.copy()
    out[grown & as_backend_array(allowed)] = int(label_id)
    return out


def shrink(work: Any, label_id: int, radius: int) -> Any:
    """Erode *label_id* by *radius* voxels (the volume's border does not erode it)."""
    work = as_backend_array(work)
    mask = work == int(label_id)
    if not bool(mask.any()):
        return work
    kept = ndi.binary_erosion(mask, structure=_ball(radius, work.ndim), border_value=1)
    out = work.copy()
    out[mask & ~kept] = 0
    return out


def smooth(work: Any, label_id: int, radius: int, allowed: Any) -> Any:
    """Round off *label_id*: an opening (drops spurs) then a closing (fills dents)."""
    work = as_backend_array(work)
    mask = work == int(label_id)
    if not bool(mask.any()):
        return work
    ball = _ball(radius, work.ndim)
    shaped = ndi.binary_closing(ndi.binary_opening(mask, structure=ball), structure=ball)
    out = work.copy()
    out[mask & ~shaped] = 0
    out[shaped & ~mask & as_backend_array(allowed)] = int(label_id)
    return out


def fill_holes(work: Any, label_id: int, allowed: Any) -> Any:
    """Fill the holes enclosed by *label_id* (in the plane, or in 3D, as *work* is)."""
    work = as_backend_array(work)
    mask = work == int(label_id)
    if not bool(mask.any()):
        return work
    filled = ndi.binary_fill_holes(mask)
    out = work.copy()
    out[filled & ~mask & as_backend_array(allowed)] = int(label_id)
    return out


def _components(mask: Any) -> tuple[Any, Any]:
    """Connected components of *mask* (full connectivity) and their sizes."""
    lab, n = ndi.label(mask, structure=_full(mask.ndim))
    sizes = np.bincount(lab.ravel(), minlength=int(n) + 1)
    sizes[0] = 0
    return lab, sizes


def keep_largest(work: Any, label_id: int) -> Any:
    """Keep only the largest connected piece of *label_id*."""
    work = as_backend_array(work)
    mask = work == int(label_id)
    if not bool(mask.any()):
        return work
    lab, sizes = _components(mask)
    out = work.copy()
    out[mask & (lab != int(np.argmax(sizes)))] = 0
    return out


def remove_islands(work: Any, label_id: int, min_voxels: int) -> Any:
    """Drop the pieces of *label_id* smaller than *min_voxels*."""
    work = as_backend_array(work)
    mask = work == int(label_id)
    if not bool(mask.any()):
        return work
    lab, sizes = _components(mask)
    small = sizes < int(min_voxels)
    small[0] = False
    out = work.copy()
    out[mask & small[lab]] = 0
    return out


def split_components(work: Any, label_id: int, first_new_id: int) -> tuple[Any, list[int]]:
    """Give each connected piece of *label_id* its own id; the largest keeps it.

    Returns the new array and the ids given to the other pieces, largest first.
    """
    work = as_backend_array(work)
    mask = work == int(label_id)
    if not bool(mask.any()):
        return work, []
    lab, sizes = _components(mask)
    # Piece sizes are a short list ranked in Python: on the host.
    sizes_host = to_numpy(sizes)
    order = sorted((c for c in range(sizes_host.size) if sizes_host[c] > 0), key=lambda c: -int(sizes_host[c]))
    out = work.copy()
    new_ids = []
    for k, comp in enumerate(order[1:]):
        nid = int(first_new_id) + k
        out[lab == comp] = nid
        new_ids.append(nid)
    return out, new_ids


def threshold_fill(work: Any, intensity: Any, low: float, high: float, label_id: int, allowed: Any) -> Any:
    """Give *label_id* to the voxels whose *intensity* lies in ``[low, high]``."""
    work = as_backend_array(work)
    values = as_backend_array(intensity)
    hit = (values >= float(low)) & (values <= float(high)) & as_backend_array(allowed)
    out = work.copy()
    out[hit] = int(label_id)
    return out


def wand_region(
    intensity: Any,
    seed: Sequence[int],
    tolerance: float,
    *,
    smooth_sigma: float = 0.0,
    seed_radius: int = 0,
    full_connectivity: bool = False,
    max_distance_mm: float = 0.0,
    spacing: Sequence[float] | None = None,
    fill_holes: bool = True,
    close_radius: int = 0,
) -> Any:
    """The magic wand's region: what grows from *seed* through intensities within
    *tolerance* of the seed's.

    Read on a copy of *intensity* smoothed by *smooth_sigma* voxels (noise no longer
    breaks the region into single voxels), with the seed value averaged over a box
    of *seed_radius* voxels around the click; grown through face neighbours, or all
    with *full_connectivity*; kept within *max_distance_mm* of the click (0: no
    limit; *spacing* in mm per axis); then gaps closed by *close_radius* voxels and
    holes filled. A boolean array like *intensity*, on the active backend.
    """
    from nvitk.segmentation.interactive import GrowSession

    session = GrowSession(intensity, smooth_sigma=smooth_sigma, spacing=spacing, full_connectivity=full_connectivity)
    return session.wand(seed, tolerance, seed_radius=seed_radius, max_distance_mm=max_distance_mm,
                        fill_holes=fill_holes, close_radius=close_radius)


def wand(
    work: Any,
    intensity: Any,
    seed: Sequence[int],
    tolerance: float,
    label_id: int,
    allowed: Any,
    *,
    remove: bool = False,
    **options: Any,
) -> tuple[Any, int]:
    """Magic wand: the region :func:`wand_region` grows from *seed* (*options* are its).

    Added to *label_id* where *allowed*, or — with *remove* — taken out of it.
    Returns the new array and the size of the region.
    """
    work = as_backend_array(work)
    region = wand_region(intensity, seed, tolerance, **options)
    out = work.copy()
    if remove:
        out[region & (work == int(label_id))] = 0
    else:
        out[region & as_backend_array(allowed)] = int(label_id)
    return out, int(np.count_nonzero(region))


def polygon_mask(shape: Sequence[int], rows: Sequence[float], cols: Sequence[float]) -> Any:
    """A filled polygon (vertices *rows*, *cols*, in pixels) on a 2D grid of *shape*."""
    from skimage.draw import polygon

    # skimage rasterises on the host; the mask goes back to the backend.
    with using("cpu"):
        mask = np.zeros(tuple(int(n) for n in shape), dtype=bool)
        rr, cc = polygon(to_numpy(list(rows)).astype(float), to_numpy(list(cols)).astype(float), shape=mask.shape)
        mask[rr, cc] = True
        # The outline itself too: a thin polygon still marks the voxels it crosses.
        for (r0, c0), (r1, c1) in zip(zip(rows, cols), list(zip(rows, cols))[1:] + [(rows[0], cols[0])]):
            n = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
            lr = np.clip(np.rint(np.linspace(r0, r1, n)).astype(int), 0, mask.shape[0] - 1)
            lc = np.clip(np.rint(np.linspace(c0, c1, n)).astype(int), 0, mask.shape[1] - 1)
            mask[lr, lc] = True
    return as_backend_array(mask)


def fill_polygon(
    layer: Any,
    index: Sequence[Any],
    rows: Sequence[float],
    cols: Sequence[float],
    label_id: int,
    allowed: Any | None = None,
) -> int:
    """Give *label_id* to the polygon (*rows*, *cols* in the plane ``data[index]``,
    whose two free axes are in order); one undo step. The voxels written."""
    data = editable_data(layer)
    work = as_backend_array(data[tuple(index)])
    if work.ndim != 2:
        raise ValueError("A polygon is drawn in a plane: a slice of the volume.")
    mask = polygon_mask(work.shape, rows, cols)
    if allowed is not None:
        mask &= as_backend_array(allowed)
    out = work.copy()
    out[mask] = int(label_id)
    return write_region(layer, index, out)


def interpolate_slices(
    work: Any,
    label_id: int,
    *,
    axis: int | None,
    method: str = "shape",
    spacing: Sequence[float] | None = None,
    preserve: bool = True,
) -> Any:
    """Fill the slices between the drawn ones of *label_id* (along *axis*, or its
    sparsest one when ``None``)."""
    from nvitk.transform.interpolation import interpolate_mask_slices
    from nvitk.types import Image

    work_host = to_numpy(work)
    metadata: dict[str, Any] = {}
    if spacing is not None and len(spacing) >= work_host.ndim:
        metadata["spacing"] = tuple(float(s) for s in spacing[: work_host.ndim])
    image = Image(data=work_host, metadata=metadata, axes="XYZ"[: work_host.ndim])
    axes = "auto" if axis is None else (int(axis),)
    filled = interpolate_mask_slices(
        image, axes=axes, method=method, label_ids=[int(label_id)], overwrite=not preserve
    )
    return as_backend_array(filled.data)


def copy_slice(
    viewer: Any, layer: Any, label_id: int, step: int, *, preserve: bool
) -> tuple[int, int | None]:
    """Copy *label_id*'s outline on the slice on screen to the slice *step* away.

    The neighbour's own *label_id* is replaced by the copy. Returns the voxels
    changed and the slice index written (``None`` past the volume's end).
    """
    axis = normal_axis(viewer, layer)
    if axis is None:
        raise ValueError("Copying a slice needs a 2D view of a 3D volume.")
    data = editable_data(layer)
    src_index = list(region(viewer, layer, "slice"))
    target = int(src_index[axis]) + int(step)
    if target < 0 or target >= data.shape[axis]:
        return 0, None
    dst_index = list(src_index)
    dst_index[axis] = target
    lid = int(label_id)
    src = as_backend_array(data[tuple(src_index)]) == lid
    dst = as_backend_array(data[tuple(dst_index)])
    out = dst.copy()
    out[(dst == lid) & ~src] = 0
    out[src & writable(dst, lid, preserve=preserve)] = lid
    return write_region(layer, dst_index, out), target


def brush_indices(layer: Any, center: Sequence[float], dims: Sequence[int]) -> tuple[Any, ...]:
    """Napari's brush at *center*: the same footprint its paint mode lays, over *dims*.

    A ball (a disc with two *dims*) of radius ``floor(brush_size / 2) + 0.5``,
    round in the layer's scale, as host index arrays into ``layer.data`` (the
    other axes fixed at *center*). Out-of-volume voxels are dropped.
    """
    from napari.layers.labels._labels_utils import sphere_indices

    dims = [int(d) for d in dims]
    shape = layer.data.shape
    radius = math.floor(float(layer.brush_size) / 2.0) + 0.5
    scale = tuple(abs(float(layer.scale[d])) or 1.0 for d in dims)
    # Napari's footprint and Napari's data: host index arithmetic.
    with using("cpu"):
        offsets = sphere_indices(radius, scale)
        points = offsets + np.array([int(round(float(center[d]))) for d in dims])
        limits = np.array([int(shape[d]) for d in dims])
        points = points[np.all((points >= 0) & (points < limits), axis=1)]
        n = int(points.shape[0])
        return tuple(
            points[:, dims.index(d)].astype(np.intp) if d in dims
            else np.full(n, int(round(float(center[d]))), dtype=np.intp)
            for d in range(len(shape))
        )


def brush_radius_voxels(layer: Any, dims: Sequence[int]) -> dict[int, float]:
    """The brush radius along each of *dims*, in voxels (for drawing its outline)."""
    radius = math.floor(float(layer.brush_size) / 2.0) + 0.5
    scale = {int(d): abs(float(layer.scale[int(d)])) or 1.0 for d in dims}
    smallest = min(scale.values()) if scale else 1.0
    return {d: radius * smallest / s for d, s in scale.items()}


def paint_at(layer: Any, indices: tuple[Any, ...], new_label: int, *, refresh: bool = True) -> int:
    """Write *new_label* at *indices* as Napari's brush would; the voxels written.

    *Preserve labels* on: painting only covers background, erasing only clears
    the selected label. Through ``data_setitem``, so it joins the history item of
    an open stroke (or makes its own).
    """
    data = editable_data(layer)
    with using("cpu"):
        if bool(getattr(layer, "preserve_labels", False)):
            current = data[indices]
            if int(new_label) == 0:
                previous = getattr(layer, "_prev_selected_label", None)
                keep = current == int(previous or layer.selected_label)
            else:
                keep = current == 0
            indices = tuple(ix[keep] for ix in indices)
        count = int(indices[0].size) if indices else 0
    if count:
        layer.data_setitem(indices, int(new_label), refresh=refresh)
        invalidate_label_ids(layer)
    return count


def fill_at(layer: Any, voxel: Sequence[int], dims: Sequence[int], new_label: int) -> int:
    """Napari's bucket fill at *voxel* over *dims* (a plane, or the volume).

    The clicked label's region becomes *new_label*: its connected piece when the
    layer fills contiguously, all of it in the plane/volume otherwise. *Preserve
    labels* on: only background (or the label just swapped out) is filled. One
    undo step; the voxels changed.
    """
    data = editable_data(layer)
    voxel = tuple(int(v) for v in voxel)
    old = int(data[voxel])
    if old == int(new_label):
        return 0
    previous = getattr(layer, "_prev_selected_label", None)
    if bool(getattr(layer, "preserve_labels", False)) and old not in (0, previous):
        return 0
    dims = sorted(int(d) for d in dims)
    index = tuple(slice(None) if d in dims else voxel[d] for d in range(data.ndim))
    work = as_backend_array(data[index])
    matches = work == old
    if bool(getattr(layer, "contiguous", True)):
        lab, n = ndi.label(matches)
        if int(n) > 1:
            matches = lab == lab[tuple(voxel[d] for d in dims)]
    out = work.copy()
    out[matches] = int(new_label)
    return write_region(layer, index, out)


def step_to_slice(viewer: Any, layer: Any, axis: int, index: int) -> None:
    """Move the viewer so slice *index* along *layer*'s *axis* is on screen."""
    point = list(current_point(viewer, layer))
    point[axis] = int(index)
    world = list(layer.data_to_world(point))
    world_axis = int(viewer.dims.ndim) - int(layer.data.ndim) + int(axis)
    viewer.dims.set_point(world_axis, float(world[axis]))


# ──────────────────────────────────────────────────────────────────────────────
# Facts about a label
# ──────────────────────────────────────────────────────────────────────────────


def label_counts(layer: Any) -> dict[int, int]:
    """Voxel count of every label in *layer*."""
    data = as_backend_array(label_source_data(layer))
    if data.size == 0:
        return {}
    if data.dtype.kind == "f":
        data = np.rint(data)
    data = data.astype(np.int64, copy=False)
    if bool((data < 0).any()):
        ids, counts = np.unique(data, return_counts=True)
        return {int(i): int(c) for i, c in zip(to_numpy(ids), to_numpy(counts)) if int(i) != 0}
    counts = to_numpy(np.bincount(data.ravel()))
    return {int(i): int(c) for i, c in enumerate(counts) if i and c}


def label_centroid(layer: Any, label_id: int) -> tuple[float, ...] | None:
    """Centre of mass of *label_id* in *layer*'s voxel coordinates (``None`` when absent)."""
    mask = as_backend_array(label_source_data(layer)) == int(label_id)
    if not bool(mask.any()):
        return None
    return tuple(float(c) for c in ndi.center_of_mass(mask))


def go_to_label(viewer: Any, layer: Any, label_id: int) -> bool:
    """Bring *label_id* on screen: its centre's slice, and the camera on it.

    A 3D+t layer keeps its current frame. False when the label is not there.
    """
    centre = label_centroid(layer, label_id)
    if centre is None:
        return False
    nd = layer.data.ndim
    t_ax = time_axis(layer)
    point = list(current_point(viewer, layer))
    for d in range(nd):
        if d != t_ax:
            point[d] = centre[d]
    world = list(layer.data_to_world(point))
    offset = int(viewer.dims.ndim) - nd
    for d in range(nd):
        if d != t_ax:
            viewer.dims.set_point(offset + d, float(world[d]))
    shown = [offset + d for d in displayed_axes(viewer, layer)]
    try:
        viewer.camera.center = tuple(float(world[a - offset]) for a in shown[-3:])
    except Exception:  # noqa: BLE001 — the slice is already right
        pass
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Editable area: where the brush and the fill bucket may paint
# ──────────────────────────────────────────────────────────────────────────────


class PaintGuard:
    """Restricts painting on a Labels layer to an editable area.

    Napari's own *preserve labels* keeps other labels intact; this adds what it
    cannot: paint only where a reference image's intensity lies in a range, only
    inside a mask, and only inside a box (:attr:`box`, an index of slices into the
    layer's data). Each stroke reaches Napari's history as one item; once
    it is committed, the voxels painted outside the area are put back and the
    item is trimmed to the rest, so Undo still undoes exactly what stayed.
    Erasing is never restricted.
    """

    def __init__(self, layer: Any) -> None:
        self._layer = weakref.ref(layer)
        self.intensity: Any | None = None
        self.low = float("-inf")
        self.high = float("inf")
        self.inside: Any | None = None
        self.box: tuple[slice, ...] | None = None
        #: Edits made by the tools are already constrained: they pass through.
        self.suspended = False
        layer.events.paint.connect(self._on_paint)

    @property
    def active(self) -> bool:
        return self.intensity is not None or self.inside is not None or self.box is not None

    def _box_bounds(self) -> list[tuple[int, int]] | None:
        layer = self._layer()
        if self.box is None or layer is None:
            return None
        return [ix.indices(int(n))[:2] for ix, n in zip(self.box, layer.data.shape)]

    def _box_mask(self, index: Sequence[Any]) -> Any:
        """Inside the box, within ``data[index]`` (a backend boolean array)."""
        layer = self._layer()
        bounds = self._box_bounds()
        shape = tuple(int(n) for n in layer.data[tuple(index)].shape)
        ok = np.ones(shape, dtype=bool)
        axis = 0
        for (lo, hi), ix, n in zip(bounds, index, layer.data.shape):
            if isinstance(ix, slice):
                start, stop, step = ix.indices(int(n))
                coords = np.arange(start, stop, step)
                inside = (coords >= lo) & (coords < hi)
                view = [1] * len(shape)
                view[axis] = int(coords.size)
                ok = ok & inside.reshape(view)
                axis += 1
            elif not lo <= int(ix) < hi:
                ok = ok & False
        return ok

    def detach(self) -> None:
        layer = self._layer()
        if layer is not None:
            try:
                layer.events.paint.disconnect(self._on_paint)
            except Exception:  # noqa: BLE001
                pass

    def allowed(self, index: Sequence[Any]) -> Any | None:
        """The editable area within ``data[index]`` (``None``: everywhere)."""
        out = None
        if self.intensity is not None:
            values = as_backend_array(self.intensity[tuple(index)])
            out = (values >= self.low) & (values <= self.high)
        if self.inside is not None:
            ok = as_backend_array(self.inside[tuple(index)]) != 0
            out = ok if out is None else (out & ok)
        if self.box is not None:
            ok = self._box_mask(index)
            out = ok if out is None else (out & ok)
        return out

    def _on_paint(self, event: Any) -> None:
        layer = self._layer()
        if layer is None or self.suspended or not self.active:
            return
        item = getattr(event, "value", None)
        if not isinstance(item, list):
            return
        # History atoms and the layer's data are NumPy: this check stays on the host.
        with using("cpu"):
            trimmed = []
            reverted = 0
            for indices, old, new in item:
                new_arr = np.broadcast_to(as_backend_array(new), np.shape(old))
                bad = np.zeros(np.shape(old), dtype=bool)
                painting = new_arr != 0
                if self.intensity is not None:
                    values = self.intensity[indices]
                    bad |= painting & ((values < self.low) | (values > self.high))
                if self.inside is not None:
                    bad |= painting & (self.inside[indices] == 0)
                bounds = self._box_bounds()
                if bounds is not None:
                    for (lo, hi), ix in zip(bounds, indices):
                        bad |= painting & ((ix < lo) | (ix >= hi))
                if not bad.any():
                    trimmed.append((indices, old, new))
                    continue
                layer.data[tuple(ix[bad] for ix in indices)] = old[bad]
                reverted += int(bad.sum())
                keep = ~bad
                kept_new = new_arr[keep] if np.ndim(new) else new
                trimmed.append((tuple(ix[keep] for ix in indices), old[keep], kept_new))
        if reverted:
            item[:] = trimmed
            invalidate_label_ids(layer)
            layer.refresh()


def paint_guard(layer: Any, *, create: bool = False) -> PaintGuard | None:
    """The :class:`PaintGuard` on *layer* (made when *create*)."""
    guard = getattr(layer, "_nvitk_paint_guard", None)
    if guard is None and create and is_labels(layer):
        guard = PaintGuard(layer)
        try:
            layer._nvitk_paint_guard = guard
        except Exception:  # noqa: BLE001
            pass
    return guard


__all__ = [
    "PaintGuard",
    "announce_history_loads",
    "pick_visible_voxel",
    "ray_through",
    "brush_indices",
    "brush_radius_voxels",
    "fill_at",
    "fill_polygon",
    "polygon_mask",
    "wand_region",
    "paint_at",
    "SCOPES",
    "copy_slice",
    "current_point",
    "delete_label",
    "displayed_axes",
    "editable_data",
    "event_voxel",
    "fill_holes",
    "go_to_label",
    "grow",
    "interpolate_slices",
    "is_3d_view",
    "is_labels",
    "keep_largest",
    "label_centroid",
    "label_counts",
    "next_free_label",
    "normal_axis",
    "paint_guard",
    "region",
    "relabel_consecutive",
    "remove_islands",
    "replace_label",
    "shrink",
    "smooth",
    "split_components",
    "step_to_slice",
    "threshold_fill",
    "time_axis",
    "voxel_volume_mm3",
    "wand",
    "writable",
]
