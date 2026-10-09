"""The Labeling tab's region tools at work, for the main canvas and the orthogonal views alike.

* :class:`RegionStroke` — one use of the magic wand, the adaptive flood or the
  vessel flood: seeds added along a drag (each new voxel the cursor reaches
  outside the region so far grows another piece), or one seed whose setting is
  tuned with the mouse. The region shows in the layer as it grows and is written
  as one undo step when the button is released.
* :class:`VesselTrace` — points clicked along a vessel, the cheapest path
  through them, and the tube filled around that path.
* :class:`SmartBrushStroke` — a brush that only paints the voxels under it that
  go with the intensity at its centre.

The growing itself is :mod:`nvitk.segmentation.interactive`'s, on the active
backend; the layer's voxels are NumPy (Napari's), written live without history
and then once through it.
"""

from __future__ import annotations

import math
import weakref
from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.gui.labels import editing as E
from nvitk.segmentation.interactive import (
    GrowSession,
    adaptive_brush_keep,
    path_radii,
    trace_vessel_path,
    tube_from_path,
)

setup(globals())

#: Tools that grow a region from a seed (a click, a drag, a tuned setting).
REGION_TOOLS = ("wand", "flood", "vessel")
#: The tab's own Draw tools (Napari's layer sits in pan/zoom under them).
EXTRA_TOOLS = ("wand", "flood", "vessel", "tracer", "smart", "polygon")
#: What Alt+drag tunes for each region tool, and how far one screen pixel moves it
#: (a fraction of the reference's display window for the wand's tolerance).
TUNED = {"wand": "tolerance", "flood": "spread", "vessel": "vesselness fraction"}


def free_dims(index: Sequence[Any]) -> list[int]:
    """The axes an index into a layer leaves whole (or as a range): the region's."""
    return [d for d, ix in enumerate(index) if isinstance(ix, slice)]


class RegionStroke:
    """One use of a region tool on *layer* within ``layer.data[index]`` (see the
    module docstring). *panel* is the Labeling tab: the reference image, the
    active label, the editable area and the tool's settings come from it."""

    def __init__(self, panel: Any, layer: Any, index: Sequence[Any], tool: str, *, remove: bool = False) -> None:
        if tool not in REGION_TOOLS:
            raise ValueError(f"Not a region tool: {tool!r}.")
        ref = panel.reference()
        if ref is None:
            raise ValueError("This tool reads a reference image: pick one under Layer.")
        self.panel = panel
        self.layer = layer
        self.index = tuple(index)
        self.tool = tool
        self.remove = bool(remove)
        self.label = int(panel.label_id())
        if self.label == 0 and not self.remove:
            raise ValueError("Pick the label to draw (not the background).")
        data = E.editable_data(layer)
        # The layer is NumPy (Napari's): a host copy to put back, a backend one to compute with.
        self.base = to_numpy(data[self.index]).copy()
        self.work = as_backend_array(self.base)
        self.free = free_dims(self.index)
        self.allowed = as_backend_array(panel._allowed(layer, self.index, self.work))
        self.options = panel.region_options(layer, tool, self.free)
        self.session = GrowSession(
            ref.data[self.index],
            smooth_sigma=self.options["smooth_sigma"],
            spacing=self.options["spacing"],
            full_connectivity=self.options["full_connectivity"],
            # Growing stops at what may not be written; taking out goes anywhere.
            allowed=None if self.remove else self.allowed,
        )
        self.value = float(self.options["tune"])
        self.union: Any | None = None
        self.seeds: list[tuple[int, ...]] = []

    # ── seeds ─────────────────────────────────────────────────────────────────

    def local(self, voxel: Sequence[int]) -> tuple[int, ...] | None:
        """*voxel* (an index into the layer) in the region's own coordinates;
        ``None`` when it lies outside the region."""
        shape = self.layer.data.shape
        out = []
        for d, ix in enumerate(self.index):
            v = int(voxel[d])
            if isinstance(ix, slice):
                start, stop, _step = ix.indices(int(shape[d]))
                if not start <= v < stop:
                    return None
                out.append(v - start)
            elif v != int(ix):
                return None
        return tuple(out)

    def grow(self, local: Sequence[int], value: float | None = None) -> Any:
        """The tool's region from the region-coordinates seed *local*."""
        o, s = self.options, self.session
        v = self.value if value is None else float(value)
        common = {"max_distance_mm": o["max_distance_mm"], "fill_holes": o["fill_holes"],
                  "close_radius": o["close_radius"]}
        if self.tool == "wand":
            return s.wand(local, v, seed_radius=o["seed_radius"], **common)
        if self.tool == "flood":
            return s.confidence(local, v, iterations=o["iterations"], seed_radius=max(o["seed_radius"], 1), **common)
        return s.vessel(local, v, radii_mm=o["radii_mm"], bright=o["bright"], tolerance=o["tolerance"],
                        seed_radius=max(o["seed_radius"], 1), **common)

    def add(self, voxel: Sequence[int]) -> bool:
        """Grow from *voxel* too, unless the region already holds it; True when it grew."""
        local = self.local(voxel)
        if local is None or (self.union is not None and bool(self.union[local])):
            return False
        region = self.grow(local)
        self.union = region if self.union is None else (self.union | region)
        self.seeds.append(local)
        self.show()
        return True

    def tune(self, value: float) -> None:
        """Grow again from the first seed with the tuned setting *value*."""
        self.value = float(value)
        if self.seeds:
            self.union = self.grow(self.seeds[0])
            self.show()

    # ── the result ────────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return 0 if self.union is None else int(np.count_nonzero(self.union))

    def result(self) -> Any:
        """``layer.data[index]`` with the region added to (taken out of) the label."""
        out = self.work.copy()
        if self.union is None:
            return out
        if self.remove:
            out[self.union & (self.work == self.label)] = 0
        else:
            out[self.union & self.allowed] = self.label
        return out

    def show(self) -> None:
        """The stroke so far, in the layer (no history yet)."""
        self.layer.data[self.index] = to_numpy(self.result())
        self.panel.live_refresh(self.layer, (self.label,))

    def commit(self) -> int:
        """Put the voxels back and write the stroke once, through the history; the
        voxels changed."""
        self.layer.data[self.index] = self.base
        if self.union is None:
            self.panel.live_refresh(self.layer, ())
            return 0
        return int(self.panel._write(self.layer, self.index, self.result()))

    def cancel(self) -> None:
        self.layer.data[self.index] = self.base
        self.panel.live_refresh(self.layer, ())


class SmartBrushStroke:
    """A brush stroke that paints, under the brush, only the voxels that go with
    the intensity at its centre (within a tolerance, or on the centre's side of
    the brush's Otsu threshold) — one undo step for the drag."""

    def __init__(self, panel: Any, layer: Any, dims: Sequence[int]) -> None:
        ref = panel.reference()
        if ref is None:
            raise ValueError("The smart brush reads a reference image: pick one under Layer.")
        self.panel = panel
        self.layer = layer
        self.ref = ref
        self.dims = [int(d) for d in dims]
        self.label = int(panel.label_id())
        if self.label == 0:
            raise ValueError("Pick the label to draw (not the background).")
        self.tolerance = panel.smart_tolerance()
        E.editable_data(layer)
        self.previous = getattr(layer, "_block_history", None)
        if self.previous is not None:
            layer._block_history = True
        self.last: tuple[int, ...] | None = None
        #: Every voxel the stroke painted (flat indices), however many dabs covered it.
        self._painted: set[int] = set()

    def _centre_value(self, voxel: Sequence[int]) -> float:
        box = tuple(slice(max(int(v) - 1, 0), int(v) + 2) if d in self.dims else int(v) for d, v in enumerate(voxel))
        return float(as_backend_array(self.ref.data[box]).mean())

    def dab(self, voxel: Sequence[int]) -> None:
        indices = E.brush_indices(self.layer, voxel, self.dims)
        if not indices or int(indices[0].size) == 0:
            return
        keep = to_numpy(adaptive_brush_keep(self.ref.data[indices], self._centre_value(voxel), self.tolerance))
        indices = tuple(ix[keep] for ix in indices)
        if int(indices[0].size) and E.paint_at(self.layer, indices, self.label, refresh=False):
            with using("cpu"):
                flat = np.ravel_multi_index(indices, self.layer.data.shape)
            self._painted.update(int(i) for i in flat)
            self.panel.live_refresh(self.layer, (self.label,))

    @property
    def painted(self) -> int:
        """Voxels the stroke has painted."""
        return len(self._painted)

    def extend(self, voxel: Sequence[int]) -> None:
        """Dabs from the last one to *voxel*, at most a quarter brush apart."""
        voxel = tuple(int(v) for v in voxel)
        if self.last is None:
            self.dab(voxel)
        else:
            span = max(abs(a - b) for a, b in zip(voxel, self.last))
            step = max(float(self.layer.brush_size) / 4.0, 1.0)
            n = max(int(math.ceil(span / step)), 1)
            for k in range(1, n + 1):
                t = k / n
                self.dab(tuple(int(round(a + (b - a) * t)) for a, b in zip(self.last, voxel)))
        self.last = voxel

    def finish(self) -> int:
        """Close the stroke as one history item; the voxels painted."""
        layer = self.layer
        if self.previous is not None:
            layer._commit_staged_history()
            layer._block_history = self.previous
        try:
            layer.refresh()
        except Exception:  # noqa: BLE001
            pass
        self.panel.after_write(layer)
        return self.painted


class VesselTrace:
    """Points clicked along a vessel on *layer*, the cheapest path through them
    (inside ``layer.data[index]``, a volume), and the tube filled around it."""

    def __init__(self, panel: Any, layer: Any, index: Sequence[Any]) -> None:
        ref = panel.reference()
        if ref is None:
            raise ValueError("The vessel tracer reads a reference image: pick one under Layer.")
        self.panel = panel
        self.layer_ref = weakref.ref(layer)
        self.index = tuple(index)
        self.free = free_dims(self.index)
        self.options = panel.region_options(layer, "vessel", self.free)
        self.session = GrowSession(ref.data[self.index], smooth_sigma=0.0, spacing=self.options["spacing"])
        self.points: list[tuple[int, ...]] = []      # layer voxels
        self._pieces: list[Any] = []                  # host (n, k) region voxels per segment

    @property
    def layer(self) -> Any | None:
        return self.layer_ref()

    def _local(self, voxel: Sequence[int]) -> tuple[int, ...] | None:
        layer = self.layer
        out = []
        for d, ix in enumerate(self.index):
            v = int(voxel[d])
            if isinstance(ix, slice):
                start, stop, _s = ix.indices(int(layer.data.shape[d]))
                if not start <= v < stop:
                    return None
                out.append(v - start)
            elif v != int(ix):
                return None
        return tuple(out)

    def _to_layer(self, local_points: Any) -> list[tuple[int, ...]]:
        """Region voxels back to layer voxels."""
        layer = self.layer
        out = []
        for p in to_numpy(local_points).astype(int).tolist():
            voxel, k = [], 0
            for d, ix in enumerate(self.index):
                if isinstance(ix, slice):
                    start = ix.indices(int(layer.data.shape[d]))[0]
                    voxel.append(int(p[k]) + start)
                    k += 1
                else:
                    voxel.append(int(ix))
            out.append(tuple(voxel))
        return out

    def add(self, voxel: Sequence[int]) -> None:
        """Another point along the vessel: the path is extended to it."""
        local = self._local(voxel)
        if local is None:
            raise ValueError("That point is outside the volume being traced (another frame?).")
        if self.points and self._local(self.points[-1]) == local:
            return
        if self.points:
            o = self.options
            piece = trace_vessel_path(self.session, [self._local(self.points[-1]), local],
                                      radii_mm=o["radii_mm"], bright=o["bright"])
            self._pieces.append(to_numpy(piece))
        self.points.append(tuple(int(v) for v in voxel))

    def undo(self) -> None:
        """Drop the last point (and the path to it)."""
        if self.points:
            self.points.pop()
        if self._pieces and len(self._pieces) >= len(self.points):
            self._pieces.pop()

    def path_local(self) -> Any:
        """The whole path, in the region's coordinates (host, ``(N, k)``)."""
        with using("cpu"):
            if not self._pieces:
                return to_numpy([list(self._local(p)) for p in self.points]).astype(int).reshape(-1, len(self.free))
            parts = [self._pieces[0]] + [p[1:] for p in self._pieces[1:]]
            return np.concatenate(parts, axis=0)

    def path(self) -> list[tuple[int, ...]]:
        """The whole path as layer voxels."""
        return self._to_layer(self.path_local())

    def tube(self) -> tuple[Any, Any]:
        """The tube around the path (region coordinates, on the backend) and the
        radii used (mm)."""
        o = self.options
        path = self.path_local()
        if o["tube_radius_mm"] > 0:
            radii = as_backend_array([float(o["tube_radius_mm"])] * len(path))
        else:
            radii = path_radii(self.session, path, bright=o["bright"],
                               min_radius_mm=min(o["radii_mm"]) * 0.5, max_radius_mm=max(o["radii_mm"]) * 1.5)
        return tube_from_path(self.session.shape, path, radii, spacing=self.session.spacing), radii

    def fill(self) -> tuple[int, float]:
        """Give the active label to the tube (preserve labels and the editable area
        apply; one undo step): the voxels changed and the median radius (mm)."""
        layer = self.layer
        if layer is None:
            raise ValueError("The layer traced on is gone.")
        if len(self.points) < 2:
            raise ValueError("Click at least two points along the vessel.")
        label = int(self.panel.label_id())
        if label == 0:
            raise ValueError("Pick the label to draw (not the background).")
        tube, radii = self.tube()
        work = as_backend_array(to_numpy(E.editable_data(layer)[self.index]))
        out = work.copy()
        out[tube & as_backend_array(self.panel._allowed(layer, self.index, work))] = label
        changed = int(self.panel._write(layer, self.index, out))
        return changed, float(np.median(radii)) if int(radii.size) else 0.0


__all__ = [
    "EXTRA_TOOLS",
    "REGION_TOOLS",
    "TUNED",
    "RegionStroke",
    "SmartBrushStroke",
    "VesselTrace",
    "free_dims",
]
