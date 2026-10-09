"""Region growing for interactive (manual) segmentation.

What the Labeling tab's region tools compute, from a voxel the user points at:

* :meth:`GrowSession.wand` — the magic wand: the connected region within a
  fixed intensity tolerance of the seed;
* :meth:`GrowSession.confidence` — an adaptive ("confidence-connected") flood:
  the intensity window starts as the seed neighbourhood's mean ± k·σ and is
  re-estimated from the region grown so far, a few rounds, so it settles on the
  structure's own spread rather than on a guessed tolerance;
* :meth:`GrowSession.vessel` — a flood through a vesselness map (Frangi's, at
  the radii of the vessels sought), so it follows a tube and stops where a blob
  or the background touches it;
* :func:`trace_vessel_path` / :func:`tube_from_path` — the cheapest path between
  points clicked along a vessel (through the vesselness), and a tube around it
  whose radius is measured from the lumen along the way;
* :func:`adaptive_brush_keep` — which voxels under a brush belong with the
  centre's intensity (the "smart brush").

A :class:`GrowSession` prepares the image once — float, smoothed, on the active
backend — so a drag that seeds the wand at every step, or a tolerance tuned with
the mouse, regrows from it without redoing that work. Everything runs on the
active backend (CuPy when the GPU is on); the vesselness filter and the
minimal-path search have no CuPy counterpart and run on the host, their results
coming back as backend arrays.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using

setup(globals())

#: Default vessel radii looked for, in mm (vesselness scales).
DEFAULT_RADII_MM = (0.5, 3.0)


def _ball(radius: int, ndim: int) -> Any:
    """Round structuring element of *radius* voxels."""
    r = max(int(radius), 1)
    axis = np.arange(-r, r + 1)
    grids = np.meshgrid(*([axis] * ndim), indexing="ij")
    return sum(g * g for g in grids) <= r * r


def _spacing(spacing: Sequence[float] | None, ndim: int) -> tuple[float, ...]:
    if spacing is None or len(list(spacing)) < ndim:
        return (1.0,) * ndim
    return tuple(abs(float(s)) or 1.0 for s in list(spacing)[:ndim])


class GrowSession:
    """An intensity image prepared once for growing regions from seeds (see the
    module docstring).

    *intensity* is a 2D plane or a 3D volume; *smooth_sigma* (voxels) smooths it
    first, so noise does not break a region into single voxels; *spacing* (mm per
    axis) makes distances and vessel radii round in the patient;
    *full_connectivity* grows through diagonals as well as faces. *allowed* (a
    boolean array like *intensity*) keeps every region inside it.
    """

    def __init__(
        self,
        intensity: Any,
        *,
        smooth_sigma: float = 0.0,
        spacing: Sequence[float] | None = None,
        full_connectivity: bool = False,
        allowed: Any | None = None,
    ) -> None:
        raw = as_backend_array(intensity).astype(np.float32, copy=False)
        self.raw = raw
        self.image = ndi.gaussian_filter(raw, float(smooth_sigma)) if float(smooth_sigma) > 0 else raw
        self.ndim = int(raw.ndim)
        self.shape = tuple(int(n) for n in raw.shape)
        self.spacing = _spacing(spacing, self.ndim)
        self.structure = ndi.generate_binary_structure(self.ndim, self.ndim if full_connectivity else 1)
        self.allowed = None if allowed is None else as_backend_array(allowed).astype(bool, copy=False)
        self._vesselness: dict[tuple, tuple[tuple[slice, ...], Any]] = {}

    # ── pieces ────────────────────────────────────────────────────────────────

    def _seed(self, seed: Sequence[int]) -> tuple[int, ...]:
        out = tuple(int(v) for v in seed)
        if len(out) != self.ndim or any(not 0 <= v < n for v, n in zip(out, self.shape)):
            raise ValueError(f"Seed {out} is outside the image {self.shape}.")
        return out

    def _box(self, seed: tuple[int, ...], radius: int) -> tuple[slice, ...]:
        r = max(int(radius), 0)
        return tuple(slice(max(c - r, 0), min(c + r + 1, n)) for c, n in zip(seed, self.shape))

    def seed_stats(self, seed: Sequence[int], radius: int = 0) -> tuple[float, float]:
        """Mean and standard deviation of the (smoothed) image around *seed*."""
        values = self.image[self._box(self._seed(seed), radius)]
        return float(values.mean()), float(values.std())

    def window(self, seed: Sequence[int], max_distance_mm: float) -> tuple[slice, ...]:
        """The part of the image within *max_distance_mm* of *seed* (all of it for 0)."""
        if float(max_distance_mm) <= 0:
            return tuple(slice(0, n) for n in self.shape)
        seed = self._seed(seed)
        return tuple(
            slice(max(c - int(math.ceil(max_distance_mm / s)), 0), min(c + int(math.ceil(max_distance_mm / s)) + 1, n))
            for c, s, n in zip(seed, self.spacing, self.shape)
        )

    def connected(self, candidate: Any, seed: Sequence[int], max_distance_mm: float = 0.0) -> Any:
        """The connected part of *candidate* (boolean, like the image) holding *seed*,
        within *max_distance_mm* of it and inside :attr:`allowed`."""
        seed = self._seed(seed)
        win = self.window(seed, max_distance_mm)
        sub = as_backend_array(candidate)[win].copy()
        local = tuple(c - w.start for c, w in zip(seed, win))
        if self.allowed is not None:
            sub &= self.allowed[win]
        if float(max_distance_mm) > 0:
            grids = np.ogrid[tuple(slice(0, w.stop - w.start) for w in win)]
            dist2 = sum(((g - c) * s) ** 2 for g, c, s in zip(grids, local, self.spacing))
            sub &= dist2 <= float(max_distance_mm) ** 2
        sub[local] = True  # the click itself, whatever smoothing did to it
        lab, _n = ndi.label(sub, structure=self.structure)
        out = np.zeros(self.shape, dtype=bool)
        out[win] = lab == lab[local]
        return out

    def finish(self, region: Any, *, close_radius: int = 0, fill_holes: bool = True) -> Any:
        """Close gaps up to *close_radius* voxels and fill the holes of *region*."""
        if int(close_radius) > 0:
            region = ndi.binary_closing(region, structure=_ball(int(close_radius), self.ndim)) | region
        if fill_holes:
            region = ndi.binary_fill_holes(region)
        if self.allowed is not None:
            region &= self.allowed
        return region

    # ── the region tools ──────────────────────────────────────────────────────

    def wand(self, seed: Sequence[int], tolerance: float, *, seed_radius: int = 0, max_distance_mm: float = 0.0,
             fill_holes: bool = True, close_radius: int = 0) -> Any:
        """The magic wand's region: connected voxels within *tolerance* of the seed's
        value (the mean over *seed_radius* voxels around it)."""
        value, _sd = self.seed_stats(seed, seed_radius)
        candidate = np.abs(self.image - value) <= float(tolerance)
        region = self.connected(candidate, seed, max_distance_mm)
        return self.finish(region, close_radius=close_radius, fill_holes=fill_holes)

    def confidence(self, seed: Sequence[int], multiplier: float = 2.5, *, iterations: int = 4, seed_radius: int = 1,
                   max_distance_mm: float = 0.0, fill_holes: bool = True, close_radius: int = 0) -> Any:
        """Adaptive flood: within *multiplier* standard deviations of the mean,
        both re-estimated from the region grown so far (*iterations* rounds)."""
        mean, sd = self.seed_stats(seed, max(int(seed_radius), 1))
        lo, hi = float(self.image.min()), float(self.image.max())
        floor = 1e-3 * max(hi - lo, 1e-6)
        region = None
        for _ in range(max(int(iterations), 0) + 1):
            half = float(multiplier) * max(sd, floor)
            region = self.connected(np.abs(self.image - mean) <= half, seed, max_distance_mm)
            values = self.image[region]
            if int(values.size) < 2:
                break
            new_mean, new_sd = float(values.mean()), float(values.std())
            if abs(new_mean - mean) < 1e-6 * max(abs(mean), 1.0) and abs(new_sd - sd) < 1e-6 * max(sd, 1.0):
                break
            mean, sd = new_mean, new_sd
        return self.finish(region, close_radius=close_radius, fill_holes=fill_holes)

    def vesselness(self, *, radii_mm: Sequence[float] = DEFAULT_RADII_MM, bright: bool = True,
                   around: Sequence[int] | None = None, half_size_mm: float = 30.0) -> tuple[tuple[slice, ...], Any]:
        """Frangi's vesselness at the vessel radii *radii_mm* (min, max), normalised to
        0…1: ``(window, values)`` — the whole image, or the window of *half_size_mm*
        around *around* (computed once per window and kept)."""
        win = self.window(around, half_size_mm) if around is not None else tuple(slice(0, n) for n in self.shape)
        r0, r1 = sorted(float(r) for r in radii_mm)
        mean_sp = float(sum(self.spacing) / self.ndim)
        sigmas = [max(r0 / mean_sp, 0.5)]
        while sigmas[-1] * 1.5 < r1 / mean_sp:
            sigmas.append(sigmas[-1] * 1.5)
        if r1 / mean_sp > sigmas[-1]:
            sigmas.append(r1 / mean_sp)
        key = (tuple((w.start, w.stop) for w in win), tuple(round(s, 4) for s in sigmas), bool(bright))
        for cached_key, (cached_win, values) in self._vesselness.items():
            # A window already computed that holds this one serves it.
            if cached_key[1:] == key[1:] and all(a[0] <= w.start and w.stop <= a[1] for a, w in zip(cached_key[0], win)):
                return cached_win, values
        from skimage.filters import frangi

        # No CuPy vesselness filter: on the host, back to the backend. Frangi's,
        # because it scores blobs low (Sato's tubeness does not tell them apart).
        with using("cpu"):
            host = to_numpy(self.raw[win]).astype(np.float32, copy=False)
            v = frangi(host, sigmas=sigmas, black_ridges=not bright).astype(np.float32)
            top = float(v.max()) if v.size else 0.0
            if top > 0:
                v /= top
        values = as_backend_array(v)
        if len(self._vesselness) > 8:
            self._vesselness.clear()
        self._vesselness[key] = (win, values)
        return win, values

    def vessel(self, seed: Sequence[int], fraction: float = 0.15, *, radii_mm: Sequence[float] = DEFAULT_RADII_MM,
               bright: bool = True, tolerance: float | None = None, seed_radius: int = 1,
               max_distance_mm: float = 0.0, fill_holes: bool = True, close_radius: int = 0) -> Any:
        """A flood through the vessel at *seed*: voxels whose vesselness is at least
        *fraction* of the seed's (and, with *tolerance*, whose intensity is no
        further than that below the seed's for a bright vessel — above, for a dark
        one), connected to the seed. The region is then grown to the vessel wall:
        the neighbours as bright as the lumen's edge."""
        seed = self._seed(seed)
        half = max(float(max_distance_mm), 4.0 * float(max(radii_mm))) * 1.5
        win, v = self.vesselness(radii_mm=radii_mm, bright=bright, around=seed, half_size_mm=max(half, 30.0))
        local = tuple(c - w.start for c, w in zip(seed, win))
        near = tuple(slice(max(c - 1, 0), c + 2) for c in local)
        v_seed = float(v[near].max())
        candidate = np.zeros(self.shape, dtype=bool)
        sub = v >= float(fraction) * max(v_seed, 1e-6)
        value, _sd = self.seed_stats(seed, seed_radius)
        image = self.image[win]
        if tolerance is not None and float(tolerance) > 0:
            sub &= (image >= value - float(tolerance)) if bright else (image <= value + float(tolerance))
        candidate[win] = sub
        region = self.connected(candidate, seed, max_distance_mm)
        # Vesselness peaks on the centreline and fades towards the wall: take in the
        # neighbours up to the vessel's edge (half way between lumen and background).
        values = self.image[region]
        if int(values.size) > 1:
            lumen = float(np.median(values))
            ring = ndi.binary_dilation(ndi.binary_dilation(region, structure=self.structure),
                                       structure=self.structure) & ~region
            if bool(ring.any()):
                outside = float(np.median(self.image[ring]))
                edge = 0.5 * (lumen + outside)
                for _ in range(3):
                    ring = ndi.binary_dilation(region, structure=self.structure) & ~region
                    grow = ring & ((self.image >= edge) if bright else (self.image <= edge))
                    if self.allowed is not None:
                        grow &= self.allowed
                    if not bool(grow.any()):
                        break
                    region = region | grow
        return self.finish(region, close_radius=close_radius, fill_holes=fill_holes)


# ──────────────────────────────────────────────────────────────────────────────
# Vessel tracing
# ──────────────────────────────────────────────────────────────────────────────


def trace_vessel_path(
    session: GrowSession,
    points: Sequence[Sequence[int]],
    *,
    radii_mm: Sequence[float] = DEFAULT_RADII_MM,
    bright: bool = True,
    margin_mm: float = 10.0,
) -> Any:
    """The cheapest path through the vessel visiting *points* in order (voxel
    indices into the session's image): an ``(N, ndim)`` integer array of voxels.

    The cost of a step is low where the vesselness and the intensity are high
    (for a *bright* vessel), so the path runs along the lumen rather than
    cutting corners; it is searched in the box around each pair of points
    (grown by *margin_mm*), with steps measured in mm.
    """
    from skimage.graph import MCP_Geometric

    pts = [session._seed(p) for p in points]
    if len(pts) < 2:
        return as_backend_array([list(p) for p in pts])
    lo_i, hi_i = float(session.image.min()), float(session.image.max())
    pieces: list[Any] = []
    for a, b in zip(pts, pts[1:]):
        win = tuple(
            slice(max(min(p, q) - int(math.ceil(margin_mm / s)), 0), min(max(p, q) + int(math.ceil(margin_mm / s)) + 1, n))
            for p, q, s, n in zip(a, b, session.spacing, session.shape)
        )
        half = max(max(w.stop - w.start for w in win) * max(session.spacing) / 2.0, 1.0)
        centre = tuple((w.start + w.stop) // 2 for w in win)
        vwin, v = session.vesselness(radii_mm=radii_mm, bright=bright, around=centre, half_size_mm=half + 1.0)
        # The cost box inside the vesselness window.
        win = tuple(slice(max(w.start, vw.start), min(w.stop, vw.stop)) for w, vw in zip(win, vwin))
        vs = v[tuple(slice(w.start - vw.start, w.stop - vw.start) for w, vw in zip(win, vwin))]
        img = (session.image[win] - lo_i) / max(hi_i - lo_i, 1e-6)
        if not bright:
            img = 1.0 - img
        speed = 0.7 * vs + 0.3 * img
        cost = 1.0 / (1e-3 + speed)
        # Minimal paths have no CuPy counterpart: on the host.
        with using("cpu"):
            host_cost = to_numpy(cost).astype(np.float64)
            start = tuple(int(p - w.start) for p, w in zip(a, win))
            end = tuple(int(q - w.start) for q, w in zip(b, win))
            mcp = MCP_Geometric(host_cost, fully_connected=True, sampling=session.spacing)
            mcp.find_costs([start], [end])
            route = np.asarray(mcp.traceback(end), dtype=np.int64) + np.asarray([w.start for w in win], dtype=np.int64)
        pieces.append(route if not pieces else route[1:])
    with using("cpu"):
        path = np.concatenate(pieces, axis=0)
    return as_backend_array(path)


def path_radii(
    session: GrowSession,
    path: Any,
    *,
    bright: bool = True,
    min_radius_mm: float = 0.3,
    max_radius_mm: float = 10.0,
    smooth: int = 5,
) -> Any:
    """The vessel's radius (mm) at each voxel of *path*: the distance from the
    centreline to the lumen's edge, where the intensity is half way between the
    lumen's (along the path) and the background's (around it); smoothed over
    *smooth* points along the path."""
    path = to_numpy(path).astype(int)
    if path.size == 0:
        return as_backend_array([])
    reach = max_radius_mm * 1.5
    win = tuple(
        slice(max(int(path[:, d].min()) - int(math.ceil(reach / s)), 0),
              min(int(path[:, d].max()) + int(math.ceil(reach / s)) + 1, n))
        for d, (s, n) in enumerate(zip(session.spacing, session.shape))
    )
    image = session.image[win]
    # Index arithmetic on the host; the indices go to the backend to read the image.
    with using("cpu"):
        local = path - np.asarray([w.start for w in win], dtype=int)[None, :]
    on_path = image[tuple(as_backend_array(local[:, d]) for d in range(local.shape[1]))]
    lumen = float(np.median(on_path))
    background = float(np.percentile(image, 25 if bright else 75))
    edge = 0.5 * (lumen + background)
    inside = (image >= edge) if bright else (image <= edge)
    depth = ndi.distance_transform_edt(inside, sampling=session.spacing)
    radii = to_numpy(depth[tuple(as_backend_array(local[:, d]) for d in range(local.shape[1]))]).astype(float)
    with using("cpu"):
        if smooth > 1 and radii.size > 2:
            k = int(smooth)
            padded = np.pad(radii, (k // 2, k // 2), mode="edge")
            radii = np.array([np.median(padded[i:i + k]) for i in range(radii.size)])
        radii = np.clip(radii, float(min_radius_mm), float(max_radius_mm))
    return as_backend_array(radii)


def tube_from_path(
    shape: Sequence[int],
    path: Any,
    radii_mm: Any,
    *,
    spacing: Sequence[float] | None = None,
) -> Any:
    """A tube along *path* (``(N, ndim)`` voxels): the union of balls of
    *radii_mm* (one per point, or one for all) around its points, round in mm."""
    shape = tuple(int(n) for n in shape)
    sp = _spacing(spacing, len(shape))
    pts = to_numpy(path).astype(int)
    radii = to_numpy(radii_mm).astype(float).ravel()
    if radii.size == 1 and len(pts) > 1:
        radii = to_numpy([float(radii[0])] * len(pts)).astype(float)
    out = np.zeros(shape, dtype=bool)
    for p, r in zip(pts, radii):
        win = tuple(slice(max(int(c) - int(math.ceil(r / s)), 0), min(int(c) + int(math.ceil(r / s)) + 1, n))
                    for c, s, n in zip(p, sp, shape))
        grids = np.ogrid[tuple(slice(w.start, w.stop) for w in win)]
        dist2 = sum(((g - int(c)) * s) ** 2 for g, c, s in zip(grids, p, sp))
        out[win] |= dist2 <= float(r) ** 2
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Smart brush
# ──────────────────────────────────────────────────────────────────────────────


def adaptive_brush_keep(values: Any, centre_value: float, tolerance: float | None = None) -> Any:
    """Which of the brush's voxel *values* go with the centre's intensity: within
    *tolerance* of *centre_value*, or — without one — on the centre's side of the
    Otsu threshold between the brush's two populations."""
    values = as_backend_array(values).astype(np.float32, copy=False)
    if tolerance is not None and float(tolerance) > 0:
        return np.abs(values - float(centre_value)) <= float(tolerance)
    if int(values.size) < 2 or float(values.max()) <= float(values.min()):
        return np.ones(values.shape, dtype=bool)
    # Otsu on a 64-bin histogram, on the backend.
    counts, edges = np.histogram(values, bins=64)
    counts = counts.astype(np.float64)
    centres = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(counts)
    w1 = w0[-1] - w0
    m0 = np.cumsum(counts * centres) / np.maximum(w0, 1e-12)
    m1 = (np.sum(counts * centres) - np.cumsum(counts * centres)) / np.maximum(w1, 1e-12)
    between = w0 * w1 * (m0 - m1) ** 2
    threshold = float(centres[int(np.argmax(between[:-1]))])
    return (values > threshold) if float(centre_value) > threshold else (values <= threshold)


__all__ = [
    "DEFAULT_RADII_MM",
    "GrowSession",
    "adaptive_brush_keep",
    "path_radii",
    "trace_vessel_path",
    "tube_from_path",
]
