"""Refine a multilabel vessel segmentation against its own image, vessel by vessel.

Description
-----------
Two steps, built for eICAB's circle-of-Willis labels on TOF but generic. Both work in physical
units and from local intensities: one vessel and its surroundings, never the whole volume's
spectrum.

**Local re-segmentation** (:func:`resegment_local`). Each label's lumen is re-drawn at a local
*half maximum*, the threshold between the vessel's own centreline intensity and the background
just around it:

    T(x) = bg(x) + α · (core(x) - bg(x))        (α = 0.5: half maximum)

- ``core`` is the image on the label's skeleton (lightly denoised);
- ``bg`` is the image in a ring around the label (1-3 mm out, unlabelled voxels only).
- Both are spread along the vessel by normalised convolution (Gaussian σ ≈ 2 mm), so ``T``
  follows the inflow decay along a long vessel.
- The new lumen is the voxels above ``T`` in the label's neighbourhood that connect to its
  skeleton. The other labels are barriers: their original voxels are never taken.
- A thick mask shrinks to the half-maximum boundary; a thin one may grow by up to ``margin_mm``
  or half its radius, whichever is smaller.
- A vessel too faint for a half maximum (contrast ``(core - bg) / bg`` below ``min_contrast``)
  keeps its mask.
- The threshold is applied to the image lightly denoised (σ = ``DENOISE_MM``), and each new mask
  goes through :func:`nvitk.morphology.siphon_cut.pretess`. Voxel noise at a half-maximum
  boundary otherwise leaves small handles (loops) that the mask did not have.

**Trunk extension** (:func:`extend_trunks`). Recovers the stretch of a trunk the segmentation
left out, such as the inferior ICAs and basilar below eICAB's circle-of-Willis box:

- A multi-scale Frangi vesselness (σ in mm, bright tubes; :func:`frangi_vesselness`) on a crop
  beyond the label's open end, and a GMM hysteresis tree (as :mod:`nvitk.segmentation.blood_flood`)
  gated by the trunk's own local intensity.
- The trunk goes to the image's edge in the extension direction along the minimum-cost path
  through the tree, cheaper along bright tubes. It starts from the label's lowest band and from
  its lowest skeleton tip (the open end), and may run up to ``UP_SLACK_MM`` above the label's
  lowest voxel: the petrous ICA often leaves the label's lowest bend sideways before it descends.
  Bright lumen (above the local half maximum) counts as tree, which bridges the Frangi gaps at
  junctions and bends. Frangi fades at the volume's edge, so the last 3 mm accept any voxel above
  the gate. The path ends on a vessel cross-section at the floor. A label may follow several
  branches (the basilar into both vertebrals): each further branch must run at least
  ``BRANCH_MIN_MM`` apart from the ones already taken.
- Without a path to the edge, the tree is followed as one tube, one mm-thick geodesic shell at a
  time, until it ends, leaks (a shell far wider than the trunk) or bifurcates (comparable pieces
  that stay apart for ``SPLIT_LOOKAHEAD_MM``; a sharp bend splits the front only briefly).
- The lumen around the route is drawn at the same local half maximum.

**Distal growth** (:func:`grow_distal`). Several labels grow along the Frangi tree beyond their own
voxels at once, in competition: each tree voxel goes to the label whose geodesic front (cheaper
along vessel centres) reaches it first. Paired neighbours, such as the two ACAs, can be kept to
their own hemisphere near the midline (:func:`head_midplane_from_image`).

Array / axis conventions
------------------------
3D volumes in array index order; voxel size per axis (mm) from the affine's column norms
(:func:`spacing_of`), so anisotropic voxels are handled. Directions (the superior axis, a plane
normal) are world-mm vectors expressed per index axis.

Backend
-------
I/O and arrays: ``np`` / ``scipy`` / ``ndi`` after ``setup(globals())`` follow
:mod:`nvitk.core.backend` (NumPy or CuPy). Volume work (filters, distance transforms,
labelling, the Frangi Hessian) runs on the active backend; inputs are brought there with
:func:`as_backend_array` and the label volumes are returned there. Host-only steps go through
:func:`to_numpy` under ``with using("cpu")``: scikit-image's minimum-cost paths
(``MCP_Geometric``), the skeleton (scikit-image, via :func:`skeletonize_binary`), the GMM
hysteresis tree, the pretess step, and the small geometry on voxel lists (end bands, polylines,
the midplane fit).
"""

from __future__ import annotations

import functools
import math
import operator
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import get_current_backend, map_in_thread_pool, setup, using
from nvitk.core.logger import Logger

setup(globals())

log = Logger()

# Defaults (see the module docstring)

ALPHA = 0.5
MARGIN_MM = 1.0
SIGMA_MM = 2.0
RING_MM = (1.0, 3.0)
MIN_KEEP_FRACTION = 0.25
#: (core - background) / background below which a vessel keeps its mask: too faint for a half maximum.
MIN_CONTRAST = 0.5
#: Gaussian σ (mm) of the light denoising the threshold is applied to.
DENOISE_MM = 0.3
#: Lateral reach of the extension crop (mm), across the extension direction.
LATERAL_MM = 40.0
FRANGI_SIGMAS_MM = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
MAX_EXTENSION_MM = 100.0
SHELL_MM = 1.0
SPLIT_RATIO = 0.2
#: How far (mm) a split has to persist to be a bifurcation rather than a bend.
SPLIT_LOOKAHEAD_MM = 3.0
LEAK_FACTOR = 3.0
DIRECTIONS = ("inferior", "superior")
#: How far (mm) above a trunk's lowest voxel its extension may still run: the petrous ICA leaves the
#: label's lowest bend sideways, at or a little above that level, before it descends.
UP_SLACK_MM = 5.0
#: A further floor branch is kept only if this much of it (mm) runs apart from the branches already
#: taken (farther than twice the trunk radius, at least 3 mm): a second vertebral, not a detour.
BRANCH_MIN_MM = 4.0
#: ... and if that part of it follows a vessel: mean step cost at most this (1 on a strong tube,
#: 5 off any). Second vertebrals measure 1.0-2.3 on PESA; the total cost is no guide, as the second
#: vertebral often reaches the floor two or three times farther than the first.
BRANCH_MAX_COST = 2.5

#: Voxel size (mm) per array axis.
Spacing = tuple[float, float, float]


# Small helpers


def spacing_of(affine: Any = None, spacing: Sequence[float] | None = None) -> Spacing:
    """Voxel size (mm) per array axis: the affine's column norms, else *spacing*, else 1 mm."""
    if affine is not None:
        aff = to_numpy(affine)
        if aff.shape == (4, 4):
            return tuple(  # type: ignore[return-value]
                math.sqrt(sum(float(aff[r, c]) ** 2 for r in range(3))) for c in range(3)
            )
    if spacing is not None and len(spacing) >= 3:
        return tuple(float(s) for s in spacing[:3])  # type: ignore[return-value]
    return (1.0, 1.0, 1.0)


def _cube() -> Any:
    """26-connectivity structuring element on the active backend."""
    return np.ones((3, 3, 3), dtype=bool)


def _vox(mm: float, sp: Spacing, *, extra: int = 0) -> tuple[int, int, int]:
    """*mm* as whole voxels along each axis (rounded up), plus *extra*."""
    return tuple(int(math.ceil(float(mm) / s)) + int(extra) for s in sp)  # type: ignore[return-value]


def _sigma_vox(mm: float, sp: Spacing, floor: float = 0.0) -> tuple[float, float, float]:
    """A Gaussian σ of *mm* in voxels along each axis, at least *floor*."""
    return tuple(max(float(mm) / s, float(floor)) for s in sp)  # type: ignore[return-value]


def _voxel_mm3(sp: Spacing) -> float:
    """Volume of one voxel (mm³)."""
    return float(sp[0] * sp[1] * sp[2])


def _isin(arr: Any, ids: Sequence[int]) -> Any:
    """``np.isin`` against a list of ids (CuPy needs an array, not a list)."""
    if not len(ids):
        return np.zeros(arr.shape, dtype=bool)
    return np.isin(arr, as_backend_array([int(i) for i in ids]))


def _any_of(masks: Sequence[Any]) -> Any:
    """Union of boolean masks (``np.logical_or.reduce`` without a stacked copy)."""
    return functools.reduce(operator.or_, masks)


def _bbox(mask: Any, pad: Sequence[int]) -> tuple[slice, slice, slice] | None:
    """Bounding box of *mask*, grown by *pad* voxels per axis and clipped; ``None`` when empty.

    Read from the three axis projections, so only three short index lists leave the device.
    """
    if not bool(mask.any()):
        return None
    out: list[slice] = []
    for ax in range(3):
        others = tuple(a for a in range(3) if a != ax)
        hit = to_numpy(np.flatnonzero(np.any(mask, axis=others)))
        lo = max(int(hit[0]) - int(pad[ax]), 0)
        hi = min(int(hit[-1]) + int(pad[ax]) + 1, int(mask.shape[ax]))
        out.append(slice(lo, hi))
    return tuple(out)  # type: ignore[return-value]


def _linear_field(shape: Sequence[int], offset: Sequence[float], direction: Sequence[float]) -> Any:
    """``Σ_ax (index_ax + offset_ax) · direction_ax`` for every voxel of *shape* (backend float64).

    With *offset* a crop's origin and *direction* the world superior row of the affine, it is each
    voxel's height in mm; with *offset* = -point and *direction* a unit normal (in mm per index
    step), its signed distance to a plane.
    """
    field_ = np.zeros(tuple(int(n) for n in shape), dtype=np.float64)
    for ax in range(3):
        idx = np.arange(int(shape[ax]), dtype=np.float64) + float(offset[ax])
        bshape = [1, 1, 1]
        bshape[ax] = int(shape[ax])
        field_ += float(direction[ax]) * idx.reshape(bshape)
    return field_


def _dilate(mask: Any, radius_mm: float, sp: Spacing) -> Any:
    """*mask* grown by *radius_mm* (exact in mm: a distance transform, not a big structuring
    element)."""
    if radius_mm <= 0:
        return mask.copy()
    return ndi.distance_transform_edt(~mask, sampling=sp) <= radius_mm


def _normalised_smooth(values: Any, weight: Any, sigma_vox: Sequence[float]) -> tuple[Any, Any]:
    """Normalised convolution: ``G(v·w) / G(w)`` and ``G(w)`` (how much support each voxel has).
    Unsupported voxels read NaN (filled by the caller)."""
    w = weight.astype(np.float64)
    num = ndi.gaussian_filter(values * w, sigma_vox, mode="constant")
    den = ndi.gaussian_filter(w, sigma_vox, mode="constant")
    # Divide only where supported: no 0/0 (CuPy has no errstate to silence it).
    return np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan), den


def _skeleton(mask: Any) -> Any:
    """Skeleton of *mask* (scikit-image on the host, returned on the backend); the deepest voxel
    when thinning leaves nothing."""
    from nvitk.morphology.centerline import skeletonize_binary

    sk = as_backend_array(skeletonize_binary(mask)).astype(bool)
    if not bool(sk.any()) and bool(mask.any()):
        d = ndi.distance_transform_edt(mask)
        sk = mask & (d >= d.max())
    return sk


def denoise(intensity: Any, sp: Spacing) -> Any:
    """The light Gaussian denoising (σ = :data:`DENOISE_MM`) thresholds are compared with."""
    return ndi.gaussian_filter(intensity, _sigma_vox(DENOISE_MM, sp, 0.3))


def local_threshold(
    intensity: Any,
    centre: Any,
    ring: Any,
    sp: Spacing,
    *,
    alpha: float = ALPHA,
    sigma_mm: float = SIGMA_MM,
) -> tuple[Any, dict[str, float]]:
    """``T = bg + α (core - bg)``: *core* spread from the *centre* voxels, *bg* from the *ring*
    voxels, both by normalised convolution with a σ = *sigma_mm* Gaussian. Where a field has no
    support it falls back to its median."""
    sig = _sigma_vox(sigma_mm, sp, 0.5)
    den_img = denoise(intensity, sp)
    core, _ = _normalised_smooth(den_img, centre, sig)
    bg, _ = _normalised_smooth(intensity, ring, sig)
    core_med = float(np.median(den_img[centre])) if bool(centre.any()) else float(np.nanmax(intensity))
    bg_med = float(np.median(intensity[ring])) if bool(ring.any()) else float(np.median(intensity))
    core = np.where(np.isfinite(core), core, core_med)
    bg = np.where(np.isfinite(bg), bg, bg_med)
    bg = np.minimum(bg, core)  # a ring brighter than the lumen (a neighbouring vessel) says nothing
    return bg + float(alpha) * (core - bg), {"core_median": core_med, "background_median": bg_med}


# Local re-segmentation


def resegment_local(
    intensity: Any,
    labels: Any,
    ids: Sequence[int] | None = None,
    *,
    affine: Any | None = None,
    spacing: Sequence[float] | None = None,
    alpha: float = ALPHA,
    margin_mm: float = MARGIN_MM,
    sigma_mm: float = SIGMA_MM,
    ring_mm: tuple[float, float] = RING_MM,
    min_keep_fraction: float = MIN_KEEP_FRACTION,
    min_contrast: float = MIN_CONTRAST,
    pretess: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Re-draw each label in *ids* (default: all) at its local half maximum (module docstring).

    Returns the new label volume (int32, on the active backend) and a per-label log (voxels
    before / after, the medians of the core and background fields, whether the original was
    kept)."""
    inten = as_backend_array(intensity).astype(np.float64, copy=False)
    orig = as_backend_array(labels).astype(np.int32)
    sp = spacing_of(affine, spacing)
    vox = _voxel_mm3(sp)
    present = [int(v) for v in to_numpy(np.unique(orig)) if int(v) > 0]
    ids = [int(i) for i in (ids if ids is not None else present) if int(i) in present]
    out = np.where(_isin(orig, ids), 0, orig).astype(np.int32)
    pad = _vox(ring_mm[1] + margin_mm + 3 * sigma_mm, sp, extra=1)
    info: dict[str, Any] = {"alpha": alpha, "margin_mm": margin_mm, "sigma_mm": sigma_mm, "ring_mm": list(ring_mm),
                            "labels": {}}
    t0 = time.time()
    for lid in ids:
        own_full = orig == lid
        box = _bbox(own_full, pad)
        own = own_full[box]
        lab = orig[box]
        img = inten[box]
        sub = out[box]  # a view: writes land in *out*
        n0 = int(own.sum())
        entry: dict[str, Any] = {"voxels_before": n0, "volume_before_mm3": round(n0 * vox, 2)}
        if n0 < 8:
            sub[own & (sub == 0)] = lid
            entry.update(voxels_after=n0, kept_original="too small")
            info["labels"][str(lid)] = entry
            continue
        sk = _skeleton(own)
        free = (lab == 0) | own  # other labels are barriers
        ring = _dilate(own, ring_mm[1], sp) & ~_dilate(own, ring_mm[0], sp) & (lab == 0)
        thr, tinfo = local_threshold(img, sk, ring, sp, alpha=alpha, sigma_mm=sigma_mm)
        contrast = (tinfo["core_median"] - tinfo["background_median"]) / max(abs(tinfo["background_median"]), 1e-9)
        entry["contrast"] = round(float(contrast), 2)
        if contrast < min_contrast:
            sub[own & (sub == 0)] = lid
            entry.update(voxels_after=n0, kept_original=f"contrast {contrast:.2f} < {min_contrast:g}")
            info["labels"][str(lid)] = entry
            continue
        depth = ndi.distance_transform_edt(own, sampling=sp)
        radius = float(np.median(depth[sk])) if bool(sk.any()) else float(depth.max())
        grow = min(float(margin_mm), 0.5 * radius)
        entry["radius_mm"] = round(radius, 2)
        hood = (_dilate(own, grow, sp) if grow >= 0.5 * min(sp) else own) & free
        smooth = denoise(img, sp)
        cand = hood & (smooth > thr)
        cc, _ = ndi.label(cand, structure=_cube())
        keep = np.unique(cc[sk & cand])
        new = np.isin(cc, keep[keep > 0])
        new = ndi.binary_fill_holes(new) & hood
        if pretess and bool(new.any()):
            from nvitk.morphology.siphon_cut import pretess as _pretess

            new, pinfo = _pretess(new, smooth, protect=~free)
            new = as_backend_array(new).astype(bool)
            entry["pretess_removed"] = int(pinfo.get("n_removed", 0))
        n1 = int(new.sum())
        entry.update(core_median=round(tinfo["core_median"], 2), background_median=round(tinfo["background_median"], 2))
        if n1 < min_keep_fraction * n0:
            new = own
            entry["kept_original"] = f"only {n1} voxels above the local threshold"
        sub[new & (sub == 0)] = lid
        n_after = int((sub == lid).sum())
        entry.update(voxels_after=n_after, volume_after_mm3=round(n_after * vox, 2),
                     added=int((new & ~own).sum()), removed=int((own & ~new).sum()))
        entry["volume_change_pct"] = round(100.0 * (n_after - n0) / n0, 2)
        info["labels"][str(lid)] = entry
    info["seconds"] = round(time.time() - t0, 2)
    return out, info


def tidy_labels(
    labels: Any, ids: Sequence[int] | None = None, *, anchors: dict[int, int] | None = None
) -> tuple[Any, dict[str, Any]]:
    """Each label in *ids* (default: all) reduced to its largest 26-connected component, then its
    enclosed holes filled (3-D; only unlabelled voxels are taken, so a neighbouring label is never
    overwritten and a loop's open hole stays open). A label in *anchors* (``{label: anchor}``, e.g.
    an M2 and its M1) keeps instead every component touching its anchor label: its branches meet
    only through the anchor. Returns the label volume (int32, on the active backend) and, per
    label, the voxels dropped and filled."""
    anchors = {int(k): int(v) for k, v in (anchors or {}).items()}
    out = as_backend_array(labels).astype(np.int32)
    present = [int(v) for v in to_numpy(np.unique(out)) if int(v) > 0]
    ids = [int(i) for i in (ids if ids is not None else present) if int(i) in present]
    info: dict[str, Any] = {}
    for lid in ids:
        box = _bbox(out == lid, (1, 1, 1))
        if box is None:
            continue
        sub = out[box]  # a view: writes land in *out*
        own = sub == lid
        cc, n = ndi.label(own, structure=_cube())
        dropped = 0
        if int(n) > 1:
            sizes = np.bincount(cc.ravel())
            sizes[0] = 0
            keep = cc == int(np.argmax(sizes))
            if lid in anchors:  # every piece on the anchor (the largest when none touches it)
                near = ndi.binary_dilation(sub == anchors[lid], structure=_cube())
                touching = np.unique(cc[near & own])
                touching = touching[touching > 0]
                if touching.size:
                    keep = np.isin(cc, touching)
            dropped = int((own & ~keep).sum())
            sub[own & ~keep] = 0
            own = keep
        holes = ndi.binary_fill_holes(own) & ~own & (sub == 0)
        sub[holes] = lid
        if dropped or bool(holes.any()):
            info[str(lid)] = {"dropped": dropped, "filled": int(holes.sum())}
    return out, info


# Frangi vesselness


def _sym3_eigvals(a11, a22, a33, a12, a13, a23):
    """Eigenvalues of symmetric 3×3 matrices given element-wise (closed form, Smith 1961),
    unsorted, as three arrays."""
    p1 = a12 * a12 + a13 * a13 + a23 * a23
    q = (a11 + a22 + a33) / 3.0
    b11, b22, b33 = a11 - q, a22 - q, a33 - q
    p2 = b11 * b11 + b22 * b22 + b33 * b33 + 2.0 * p1
    p = np.sqrt(p2 / 6.0)
    safe = np.where(p > 1e-12, p, 1.0)
    c11, c22, c33, c12, c13, c23 = b11 / safe, b22 / safe, b33 / safe, a12 / safe, a13 / safe, a23 / safe
    det = (c11 * (c22 * c33 - c23 * c23) - c12 * (c12 * c33 - c23 * c13) + c13 * (c12 * c23 - c22 * c13)) / 2.0
    phi = np.arccos(np.clip(det, -1.0, 1.0)) / 3.0
    e1 = q + 2.0 * p * np.cos(phi)
    e3 = q + 2.0 * p * np.cos(phi + 2.0 * math.pi / 3.0)
    e2 = 3.0 * q - e1 - e3
    return e1, e2, e3


def frangi_vesselness(
    intensity: Any, sp: Spacing, sigmas_mm: Sequence[float] = FRANGI_SIGMAS_MM,
    *, alpha: float = 0.5, beta: float = 0.5, workers: int = 6,
) -> Any:
    """Multi-scale Frangi vesselness of bright tubes, σ in mm (per axis, so anisotropic voxels
    are fine), normalised to [0, 1] (float32, on the active backend).

    The same measure as scikit-image's ``frangi(black_ridges=False)``, which the blood flood uses:
    Gaussian derivatives as two first-order passes of σ/√2, and γ = half the largest Hessian norm
    of the first scale, kept for the others. It is computed faster: the eigenvalues in closed
    form, and on the CPU backend the Hessian components in parallel threads (~6× on a TOF crop);
    on CuPy the filters run on the device, one after another."""
    from itertools import combinations_with_replacement

    vol = np.nan_to_num(as_backend_array(intensity).astype(np.float64))
    # Clipped at the 99.9th percentile, not 99.5 (the blood flood's): in a head-sized crop the
    # large arteries are more than 0.5 % of the voxels, and clipping them flat leaves their core
    # with no Hessian response.
    lo, hi = float(np.percentile(vol, 1.0)), float(np.percentile(vol, 99.9))
    img = -np.clip((vol - lo) / max(hi - lo, 1e-8), 0.0, 1.0).astype(np.float32)  # bright -> dark ridges
    del vol
    out = np.zeros(img.shape, dtype=np.float32)
    orders = [tuple(int(i == d) for i in range(3)) for d in range(3)]
    pairs = list(combinations_with_replacement(range(3), 2))  # (0,0),(0,1),(0,2),(1,1),(1,2),(2,2)
    # Threads only pay on the host: device kernels queue on one stream anyway.
    n_workers = max(1, int(workers)) if get_current_backend() == "numpy" else 1
    gamma: float | None = None
    for s_mm in sigmas_mm:
        sig = tuple(float(s_mm) / float(x) for x in sp)
        truncate = 8 if all(x > 1 for x in sig) else 100
        half = tuple(x / math.sqrt(2.0) for x in sig)

        def g(arr: Any, order: tuple[int, ...]) -> Any:
            """One first-order Gaussian derivative pass (σ/√2) of *arr* along *order*."""
            return ndi.gaussian_filter(arr, sigma=half, order=order, mode="reflect", truncate=truncate)

        grads = map_in_thread_pool(lambda o: g(img, o), orders, max_workers=n_workers)
        hess = dict(zip(pairs, map_in_thread_pool(lambda ab, gr=grads: g(gr[ab[0]], orders[ab[1]]), pairs,
                                                  max_workers=n_workers)))
        del grads
        ev = _sym3_eigvals(hess[(0, 0)], hess[(1, 1)], hess[(2, 2)], hess[(0, 1)], hess[(0, 2)], hess[(1, 2)])
        del hess
        ev = np.stack(ev)
        ev = np.take_along_axis(ev, np.abs(ev).argsort(0), 0)
        l1 = ev[0]
        l2 = np.maximum(ev[1], 1e-10)
        l3 = np.maximum(ev[2], 1e-10)
        r_a = l2 / l3
        r_b = np.abs(l1) / np.sqrt(l2 * l3)
        strength = np.sqrt((ev ** 2).sum(0))
        del ev
        if gamma is None:  # as scikit-image: set at the first scale, kept for the others
            gamma = float(strength.max()) / 2.0 or 1.0
        v = (1.0 - np.exp(-(r_a ** 2) / (2 * alpha ** 2))) * np.exp(-(r_b ** 2) / (2 * beta ** 2))
        v *= 1.0 - np.exp(-(strength ** 2) / (2 * gamma ** 2))
        np.maximum(out, v.astype(np.float32), out=out)
    m = float(out.max())
    return out / m if m > 0 else out


# Trunk extension


@dataclass
class Extension:
    """One trunk's extension: the route (centroids, host voxel coordinates), the new voxels and
    why it stopped."""

    label: int
    centroids: list[Any] = field(default_factory=list)
    length_mm: float = 0.0
    stop: str = ""
    n_voxels: int = 0
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able summary for the post-process report."""
        return {"label": self.label, "length_mm": round(self.length_mm, 1), "stop": self.stop,
                "voxels_added": self.n_voxels, **self.info}


def _next_shell(front: Any, allowed: Any, taken: Any, per_shell: int) -> Any:
    """The next geodesic shell (*per_shell* 26-connected steps) from *front* inside *allowed*."""
    shell = np.zeros_like(allowed)
    cur = front
    for _ in range(per_shell):
        nxt = ndi.binary_dilation(cur, structure=_cube()) & allowed & ~taken & ~shell
        if not bool(nxt.any()):
            break
        shell |= nxt
        cur = nxt
    return shell


def _pieces(shell: Any) -> tuple[Any, Any]:
    """Components of *shell* (26-connected, labels 1..n; backend) and their sizes (host)."""
    cc, _n = ndi.label(shell, structure=_cube())
    sizes = to_numpy(np.bincount(cc.ravel()))[1:]
    return cc, sizes


def _persistent_split(
    pieces: list[Any], allowed: Any, taken: Any, per_shell: int, n_ahead: int,
    split_ratio: float, min_piece: int,
) -> bool:
    """Whether the comparable *pieces* of a shell stay apart for *n_ahead* more shells (a
    bifurcation) instead of merging again (a bend)."""
    fronts = [p.copy() for p in pieces]
    seen = taken | _any_of(pieces)
    for _ in range(n_ahead):
        nxt = [_next_shell(f, allowed, seen, per_shell) for f in fronts]
        sizes = [int(x.sum()) for x in nxt]
        if sum(sizes) == 0:
            return False  # both end: no bifurcation to stop at
        _cc, csz = _pieces(_any_of(nxt))
        csz = sorted((int(c) for c in csz), reverse=True)
        if len(csz) < 2 or csz[1] < split_ratio * csz[0] or csz[1] < min_piece:
            return False  # merged again, or one piece died out: a bend or a side twig
        for x in nxt:
            seen |= x
        fronts = nxt
    return True


def _follow_tube(
    tree: Any,
    seeds: Any,
    sp: Spacing,
    *,
    max_mm: float,
    shell_mm: float,
    split_ratio: float,
    leak_factor: float,
    seed_area: int,
) -> tuple[Any, list[Any], str]:
    """Follow *tree* from *seeds* one geodesic shell at a time as a single tube (module docstring).
    Returns the route voxels (backend), the shell centroids (host vectors) and the stop reason."""
    step_mm = sum(sp) / 3.0
    per_shell = max(1, int(round(shell_mm / step_mm)))
    n_ahead = max(1, int(round(SPLIT_LOOKAHEAD_MM / (per_shell * step_mm))))
    allowed = tree & ~seeds
    route = np.zeros_like(tree)
    front = seeds.copy()
    centroids: list[Any] = []
    widths: list[int] = []
    travelled = 0.0
    stop = "max length"
    min_piece = max(4, seed_area // 8)
    while travelled < max_mm:
        shell = _next_shell(front, allowed, route | seeds, per_shell)
        if not bool(shell.any()):
            stop = "vessel ends (no more tree)"
            break
        cc, sizes = _pieces(shell)
        order = sizes.argsort()[::-1]
        part = cc == int(order[0]) + 1
        if len(sizes) > 1 and sizes[order[1]] >= split_ratio * sizes[order[0]] and sizes[order[1]] >= min_piece:
            comparable = [cc == int(i) + 1 for i in order if sizes[i] >= split_ratio * sizes[order[0]]]
            if _persistent_split(comparable, allowed, route | seeds, per_shell, n_ahead, split_ratio, min_piece):
                stop = "bifurcation"
                break
            part = _any_of(comparable)  # a bend: keep the front whole
        n_part = int(part.sum())
        ref = statistics.median(widths[-5:]) if widths else max(seed_area, 1)
        if n_part > leak_factor * ref * per_shell and travelled > 0:
            stop = "leak (shell much wider than the trunk)"
            break
        route |= part
        widths.append(n_part // per_shell)
        centroids.append(to_numpy(np.argwhere(part).mean(0)))
        front = part
        travelled += per_shell * step_mm
        box = _bbox(part, (0, 0, 0))
        if box is not None and any(sl.start == 0 or sl.stop == d for sl, d in zip(box, tree.shape)):
            stop = "edge of the field of view"
            break
    return route, centroids, stop


def _polyline_mask(points: list[Any], shape: tuple[int, ...]) -> Any:
    """Host boolean mask of the straight segments through *points* (voxel coordinates)."""
    with using("cpu"):
        out = np.zeros(shape, dtype=bool)
        for a, b in zip(points[:-1], points[1:]):
            n = int(np.ceil(np.abs(b - a).max())) + 1
            for t in np.linspace(0.0, 1.0, max(n, 2)):
                p = np.rint(a + t * (b - a)).astype(int)
                if all(0 <= p[i] < shape[i] for i in range(3)):
                    out[tuple(p)] = True
        if len(points) == 1:
            p = np.rint(points[0]).astype(int)
            out[tuple(np.clip(p, 0, to_numpy(shape) - 1))] = True
        return out


def _path_length_mm(path: Any, sp: Spacing) -> float:
    """Length (mm) of a host ``(N, 3)`` voxel polyline."""
    with using("cpu"):
        if len(path) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(path, axis=0) * to_numpy(sp), axis=1).sum())


def _branch_part(path: Any, others: list[Any], sp: Spacing, min_sep_mm: float, cost: Any) -> tuple[float, float]:
    """Host: the part of *path* farther than *min_sep_mm* from every path in *others*; its length
    (mm) and the mean step *cost* on it (``inf`` when there is none)."""
    with using("cpu"):
        sp_h = to_numpy(sp)
        p = path * sp_h
        far = np.ones(len(p), dtype=bool)
        for o in others:
            q = o * sp_h
            d = np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1)).min(1)
            far &= d > min_sep_mm
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        length = float(seg[far[1:] & far[:-1]].sum())
        if not far.any():
            return length, float("inf")
        vox = tuple(np.rint(path[far]).astype(int).T)
        return length, float(cost[vox].mean())


def _paths_to_floor(
    allowed: Any,
    seeds: Any,
    cost: Any,
    floors: Sequence[Any],
    sp: Spacing,
    *,
    max_paths: int,
    max_mm: float,
    min_sep_mm: float = 3.0,
) -> tuple[list[Any], dict[str, Any]]:
    """Minimum-cost paths through *allowed* from *seeds* to the separate pieces of the floor: the
    cheapest one, and with *max_paths* > 1 the next pieces (by cost) whose own part (where the path
    runs farther than *min_sep_mm* from those already taken) is at least :data:`BRANCH_MIN_MM`
    long and follows a vessel (mean step cost <= :data:`BRANCH_MAX_COST`): a second branch, not a
    detour to a neighbouring piece. *floors* are tried in order until one is reached (vessel
    cross-sections at the floor, then any allowed voxel there). Each path is a host ``(N, 3)``
    voxel array, seed end first (scikit-image ``MCP_Geometric``: host only)."""
    from skimage.graph import MCP_Geometric

    allowed_h, seeds_h, cost_h = (to_numpy(a) for a in (allowed, seeds, cost))
    floors_h = [to_numpy(f) for f in floors]
    with using("cpu"):
        costs = np.where(allowed_h | seeds_h, cost_h, np.inf)
        starts = [tuple(int(v) for v in c) for c in np.argwhere(seeds_h)]
        mcp = MCP_Geometric(costs, fully_connected=True, sampling=tuple(float(x) for x in sp))
        cum, _ = mcp.find_costs(starts)
        ends: list[tuple[float, tuple[int, ...]]] = []
        for k, floor_h in enumerate(floors_h):
            fl, n = ndi.label(floor_h & np.isfinite(cum), structure=_cube())
            for c in range(1, int(n) + 1):
                idx = np.argwhere(fl == c)
                best = idx[int(np.argmin(cum[tuple(idx.T)]))]
                ends.append((float(cum[tuple(best)]), tuple(int(v) for v in best)))
            if ends:
                break
        ends.sort()
        info: dict[str, Any] = {"floor_pieces_reached": len(ends), "floor_target": int(k) if floors_h else None}
        if not ends:
            return [], info
        paths: list[Any] = []
        costs_taken: list[float] = []
        branches: list[dict[str, float]] = []
        for cst, e in ends[:40]:
            path = np.array(mcp.traceback(e), dtype=float)
            if _path_length_mm(path, sp) > max_mm:
                continue
            if paths:
                apart, step = _branch_part(path, paths, sp, min_sep_mm, cost_h)
                if apart < BRANCH_MIN_MM or step > BRANCH_MAX_COST:
                    continue
                branches.append({"apart_mm": round(apart, 1), "mean_step_cost": round(step, 2)})
            paths.append(path)
            costs_taken.append(cst)
            if len(paths) >= int(max_paths):
                break
        info["path_costs"] = [round(c, 1) for c in costs_taken]
        if branches:
            info["further_branches"] = branches
        return paths, info


def _end_band(coords: Any, sup: Any, sign: float, seed_band_mm: float) -> tuple[float, Any]:
    """Host: the farthest height of a label's voxels (*coords*, ``(N, 3)``) along ``sign · sup``,
    and the voxels within *seed_band_mm* of it (its open end)."""
    with using("cpu"):
        height = coords @ sup * sign  # larger = further in the extension direction
        end = float(height.max())
        return end, coords[height >= end - seed_band_mm]


def _extension_box(band: Any, sup: Any, sign: float, sp: Spacing, shape: Sequence[int],
                   max_extension_mm: float | None) -> tuple[Any, Any]:
    """Host: the crop beyond a trunk's open end, as ``(lo, hi)`` index vectors: *max_extension_mm*
    ahead along the extension direction (``None``: to the image's edge), from :data:`UP_SLACK_MM`
    + 3 mm back, and :data:`LATERAL_MM` across it."""
    with using("cpu"):
        sp_h = to_numpy(sp)
        shape_h = to_numpy(tuple(int(n) for n in shape))
        if max_extension_mm is None:
            reach = shape_h.copy()
        else:
            reach = np.ceil((max_extension_mm + 5.0) / sp_h).astype(int)
        lo = np.maximum(band.min(0) - reach, 0)
        hi = np.minimum(band.max(0) + reach + 1, shape_h)
        axis_w = np.abs(sup) / (np.abs(sup).sum() + 1e-12)  # how much each index axis is along the direction
        lat = np.ceil((LATERAL_MM + 5.0) / sp_h).astype(int)
        for ax_i in range(3):
            if axis_w[ax_i] < 0.5:  # mostly across the direction: lateral reach only
                lo[ax_i] = max(int(band[:, ax_i].min()) - int(lat[ax_i]), 0)
                hi[ax_i] = min(int(band[:, ax_i].max()) + int(lat[ax_i]) + 1, int(shape_h[ax_i]))
            else:  # along the direction: from a little above the end to max_extension_mm ahead
                back = int(np.ceil((UP_SLACK_MM + 3.0) / sp_h[ax_i]))
                if np.sign(sup[ax_i]) * sign > 0:
                    lo[ax_i] = max(int(band[:, ax_i].min()) - back, 0)
                else:
                    hi[ax_i] = min(int(band[:, ax_i].max()) + back + 1, int(shape_h[ax_i]))
        return lo, hi


def _lowest_tip(own: Any, h: Any) -> tuple[int, int, int] | None:
    """The skeleton leaf of *own* farthest along *h* (its open end), as host voxel coordinates;
    ``None`` when the skeleton has no leaf."""
    sk = _skeleton(own)
    if not bool(sk.any()):
        return None
    n_nb = ndi.convolve(sk.astype(np.int16), np.ones((3, 3, 3), dtype=np.int16), mode="constant") - 1
    leaves = sk & (n_nb == 1)
    if not bool(leaves.any()):
        return None
    hl = np.where(leaves, h, -np.inf)
    return tuple(int(v) for v in to_numpy(np.argwhere(hl == hl.max())[0]))  # type: ignore[return-value]


def _distance_to_point_mm(shape: Sequence[int], point: Sequence[int], sp: Spacing) -> Any:
    """Distance (mm) of every voxel of *shape* to voxel *point* (backend float64)."""
    d2 = np.zeros(tuple(int(n) for n in shape), dtype=np.float64)
    for ax in range(3):
        idx = (np.arange(int(shape[ax]), dtype=np.float64) - float(point[ax])) * float(sp[ax])
        bshape = [1, 1, 1]
        bshape[ax] = int(shape[ax])
        d2 += (idx * idx).reshape(bshape)
    return np.sqrt(d2)


#: How far (mm) along the floor an edge trunk's lumen is followed past where its route met the floor.
FLOOR_RUN_MM = 25.0


def _floor_lumen(new: Any, above: Any, h: Any, h_floor: float, region: Any, line: Any, sp: Spacing,
                 r_trunk: float) -> Any:
    """The lumen (*above* the local threshold) that runs on along the image's floor from an edge
    trunk's new voxels: the petrous ICA often lies along the lowest slices past the point where its
    route met the floor. Kept: the part of the bottom band (2 trunk radii + 1 mm) connected to
    *new* and within :data:`FLOOR_RUN_MM` of the route."""
    band = above & region & (h >= h_floor - (2.0 * r_trunk + 1.0))
    seed = new & band
    if not bool(seed.any()):
        return np.zeros_like(new)
    near = ndi.distance_transform_edt(~line, sampling=sp) <= FLOOR_RUN_MM
    cc, _ = ndi.label(band & near, structure=_cube())
    keep = np.unique(cc[seed])
    return np.isin(cc, keep[keep > 0]) & ~new


def extend_trunks(
    intensity: Any,
    labels: Any,
    ids: Sequence[int],
    *,
    affine: Any | None = None,
    spacing: Sequence[float] | None = None,
    direction: str = "inferior",
    frangi_sigmas_mm: Sequence[float] = FRANGI_SIGMAS_MM,
    max_extension_mm: float = MAX_EXTENSION_MM,
    alpha: float = ALPHA,
    sigma_mm: float = SIGMA_MM,
    shell_mm: float = SHELL_MM,
    split_ratio: float | dict[int, float] = SPLIT_RATIO,
    leak_factor: float = LEAK_FACTOR,
    seed_band_mm: float = 1.5,
    to_floor: bool = True,
    floor_branches: dict[int, int] | None = None,
    edge_ids: Sequence[int] = (),
) -> tuple[Any, dict[str, Any]]:
    """Extend each trunk in *ids* beyond its open end in *direction* (module docstring).

    *direction* is along the world superior axis (the affine's third row): ``inferior`` extends
    from each label's lowest end, as the ICAs and basilar need below eICAB's circle-of-Willis box.
    Every other label, and every route already added, is a barrier.

    With *to_floor* (default) a trunk goes on to the edge of the image in *direction*: the
    minimum-cost path through the Frangi tree from its end to the image's last slices, cheaper
    along bright tubes (cost 1-5 per mm by vesselness). Frangi fades at the volume's edge, so the
    last 3 mm accept any voxel above the trunk's gate. *floor_branches* (``{label: n}``, default 1)
    lets a trunk follow *n* pieces down, e.g. the basilar into both vertebrals. A trunk with no path
    to the floor falls back to following the tree as one tube; there *split_ratio* (per label
    allowed) sets what counts as a bifurcation. Returns the new label volume (int32, on the active
    backend) and a per-label log (length, voxels added, how it ended, and whether the lumen reached
    the floor).

    *edge_ids* (the ICAs) ignore *max_extension_mm*: their crop and route run to the image's edge
    however far it is (a route may be up to 2.5 times the straight drop, plus 20 mm, long)."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    from nvitk.segmentation.blood_flood import hysteresis_vessel_tree

    inten = as_backend_array(intensity).astype(np.float64, copy=False)
    out = as_backend_array(labels).astype(np.int32)
    sp = spacing_of(affine, spacing)
    mean_sp = sum(sp) / 3.0
    with using("cpu"):
        sup = to_numpy(affine).astype(float)[2, :3] if affine is not None else to_numpy((0.0, 0.0, 1.0))
    sign = -1.0 if direction == "inferior" else 1.0
    info: dict[str, Any] = {"direction": direction, "frangi_sigmas_mm": list(frangi_sigmas_mm),
                            "max_extension_mm": max_extension_mm, "to_floor": bool(to_floor), "labels": {}}
    t0 = time.time()
    for lid in [int(i) for i in ids]:
        own = out == lid
        ext = Extension(lid)
        if int(own.sum()) < 8:
            ext.stop = "label absent"
            info["labels"][str(lid)] = ext.to_dict()
            continue
        # 1. Open end and crop (host geometry on the label's voxel list)
        coords = to_numpy(np.argwhere(own))
        end, band = _end_band(coords, sup, sign, seed_band_mm)
        to_edge = lid in [int(i) for i in edge_ids]
        lo, hi = _extension_box(band, sup, sign, sp, out.shape, None if to_edge else max_extension_mm)
        with using("cpu"):
            band_centre = band.mean(0) - lo
        box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        img = inten[box]
        lab = out[box]
        own_c = lab == lid
        # 2. Heights, seeds and the trunk's own gate (backend)
        h = _linear_field(img.shape, [int(v) for v in lo], [sign * float(v) for v in sup])
        # Up to UP_SLACK_MM above the lowest voxel: the vessel may leave the label sideways.
        beyond = h > end - UP_SLACK_MM
        seeds = own_c & (h >= end - seed_band_mm)
        r_end = float(ndi.distance_transform_edt(own_c, sampling=sp)[seeds].max()) if bool(seeds.any()) else 1.0
        tip = _lowest_tip(own_c, h)
        if tip is not None:  # the open end, which on a bend is not the lowest band
            seeds |= own_c & (_distance_to_point_mm(img.shape, tip, sp) <= r_end + 1.0)
            ext.info["open_end_vox"] = [int(a) + int(b) for a, b in zip(tip, lo)]
        sk = _skeleton(own_c) & (h >= end - 5.0)
        ring = _dilate(own_c, RING_MM[1], sp) & ~_dilate(own_c, RING_MM[0], sp) & (lab == 0) & (h >= end - 5.0)
        core = float(np.median(img[sk])) if bool(sk.any()) else float(np.median(img[seeds]))
        bg = float(np.median(img[ring])) if bool(ring.any()) else float(np.median(img))
        gate = bg + 0.5 * alpha * (core - bg)
        # 3. Frangi tree beyond the end
        vess = frangi_vesselness(img, sp, frangi_sigmas_mm)
        region = beyond & (lab == 0)
        tree, _tmeta = hysteresis_vessel_tree(vess, region)
        smooth = denoise(img, sp)
        # Bright lumen (above the trunk's half maximum) is tree whatever its vesselness: Frangi
        # drops at junctions (the vertebrobasilar confluence) and in tight bends.
        lumen = region & (smooth > bg + 0.5 * (core - bg))
        tree = (as_backend_array(tree).astype(bool) & (img > gate) | lumen) & region
        ext.info.update(core=round(core, 1), background=round(bg, 1), gate=round(gate, 1),
                        tree_voxels=int(tree.sum()))
        # The image's floor: its lowest voxels with real signal. Resampling leaves a slice of
        # near-zero values below the first acquired one, which is no floor to reach.
        pos = img > 0
        inside = img > (0.05 * float(np.median(img[pos])) if bool(pos.any()) else 0.0)
        h_floor = float(h[inside].max()) if bool(inside.any()) else float(h.max())
        max_mm = float(max_extension_mm)
        if to_edge:  # the whole drop to the image's edge, with room for the bends
            max_mm = max(max_mm, 2.5 * max(h_floor - end, 0.0) + 20.0)
            ext.info["drop_to_floor_mm"] = round(max(h_floor - end, 0.0), 1)
        # 4. Route: minimum-cost paths to the floor, else the tube follower
        paths: list[Any] = []
        if to_floor:
            edge = region & inside & (h >= h_floor - 3.0) & (smooth > gate)
            allowed = tree | edge
            v_ref = float(np.percentile(vess[tree], 95)) if bool(tree.any()) else 1.0
            cost = 1.0 + 4.0 * (1.0 - np.clip(vess / max(v_ref, 1e-9), 0.0, 1.0))
            floor_band = allowed & inside & (h >= h_floor - 1.0)
            n_paths = int((floor_branches or {}).get(lid, 1))
            # Targets: vessel cross-sections at the floor; any allowed floor voxel if none is reached.
            targets = (floor_band & lumen, floor_band)
            if to_edge:  # the lowest slice itself first (its lumen is often dim at the slab's edge)
                last = floor_band & (h >= h_floor - 0.5 * min(sp))
                targets = (last & (smooth > gate), *targets)
            paths, pinfo = _paths_to_floor(allowed, seeds, cost, targets, sp,
                                           max_paths=n_paths, max_mm=max_mm,
                                           min_sep_mm=max(2.0 * r_end, 3.0))
            ext.info.update(pinfo)
            if paths:
                ext.stop = "floor of the image" + (f" ({len(paths)} branches)" if len(paths) > 1 else "")
        if paths:
            with using("cpu"):
                pts_list = [np.vstack([band_centre, p]) for p in paths]
                ext.centroids = [q + lo for q in pts_list[0]]
            ext.length_mm = max(_path_length_mm(p, sp) for p in pts_list)
            line = as_backend_array(functools.reduce(operator.or_, [_polyline_mask(list(p), img.shape)
                                                                    for p in pts_list]))
            ahead = np.zeros(img.shape, dtype=bool)
        else:
            seed_area = int(int(seeds.sum()) / max(1.0, seed_band_mm / mean_sp))
            ratio = float(split_ratio.get(lid, SPLIT_RATIO)) if isinstance(split_ratio, dict) else float(split_ratio)
            _route, cents, stop = _follow_tube(tree | seeds, seeds, sp, max_mm=max_mm, shell_mm=shell_mm,
                                               split_ratio=ratio, leak_factor=leak_factor, seed_area=seed_area)
            ext.info["split_ratio"] = ratio
            ext.stop = (stop + " (no path to the floor)") if to_floor else stop
            if len(cents) < 2:
                info["labels"][str(lid)] = ext.to_dict()
                continue
            pts = [band_centre] + cents
            with using("cpu"):
                ext.centroids = [q + lo for q in pts]
                tang = pts[-1] - pts[-2]
                tang = tang / (np.linalg.norm(tang) + 1e-9)
            ext.length_mm = _path_length_mm(to_numpy(pts), sp)
            line = as_backend_array(_polyline_mask(pts, img.shape))
            # No cap beyond the last centroid: cut the hood with the plane through it.
            ahead = _linear_field(img.shape, [-float(v) for v in pts[-1]], [float(v) for v in tang]) > 0.5
        # 5. Lumen around the route at the local half maximum (~2 trunk radii)
        d_line = ndi.distance_transform_edt(~line, sampling=sp)
        r_trunk = float(np.median(ndi.distance_transform_edt(own_c, sampling=sp)[sk])) if bool(sk.any()) else 2.0
        r_lim = max(2.0 * r_trunk, r_trunk + 1.0)
        ring_r = (d_line > r_lim) & (d_line <= r_lim + 2.0) & (lab == 0)
        thr, _ = local_threshold(img, line | sk, ring_r | ring, sp, alpha=alpha, sigma_mm=sigma_mm)
        hood = (d_line <= r_lim) & beyond & (lab == 0) & ~ahead
        # The route itself is lumen where it is above the trunk's gate: a stretch below the local
        # half maximum (a dim bend) must not cut off the rest of the route from the label.
        bridge = _dilate(line, 0.5 * r_trunk, sp) & hood & (smooth > gate)
        cand = (hood & (smooth > thr)) | own_c | bridge
        cc, _ = ndi.label(cand, structure=_cube())
        keep = np.unique(cc[own_c | bridge])
        new = np.isin(cc, keep[keep > 0]) & hood
        new = ndi.binary_fill_holes(new | own_c) & (hood | own_c) & ~own_c
        if to_edge and paths:
            new |= _floor_lumen(new, smooth > thr, h, h_floor, region, line, sp, r_trunk)
        lab[new & (lab == 0)] = lid  # *lab* is a view of *out*
        ext.n_voxels = int(new.sum())
        ext.info.update(trunk_radius_mm=round(r_trunk, 2), volume_added_mm3=round(ext.n_voxels * _voxel_mm3(sp), 1))
        # Did the lumen reach the floor? (the gap left, in mm along the extension direction)
        reached = np.where(lab == lid, h, -np.inf)
        gap = max(h_floor - float(reached.max()), 0.0) if bool((lab == lid).any()) else float("nan")
        ext.info.update(floor_gap_mm=round(gap, 1), reached_floor=bool(gap <= 1.5 * max(sp)))
        info["labels"][str(lid)] = ext.to_dict()
        log.step(f"[label {lid}] extended {ext.length_mm:.1f} mm {direction} ({ext.n_voxels} voxels): {ext.stop}")
    info["seconds"] = round(time.time() - t0, 2)
    return out, info


# Distal growth

#: Distal growth defaults (:func:`grow_distal`).
DISTAL_SIGMAS_MM = (0.5, 1.0, 1.5, 2.0)
DISTAL_MAX_MM = 60.0
#: How far (mm) a hemisphere-paired label (the ACAs) may grow past the mid-sagittal plane.
MIDLINE_SLACK_MM = 5.0


def _compete(
    tree: Any, lab: Any, ids: Sequence[int], step_cost: Any, sp: Spacing, max_cost: float,
    forbid: dict[int, Any] | None = None, extra_seeds: dict[int, Any] | None = None,
) -> tuple[Any, Any]:
    """Grow every label in *ids* into *tree* by weighted geodesic distance at once; each voxel goes
    to the label that reaches it first, up to *max_cost*. *extra_seeds* (``{label: mask}``) start a
    label's front from voxels it does not own as well. Returns ``(owner, cost)`` on the active
    backend (scikit-image ``MCP_Geometric`` runs on the host)."""
    from skimage.graph import MCP_Geometric

    tree_h, lab_h, cost_h = to_numpy(tree), to_numpy(lab), to_numpy(step_cost)
    forbid_h = {int(k): to_numpy(v) for k, v in (forbid or {}).items()}
    extra_h = {int(k): to_numpy(v) for k, v in (extra_seeds or {}).items()}
    with using("cpu"):
        best = np.full(lab_h.shape, np.inf)
        owner = np.zeros(lab_h.shape, dtype=np.int32)
        for lid in ids:
            own = lab_h == lid
            if lid in extra_h:
                own = own | extra_h[lid]
            ok = tree_h & ~forbid_h[lid] if lid in forbid_h else tree_h
            costs = np.where(own, 1.0, np.where(ok, cost_h, np.inf))
            mcp = MCP_Geometric(costs, fully_connected=True, sampling=tuple(float(x) for x in sp))
            cum, _ = mcp.find_costs([tuple(int(v) for v in c) for c in np.argwhere(own)])
            better = ok & (cum < best) & (cum <= max_cost)
            best[better] = cum[better]
            owner[better] = lid
    return as_backend_array(owner), as_backend_array(best)


def head_midplane(
    labels: Any, refs: Sequence[tuple[int, int]], sp: Spacing
) -> tuple[Any, Any] | None:
    """``(point, normal)`` of the head's mid-sagittal plane from paired left / right labels (*refs*,
    e.g. the ICAs, MCAs, PCAs): through the mean of each pair's midpoint, normal the mean of their
    left-minus-right directions (mm, unit, pointing to the left). Host vectors; ``None`` without a
    present pair."""
    lab = as_backend_array(labels)
    mids, dirs = [], []
    for left, right in refs:
        a, b = to_numpy(np.argwhere(lab == left)), to_numpy(np.argwhere(lab == right))
        if len(a) < 5 or len(b) < 5:
            continue
        with using("cpu"):
            ca, cb = a.mean(0), b.mean(0)
            d = (ca - cb) * to_numpy(sp)
            if np.linalg.norm(d) < 1e-6:
                continue
            mids.append(0.5 * (ca + cb))
            dirs.append(d / np.linalg.norm(d))
    if not dirs:
        return None
    with using("cpu"):
        n = np.mean(dirs, axis=0)
        if np.linalg.norm(n) < 1e-6:
            return None
        return np.mean(mids, axis=0), n / np.linalg.norm(n)


def head_midplane_from_image(intensity: Any, sp: Spacing, *, lr_axis: int = 0) -> tuple[Any, Any] | None:
    """``(point, normal)`` of the head's mid-sagittal plane from the image: for every row across
    the left-right index axis *lr_axis*, the midpoint between the head's two edges (the head mask
    is the image above 10 % of its 99th percentile, closed, largest component), then a
    least-squares plane through those midpoints. The normal (mm, unit) points to +*lr_axis*. The
    head mask runs on the backend; the plane fit on the host (host vectors returned)."""
    img = as_backend_array(intensity).astype(np.float64, copy=False)
    if img.ndim != 3:
        return None
    pos = img > 0
    thr = 0.1 * float(np.percentile(img[pos], 99)) if bool(pos.any()) else 0.0
    head = ndi.binary_closing(img > thr, structure=_cube(), iterations=2, brute_force=True)
    lab, n = ndi.label(head)
    if int(n) == 0:
        return None
    head = lab == (int(np.argmax(np.bincount(lab.ravel())[1:])) + 1)
    hm = np.moveaxis(head, lr_axis, 0)  # (lr, a, b)
    any_ = hm.any(0)
    if int(any_.sum()) < 50:
        return None
    first = to_numpy(np.argmax(hm, axis=0)).astype(float)
    last = to_numpy(hm.shape[0] - 1 - np.argmax(hm[::-1], axis=0)).astype(float)
    any_h = to_numpy(any_)
    with using("cpu"):
        a_idx, b_idx = np.nonzero(any_h & ((last - first) > 20))  # rows that cross a real width of head
        mid = 0.5 * (first + last)[a_idx, b_idx]
        A = np.c_[np.ones_like(mid), a_idx.astype(float), b_idx.astype(float)]
        coef, *_ = np.linalg.lstsq(A, mid, rcond=None)
        for _ in range(2):  # drop outlying rows (ears, shoulders), refit
            res = mid - A @ coef
            keep = np.abs(res) <= 3.0 * max(float(np.std(res)), 1e-6)
            coef, *_ = np.linalg.lstsq(A[keep], mid[keep], rcond=None)
        other = [ax for ax in range(3) if ax != lr_axis]
        # Plane lr = c0 + c1 a + c2 b  ->  normal ∝ (1, -c1, -c2) in index units, then to mm.
        n_idx = np.zeros(3)
        n_idx[lr_axis] = 1.0
        n_idx[other[0]] = -coef[1]
        n_idx[other[1]] = -coef[2]
        normal = n_idx / to_numpy(sp)  # gradient in mm of (lr - plane(a, b)) per index step / spacing
        normal = normal / np.linalg.norm(normal)
        p0 = np.zeros(3)
        ca, cb = float(np.mean(a_idx)), float(np.mean(b_idx))
        p0[other[0]], p0[other[1]] = ca, cb
        p0[lr_axis] = coef[0] + coef[1] * ca + coef[2] * cb
        return p0, normal


def grow_distal(
    intensity: Any,
    labels: Any,
    ids: Sequence[int],
    *,
    affine: Any | None = None,
    spacing: Sequence[float] | None = None,
    frangi_sigmas_mm: Sequence[float] = DISTAL_SIGMAS_MM,
    max_mm: float = DISTAL_MAX_MM,
    alpha: float = ALPHA,
    sigma_mm: float = SIGMA_MM,
    hemisphere_pairs: Sequence[tuple[int, int, int | None]] = (),
    midline_refs: Sequence[tuple[int, int]] = (),
    midline_slack_mm: float = MIDLINE_SLACK_MM,
    relabel: dict[int, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Grow the labels in *ids* along the multi-scale Frangi tree beyond their own voxels, all at
    once and in competition, as the blood flood floods its markers. *relabel* (``{label: new}``)
    writes what a label gains under another id, e.g. an ACA's growth as its A2.

    The tree is a GMM hysteresis tree of the Frangi vesselness (σ *frangi_sigmas_mm*) on a crop
    around the labels, gated at 0.15 of the way from the background to the labels' centreline
    intensity. Every label grows into unlabelled tree voxels by geodesic distance, weighted 1-5
    per mm by vesselness so the fronts keep to vessel centres. Each voxel goes to the label that
    reaches it first (the fronts advance together, step for step), up to *max_mm*. Labels not in
    *ids* are barriers.

    The competition is what keeps two neighbours running side by side apart, such as the A2s of
    the two ACAs: each front advances along its own lumen, so each A2 is reached first by its own
    ACA. *hemisphere_pairs* (``(left, right, comm)``, e.g. the ACAs and the AComm) add two things.
    The communicating label does not block the pair, and each side of it starts that side's front,
    since an A2 often begins at the AComm rather than at its ACA's own voxels. And a guard against
    invading the other hemisphere: neither may grow more than *midline_slack_mm* past a midline
    plane:
    - oriented as the head's mid-sagittal plane (:func:`head_midplane_from_image`; from the paired
      labels *midline_refs* if that fails);
    - anchored at the communicating label (else midway between the pair), because the ACA
      complex often sits a few mm off the head's geometric midline.
    The A2s, which hug the midline, are thus left to the competition. The lumen of what each label
    gained is drawn at its local half maximum.
    Returns the new label volume (int32, on the active backend) and a per-label log (voxels and
    mm³ added, farthest reach)."""
    from nvitk.segmentation.blood_flood import hysteresis_vessel_tree

    inten = as_backend_array(intensity).astype(np.float64, copy=False)
    out = as_backend_array(labels).astype(np.int32)
    sp = spacing_of(affine, spacing)
    ids = [int(i) for i in ids if int((out == int(i)).sum()) >= 8]
    info: dict[str, Any] = {"frangi_sigmas_mm": list(frangi_sigmas_mm), "max_mm": max_mm, "labels": {}}
    if not ids:
        return out, info
    t0 = time.time()
    # 1. Crop, centreline intensity and gate
    box = _bbox(_isin(out, ids), _vox(max_mm + 5.0, sp))
    img = inten[box]
    lab = out[box]  # a view: the lumen written below lands in *out*
    smooth = denoise(img, sp)
    inside = img > 0
    mine = _isin(lab, ids)
    sk_all = np.zeros(lab.shape, dtype=bool)
    for lid in ids:
        sub = _bbox(lab == lid, (2, 2, 2))
        sk_all[sub] |= _skeleton(lab[sub] == lid)
    core = float(np.median(smooth[sk_all])) if bool(sk_all.any()) else float(np.percentile(smooth[inside], 99))
    bg_mask = inside & (lab == 0)
    bg = float(np.median(img[bg_mask])) if bool(bg_mask.any()) else 0.0
    gate = bg + 0.15 * (core - bg)
    # 2. Frangi tree, other labels as barriers
    vess = frangi_vesselness(img, sp, frangi_sigmas_mm)
    comm_ids = [int(c) for _l, _r, c in hemisphere_pairs if c is not None]
    others = (lab != 0) & ~mine & ~_isin(lab, comm_ids)  # a pair's communicating label does not block it
    barrier = ndi.binary_dilation(others, structure=_cube())
    region = inside & (lab == 0) & ~barrier
    tree, meta = hysteresis_vessel_tree(vess, region)
    # Bright lumen (above the vessels' half maximum, at least weakly tubular) belongs to the tree
    # whatever its vesselness: a saturated core has little Hessian response.
    bright = (smooth > bg + 0.5 * (core - bg)) & (vess > float(meta.get("lowt", 0.0)))
    tree = (as_backend_array(tree).astype(bool) | bright) & region & (smooth > gate)
    v_ref = float(np.percentile(vess[tree], 95)) if bool(tree.any()) else 1.0
    step_cost = (1.0 + 4.0 * (1.0 - np.clip(vess / max(v_ref, 1e-9), 0.0, 1.0))).astype(np.float64)
    del vess
    info.update(core=round(core, 1), background=round(bg, 1), gate=round(gate, 1), tree_voxels=int(tree.sum()))
    # 3. Hemisphere guard: past the midline plane (+ slack), a paired label may not grow
    forbid: dict[int, Any] = {}
    extra: dict[int, Any] = {}
    origin = [int(b.start) for b in box]
    if hemisphere_pairs:
        plane = head_midplane_from_image(inten, sp) or head_midplane(out, midline_refs, sp)
        if plane is not None:
            _p_head, n = plane
            info["midplanes"] = []
            for left, right, comm in hemisphere_pairs:
                a = to_numpy(np.argwhere(lab == int(left)))
                b = to_numpy(np.argwhere(lab == int(right)))
                if len(a) < 5 or len(b) < 5:
                    continue
                cc = to_numpy(np.argwhere(lab == int(comm))) if comm is not None else None
                with using("cpu"):
                    p0 = cc.mean(0) if cc is not None and len(cc) >= 3 else 0.5 * (a.mean(0) + b.mean(0))
                    left_dir = float(((a.mean(0) - b.mean(0)) * to_numpy(sp)) @ n)
                    nn = n if left_dir > 0 else -n  # point to *left*
                signed = _linear_field(lab.shape, [-float(v) for v in p0], [float(s) * float(v) for s, v in zip(sp, nn)])
                forbid[int(left)] = signed < -float(midline_slack_mm)
                forbid[int(right)] = signed > float(midline_slack_mm)
                if comm is not None:  # each side of the communicating label starts that side's front
                    cm = lab == int(comm)
                    extra[int(left)] = cm & (signed >= 0)
                    extra[int(right)] = cm & (signed < 0)
                info["midplanes"].append({"left": int(left), "right": int(right),
                                          "point_vox": [round(float(v) + o, 1) for v, o in zip(p0, origin)],
                                          "normal": [round(float(v), 3) for v in nn],
                                          "slack_mm": float(midline_slack_mm)})
    # 4. Competition (~1 per mm on a centreline, so 2 × max_mm)
    owner, best = _compete(tree, lab, ids, step_cost, sp, 2.0 * max_mm, forbid, extra)
    # 5. The lumen of what each label gained, at its local half maximum
    for lid in ids:
        own = lab == lid
        grown = owner == lid
        entry: dict[str, Any] = {"tree_voxels": int(grown.sum())}
        if not bool(grown.any()):
            entry["voxels_added"] = 0
            info["labels"][str(lid)] = entry
            continue
        sub = _bbox(grown | own, _vox(RING_MM[1] + 3 * sigma_mm, sp, extra=1))
        g_s, o_s, l_s, ow_s = grown[sub], own[sub], lab[sub], owner[sub]
        centre = _skeleton(g_s | o_s)
        hood = _dilate(g_s, 1.0, sp) & (l_s == 0) & ((ow_s == lid) | (ow_s == 0)) & ~barrier[sub]
        ring = _dilate(g_s, RING_MM[1], sp) & ~_dilate(g_s, RING_MM[0], sp) & (l_s == 0) & (ow_s == 0)
        thr, _ = local_threshold(img[sub], centre, ring, sp, alpha=alpha, sigma_mm=sigma_mm)
        cand = (hood & (smooth[sub] > thr)) | o_s
        cc, _ = ndi.label(cand, structure=_cube())
        keep = np.unique(cc[o_s])
        new = np.isin(cc, keep[keep > 0]) & hood & ~o_s
        l_s[new & (l_s == 0)] = int((relabel or {}).get(lid, lid))  # a view of *lab*, itself a view of *out*
        reach = float(np.max(best[grown]))
        n_new = int(new.sum())
        entry.update(voxels_added=n_new, volume_added_mm3=round(n_new * _voxel_mm3(sp), 1),
                     reach_cost=round(reach, 1))
        if relabel and lid in relabel:
            entry["written_as"] = int(relabel[lid])
        info["labels"][str(lid)] = entry
        log.step(f"[label {lid}] grew {entry['volume_added_mm3']} mm³ along the Frangi tree")
    info["seconds"] = round(time.time() - t0, 2)
    return out, info


__all__ = [
    "ALPHA",
    "DIRECTIONS",
    "Extension",
    "FRANGI_SIGMAS_MM",
    "MAX_EXTENSION_MM",
    "extend_trunks",
    "tidy_labels",
    "frangi_vesselness",
    "grow_distal",
    "head_midplane",
    "head_midplane_from_image",
    "local_threshold",
    "resegment_local",
    "spacing_of",
]
