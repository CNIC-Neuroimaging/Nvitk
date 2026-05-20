"""Topological siphon correction for ICA-like vessel centerlines.

Problem
-------
The cavernous siphon of the Internal Carotid Artery (ICA) often closes into a
donut on TOF MRA: two adjacent passes of the vessel merge into a single mask
because of partial-volume effects, turning the topological hook
(:math:`\\beta_1 = 0`) into a torus (:math:`\\beta_1 \\ge 1`). A standard
skeleton-diameter centerline then **shortcuts across the false bridge** instead
of following the true curl.

Strategy (purely topological — no orientation prior)
----------------------------------------------------
For each vessel label requested via ``correction_ids`` we:

1. Skeletonize the binary mask.
2. Build a 26-connected ``networkx`` graph of the skeleton voxels.
3. Find every cycle (``minimum_cycle_basis``).
4. Split each cycle into **two arcs** at *anchors*:

   - degree-3+ junctions on the cycle (real branch points where stem / tip
     attach to the loop), or
   - one junction + the max-Z cycle voxel, or
   - the min-Z and max-Z cycle voxels (pure floating loop, no junctions).

5. Drop the **shorter arc** from the skeleton — the bridge is by anatomical
   definition a short chord across the donut hole; the curl is the long way
   around. The original mask is **not** modified.
6. Trace the centerline as the unique shortest path between the **min-Z leaf**
   (skull-base entry) and the **max-Z leaf** (ICA bifurcation tip) of the
   pruned skeleton.

Outputs
-------
:func:`correct_siphon_centerlines` produces, in the input TOF image space:

- ``corrected_centerlines.nii.gz`` — per-label centerline mask (siphon-
  corrected for ``correction_ids``, default ``compute_centerlines`` for the
  rest).
- ``removed_bridges.nii.gz`` — per-label bridge-voxel mask (same label IDs as
  ``correction_ids``).
- ``vessel_mask_corrected.nii.gz`` — full multilabel mask with ICAs replaced by
  the Otsu+erode (+ optional donut-cut) lumen mask (same label IDs).
- ``seg_ica_repaired.nii.gz`` — ICA-only multilabel volume (corrected labels only).
- ``siphon_correction.json`` — per-label metadata (cycles, arc lengths,
  endpoints, warnings).
- ``qc_siphon_correction.png`` — 3D matplotlib QC figure (when
  ``save_qc=True``).
- ``qc_ica_overview.png`` — axial Otsu → erode → repaired+CL montage (when
  ``save_qc=True``).

At the end of each run a notebook-style summary table (voxels, β₁, cycles,
centerline length, repair action) is printed to stdout.

Backend
-------
The module is registered for backend-proxy switching (``np`` / ``scipy`` /
``ndi`` track ``nvitk.core.backend`` between NumPy and CuPy). CPU-only
libraries (``scikit-image`` skeleton, ``networkx``, ``marching_cubes``,
``matplotlib``) are fed via :func:`to_numpy`; their results are routed back to
the active backend via :func:`as_backend_array`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from nvitk.core.array import as_backend_array, to_numpy
from nvitk.core.backend import setup, using
from nvitk.core.exceptions import ValidationError
from nvitk.core.logger import Logger
from nvitk.morphology.centerline import compute_centerlines, skeletonize_binary
from nvitk.morphology.components import (
    keep_components_touching_seeds,
    label_connected,
    remove_small_components_by_fraction,
)
from nvitk.types import Image

try:
    from nvitk.pipes.bbtpy.labels import bb_vessel_name
except ImportError:

    def bb_vessel_name(label_id: int) -> str:
        return f"label_{int(label_id)}"

setup(globals())

log = Logger()

CROP_PAD = 7
MIN_COMPONENT_FRAC = 0.005
CL_BARRIER_RADIUS = 5
EROSION_ITERS = 2
MAX_REPAIR_ITERS = 3
CUT_RADII = (1, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GenusReport:
    """Topology summary for a 3D binary mask.

    For a tube-like vessel we assume :math:`\\beta_2 \\approx 0` (no internal
    cavities), so :math:`\\beta_1 = \\beta_0 - \\chi`. ``suspect`` flips True
    iff :math:`\\beta_1 > 0` (a topological handle / donut).
    """

    label_name: str
    n_voxels: int
    n_components: int
    euler_chi: float
    beta0: int
    beta1: int
    skeleton_cycles: int
    skeleton_voxels: int

    @property
    def suspect(self) -> bool:
        return self.beta1 > 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["suspect"] = bool(self.suspect)
        return d


@dataclass
class SiphonCorrectionResult:
    """Per-label outcome of :func:`correct_siphon_centerlines`."""

    label: int
    label_name: str
    n_skel: int = 0
    n_skel_pruned: int = 0
    n_bridge: int = 0
    n_pts: int = 0
    base: tuple[int, int, int] | None = None
    tip: tuple[int, int, int] | None = None
    cycles: list[dict] = field(default_factory=list)
    bridge_voxels: list[tuple[int, int, int]] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict:
        return {
            "label": int(self.label),
            "label_name": self.label_name,
            "n_skel": int(self.n_skel),
            "n_skel_pruned": int(self.n_skel_pruned),
            "n_bridge": int(self.n_bridge),
            "n_pts": int(self.n_pts),
            "base": list(self.base) if self.base is not None else None,
            "tip": list(self.tip) if self.tip is not None else None,
            "cycles": list(self.cycles),
            "bridge_voxels": [list(v) for v in self.bridge_voxels],
            "warning": self.warning,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Genus check — helper
# ──────────────────────────────────────────────────────────────────────────────


def compute_mask_genus(
    mask: Any,
    *,
    label_name: str = "vessel",
    connectivity: int = 1,
) -> GenusReport:
    """Return :class:`GenusReport` for *mask* (3D bool / int, NumPy or CuPy).

    The mask is moved to NumPy internally (``skimage`` / ``networkx`` are CPU
    only). The returned report is JSON-serialisable via ``report.to_dict()``.

    Parameters
    ----------
    mask
        3D mask; non-zero voxels are foreground.
    label_name
        Free-form name recorded in the report (e.g. ``"LICA"``).
    connectivity
        ``ndi``-style voxel connectivity for the CC count
        (1 = 6-conn, 2 = 18-conn, 3 = 26-conn).
    """
    from skimage.measure import euler_number

    import networkx as nx

    m = to_numpy(mask).astype(bool, copy=False)
    n_vox = int(m.sum())
    if n_vox == 0:
        return GenusReport(label_name, 0, 0, 0.0, 0, 0, 0, 0)
    labeled, n_cc = label_connected(m, connectivity=connectivity)
    labeled_np = to_numpy(labeled)
    chi_sum = 0.0
    for cid in range(1, int(n_cc) + 1):
        chi_sum += float(euler_number(labeled_np == cid, connectivity=connectivity))
    beta0 = int(n_cc)
    beta1 = max(0, int(beta0 - int(round(chi_sum))))
    sk = to_numpy(skeletonize_binary(m)).astype(bool, copy=False)
    sk_vox = int(sk.sum())
    if sk_vox < 3:
        cyc = 0
    else:
        G = _skeleton_to_graph(sk)
        cyc = max(
            0,
            int(
                G.number_of_edges()
                - G.number_of_nodes()
                + nx.number_connected_components(G)
            ),
        )
    return GenusReport(
        label_name=label_name,
        n_voxels=n_vox,
        n_components=beta0,
        euler_chi=float(chi_sum),
        beta0=beta0,
        beta1=int(beta1),
        skeleton_cycles=int(cyc),
        skeleton_voxels=int(sk_vox),
    )


# ──────────────────────────────────────────────────────────────────────────────
# ICA mask preparation (Otsu + erode + optional donut cut)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class RepairLog:
    """Per-ICA donut-repair summary."""

    label_name: str
    action: str = "none"
    n_voxels_before: int = 0    
    n_voxels_after: int = 0
    before: GenusReport | None = None
    after: GenusReport | None = None
    iters: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "label_name": self.label_name,
            "action": self.action,
            "n_voxels_before": int(self.n_voxels_before),
            "n_voxels_after": int(self.n_voxels_after),
            "before": self.before.to_dict() if self.before else None,
            "after": self.after.to_dict() if self.after else None,
            "iters": list(self.iters),
            "notes": list(self.notes),
        }


def _bbox_with_padding(
    roi: Any,
    full_shape: tuple[int, int, int],
    pad: int,
) -> tuple[int, int, int, int, int, int] | None:
    """Tight bbox of *roi* voxels padded by *pad*."""
    roi_np = to_numpy(roi)
    xs, ys, zs = np.nonzero(roi_np)
    if xs.size == 0:
        return None
    nx_, ny_, nz_ = full_shape
    p = max(0, int(pad))
    return (
        max(0, int(xs.min()) - p),
        min(nx_ - 1, int(xs.max()) + p),
        max(0, int(ys.min()) - p),
        ny_ - 1,
        max(0, int(zs.min()) - p),
        min(nz_ - 1, int(zs.max()) + p),
    )


def _other_cl_barrier(
    cl_mask: Any,
    bbox: tuple[int, int, int, int, int, int],
    lid: int,
    radius: int,
) -> Any:
    """Voxels in the crop too close to other labels' seed centerlines."""
    i0, i1, j0, j1, k0, k1 = bbox
    cl_np = to_numpy(cl_mask)
    other = (cl_np != 0) & (cl_np != int(lid))
    if int(radius) > 0 and np.any(other):
        from scipy import ndimage as ndi_cpu

        other = ndi_cpu.binary_dilation(other, iterations=int(radius))
    return other[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]


def ica_otsu_mask(
    wvi: Any,
    cl_mask: Any,
    lid: int,
    *,
    pad: int = CROP_PAD,
    min_cc_frac: float = MIN_COMPONENT_FRAC,
    barrier_r: int = CL_BARRIER_RADIUS,
    erode_iters: int = EROSION_ITERS,
) -> tuple[Any, Any, dict]:
    """Per-ICA local Otsu on TOF inside the seed-CL bbox (+ erosion).

    Returns eroded mask (for genus check / repair), pre-erode mask, and an info dict.
    """
    from skimage.filters import threshold_otsu

    name = bb_vessel_name(int(lid))
    wvi_np = to_numpy(wvi)
    shape = tuple(int(s) for s in wvi_np.shape[:3])
    full_post = np.zeros(shape, dtype=bool)
    full_pre = np.zeros(shape, dtype=bool)

    cl_np = to_numpy(cl_mask)
    seed = cl_np == int(lid)
    bbox = _bbox_with_padding(seed, shape, pad)
    if bbox is None:
        return (
            as_backend_array(full_post),
            as_backend_array(full_pre),
            {"label": name, "warning": "no seed centerline", "n_voxels": 0},
        )

    i0, i1, j0, j1, k0, k1 = bbox
    crop = wvi_np[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1].astype(np.float64)
    pos = crop[crop > 0]
    if pos.size < 16:
        return (
            as_backend_array(full_post),
            as_backend_array(full_pre),
            {
                "label": name,
                "warning": "too few positive voxels",
                "bbox": bbox,
                "n_voxels": 0,
            },
        )

    t = float(threshold_otsu(pos))
    mask_crop = crop > t
    if min_cc_frac > 0 and mask_crop.any():
        mask_crop = to_numpy(
            remove_small_components_by_fraction(
                mask_crop, min_fraction=float(min_cc_frac), connectivity=1
            )
        ).astype(bool)

    forbidden = to_numpy(_other_cl_barrier(cl_mask, bbox, int(lid), barrier_r))
    mask_crop = mask_crop & (~forbidden)

    seed_crop = seed[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1]
    if mask_crop.any() and seed_crop.any():
        mask_crop = to_numpy(
            keep_components_touching_seeds(mask_crop, seed_crop, connectivity=1)
        ).astype(bool)

    mask_pre_erode = mask_crop.copy()
    full_pre[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1] = mask_pre_erode

    if int(erode_iters) > 0 and mask_crop.any():
        from scipy import ndimage as ndi_cpu

        mask_crop = ndi_cpu.binary_erosion(mask_crop, iterations=int(erode_iters))
        if mask_crop.any() and seed_crop.any():
            mask_crop = to_numpy(
                keep_components_touching_seeds(mask_crop, seed_crop, connectivity=1)
            ).astype(bool)
        if min_cc_frac > 0 and mask_crop.any():
            mask_crop = to_numpy(
                remove_small_components_by_fraction(
                    mask_crop, min_fraction=float(min_cc_frac), connectivity=1
                )
            ).astype(bool)

    full_post[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1] = mask_crop
    return (
        as_backend_array(full_post),
        as_backend_array(full_pre),
        {
            "label": name,
            "otsu_thresh": t,
            "bbox": bbox,
            "n_voxels": int(mask_crop.sum()),
            "n_voxels_pre_erode": int(mask_pre_erode.sum()),
            "erode_iters": int(erode_iters),
        },
    )


def _ball_offsets(radius: int) -> np.ndarray:
    r = int(radius)
    if r <= 0:
        return np.zeros((1, 3), dtype=np.int32)
    rr = np.arange(-r, r + 1, dtype=np.int32)
    gx, gy, gz = np.meshgrid(rr, rr, rr, indexing="ij")
    sel = (gx * gx + gy * gy + gz * gz) <= (r * r)
    return np.stack([gx[sel], gy[sel], gz[sel]], axis=1).astype(np.int32)


def _smallest_cycle(G: Any) -> list[tuple[int, int, int]] | None:
    import networkx as nx

    if G.number_of_edges() == 0:
        return None
    try:
        cycles = list(nx.minimum_cycle_basis(G))
    except Exception:
        cycles = []
    if not cycles:
        try:
            cycles = list(nx.cycle_basis(G))
        except Exception:
            cycles = []
    if not cycles:
        return None
    cycles.sort(key=len)
    return [tuple(int(v) for v in node) for node in cycles[0]]


def repair_ica_donut_3d(
    mask_bool: Any,
    seed_bool: Any,
    *,
    label_name: str,
    ckpt_dir: Path | None = None,
) -> tuple[Any, RepairLog]:
    """Optional 3D donut cut on an eroded ICA mask.

    When ``ckpt_dir`` is ``None`` no checkpoint files are written.
    """
    rlog = RepairLog(label_name=label_name)
    mask = to_numpy(mask_bool).astype(bool, copy=False)
    seed = to_numpy(seed_bool).astype(bool, copy=False)
    rlog.n_voxels_before = int(mask.sum())
    rlog.before = compute_mask_genus(mask, label_name=label_name)

    if not rlog.before.suspect:
        rlog.action = "skipped"
        rlog.after = rlog.before
        rlog.n_voxels_after = rlog.n_voxels_before
        log.step(f"[{label_name}] genus 0 → skipping repair")
        return as_backend_array(mask), rlog

    for it in range(MAX_REPAIR_ITERS):
        sk = to_numpy(skeletonize_binary(mask)).astype(bool)
        G = _skeleton_to_graph(sk)
        cycle = _smallest_cycle(G)
        if cycle is None:
            log.step(f"[{label_name}] iter {it}: no skeleton cycle → stop")
            rlog.notes.append(f"iter {it}: no cycle")
            break

        cycle_arr = np.array(cycle, dtype=np.int32)
        log.step(
            f"[{label_name}] iter {it}: cycle len={len(cycle)} "
            f"x[{cycle_arr[:, 0].min()},{cycle_arr[:, 0].max()}] "
            f"y[{cycle_arr[:, 1].min()},{cycle_arr[:, 1].max()}] "
            f"z[{cycle_arr[:, 2].min()},{cycle_arr[:, 2].max()}]"
        )

        top_y = int(cycle_arr[:, 1].max())
        top_band = cycle_arr[cycle_arr[:, 1] == top_y]
        from scipy import ndimage as ndi_cpu

        dt = ndi_cpu.distance_transform_edt(mask)
        if top_band.shape[0] > 1:
            dt_top = dt[top_band[:, 0], top_band[:, 1], top_band[:, 2]]
            anchor = tuple(int(v) for v in top_band[int(np.argmin(dt_top))])
            anchor_dt = float(dt_top.min())
        else:
            anchor = tuple(int(v) for v in top_band[0])
            anchor_dt = float(dt[anchor])
        log.step(
            f"[{label_name}] iter {it}: anchor={anchor} top_y={top_y} dt={anchor_dt:.2f}"
        )

        beta1_pre = (
            int(rlog.before.beta1)
            if it == 0
            else int(compute_mask_genus(mask, label_name=label_name).beta1)
        )
        mask_try = mask
        rep_try = compute_mask_genus(mask, label_name=label_name)
        best_score: tuple[int, int] | None = None
        chosen_radius = int(CUT_RADII[0])
        for r in CUT_RADII:
            offs = _ball_offsets(int(r)) + np.array(anchor, dtype=np.int32)
            valid = (
                (offs[:, 0] >= 0)
                & (offs[:, 0] < mask.shape[0])
                & (offs[:, 1] >= 0)
                & (offs[:, 1] < mask.shape[1])
                & (offs[:, 2] >= 0)
                & (offs[:, 2] < mask.shape[2])
            )
            cuts = offs[valid]
            candidate = mask.copy()
            candidate[cuts[:, 0], cuts[:, 1], cuts[:, 2]] = False
            if seed.any():
                candidate = to_numpy(
                    keep_components_touching_seeds(candidate, seed, connectivity=1)
                ).astype(bool)
            if MIN_COMPONENT_FRAC > 0 and candidate.any():
                candidate = to_numpy(
                    remove_small_components_by_fraction(
                        candidate,
                        min_fraction=MIN_COMPONENT_FRAC,
                        connectivity=1,
                    )
                ).astype(bool)
            rep_candidate = compute_mask_genus(candidate, label_name=label_name)
            log.step(
                f"[{label_name}] iter {it} r={r}: vox={rep_candidate.n_voxels} "
                f"β₁={rep_candidate.beta1} cycles={rep_candidate.skeleton_cycles}"
            )
            score = (int(rep_candidate.beta1), -int(rep_candidate.n_voxels))
            if best_score is None or score < best_score:
                best_score = score
                mask_try = candidate
                rep_try = rep_candidate
                chosen_radius = int(r)
            if rep_candidate.beta1 == 0:
                break

        if rep_try.beta1 >= beta1_pre:
            log.warning(
                f"[{label_name}] iter {it}: best candidate β₁={rep_try.beta1} "
                f"≥ pre-cut β₁={beta1_pre} → REJECTING the cut for this iter"
            )
            rlog.iters.append(
                {
                    "iter": int(it),
                    "anchor": list(anchor),
                    "radius": int(chosen_radius),
                    "accepted": False,
                    "rejected_reason": "no β₁ reduction",
                }
            )
            break

        mask = mask_try
        rlog.iters.append(
            {
                "iter": int(it),
                "anchor": list(anchor),
                "top_y": int(top_y),
                "cycle_len": int(len(cycle)),
                "anchor_dt": float(anchor_dt),
                "radius": int(chosen_radius),
                "voxels_after": int(mask.sum()),
                "beta1_after": int(rep_try.beta1),
                "accepted": bool(rep_try.beta1 == 0),
            }
        )
        if rep_try.beta1 == 0:
            log.ok(f"[{label_name}] iter {it}: genus cleared with r={chosen_radius}")
            break

    rlog.after = compute_mask_genus(mask, label_name=label_name)
    rlog.n_voxels_after = int(mask.sum())
    rlog.action = "repaired" if not rlog.after.suspect else "partial"
    log.step(f"[{label_name}] repair done: action={rlog.action}")
    return as_backend_array(mask), rlog


def _prepare_ica_mask_for_centerline(
    wvi: Any,
    cl_mask: Any,
    lid: int,
    *,
    ckpt_dir: Path | None = None,
) -> tuple[Any, dict]:
    """ICA path: Otsu+erode → optional donut repair → mask for CL."""
    name = bb_vessel_name(int(lid))
    log.step(f"--- {name} (id={lid}) ---")
    t0 = time.time()
    eroded_mask, otsu_mask, otsu_info = ica_otsu_mask(wvi, cl_mask, int(lid))
    log.step(
        f"[{name}] Otsu+erode in {time.time() - t0:.2f}s: "
        f"thr={otsu_info.get('otsu_thresh')} "
        f"vox_pre={otsu_info.get('n_voxels_pre_erode')} → "
        f"vox_post={otsu_info.get('n_voxels')} "
        f"(erode_iters={otsu_info.get('erode_iters')}) "
        f"bbox={otsu_info.get('bbox')}"
    )
    if otsu_info.get("warning"):
        log.warning(f"[{name}] Otsu skipped: {otsu_info['warning']}")
        empty = np.zeros(to_numpy(wvi).shape[:3], dtype=bool)
        return as_backend_array(empty), {
            "otsu_info": otsu_info,
            "otsu_mask": empty,
            "eroded_mask": empty,
            "repair": None,
        }

    otsu_report = compute_mask_genus(otsu_mask, label_name=name)
    eroded_report = compute_mask_genus(eroded_mask, label_name=name)
    log.step(
        f"[{name}] eroded (iters={otsu_info.get('erode_iters', 0)}): "
        f"voxels={eroded_report.n_voxels} β₀={eroded_report.beta0} "
        f"χ={eroded_report.euler_chi:.1f} β₁={eroded_report.beta1} "
        f"skel_cycles={eroded_report.skeleton_cycles} suspect={eroded_report.suspect}"
    )

    cl_np = to_numpy(cl_mask)
    seed_full = cl_np == int(lid)
    if eroded_report.suspect:
        log.step(f"[{name}] eroded mask still suspect → running 3D donut cut")
        ica_ckpt = (ckpt_dir / name) if ckpt_dir is not None else None
        repaired, rlog = repair_ica_donut_3d(
            eroded_mask,
            seed_full,
            label_name=name,
            ckpt_dir=ica_ckpt,
        )
        log.step(f"[{name}] repair action={rlog.action}")
    else:
        repaired = eroded_mask
        rlog = RepairLog(
            label_name=name,
            action="skipped (erosion alone cleared)",
            before=eroded_report,
            after=eroded_report,
            n_voxels_before=int(to_numpy(eroded_mask).sum()),
            n_voxels_after=int(to_numpy(eroded_mask).sum()),
        )
        log.ok(f"[{name}] erosion alone cleared topology → no cut needed")

    rep_report = compute_mask_genus(repaired, label_name=name)
    log.step(
        f"[{name}] repaired: voxels={rep_report.n_voxels} β₁={rep_report.beta1} "
        f"skel_cycles={rep_report.skeleton_cycles} suspect={rep_report.suspect}"
    )
    return repaired, {
        "otsu_info": otsu_info,
        "otsu_mask": to_numpy(otsu_mask).astype(bool, copy=False),
        "eroded_mask": to_numpy(eroded_mask).astype(bool, copy=False),
        "otsu_report": otsu_report.to_dict(),
        "eroded_report": eroded_report.to_dict(),
        "repair": rlog.to_dict(),
        "repaired_report": rep_report.to_dict(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Skeleton-graph + cycle utilities (CPU only)
# ──────────────────────────────────────────────────────────────────────────────


_NEI26 = np.array(
    [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ],
    dtype=np.int32,
)


def _skeleton_to_graph(sk: Any) -> Any:
    """Build a 26-connected ``networkx.Graph`` over the skeleton voxels.

    *sk* may be NumPy or CuPy; it is materialised on CPU first. Nodes are
    ``(i, j, k)`` int tuples; node attributes ``x, y, z`` carry the integer
    coordinates (e.g. for GraphML export).
    """
    import networkx as nx

    sk_np = to_numpy(sk).astype(bool, copy=False)
    G = nx.Graph()
    coords = np.argwhere(sk_np)
    nodes = [tuple(int(v) for v in row) for row in coords]
    node_set = set(nodes)
    for n in nodes:
        G.add_node(n, x=int(n[0]), y=int(n[1]), z=int(n[2]))
    shape = sk_np.shape
    for n in nodes:
        for d in _NEI26:
            v = (int(n[0] + d[0]), int(n[1] + d[1]), int(n[2] + d[2]))
            if not (
                0 <= v[0] < shape[0] and 0 <= v[1] < shape[1] and 0 <= v[2] < shape[2]
            ):
                continue
            if v in node_set and (v > n):
                G.add_edge(n, v)
    return G


def _walk_cycle(
    G: Any, cycle_nodes: list[tuple[int, int, int]]
) -> list[tuple[int, int, int]]:
    """Walk the cycle subgraph so consecutive returned nodes form edges.

    ``networkx``'s ``minimum_cycle_basis`` does not contractually guarantee
    cyclic ordering; this helper reconstructs it. Falls back to the input
    order if a chord prevents a clean walk.
    """
    n = len(cycle_nodes)
    if n < 2:
        return list(cycle_nodes)
    cycle_set = set(cycle_nodes)
    adj = {v: [u for u in G.neighbors(v) if u in cycle_set] for v in cycle_nodes}
    start = cycle_nodes[0]
    if not adj[start]:
        return list(cycle_nodes)
    order: list[tuple[int, int, int]] = [start]
    prev: tuple[int, int, int] | None = None
    curr = start
    while True:
        nxt = None
        for nb in adj[curr]:
            if nb != prev:
                nxt = nb
                break
        if nxt is None or nxt == start:
            break
        order.append(nxt)
        prev, curr = curr, nxt
        if len(order) > n:
            break
    if len(order) != n:
        return list(cycle_nodes)
    return order


def _split_cycle_into_arcs(
    G: Any, cycle: list[tuple[int, int, int]]
) -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
]:
    """Split *cycle* into ``(bridge_arc, curl_arc, anchors)``.

    Anchors are the two cycle nodes we want to **keep** (they terminate the
    curl arc). Selection priority:

    1. **Degree-3+ junctions** on the cycle (real branch points where stem /
       tip attach). If > 2 such junctions exist, pick the pair with the
       largest Z-spread.
    2. **One junction + virtual** — single junction + max-Z cycle voxel.
    3. **No junctions** (pure floating loop) — min-Z and max-Z cycle voxels.

    The cycle then splits into two arcs in cyclic order at the anchors. The
    bridge is the **shorter** arc; the curl is the longer one.
    """
    ordered = _walk_cycle(G, cycle)
    cyc_arr = np.array(ordered, dtype=np.int32)
    junctions = [n for n in ordered if G.degree(n) > 2]
    if len(junctions) >= 2:
        if len(junctions) > 2:
            j_arr = np.array(junctions, dtype=np.int32)
            anchors = [
                junctions[int(np.argmin(j_arr[:, 2]))],
                junctions[int(np.argmax(j_arr[:, 2]))],
            ]
        else:
            anchors = list(junctions)
    elif len(junctions) == 1:
        j = junctions[0]
        others = [n for n in ordered if n != j]
        if not others:
            return [], list(ordered), [j]
        oa = np.array(others, dtype=np.int32)
        virtual = others[int(np.argmax(oa[:, 2]))]
        anchors = [j, virtual]
    else:
        zmin = ordered[int(np.argmin(cyc_arr[:, 2]))]
        zmax = ordered[int(np.argmax(cyc_arr[:, 2]))]
        anchors = [zmin, zmax]
    a_idx = ordered.index(anchors[0])
    b_idx = ordered.index(anchors[1])
    if a_idx == b_idx:
        return [], list(ordered), list(anchors)
    if a_idx > b_idx:
        a_idx, b_idx = b_idx, a_idx
    arc1 = ordered[a_idx + 1 : b_idx]
    arc2 = ordered[b_idx + 1 :] + ordered[: a_idx]
    bridge = arc1 if len(arc1) <= len(arc2) else arc2
    curl = arc2 if bridge is arc1 else arc1
    return bridge, curl, list(anchors)


# ──────────────────────────────────────────────────────────────────────────────
# Public primitives
# ──────────────────────────────────────────────────────────────────────────────


def prune_skeleton_shortest_arc(
    mask: Any,
    *,
    label_name: str = "vessel",
) -> tuple[Any, list[tuple[int, int, int]], dict]:
    """Skeleton with each cycle's *shortest* arc removed (the false bridge).

    Operates on a NumPy or CuPy mask; the returned skeleton is moved back to
    the active backend via :func:`as_backend_array`.

    Returns
    -------
    sk_pruned : array (current backend, bool)
        Skeleton with bridge voxels deleted (mask itself is **not** modified).
    bridge_voxels : list[(i, j, k)]
        Deleted voxel coordinates (union across cycles).
    info : dict
        Per-cycle arc lengths, anchors, Y/Z ranges.
    """
    import networkx as nx

    m = to_numpy(mask).astype(bool, copy=False)
    sk_np = to_numpy(skeletonize_binary(m)).astype(bool, copy=False)
    info: dict = {"n_skel": int(sk_np.sum()), "cycles": []}
    if not sk_np.any():
        return as_backend_array(sk_np), [], info
    G = _skeleton_to_graph(sk_np)
    cycles: list[list[tuple[int, int, int]]] = []
    if G.number_of_edges() > 0:
        try:
            cycles = [
                [tuple(int(v) for v in n) for n in c]
                for c in nx.minimum_cycle_basis(G)
            ]
        except Exception:
            cycles = []
        if not cycles:
            try:
                cycles = [
                    [tuple(int(v) for v in n) for n in c]
                    for c in nx.cycle_basis(G)
                ]
            except Exception:
                cycles = []
    if not cycles:
        log.step(f"[{label_name}] no skeleton cycles → no pruning needed")
        return as_backend_array(sk_np), [], info

    sk_pruned = sk_np.copy()
    bridge_voxels: list[tuple[int, int, int]] = []
    for ci, cycle in enumerate(cycles):
        bridge, curl, anchors = _split_cycle_into_arcs(G, cycle)
        cyc_arr = np.array(cycle, dtype=np.int32)
        br_y = br_z = None
        if bridge:
            br_arr = np.array(bridge, dtype=np.int32)
            br_y = [int(br_arr[:, 1].min()), int(br_arr[:, 1].max())]
            br_z = [int(br_arr[:, 2].min()), int(br_arr[:, 2].max())]
        for v in bridge:
            sk_pruned[v[0], v[1], v[2]] = False
            bridge_voxels.append(v)
        info["cycles"].append(
            {
                "idx": int(ci),
                "len": int(len(cycle)),
                "bridge_len": int(len(bridge)),
                "curl_len": int(len(curl)),
                "anchors": [list(a) for a in anchors],
                "bridge_y_range": br_y,
                "bridge_z_range": br_z,
                "cycle_y_range": [
                    int(cyc_arr[:, 1].min()),
                    int(cyc_arr[:, 1].max()),
                ],
                "cycle_z_range": [
                    int(cyc_arr[:, 2].min()),
                    int(cyc_arr[:, 2].max()),
                ],
            }
        )
        log.step(
            f"[{label_name}] cycle #{ci}: len={len(cycle)} "
            f"anchors={[list(a) for a in anchors]} "
            f"bridge={len(bridge)} curl={len(curl)} → pruned shorter arc"
        )
    return as_backend_array(sk_pruned), bridge_voxels, info


def compute_corrected_centerline(
    mask: Any,
    *,
    label_name: str = "vessel",
) -> tuple[Any, Any, dict]:
    """Corrected centerline as ``min-Z leaf → max-Z leaf`` shortest path.

    1. :func:`prune_skeleton_shortest_arc` removes false bridges.
    2. ``base = min-Z degree-1 leaf``, ``tip = max-Z degree-1 leaf`` in the
       pruned skeleton (fallback: extremal-Z over all nodes if no leaf).
    3. ``networkx.shortest_path(G, base, tip)`` is unique on a tree, so the
       centerline always terminates at the vessel's anatomical extremes
       (skull-base entry and ICA bifurcation tip).

    Returns
    -------
    path : array (current backend, float32, shape (N, 3))
        Ordered voxel coords (i, j, k), base → tip.
    sk_pruned : array (current backend, bool)
        Skeleton with bridge voxels removed.
    info : dict
        Pruning + endpoint metadata.
    """
    import networkx as nx

    mask_b = to_numpy(mask).astype(bool, copy=False)
    sk_pruned, bridge_voxels, prune_info = prune_skeleton_shortest_arc(
        mask_b, label_name=label_name
    )
    info: dict = {
        "prune": prune_info,
        "bridge_voxels": [list(v) for v in bridge_voxels],
    }
    sk_np = to_numpy(sk_pruned).astype(bool, copy=False)
    if not sk_np.any():
        info["error"] = "empty pruned skeleton"
        empty = np.empty((0, 3), dtype=np.float32)
        return as_backend_array(empty), as_backend_array(sk_np), info

    G = _skeleton_to_graph(sk_np)
    leaves = [n for n in G.nodes() if G.degree(n) == 1]
    info["n_nodes"] = int(G.number_of_nodes())
    info["n_edges"] = int(G.number_of_edges())
    info["n_leaves"] = int(len(leaves))

    candidates = leaves if leaves else list(G.nodes())
    if not candidates:
        info["error"] = "no candidate endpoints"
        empty = np.empty((0, 3), dtype=np.float32)
        return as_backend_array(empty), as_backend_array(sk_np), info

    cand_arr = np.array(candidates, dtype=np.int32)
    base = candidates[int(np.argmin(cand_arr[:, 2]))]
    tip = candidates[int(np.argmax(cand_arr[:, 2]))]
    info["base"] = list(base)
    info["tip"] = list(tip)
    log.step(
        f"[{label_name}] endpoints: base(min-Z)={base} tip(max-Z)={tip} "
        f"({len(leaves)} leaves / {G.number_of_nodes()} nodes)"
    )
    if base == tip:
        info["error"] = "base == tip"
        empty = np.empty((0, 3), dtype=np.float32)
        return as_backend_array(empty), as_backend_array(sk_np), info
    if not nx.has_path(G, base, tip):
        info["warning"] = "base/tip in different CCs; using largest-CC Z-extremes"
        ccs = sorted(nx.connected_components(G), key=len, reverse=True)
        sub = G.subgraph(ccs[0]).copy()
        sub_nodes = list(sub.nodes())
        sub_arr = np.array(sub_nodes, dtype=np.int32)
        base = sub_nodes[int(np.argmin(sub_arr[:, 2]))]
        tip = sub_nodes[int(np.argmax(sub_arr[:, 2]))]
        info["base"] = list(base)
        info["tip"] = list(tip)
        if not nx.has_path(sub, base, tip):
            info["error"] = "no path even within largest CC"
            empty = np.empty((0, 3), dtype=np.float32)
            return as_backend_array(empty), as_backend_array(sk_np), info
        path = nx.shortest_path(sub, base, tip)
    else:
        path = nx.shortest_path(G, source=base, target=tip)
    info["n_pts"] = int(len(path))
    path_arr = np.array(path, dtype=np.float32)
    return as_backend_array(path_arr), as_backend_array(sk_np), info


# ──────────────────────────────────────────────────────────────────────────────
# Public top-level functions
# ──────────────────────────────────────────────────────────────────────────────


def correct_siphon_centerlines(
    tof: Any,
    vessel_mask: Any,
    *,
    correction_ids: Sequence[int] = (1, 2),
    out_dir: str | Path | None = None,
    save_qc: bool = False,
    min_points: int = 3,
) -> dict[str, Any]:
    """
    End-to-end siphon centerline correction (matches ``eicab_reseg.ipynb``).

    For each label in *correction_ids*, the notebook's pipeline:

    1. Seed centerlines from input multilabel *vessel_mask*.
    2. Per-ICA local Otsu on TOF inside the seed-CL bbox (+2-iter erosion).
    3. Optional 3D donut cut when β₁ > 0 after erosion.
    4. Topological bridge prune + ``min-Z → max-Z`` centerline on the
       **repaired Otsu mask** (not the raw vessel label).

    Labels outside *correction_ids* keep the default
    :func:`~nvitk.morphology.centerline.compute_centerlines` on *vessel_mask*.
    When *out_dir* is set, a merged ``vessel_mask_corrected.nii.gz`` replaces
    each corrected ICA with its final Otsu lumen mask (eroded, or donut-cut
    when needed); non-ICA labels are copied unchanged from *vessel_mask*.

    Parameters
    ----------
    tof
        TOF MRA (``Image``, path, or array); drives Otsu. Spatial metadata (affine, zooms, axes)
        is copied to all NIfTI outputs.
    vessel_mask
        Multilabel vessel mask on the same grid as *tof*.
    correction_ids
        Label IDs to siphon-correct (default ``(1, 2)`` = RICA/LICA).
    out_dir
        If set, writes centerlines, bridges, corrected vessel/ICA masks,
        ``siphon_correction.json``, and optionally QC figures.
    save_qc
        Save 3D QC and ICA overview figures; always prints the summary table.
    min_points
        Minimum centerline length (voxels) for inclusion.

    Returns
    -------
    dict
        ``centerlines``, ``bridges``, ``details``, rasterised masks, and
        ``output_paths``.
    """
    # Migrate to CPU backend 
    # TBA: (No GPU processing implemented). 
    with using('cpu'):
        # Load and verify input images; check dimensions before proceeding.
        tof_img = _to_image(tof, name="tof")
        mask_img = _to_image(vessel_mask, name="vessel_mask")
        wvi = as_backend_array(to_numpy(tof_img.data).astype(np.float32))
        mask_data = as_backend_array(mask_img.data)
        shape = tuple(int(s) for s in mask_data.shape[:3])
        if tuple(int(s) for s in wvi.shape[:3]) != shape:
            raise ValidationError(
                f"tof shape {tuple(wvi.shape[:3])} != "
                f"vessel_mask shape {tuple(mask_data.shape[:3])}"
            )
        correction_ids = tuple(int(v) for v in correction_ids)
        log.step(
            f"correct_siphon_centerlines: shape={shape} correction_ids={correction_ids}"
        )

        # Determine all present labels and which ones will/won't be corrected.
        all_labels = sorted(
            int(v) for v in np.unique(to_numpy(mask_data)) if int(v) > 0
        )
        corr_present = [lid for lid in correction_ids if lid in all_labels]
        non_corr = [lid for lid in all_labels if lid not in correction_ids]
        log.step(
            f"labels present: {all_labels} | siphon-corrected: {corr_present} | "
            f"default: {non_corr}"
        )

        # Seed centerlines determine bounds for Otsu per-ICA cropping.
        log.step("=== Seed centerlines from vessel_mask ===")
        t_seed = time.time()
        seed_cls = compute_centerlines(
            mask_data, labels=all_labels, min_points=int(min_points)
        )
        cl_seed_mask = _rasterize_centerlines_mask(shape, seed_cls)
        log.step(
            f"seed centerlines done in {time.time() - t_seed:.2f}s "
            f"({len(seed_cls)} labels)"
        )

        centerlines: dict[int, Any] = {}
        # Default centerlines for any label not needing ICA/siphon correction
        if non_corr:
            log.step("=== Default centerlines for non-ICA labels ===")
            centerlines.update(
                compute_centerlines(
                    mask_data, labels=non_corr, min_points=int(min_points)
                )
            )

        details: dict[int, dict] = {}
        bridges_by_label: dict[int, list[tuple[int, int, int]]] = {}
        pruned_skeletons: dict[int, Any] = {}
        repaired_masks: dict[int, Any] = {}
        uncorrected_cls: dict[int, Any] = {}
        seg_ica_otsu = np.zeros(shape, dtype=np.int32)
        seg_ica_eroded = np.zeros(shape, dtype=np.int32)
        seg_ica_repaired = np.zeros(shape, dtype=np.int32)

        out_path = Path(out_dir) if out_dir is not None else None
        ckpt_root = out_path if save_qc else None

        log.step("=== ICA Otsu + repair + siphon centerlines ===")
        for lid in correction_ids:
            name = bb_vessel_name(int(lid))
            if int(lid) not in all_labels:
                log.warning(f"[{name}] label {lid} not in mask — skipping")
                continue

            # Prepare ICA region: Otsu, erosion, repair and save intermediate masks.
            prep_info: dict = {}
            repaired_mask, prep_info = _prepare_ica_mask_for_centerline(
                wvi,
                cl_seed_mask,
                int(lid),
                ckpt_dir=ckpt_root,
            )
            repaired_masks[int(lid)] = repaired_mask
            otsu_m = prep_info.get("otsu_mask")
            eroded_m = prep_info.get("eroded_mask")
            if otsu_m is not None and bool(np.asarray(otsu_m).any()):
                seg_ica_otsu[np.asarray(otsu_m, dtype=bool)] = int(lid)
            if eroded_m is not None and bool(np.asarray(eroded_m).any()):
                seg_ica_eroded[np.asarray(eroded_m, dtype=bool)] = int(lid)
            rep_np = to_numpy(repaired_mask).astype(bool, copy=False)
            if rep_np.any():
                seg_ica_repaired[rep_np] = int(lid)

            # Optionally compute unpruned centerline on repaired mask for QC
            if save_qc:
                try:
                    uncorr = compute_centerlines(
                        to_numpy(repaired_mask).astype(np.int32),
                        labels=[1],
                        min_points=int(min_points),
                    )
                    uncorrected_cls[int(lid)] = uncorr.get(1)
                except Exception as exc:
                    log.warning(f"[{name}] uncorrected CL on repaired mask failed: {exc}")

            # Siphon-corrected centerline extraction: prune bridges and extract min-Z→max-Z path.
            t3 = time.time()
            path, sk_pruned, cl_info = compute_corrected_centerline(
                repaired_mask, label_name=name
            )
            pruned_skeletons[int(lid)] = sk_pruned
            path_np = to_numpy(path)
            bridge_vox = [
                tuple(int(v) for v in c) for c in cl_info.get("bridge_voxels", [])
            ]
            bridges_by_label[int(lid)] = bridge_vox
            log.step(
                f"[{name}] centerline in {time.time() - t3:.2f}s pts={path_np.shape[0]}"
            )
            if path_np.shape[0] >= int(min_points):
                centerlines[int(lid)] = path
            else:
                log.warning(f"[{name}] centerline too short ({path_np.shape[0]} pts)")

            sk_full = int(to_numpy(skeletonize_binary(repaired_mask)).sum())
            sk_pruned_n = int(to_numpy(sk_pruned).sum())
            res = SiphonCorrectionResult(
                label=int(lid),
                label_name=name,
                n_skel=sk_full,
                n_skel_pruned=sk_pruned_n,
                n_bridge=len(bridge_vox),
                n_pts=int(path_np.shape[0]),
                base=tuple(int(v) for v in cl_info["base"]) if cl_info.get("base") else None,
                tip=tuple(int(v) for v in cl_info["tip"]) if cl_info.get("tip") else None,
                cycles=cl_info.get("prune", {}).get("cycles", []),
                bridge_voxels=bridge_vox,
                warning=cl_info.get("warning") or cl_info.get("error"),
            )
            log.step(
                f"[{name}] skeleton pruning: {sk_full} → {sk_pruned_n} voxels "
                f"({len(bridge_vox)} bridge voxel(s) removed); "
                f"endpoints base={res.base} tip={res.tip} (leaves={cl_info.get('n_leaves')})"
            )
            details[int(lid)] = {**res.to_dict(), "prep": prep_info}

        cl_mask = _rasterize_centerlines_mask(shape, centerlines)
        bridges_mask = _rasterize_bridges_mask(shape, bridges_by_label)
        vessel_mask_corrected = _merge_ica_into_vessel_mask(
            mask_data, repaired_masks, correction_ids
        )

        out_paths: dict[str, str] = {}
        # I/O: Only if output directory given.
        if out_path is not None:
            import nvitk.io as io

            out_path.mkdir(parents=True, exist_ok=True)
            cl_path = out_path / "corrected_centerlines.nii.gz"
            br_path = out_path / "removed_bridges.nii.gz"
            vm_path = out_path / "vessel_mask_corrected.nii.gz"
            ica_path = out_path / "seg_ica_repaired.nii.gz"
            io.imsave(cl_path, cl_mask, metadata=tof.metadata, axes="XYZ")
            io.imsave(br_path, bridges_mask, metadata=tof.metadata, axes="XYZ")
            io.imsave(
                vm_path,
                vessel_mask_corrected,
                metadata=tof.metadata,
                axes="XYZ",
            )
            io.imsave(
                ica_path,
                seg_ica_repaired,
                metadata=tof.metadata,
                axes="XYZ",
            )
            out_paths["centerlines"] = str(cl_path)
            out_paths["bridges"] = str(br_path)
            out_paths["vessel_mask"] = str(vm_path)
            out_paths["seg_ica_repaired"] = str(ica_path)
            log.ok(
                f"wrote {cl_path.name} + {br_path.name} + "
                f"{vm_path.name} + {ica_path.name} (TOF spatial metadata)"
            )

            meta_path = out_path / "siphon_correction.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "correction_ids": list(correction_ids),
                        "shape": list(shape),
                        "details": {str(lid): details[lid] for lid in details},
                    },
                    indent=2,
                    default=_jsonable_default,
                ),
                encoding="utf-8",
            )
            out_paths["meta"] = str(meta_path)

            if save_qc:
                # Save main QC figure overlaying masks/centerlines/bridges/etc.
                qc_path = out_path / "qc_siphon_correction.png"
                try:
                    _save_qc_figure(
                        qc_path,
                        mask_by_label=repaired_masks,
                        correction_ids=correction_ids,
                        centerlines=centerlines,
                        pruned_skeletons=pruned_skeletons,
                        bridges_by_label=bridges_by_label,
                        uncorrected_centerlines=uncorrected_cls,
                    )
                    out_paths["qc"] = str(qc_path)
                    log.ok(f"wrote QC figure {qc_path.name}")
                except Exception as exc:
                    log.warning(f"QC figure generation failed: {exc}")

                # Save ICA overview figure.
                try:
                    ov_path = out_path / "qc_ica_overview.png"
                    _save_ica_overview_figure(
                        ov_path,
                        wvi=wvi,
                        seg_otsu=seg_ica_otsu,
                        seg_eroded=seg_ica_eroded,
                        seg_repaired=seg_ica_repaired,
                        correction_ids=correction_ids,
                        centerlines=centerlines,
                        shape=shape,
                    )
                    out_paths["qc_overview"] = str(ov_path)
                    log.ok(f"wrote ICA overview {ov_path.name}")
                except Exception as exc:
                    log.warning(f"ICA overview figure failed: {exc}")

        # (Always) print ICA summary table - see @stage3_centerline.py (142)
        _print_ica_summary_table(correction_ids, details, centerlines)

        return {
            "centerlines": {int(k): as_backend_array(v) for k, v in centerlines.items()},
            "bridges": {int(k): list(v) for k, v in bridges_by_label.items()},
            "details": details,
            "corrected_centerlines_mask": as_backend_array(cl_mask),
            "removed_bridges_mask": as_backend_array(bridges_mask),
            "vessel_mask_corrected": as_backend_array(vessel_mask_corrected),
            "seg_ica_repaired": as_backend_array(seg_ica_repaired),
            "output_paths": out_paths,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _jsonable_default(o: Any) -> Any:
    if isinstance(o, (np.integer, np.floating)):
        return int(o) if isinstance(o, np.integer) else float(o)
    if isinstance(o, np.ndarray):
        return to_numpy(o).tolist()
    return str(o)


def _spatial_metadata_from_image(
    ref: Image,
    *,
    fallback: Image | None = None,
) -> dict[str, Any]:
    """Copy affine / zooms / axes from a reference ``Image`` (notebook ``ref=tof``)."""
    for img in (ref, fallback):
        if img is None:
            continue
        meta = dict(img.metadata or {})
        affine = meta.get("affine")
        if affine is None:
            continue
        out: dict[str, Any] = {"affine": to_numpy(affine)}
        axes = img.axes or meta.get("axes")
        if axes is not None:
            out["axes"] = axes
        for key in (
            "orientation",
            "x_res",
            "y_res",
            "z_res",
            "t_res",
            "temporal_resolution",
        ):
            if key in meta and meta[key] is not None:
                out[key] = meta[key]
        return out
    raise ValidationError(
        "TOF reference has no 'affine' in metadata. Load TOF with "
        "nv.imread(path) so nibabel populates affine/zooms before calling "
        "correct_siphon_centerlines."
    )


def _to_image(obj: Any, *, name: str) -> Image:
    """Coerce path / array / :class:`Image` to an :class:`Image` (with metadata)."""
    if isinstance(obj, Image):
        return obj
    if isinstance(obj, (str, Path)):
        from nvitk.io.imageio import imread  

        loaded = imread(str(obj))
        if isinstance(loaded, list):
            raise ValidationError(
                f"{name}: path '{obj}' resolved to multiple series."
            )
        return loaded
    return Image(data=obj, metadata={})


def _merge_ica_into_vessel_mask(
    vessel_mask: Any,
    repaired_masks: dict[int, Any],
    correction_ids: Sequence[int],
) -> np.ndarray:
    """Replace ICA labels in *vessel_mask* with per-ICA Otsu lumen masks.

    Skips labels with an empty repaired mask (Otsu failure) and leaves the
    original *vessel_mask* voxels for that ICA unchanged.
    """
    merged = to_numpy(vessel_mask).astype(np.int32, copy=True)
    for lid in correction_ids:
        rep = repaired_masks.get(int(lid))
        if rep is None:
            continue
        rep_np = to_numpy(rep).astype(bool, copy=False)
        if not rep_np.any():
            continue
        merged[merged == int(lid)] = 0
        merged[rep_np] = int(lid)
    return merged


def _rasterize_centerlines_mask(
    shape: tuple[int, int, int], centerlines: dict[int, Any]
) -> Any:
    """Per-label voxel mask with vessel id on each centerline point (CPU NumPy)."""
    mask = np.zeros(shape, dtype=np.int32)
    for vid, pts in sorted(centerlines.items()):
        p = to_numpy(pts)
        if p.size == 0:
            continue
        ii = np.rint(p[:, 0]).astype(np.int32)
        jj = np.rint(p[:, 1]).astype(np.int32)
        kk = np.rint(p[:, 2]).astype(np.int32)
        keep = (
            (ii >= 0) & (ii < shape[0])
            & (jj >= 0) & (jj < shape[1])
            & (kk >= 0) & (kk < shape[2])
        )
        mask[ii[keep], jj[keep], kk[keep]] = int(vid)
    return as_backend_array(mask)


def _rasterize_bridges_mask(
    shape: tuple[int, int, int],
    bridges_by_label: dict[int, list[tuple[int, int, int]]],
) -> Any:
    """Per-label voxel mask of removed-bridge voxels (CPU NumPy)."""
    mask = np.zeros(shape, dtype=np.int32)
    for vid, voxels in bridges_by_label.items():
        for v in voxels:
            i, j, k = int(v[0]), int(v[1]), int(v[2])
            if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
                mask[i, j, k] = int(vid)
    return as_backend_array(mask)


def _print_ica_summary_table(
    correction_ids: Sequence[int],
    details: dict[int, dict],
    centerlines: dict[int, Any],
) -> str:
    """Print notebook Cell 9 summary (voxels, genus, cycles, CL length, action)."""
    lines: list[str] = [
        "",
        "=" * 110,
        (
            f"{'ICA':6s} {'vox_o':>7s} {'vox_e':>7s} {'vox_r':>7s} "
            f"{'β₁ o→e→r':>10s} {'cyc o→e→r':>12s} "
            f"{'CL_pts':>7s} {'action':>32s}"
        ),
        "-" * 110,
    ]
    for lid in correction_ids:
        name = bb_vessel_name(int(lid))
        info = details.get(int(lid), {})
        prep = info.get("prep") or {}
        otsu_info = prep.get("otsu_info") or {}
        warning = otsu_info.get("warning") or info.get("warning")
        if not prep.get("repair") or warning:
            lines.append(
                f"{name:6s} (skipped: {warning or 'missing'})"
            )
            continue
        o = prep["otsu_report"]
        e = prep["eroded_report"]
        a = prep["repaired_report"]
        n_o = int(otsu_info.get("n_voxels_pre_erode", o["n_voxels"]))
        n_e = int(otsu_info["n_voxels"])
        n_r = int(a["n_voxels"])
        cl = centerlines.get(int(lid))
        n_cl = int(to_numpy(cl).shape[0]) if cl is not None else 0
        action = prep["repair"]["action"]
        b1_str = f"{o['beta1']}→{e['beta1']}→{a['beta1']}"
        cyc_str = (
            f"{o['skeleton_cycles']}→{e['skeleton_cycles']}→{a['skeleton_cycles']}"
        )
        lines.append(
            f"{name:6s} {n_o:7d} {n_e:7d} {n_r:7d} "
            f"{b1_str:>10s} {cyc_str:>12s} {n_cl:7d} {action:>32s}"
        )
    lines.append("=" * 110)
    text = "\n".join(lines)
    print(text)
    return text


def _save_ica_overview_figure(
    out_path: Path,
    *,
    wvi: Any,
    seg_otsu: Any,
    seg_eroded: Any,
    seg_repaired: Any,
    correction_ids: Sequence[int],
    centerlines: dict[int, Any],
    shape: tuple[int, int, int],
) -> None:
    """Axial montage: Otsu → eroded → repaired mask + centerline per ICA."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    wvi_np = to_numpy(wvi)
    seg_o = to_numpy(seg_otsu)
    seg_e = to_numpy(seg_eroded)
    seg_r = to_numpy(seg_repaired)
    ids = [int(lid) for lid in correction_ids]
    ncols = max(1, len(ids))
    fig, axes = plt.subplots(3, ncols, figsize=(6 * ncols, 13))
    if ncols == 1:
        axes = np.array([[axes[0]], [axes[1]], [axes[2]]])

    for col, lid in enumerate(ids):
        name = bb_vessel_name(lid)
        coords_o = np.argwhere(seg_o == lid)
        coords_r = np.argwhere(seg_r == lid)
        if coords_o.shape[0] > 0:
            k = int(coords_o[:, 2].mean().round())
        elif coords_r.shape[0] > 0:
            k = int(coords_r[:, 2].mean().round())
        else:
            k = shape[2] // 2

        for row, (seg, label) in enumerate(
            [
                (seg_o, "Otsu (pre-erode)"),
                (seg_e, f"eroded ({EROSION_ITERS} iters)"),
                (seg_r, "repaired + CL"),
            ]
        ):
            axes[row, col].imshow(wvi_np[:, :, k].T, cmap="gray", origin="lower")
            axes[row, col].imshow(
                np.ma.masked_where(seg[:, :, k] != lid, seg[:, :, k]).T,
                alpha=0.5,
                cmap="cool",
                origin="lower",
            )
            axes[row, col].set_title(f"{name} — {label} (z={k})")
            axes[row, col].axis("off")

        path = centerlines.get(lid)
        if path is not None:
            path_np = to_numpy(path)
            if path_np.shape[0] > 0:
                near = np.abs(path_np[:, 2] - k) <= 1.5
                if near.any():
                    axes[2, col].plot(
                        path_np[near, 0], path_np[near, 1], "r.", ms=3
                    )

    plt.suptitle(
        "ICA: Otsu → erode (2 iters) → 3D donut cut → recomputed centerline"
    )
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


def _save_qc_figure(
    out_path: Path,
    *,
    mask_by_label: dict[int, Any],
    correction_ids: Sequence[int],
    centerlines: dict[int, Any],
    pruned_skeletons: dict[int, Any],
    bridges_by_label: dict[int, list[tuple[int, int, int]]],
    uncorrected_centerlines: dict[int, Any],
) -> None:
    """Render the 3D matplotlib QC figure (offline; headless-safe)."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    n = max(1, len(correction_ids))
    fig = plt.figure(figsize=(7 * n, 8))
    for col, lid in enumerate(correction_ids):
        name = bb_vessel_name(int(lid))
        ax = fig.add_subplot(1, n, col + 1, projection="3d")
        ax.set_title(
            f"{name}: surface (light blue) + skeleton (green) + centerline (red)"
        )

        repaired = mask_by_label.get(int(lid))
        if repaired is None:
            ax.set_title(f"{name}: missing repaired mask")
            continue
        roi = to_numpy(repaired).astype(bool, copy=False)
        if not roi.any():
            ax.set_title(f"label {int(lid)}: empty mask")
            continue

        # 1) Surface — downsampled mask voxels (no marching-cubes dependency for QC)
        surf = np.argwhere(roi)
        if surf.shape[0] > 8000:
            idx = np.random.default_rng(0).choice(
                surf.shape[0], 8000, replace=False
            )
            surf = surf[idx]
        ax.scatter(
            surf[:, 0], surf[:, 1], surf[:, 2],
            s=1, c="lightblue", alpha=0.25, label="surface",
        )

        # 2) Pruned skeleton
        sk_pruned = pruned_skeletons.get(int(lid))
        if sk_pruned is not None:
            sk_np = to_numpy(sk_pruned).astype(bool, copy=False)
            sk_coords = np.argwhere(sk_np)
            if sk_coords.shape[0] > 0:
                ax.scatter(
                    sk_coords[:, 0], sk_coords[:, 1], sk_coords[:, 2],
                    s=12, c="green", label="pruned skeleton",
                )

        # 3) Corrected centerline + endpoints
        cl = centerlines.get(int(lid))
        if cl is not None:
            cl_np = to_numpy(cl)
            if cl_np.shape[0] > 0:
                ax.plot(
                    cl_np[:, 0], cl_np[:, 1], cl_np[:, 2],
                    "-", c="red", lw=2.5,
                    label=f"centerline ({cl_np.shape[0]} pts)",
                )
                ax.scatter(
                    cl_np[:1, 0], cl_np[:1, 1], cl_np[:1, 2],
                    s=120, c="navy", marker="^", edgecolors="white",
                    label=f"base (min-Z)  {tuple(int(v) for v in cl_np[0])}",
                )
                ax.scatter(
                    cl_np[-1:, 0], cl_np[-1:, 1], cl_np[-1:, 2],
                    s=120, c="darkred", marker="v", edgecolors="white",
                    label=f"tip  (max-Z)  {tuple(int(v) for v in cl_np[-1])}",
                )

        # 4) Removed bridge voxels
        bv = bridges_by_label.get(int(lid)) or []
        if bv:
            bv_arr = np.asarray(bv, dtype=np.int32)
            ax.scatter(
                bv_arr[:, 0], bv_arr[:, 1], bv_arr[:, 2],
                s=80, c="magenta", marker="x",
                label=f"removed bridge ({bv_arr.shape[0]} vox)",
            )

        # 5) Uncorrected reference centerline (skeleton-diameter via the default
        #    compute_centerlines) — orange dashed.
        ref = uncorrected_centerlines.get(int(lid))
        if ref is not None:
            ref_np = to_numpy(ref)
            if ref_np.shape[0] > 0:
                ax.plot(
                    ref_np[:, 0], ref_np[:, 1], ref_np[:, 2],
                    "--", c="orange", lw=1.5, alpha=0.7,
                    label="uncorrected CL (diameter)",
                )

        ax.set_xlabel("X (L↔R)")
        ax.set_ylabel("Y (post↔ant)")
        ax.set_zlabel("Z (inf↔sup)")
        ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "GenusReport",
    "RepairLog",
    "SiphonCorrectionResult",
    "compute_corrected_centerline",
    "compute_mask_genus",
    "correct_siphon_centerlines",
    "ica_otsu_mask",
    "prune_skeleton_shortest_arc",
    "repair_ica_donut_3d",
]
