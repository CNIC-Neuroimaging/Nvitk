"""Precompute the real ICA siphon-correction stages on a PESA subject for ``siphon_cut_manim.py``.

The animation must show what :mod:`nvitk.morphology.siphon_cut` does on a real looped
ICA, not a cartoon. This script runs the library on PESA13727025's eICAB mask (whose right ICA
closes a siphon loop) and records every intermediate:

- the mask before / after :func:`~nvitk.morphology.siphon_cut.pretess` and after
  :func:`~nvitk.morphology.siphon_cut.repair_loops`, and their Betti numbers;
- what ``repair_loops`` hands its helpers, captured by wrapping them for the run (nothing is
  re-implemented): the skeleton loop and its bridge / curl arcs (``loop_candidates``), the
  contact run (``contact_run``), the voxel cost field (``voxel_weights``), and the region, source,
  sink and cut of every minimum-cut attempt (``_min_vertex_cut``);
- the centerline before (plain skeleton centerline, which shortcuts across the contact) and
  after (:func:`~nvitk.morphology.siphon_cut.siphon_centerline` on the repaired mask).

Arrays are written at the mask's resolution for the small sets (the cut, the pretess voxels, the
loop) and as 1 mm display blocks (occupancy fractions) for the vessel itself, so the renderer
draws a few hundred cubes. The renderer needs numpy only.

Run (nvitk environment)::

    python scripts/anim/siphon_cut_precompute.py -o scripts/anim/siphon_cut_stages.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

DEFAULT_EICAB_DIR = Path("/home/imarcoss/DATA/PESA-Brain/Morphometrics/RESULTS/PESA13727025/eicab")
#: Display block (voxels per side): 2 -> 1 mm cubes on the 0.5 mm grid.
BLOCK = 2
#: Crop margin around the label (voxels).
PAD = 4
LABEL_NAMES = {1: "LICA", 2: "RICA"}


class Recorder:
    """Wraps siphon_cut's helpers for one ``repair_loops`` run and keeps what they were given."""

    NAMES = ("loop_candidates", "contact_run", "voxel_weights", "_min_vertex_cut")

    def __init__(self, sc):
        self.sc = sc
        self.orig = {n: getattr(sc, n) for n in self.NAMES}
        self.loop_calls: list = []
        self.groups: list[dict] = []

    def __enter__(self) -> "Recorder":
        o = self.orig

        def loop_candidates(*a, **k):
            out = o["loop_candidates"](*a, **k)
            self.loop_calls.append(out)
            return out

        def contact_run(loop, depth, intensity):
            run, info = o["contact_run"](loop, depth, intensity)
            self.groups.append({"loop": loop, "run": list(run), "contact": dict(info), "calls": []})
            return run, info

        def voxel_weights(mask, intensity, centres, spacing, depth=None):
            w, info = o["voxel_weights"](mask, intensity, centres, spacing, depth)
            self.groups[-1].update(weights=np.array(w), winfo=dict(info), centres=np.array(centres, bool),
                                   mask=np.array(mask, bool))
            return w, info

        def min_vertex_cut(region, source, sink, weights, uncuttable=None, scale=100.0):
            cut, cost, ok = o["_min_vertex_cut"](region, source, sink, weights, uncuttable, scale)
            self.groups[-1]["calls"].append({"region": np.array(region, bool), "source": np.array(source, bool),
                                             "sink": np.array(sink, bool), "cut": np.array(cut, bool),
                                             "cost": float(cost), "ok": bool(ok)})
            return cut, cost, ok

        self.sc.loop_candidates = loop_candidates
        self.sc.contact_run = contact_run
        self.sc.voxel_weights = voxel_weights
        self.sc._min_vertex_cut = min_vertex_cut
        return self

    def __exit__(self, *exc) -> None:
        for n, f in self.orig.items():
            setattr(self.sc, n, f)


def _bbox(mask: np.ndarray, pad: int) -> tuple[slice, slice, slice]:
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _blocks(arr: np.ndarray, b: int) -> np.ndarray:
    """Block means of *arr* (padded with zeros to a multiple of *b*)."""
    shape = [int(np.ceil(n / b)) * b for n in arr.shape]
    p = np.zeros(shape, dtype=float)
    p[: arr.shape[0], : arr.shape[1], : arr.shape[2]] = arr
    s = p.reshape(shape[0] // b, b, shape[1] // b, b, shape[2] // b, b)
    return s.mean(axis=(1, 3, 5))


def _block_mean_where(values: np.ndarray, where: np.ndarray, b: int) -> np.ndarray:
    """Mean of *values* over the *where* voxels of each block (0 where a block has none)."""
    num = _blocks(np.where(where, values, 0.0), b)
    den = _blocks(where.astype(float), b)
    return np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)


def compute(eicab_dir: Path, label: int | None) -> tuple[dict[str, np.ndarray], dict]:
    """Run pretess + repair_loops on the looped ICA of *eicab_dir* and collect every stage."""
    from nvitk.core.backend import set_global_backend

    set_global_backend("numpy")
    import nvitk.morphology.siphon_cut as sc
    from nvitk.morphology.centerline import compute_centerlines

    seg_img = nib.load(eicab_dir / "TOF_eICAB_CW.nii.gz")
    aff = seg_img.affine
    lab = np.asanyarray(seg_img.dataobj).astype(np.int32)
    tof = np.asanyarray(nib.load(eicab_dir / "TOF_resampled.nii.gz").dataobj).astype(np.float64)
    sp = np.asarray(sc.spacing_of(aff), float)

    # 1. The ICA to show: the one that still has a loop after pretess
    candidates = [label] if label else [2, 1]
    lid = None
    for c in candidates:
        own = lab == c
        if not own.any():
            continue
        box = _bbox(own, PAD)
        m_try, _ = sc.pretess(own[box], tof[box], mode="remove", protect=(lab[box] != 0) & ~own[box])
        if sc.betti(m_try).b1 > 0:
            lid = c
            break
    if lid is None:
        raise SystemExit("no ICA with a loop left after pretess")
    own = lab == lid
    box = _bbox(own, PAD)
    m0 = own[box].copy()
    inten = tof[box]
    others = (lab[box] != 0) & ~m0
    origin = np.array([s.start for s in box])

    # 2. pretess, then the loop repair (instrumented)
    b_raw = sc.betti(m0)
    m1, pinfo = sc.pretess(m0, inten, mode="remove", protect=others)
    b_pre = sc.betti(m1)
    with Recorder(sc) as rec:
        m2, cut, rep = sc.repair_loops(m1, inten, affine=aff, spacing=sp, label_name=LABEL_NAMES.get(lid, str(lid)))
    b_post = sc.betti(m2)
    inner = sc._bbox(sc._host_bool(m1), 3)
    off = np.array([s.start for s in inner])

    def to_crop(a: np.ndarray) -> np.ndarray:
        out = np.zeros(m1.shape, dtype=a.dtype)
        out[inner] = a
        return out

    # the accepted attempt: the call whose cut is (part of) the final removed voxels
    accepted = None
    for g in rec.groups:
        for c in g["calls"]:
            cc = to_crop(c["cut"])
            if c["ok"] and cc.any() and not (cc & ~cut).any():
                accepted = (g, c)
                break
        if accepted:
            break
    if accepted is None:
        raise SystemExit("no accepted cut recorded")
    g, call = accepted
    loop = g["loop"]
    loop_nodes = np.array(loop.nodes, float) + off
    roles = np.zeros(len(loop.nodes), dtype=np.int8)  # 0 curl, 1 bridge
    roles[list(loop.bridge)] = 1
    run_mask = np.zeros(len(loop.nodes), bool)
    run_mask[list(g["run"])] = True
    anchors = np.array([loop.nodes[i] for i in loop.anchors], float) + off
    skeleton = to_crop(rec.loop_calls[0][1]["skeleton"].astype(bool))

    region = to_crop(call["region"])
    mask_inner = to_crop(g["mask"])
    slab = mask_inner & ~region
    source, sink = to_crop(call["source"]), to_crop(call["sink"])
    weights = to_crop(g["weights"])
    cut_voxels = np.argwhere(to_crop(call["cut"])).astype(float)
    pretess_removed = np.argwhere(m0 & ~m1).astype(float)

    # 3. Centerlines before / after
    before = compute_centerlines(m0.astype(np.int32), labels=[1], min_points=3).get(1)
    before = np.zeros((0, 3)) if before is None else np.asarray(before, float)[:, :3]
    after, after_info = sc.siphon_centerline(m2, affine=aff, spacing=sp)
    after = np.asarray(after, float)

    def length_mm(p: np.ndarray) -> float:
        return float(np.linalg.norm(np.diff(p, axis=0) * sp, axis=1).sum()) if len(p) > 1 else 0.0

    # 4. Display blocks (1 mm)
    b = BLOCK
    occ0, occ1, occ2 = _blocks(m0, b), _blocks(m1, b), _blocks(m2, b)
    present = (occ0 > 0) | (occ1 > 0) | (occ2 > 0)
    bidx = np.argwhere(present)
    centres = (bidx + 0.5) * b - 0.5  # in crop voxel units
    sel = tuple(bidx.T)
    ref = float(np.median(inten[skeleton & m1])) if (skeleton & m1).any() else float(np.median(inten[m1]))
    tof_blk = _block_mean_where(inten / max(ref, 1e-9), m0, b)[sel]
    w_blk = _block_mean_where(weights, mask_inner, b)[sel]
    arrays = {
        "blk_centres": centres.astype(np.float32),
        "occ_raw": occ0[sel].astype(np.float32),
        "occ_pretess": occ1[sel].astype(np.float32),
        "occ_repaired": occ2[sel].astype(np.float32),
        "tof_rel": tof_blk.astype(np.float32),
        "weight": w_blk.astype(np.float32),
        "slab_frac": _blocks(slab, b)[sel].astype(np.float32),
        "source_frac": _blocks(source, b)[sel].astype(np.float32),
        "sink_frac": _blocks(sink, b)[sel].astype(np.float32),
        "loop_nodes": loop_nodes.astype(np.float32),
        "loop_bridge": roles,
        "loop_contact": run_mask,
        "loop_arc_mm": np.asarray(loop.arc_mm, np.float32),
        "loop_anchors": anchors.astype(np.float32),
        "skeleton": np.argwhere(skeleton).astype(np.float32),
        "cut_voxels": cut_voxels.astype(np.float32),
        "pretess_removed": pretess_removed.astype(np.float32),
        "cl_before": before.astype(np.float32),
        "cl_after": after.astype(np.float32),
    }
    acc_loop = next((e for e in rep["loops"] if e.get("resolved")), {})
    acc_att = next((a for a in acc_loop.get("attempts", []) if a.get("accepted")), {})
    vox = float(np.prod(sp))
    meta = {
        "subject": eicab_dir.parent.name,
        "label": int(lid),
        "label_name": LABEL_NAMES.get(lid, str(lid)),
        "spacing_mm": [float(s) for s in sp],
        "block": b,
        "crop_origin_vox": [int(v) for v in origin],
        "betti_raw": b_raw.to_dict(),
        "betti_pretess": b_pre.to_dict(),
        "betti_repaired": b_post.to_dict(),
        "n_voxels": int(m0.sum()),
        "pretess_removed": int(pinfo.get("n_removed", 0)),
        "loop_length_mm": round(float(loop.length_mm), 1),
        "bridge_mm": acc_loop.get("bridge_mm"),
        "curl_mm": acc_loop.get("curl_mm"),
        "anchor_rule": loop.anchor_rule,
        "contact_min_score": g["contact"].get("min_score"),
        "contact_nodes": int(run_mask.sum()),
        "radius_mm": round(float(g["winfo"]["radius_mm"]), 2),
        "intensity_ref": round(float(g["winfo"]["intensity_ref"]), 1),
        "attempt_where": acc_att.get("where"),
        "cut_cost": acc_att.get("cost"),
        "cut_cost_ratio": acc_att.get("cost_ratio"),
        "slice_cost": acc_loop.get("slice_cost"),
        "cut_voxels": int(len(cut_voxels)),
        "cut_mm3": round(len(cut_voxels) * vox, 3),
        "removed_mm3": round(float(rep.get("removed_mm3", 0.0)), 3),
        "volume_change_pct": round(100.0 * (int(m2.sum()) - int(m0.sum())) / int(m0.sum()), 3),
        "cl_before_mm": round(length_mm(before), 1),
        "cl_after_mm": round(length_mm(after), 1),
        "n_blocks": int(len(bidx)),
        "attempts": [{k: v for k, v in a.items() if k != "cut"} for a in acc_loop.get("attempts", [])],
    }
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--eicab-dir", type=Path, default=DEFAULT_EICAB_DIR,
                    help="eICAB output folder holding TOF_eICAB_CW and TOF_resampled.")
    ap.add_argument("--label", type=int, default=None, help="ICA label id (default: the looped one, RICA first).")
    args = ap.parse_args()
    arrays, meta = compute(args.eicab_dir, args.label)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta, default=str)), **arrays)
    print(json.dumps({k: v for k, v in meta.items() if k != "attempts"}, indent=1, default=str))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
