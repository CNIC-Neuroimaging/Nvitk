"""Precompute the real eICAB post-process stages on a PESA subject for ``eicab_pp_manim.py``.

Runs :func:`nvitk.pipes.qvtpy.util.eicab.eicab_postprocess.postprocess_eicab_labels` once, as
stage 1 does, on a PESA subject's eICAB Circle-of-Willis mask (PESA10061584 by default), and records every stage by wrapping
the functions it calls for that run (nothing is re-implemented):

- the label volume after each step: anterior choroidals dropped, local re-segmentation, ICAs
  to the image floor, distal growth, siphon correction;
- inside the trunk extension, each trunk's crop and its minimum-cost paths to the floor;
- inside the distal growth, the Frangi tree the labels compete for, the owner of each tree voxel
  and the geodesic cost at which its front arrived (the flood order), and the ACA midline plane.

The label volumes go out as 3 mm display blocks (most frequent label + occupancy per block), so
the renderer draws a couple of thousand cubes; two 2D insets (the half-maximum profile across the
basilar, a coronal view of the trunk extension) go out as RGB images. The renderer needs numpy
only.

Run (nvitk environment)::

    python scripts/anim/eicab_pp_precompute.py -o scripts/anim/eicab_pp_stages.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

#: PESA10061584 shows every step: ICAs 37-47 mm below eICAB's box, both ACAs with the AComm (the
#: midline guard), and a right ICA siphon loop.
DEFAULT_EICAB_DIR = Path("/home/imarcoss/DATA/PESA-Brain/Morphometrics/RESULTS/PESA10061584/eicab")
#: Display block (voxels per side): 6 -> 3 mm cubes on the 0.5 mm grid (~1,600 cubes).
BLOCK = 6
PAD = 6
STAGES = ("raw", "noacha", "reseg", "ext", "distal", "final")
NAMES = {1: "LICA", 2: "RICA", 3: "BA", 4: "AComm", 5: "LACA", 6: "RACA", 7: "LMCA", 8: "RMCA", 9: "LPComm",
         10: "RPComm", 11: "LPCA P1", 12: "RPCA P1", 13: "LPCA P2", 14: "RPCA P2", 15: "LSCA", 16: "RSCA",
         17: "LAChA", 18: "RAChA", 19: "LVA", 20: "RVA"}


class Recorder:
    """Wraps the post-process steps (and two helpers inside them) for one run."""

    def __init__(self, vr, ep):
        self.vr, self.ep = vr, ep
        self.snap: dict[str, np.ndarray] = {}
        self.info: dict[str, dict] = {}
        self.ext_boxes: list = []
        self.ext_paths: list = []
        self.compete: dict = {}
        self._orig: dict = {}

    def _wrap(self, mod, name, fn):
        self._orig[(mod, name)] = getattr(mod, name)
        setattr(mod, name, fn)

    def __enter__(self) -> "Recorder":
        vr, ep, o = self.vr, self.ep, self._orig

        def resegment_local(intensity, labels, *a, **k):
            self.snap["noacha"] = np.array(labels, np.int32)
            out, info = o[(vr, "resegment_local")](intensity, labels, *a, **k)
            self.snap["reseg"], self.info["reseg"] = np.array(out, np.int32), info
            return out, info

        def extend_trunks(intensity, labels, *a, **k):
            out, info = o[(vr, "extend_trunks")](intensity, labels, *a, **k)
            self.snap["ext"], self.info["ext"] = np.array(out, np.int32), info
            return out, info

        def extension_box(*a, **k):
            lo, hi = o[(vr, "_extension_box")](*a, **k)
            self.ext_boxes.append(np.array(lo))
            return lo, hi

        def paths_to_floor(*a, **k):
            paths, info = o[(vr, "_paths_to_floor")](*a, **k)
            self.ext_paths.append([np.array(p) + self.ext_boxes[-1] for p in paths])
            return paths, info

        def grow_distal(intensity, labels, ids, *a, **k):
            self.compete["ids"] = [int(i) for i in ids]
            out, info = o[(vr, "grow_distal")](intensity, labels, ids, *a, **k)
            self.snap["distal"], self.info["distal"] = np.array(out, np.int32), info
            return out, info

        def compete(tree, lab, ids, step_cost, sp, max_cost, forbid=None, extra_seeds=None):
            owner, best = o[(vr, "_compete")](tree, lab, ids, step_cost, sp, max_cost, forbid, extra_seeds)
            self.compete.update(tree=np.array(tree, bool), owner=np.array(owner, np.int32), best=np.array(best),
                                lab=np.array(lab, np.int32))
            return owner, best

        def repair_icas(intensity, labels, ica_ids, affine, *, pretess):
            self.snap["pre_siphon"] = np.array(labels, np.int32)
            out = o[(ep, "_repair_icas_mincut")](intensity, labels, ica_ids, affine, pretess=pretess)
            self.snap["final"], self.info["siphon"] = np.array(labels, np.int32), out
            return out

        self._wrap(vr, "resegment_local", resegment_local)
        self._wrap(vr, "extend_trunks", extend_trunks)
        self._wrap(vr, "_extension_box", extension_box)
        self._wrap(vr, "_paths_to_floor", paths_to_floor)
        self._wrap(vr, "grow_distal", grow_distal)
        self._wrap(vr, "_compete", compete)
        self._wrap(ep, "_repair_icas_mincut", repair_icas)
        return self

    def __exit__(self, *exc) -> None:
        for (mod, name), fn in self._orig.items():
            setattr(mod, name, fn)


def _blocks_label(lab: np.ndarray, b: int) -> tuple[np.ndarray, np.ndarray]:
    """Per block: the most frequent non-zero label and the labelled fraction."""
    shape = [int(np.ceil(n / b)) * b for n in lab.shape]
    p = np.zeros(shape, dtype=np.int32)
    p[: lab.shape[0], : lab.shape[1], : lab.shape[2]] = lab
    s = p.reshape(shape[0] // b, b, shape[1] // b, b, shape[2] // b, b).transpose(0, 2, 4, 1, 3, 5)
    s = s.reshape(s.shape[0], s.shape[1], s.shape[2], -1)
    occ = (s > 0).mean(-1)
    n_lab = int(lab.max()) + 1
    counts = np.stack([(s == k).sum(-1) for k in range(1, n_lab)], -1)
    mode = np.where(occ > 0, counts.argmax(-1) + 1, 0)
    return mode.astype(np.int16), occ.astype(np.float32)


def _blocks_min(values: np.ndarray, where: np.ndarray, b: int) -> np.ndarray:
    """Per block: the minimum of *values* over *where* (inf where a block has none)."""
    shape = [int(np.ceil(n / b)) * b for n in values.shape]
    p = np.full(shape, np.inf)
    v = np.where(where, values, np.inf)
    p[: v.shape[0], : v.shape[1], : v.shape[2]] = v
    s = p.reshape(shape[0] // b, b, shape[1] // b, b, shape[2] // b, b)
    return s.min(axis=(1, 3, 5))


def _blocks_any(mask: np.ndarray, b: int) -> np.ndarray:
    shape = [int(np.ceil(n / b)) * b for n in mask.shape]
    p = np.zeros(shape, bool)
    p[: mask.shape[0], : mask.shape[1], : mask.shape[2]] = mask
    return p.reshape(shape[0] // b, b, shape[1] // b, b, shape[2] // b, b).mean(axis=(1, 3, 5))


def _inset_reseg(tof: np.ndarray, raw: np.ndarray, reseg: np.ndarray, info: dict, lid: int = 3) -> np.ndarray:
    """Axial slice through the basilar: eICAB's outline vs the half-maximum one, and the
    profile across it with core, background and the threshold."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = raw == lid
    ks = np.argwhere(m)[:, 2]
    k = int(np.median(ks))
    sl_raw, sl_new, img = m[:, :, k], (reseg == lid)[:, :, k], tof[:, :, k]
    ci, cj = np.argwhere(sl_raw).mean(0).round().astype(int)
    h = 14
    win = (slice(ci - h, ci + h), slice(cj - h, cj + h))
    e = info["labels"][str(lid)]
    core, bg = float(e["core_median"]), float(e["background_median"])
    thr = bg + 0.5 * (core - bg)
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), dpi=130, gridspec_kw={"width_ratios": [1, 1.35]})
    ax[0].imshow(img[win].T, cmap="gray", origin="lower", interpolation="nearest")
    ax[0].contour(sl_raw[win].T, levels=[0.5], colors="#EFA43A", linewidths=1.6, origin="lower")
    ax[0].contour(sl_new[win].T, levels=[0.5], colors="#2E9E4F", linewidths=1.6, origin="lower")
    ax[0].plot([0, 2 * h - 1], [h, h], color="#2F7FD6", lw=0.8, ls="--")
    ax[0].set_title("basilar, axial slice", fontsize=9)
    ax[0].axis("off")
    x = np.arange(ci - h, ci + h)
    prof = tof[x, cj, k] / 1000.0
    mm = (x - ci) * 0.5
    ax[1].plot(mm, prof, color="#1B1D23", lw=1.6)
    for arr, c, lab in ((raw[x, cj, k] == lid, "#EFA43A", "eICAB"), (reseg[x, cj, k] == lid, "#2E9E4F", "half max")):
        if arr.any():
            xs = mm[arr]
            ax[1].axvspan(xs.min() - 0.25, xs.max() + 0.25, color=c, alpha=0.18, lw=0, label=lab)
    for v, c, lab in ((core, "#22456E", "core"), (thr, "#2E9E4F", "T = bg + ½(core − bg)"), (bg, "#8B9199", "bg")):
        ax[1].axhline(v / 1000.0, color=c, lw=1.0, ls=":")
        ax[1].text(mm[0], v / 1000.0, lab, fontsize=7, color=c, va="bottom")
    ax[1].set_xlabel("mm across the vessel", fontsize=8)
    ax[1].set_ylabel("TOF (×1000)", fontsize=8)
    ax[1].tick_params(labelsize=7)
    ax[1].legend(fontsize=7, loc="upper right", frameon=False)
    for s in ("top", "right"):
        ax[1].spines[s].set_visible(False)
    fig.tight_layout()
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return out


def _inset_extension(tof: np.ndarray, before: np.ndarray, after: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Coronal MIPs of the lower half: eICAB's trunks vs the trunks taken to the floor."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {1: (0.83, 0.63, 0.09), 2: (0.91, 0.77, 0.28), 3: (0.88, 0.42, 0.62), 19: (0.36, 0.31, 0.77),
              20: (0.60, 0.57, 0.88)}
    ap = int(np.argmax(np.abs(affine[1, :3])))
    trunks = list(colors)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), dpi=130)
    sel = np.isin(after, trunks) | np.isin(before, trunks)
    zs = np.argwhere(sel)[:, 2]
    zlo, zhi = 0, int(np.percentile(zs, 90)) + 6
    xs = np.argwhere(sel)[:, 0]
    xlo, xhi = max(int(xs.min()) - 25, 0), min(int(xs.max()) + 25, tof.shape[0])
    mip = tof.max(axis=ap)[xlo:xhi, zlo:zhi]
    lo, hi = np.percentile(mip, [1, 99.7])
    g = np.clip((mip - lo) / (hi - lo), 0, 1)
    for ax, lab, title in ((axes[0], before, "eICAB: cut at its box"), (axes[1], after, "post-process: to the floor")):
        rgbimg = np.stack([g] * 3, -1)
        proj = np.zeros(mip.shape, np.int32)
        for k in trunks:  # later ids drawn over earlier ones
            p = (lab == k).any(axis=ap)[xlo:xhi, zlo:zhi]
            proj[p] = k
        for k, c in colors.items():
            mk = proj == k
            rgbimg[mk] = 0.35 * rgbimg[mk] + 0.65 * np.array(c)
        ax.imshow(np.transpose(rgbimg, (1, 0, 2)), origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return out


def compute(eicab_dir: Path, block: int = BLOCK) -> tuple[dict[str, np.ndarray], dict]:
    """Run the post-process with the recorder on and collect every stage."""
    from nvitk.core.backend import set_global_backend

    set_global_backend("numpy")
    import nvitk.pipes.qvtpy.util.eicab.eicab_postprocess as ep
    import nvitk.segmentation.vessel_refine as vr

    seg_img = nib.load(eicab_dir / "TOF_eICAB_CW.nii.gz")
    aff = seg_img.affine
    raw = np.asanyarray(seg_img.dataobj).astype(np.int32)
    tof = np.asanyarray(nib.load(eicab_dir / "TOF_resampled.nii.gz").dataobj).astype(np.float64)
    labels = raw.copy()
    with Recorder(vr, ep) as rec:
        summary = ep.postprocess_eicab_labels(tof, labels, affine=aff)
    snaps = {"raw": raw, **rec.snap}
    assert all(k in snaps for k in STAGES), sorted(snaps)

    # distal growth in the full frame: tree, owner, arrival cost
    sp = vr.spacing_of(aff)
    dbox = vr._bbox(vr._isin(rec.snap["ext"], rec.compete["ids"]), vr._vox(vr.DISTAL_MAX_MM + 5.0, sp))
    tree = np.zeros(raw.shape, bool)
    owner = np.zeros(raw.shape, np.int32)
    best = np.full(raw.shape, np.inf)
    tree[dbox], owner[dbox], best[dbox] = rec.compete["tree"], rec.compete["owner"], rec.compete["best"]

    # display crop and blocks
    union = np.zeros(raw.shape, bool)
    for k in STAGES:
        union |= snaps[k] > 0
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - PAD, 0)
    hi = np.minimum(idx.max(0) + PAD + 1, raw.shape)
    crop = tuple(slice(int(a), int(c)) for a, c in zip(lo, hi))
    b = int(block)
    modes, occs = {}, {}
    for k in STAGES:
        modes[k], occs[k] = _blocks_label(snaps[k][crop], b)
    tree_frac = _blocks_any(tree[crop] & (snaps["ext"][crop] == 0), b)
    present = np.zeros(modes["raw"].shape, bool)
    for k in STAGES:
        present |= occs[k] > 0
    present |= tree_frac > 0
    bidx = np.argwhere(present)
    sel = tuple(bidx.T)
    centres = (bidx + 0.5) * b - 0.5 + lo  # full-volume voxel units
    world = (aff[:3, :3] @ centres.T).T + aff[:3, 3]
    grown = (owner > 0) & (snaps["ext"] == 0)
    arrival = _blocks_min(best[crop], grown[crop], b)[sel]
    removed_siphon = np.argwhere((snaps["pre_siphon"] > 0) & (snaps["final"] == 0) & np.isin(snaps["pre_siphon"], [1, 2]))

    arrays: dict[str, np.ndarray] = {
        "blk_centres": centres.astype(np.float32),
        "blk_world": world.astype(np.float32),
        "tree_frac": tree_frac[sel].astype(np.float32),
        "arrival": np.where(np.isfinite(arrival), arrival, -1.0).astype(np.float32),
        "siphon_removed": removed_siphon.astype(np.float32),
        "inset_reseg": _inset_reseg(tof, snaps["noacha"], snaps["reseg"], rec.info["reseg"]),
        "inset_ext": _inset_extension(tof, snaps["reseg"], snaps["ext"], aff),
    }
    for k in STAGES:
        arrays[f"label_{k}"] = modes[k][sel]
        arrays[f"occ_{k}"] = occs[k][sel]
    paths = []
    for group in rec.ext_paths:
        for p in group:
            paths.append(p)
    arrays["ext_paths"] = np.concatenate([np.c_[p, np.full(len(p), i)] for i, p in enumerate(paths)]).astype(
        np.float32) if paths else np.zeros((0, 4), np.float32)

    vox = float(np.prod(sp))

    def vol(lab: np.ndarray) -> float:
        return round(float((lab > 0).sum()) * vox, 1)

    ext_labels = rec.info["ext"]["labels"]
    dg = rec.info["distal"]
    mids = dg.get("midplanes") or []
    meta = {
        "subject": eicab_dir.parent.name,
        "spacing_mm": [float(s) for s in sp],
        "block": b,
        "affine": aff.tolist(),
        "names": {str(k): v for k, v in NAMES.items()},
        "volume_mm3": {k: vol(snaps[k]) for k in STAGES},
        "acha_mm3": round(float(np.isin(raw, [17, 18]).sum()) * vox, 1),
        "n_raw_labels": int(len([v for v in np.unique(raw) if v > 0])),
        "reseg": {lid: {"before": e.get("volume_before_mm3"), "after": e.get("volume_after_mm3"),
                        "change_pct": e.get("volume_change_pct"), "kept": e.get("kept_original")}
                  for lid, e in rec.info["reseg"]["labels"].items()},
        "extension": {lid: {"length_mm": e.get("length_mm"), "stop": e.get("stop"),
                            "added_mm3": e.get("volume_added_mm3"), "branches": len(e.get("path_costs") or [])}
                      for lid, e in ext_labels.items()},
        "distal": {lid: e.get("volume_added_mm3", 0.0) for lid, e in dg.get("labels", {}).items()},
        "distal_tree_voxels": dg.get("tree_voxels"),
        "distal_gate": dg.get("gate"),
        "midplane": mids[0] if mids else None,
        "siphon": rec.info["siphon"],
        "seconds": summary.get("seconds"),
        "n_blocks": int(len(bidx)),
    }
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--eicab-dir", type=Path, default=DEFAULT_EICAB_DIR,
                    help="eICAB output folder holding TOF_eICAB_CW and TOF_resampled.")
    ap.add_argument("--block", type=int, default=BLOCK, help="Display block side in voxels (6 = 3 mm).")
    args = ap.parse_args()
    arrays, meta = compute(args.eicab_dir, args.block)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta, default=str)), **arrays)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("affine", "names")}, indent=1, default=str))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
