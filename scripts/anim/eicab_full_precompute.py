"""Precompute the complete eICAB + post-process run for ``eicab_full_manim.py`` (dense voxels).

Runs :func:`nvitk.pipes.qvtpy.util.eicab.eicab_postprocess.postprocess_eicab_variants` once, as
stage 1 does (both masks: CW_pp and the distal-grown WB_pp), on the example subject's eICAB
circle-of-Willis labels and TOF, and records every step by wrapping the functions it calls for
that run (nothing is re-implemented):

- the label volume after each step: anterior choroidals dropped, local half-maximum
  re-segmentation, ICAs to the image floor, the ICA loop fill below the siphon, the distal growth,
  the siphon correction (WB and CW masks);
- inside the trunk extension, the minimum-cost paths to the floor;
- inside the distal growth, the Frangi tree the labels compete for, the owner of each tree voxel and
  the geodesic cost at which its front arrived (the flood order), and the ACA midline plane.

Unlike ``eicab_pp_precompute.py`` (3 mm display blocks), every 0.5 mm voxel goes out: the voxel
indices of the union of all stages, one label per voxel and stage, plus the TOF as an 8-bit volume
for the GPU maximum-intensity projection. The renderer needs numpy only.

The quick siphon step's data (``sip_*``) come from ``siphon_cut_precompute.compute`` at 1-voxel
blocks, on the mask stage 1 cut on the subject given by ``--siphon-eicab-dir`` (the fetal example;
the stenosis example has no siphon loop).

Run (nvitk environment)::

    python scripts/anim/eicab_full_precompute.py -o scripts/anim/eicab_full_stages.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

#: The stenosis example shipped with the animations.
DEFAULT_EICAB_DIR = Path("/home/imarcoss/placeholder/animations/stenosis example/eicab")
#: The siphon step needs an ICA whose siphon closes a loop; the stenosis example has none, so it
#: shows the fetal example's left ICA, as the post-process left it just before its siphon cut.
DEFAULT_SIPHON_DIR = Path("/home/imarcoss/placeholder/animations/fetal example/eicab")
PAD_MM = 6.0
STAGES = ("raw", "noacha", "reseg", "ext", "loopfill", "distal", "wb", "cw")
NAMES = {1: "LICA", 2: "RICA", 3: "BA", 4: "AComm", 5: "LACA", 6: "RACA", 7: "LMCA", 8: "RMCA", 9: "LPComm",
         10: "RPComm", 11: "LPCA P1", 12: "RPCA P1", 13: "LPCA P2", 14: "RPCA P2", 15: "LSCA", 16: "RSCA",
         17: "LAChA", 18: "RAChA", 19: "LVA", 20: "RVA", 21: "LACA A2", 22: "RACA A2", 23: "LMCA M2",
         24: "RMCA M2"}


class Recorder:
    """Wraps the post-process steps (and helpers inside them) for one run."""

    def __init__(self, vr, ep):
        self.vr, self.ep = vr, ep
        self.snap: dict[str, np.ndarray] = {}
        self.info: dict[str, dict] = {}
        self.ext_boxes: list = []
        self.ext_paths: list = []
        self.compete: dict = {}
        self.siphon_calls: list = []
        self._orig: dict = {}

    def _wrap(self, mod, name, fn):
        self._orig[(mod, name)] = getattr(mod, name)
        setattr(mod, name, fn)

    def __enter__(self) -> "Recorder":
        vr, ep, o = self.vr, self.ep, self._orig
        host = lambda a: np.array(a, np.int32)  # noqa: E731

        def resegment_local(intensity, labels, *a, **k):
            self.snap["noacha"] = host(labels)
            out, info = o[(vr, "resegment_local")](intensity, labels, *a, **k)
            self.info["reseg"] = info
            return out, info

        def extend_trunks(intensity, labels, *a, **k):
            self.snap["reseg"] = host(labels)  # after the re-segmentation's tidy step
            out, info = o[(vr, "extend_trunks")](intensity, labels, *a, **k)
            self.info["ext"] = info
            return out, info

        def extension_box(*a, **k):
            lo, hi = o[(vr, "_extension_box")](*a, **k)
            self.ext_boxes.append(np.array(lo))
            return lo, hi

        def paths_to_floor(*a, **k):
            paths, info = o[(vr, "_paths_to_floor")](*a, **k)
            self.ext_paths.append([np.array(p) + self.ext_boxes[-1] for p in paths])
            return paths, info

        def fill_loops(labels, ica_ids, affine):
            self.snap["ext"] = host(labels)
            out = o[(ep, "_fill_ica_loops")](labels, ica_ids, affine)
            self.snap["loopfill"] = host(labels)
            self.info["loopfill"] = out
            return out

        def grow_distal(intensity, labels, ids, *a, **k):
            self.compete["ids"] = [int(i) for i in ids]
            out, info = o[(vr, "grow_distal")](intensity, labels, ids, *a, **k)
            self.info["distal"] = info
            return out, info

        def compete(tree, lab, ids, step_cost, sp, max_cost, forbid=None, extra_seeds=None):
            owner, best = o[(vr, "_compete")](tree, lab, ids, step_cost, sp, max_cost, forbid, extra_seeds)
            self.compete.update(tree=np.array(tree, bool), owner=np.array(owner, np.int32), best=np.array(best),
                                shape=np.array(lab).shape)
            return owner, best

        def repair_icas(intensity, labels, ica_ids, affine, *, pretess):
            before = host(labels)
            out = o[(ep, "_repair_icas_mincut")](intensity, labels, ica_ids, affine, pretess=pretess)
            self.siphon_calls.append((before, host(labels), out))
            return out

        self._wrap(vr, "resegment_local", resegment_local)
        self._wrap(vr, "extend_trunks", extend_trunks)
        self._wrap(vr, "_extension_box", extension_box)
        self._wrap(vr, "_paths_to_floor", paths_to_floor)
        self._wrap(vr, "grow_distal", grow_distal)
        self._wrap(vr, "_compete", compete)
        self._wrap(ep, "_fill_ica_loops", fill_loops)
        self._wrap(ep, "_repair_icas_mincut", repair_icas)
        return self

    def __exit__(self, *exc) -> None:
        for (mod, name), fn in self._orig.items():
            setattr(mod, name, fn)


def compute(eicab_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    from scipy import ndimage as ndi

    from nvitk.core.backend import set_global_backend

    set_global_backend("numpy")
    import nvitk.pipes.qvtpy.util.eicab.eicab_postprocess as ep
    import nvitk.segmentation.vessel_refine as vr
    from eicab_pp_precompute import _inset_reseg

    tof_img = nib.load(eicab_dir / "TOF_resampled.nii.gz")
    aff = tof_img.affine  # eICAB's grid frame: every mask and the TOF share these voxels
    raw = np.asanyarray(nib.load(eicab_dir / "TOF_eICAB_CW.nii.gz").dataobj).astype(np.int32)
    tof = np.asanyarray(tof_img.dataobj).astype(np.float64)
    with Recorder(vr, ep) as rec:
        masks, summary = ep.postprocess_eicab_variants(tof, raw.copy(), affine=aff, distal=True, cw=True)
    (d_before, wb_cut, wb_log), (c_before, cw_cut, cw_log) = rec.siphon_calls
    snaps = {"raw": raw, **rec.snap, "distal": d_before, "wb": np.asarray(masks["wb"], np.int32),
             "cw": np.asarray(masks["cw"], np.int32)}
    assert all(k in snaps for k in STAGES), sorted(snaps)

    # distal growth in the full frame: tree, owner, arrival cost
    sp = vr.spacing_of(aff)
    dbox = vr._bbox(vr._isin(snaps["loopfill"], rec.compete["ids"]), vr._vox(vr.DISTAL_MAX_MM + 5.0, sp))
    tree = np.zeros(raw.shape, bool)
    owner = np.zeros(raw.shape, np.int32)
    best = np.full(raw.shape, np.inf)
    tree[dbox], owner[dbox], best[dbox] = rec.compete["tree"], rec.compete["owner"], rec.compete["best"]
    tree &= snaps["loopfill"] == 0

    union = np.zeros(raw.shape, bool)
    for k in STAGES:
        union |= snaps[k] > 0
    union |= tree
    idx = np.argwhere(union)
    sel = tuple(idx.T)

    # depth of each voxel inside eICAB's mask (outer shell first when the re-segmentation peels it)
    depth = ndi.distance_transform_edt(snaps["noacha"] > 0, sampling=sp)
    # growth order of the distal lumen: the arrival cost of the nearest tree voxel a front owned
    owned = (owner > 0) & np.isfinite(best)
    near = ndi.distance_transform_edt(~owned, return_distances=False, return_indices=True)
    grow_order = best[tuple(near)]
    # extension order: position along the minimum-cost path (0 at the label's end, 1 at the floor)
    paths = [p for group in rec.ext_paths for p in group]
    on_path = np.zeros(raw.shape, bool)
    pos = np.zeros(raw.shape, np.float32)
    for p in paths:
        q = np.clip(np.rint(p).astype(int), 0, np.array(raw.shape) - 1)
        on_path[tuple(q.T)] = True
        pos[tuple(q.T)] = np.linspace(0.0, 1.0, len(q))
    near = ndi.distance_transform_edt(~on_path, return_distances=False, return_indices=True) if paths else None
    ext_order = pos[tuple(near)] if paths else np.zeros(raw.shape, np.float32)

    # TOF for the MIP: the labels' box with a margin, 8 bit
    pad = np.ceil(np.array([PAD_MM * 6] * 3) / np.array(sp)).astype(int)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, raw.shape)
    crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    sub = tof[crop]
    p_lo, p_hi = np.percentile(sub[sub > 0], [40.0, 99.9])
    vol8 = (np.clip((sub - p_lo) / (p_hi - p_lo), 0, 1) * 255).astype(np.uint8)
    vol_aff = aff.copy()
    vol_aff[:3, 3] = aff[:3, :3] @ lo + aff[:3, 3]

    arrays: dict[str, np.ndarray] = {
        "vox": idx.astype(np.int16),
        "tree": tree[sel],
        "owner": owner[sel].astype(np.int8),
        "arrival": np.where(np.isfinite(best[sel]), best[sel], -1.0).astype(np.float32),
        "depth": depth[sel].astype(np.float32),
        "grow_order": np.where(np.isfinite(grow_order[sel]), grow_order[sel], -1.0).astype(np.float32),
        "ext_order": ext_order[sel].astype(np.float32),
        "tof8": vol8,
        "tof_affine": vol_aff,
        "inset_reseg": _inset_reseg(tof, snaps["noacha"], snaps["reseg"], rec.info["reseg"]),
    }
    for k in STAGES:
        arrays[f"lab_{k}"] = snaps[k][sel].astype(np.int8)
    arrays["ext_paths"] = (np.concatenate([np.c_[p, np.full(len(p), i)] for i, p in enumerate(paths)]).astype(
        np.float32) if paths else np.zeros((0, 4), np.float32))

    vox_mm3 = float(np.prod(sp))

    def vol(lab: np.ndarray) -> float:
        return round(float((lab > 0).sum()) * vox_mm3, 1)

    dg = rec.info["distal"]
    mids = dg.get("midplanes") or []
    box_lo = np.argwhere(raw > 0).min(0)
    box_hi = np.argwhere(raw > 0).max(0)
    meta = {
        "spacing_mm": [float(s) for s in sp],
        "affine": aff.tolist(),
        "names": {str(k): v for k, v in NAMES.items()},
        "volume_mm3": {k: vol(snaps[k]) for k in STAGES},
        "acha_mm3": round(float(np.isin(raw, [17, 18]).sum()) * vox_mm3, 1),
        "n_raw_labels": int(len([v for v in np.unique(raw) if v > 0])),
        "eicab_box_vox": [box_lo.tolist(), box_hi.tolist()],
        "reseg": {lid: {"before": e.get("volume_before_mm3"), "after": e.get("volume_after_mm3"),
                        "change_pct": e.get("volume_change_pct"), "kept": e.get("kept_original")}
                  for lid, e in rec.info["reseg"]["labels"].items()},
        "extension": {lid: {"length_mm": e.get("length_mm"), "stop": e.get("stop"),
                            "added_mm3": e.get("volume_added_mm3"), "branches": len(e.get("path_costs") or [])}
                      for lid, e in rec.info["ext"]["labels"].items()},
        "loopfill": rec.info["loopfill"],
        "distal": {lid: e.get("volume_added_mm3", 0.0) for lid, e in dg.get("labels", {}).items()},
        "distal_tree_voxels": dg.get("tree_voxels"),
        "midplane": mids[0] if mids else None,
        "siphon_wb": wb_log,
        "siphon_cw": cw_log,
        "seconds": summary.get("seconds"),
        "n_voxels": int(len(idx)),
    }
    return arrays, meta


def siphon(eicab_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    """The siphon min-cut of ``siphon_cut_precompute.py`` (pretess + one ``repair_loops`` run,
    recorded), at 1-voxel blocks, on the mask the post-process cut: *eicab_dir*'s WB mask just
    before its siphon correction (the post-process run with the recorder), on its looped ICA."""
    import shutil
    import tempfile

    import siphon_cut_precompute as scp

    import nvitk.pipes.qvtpy.util.eicab.eicab_postprocess as ep
    import nvitk.segmentation.vessel_refine as vr

    tof_img = nib.load(eicab_dir / "TOF_resampled.nii.gz")
    raw = np.asanyarray(nib.load(eicab_dir / "TOF_eICAB_CW.nii.gz").dataobj).astype(np.int32)
    tof = np.asanyarray(tof_img.dataobj).astype(np.float64)
    with Recorder(vr, ep) as rec:
        ep.postprocess_eicab_variants(tof, raw.copy(), affine=tof_img.affine, distal=True, cw=False)
    before, _after, log = rec.siphon_calls[0]
    looped = [int(k) for k, x in log.items() if (x.get("loops_cut") or 0) > 0]
    if not looped:
        raise SystemExit(f"no ICA siphon loop was cut in {eicab_dir}")
    tmp = Path(tempfile.mkdtemp(prefix="siphon_demo_"))
    try:
        nib.save(nib.Nifti1Image(before.astype(np.int32), tof_img.affine), tmp / "TOF_eICAB_CW.nii.gz")
        (tmp / "TOF_resampled.nii.gz").symlink_to(eicab_dir / "TOF_resampled.nii.gz")
        scp.BLOCK = 1
        arrays, meta = scp.compute(tmp, looped[0])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    meta.update(subject="fetal example", source="WB mask before the siphon correction", pipeline_log=log[str(looped[0])])
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--eicab-dir", type=Path, default=DEFAULT_EICAB_DIR,
                    help="eICAB output folder holding TOF_eICAB_CW and TOF_resampled.")
    ap.add_argument("--siphon-eicab-dir", type=Path, default=DEFAULT_SIPHON_DIR,
                    help="eICAB folder of the subject whose ICA siphon loop the quick siphon step shows.")
    args = ap.parse_args()
    arrays, meta = compute(args.eicab_dir)
    s_arrays, s_meta = siphon(args.siphon_eicab_dir)
    arrays.update({f"sip_{k}": v for k, v in s_arrays.items()})
    meta["siphon_demo"] = s_meta
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta, default=str)), **arrays)
    print(json.dumps({k: v for k, v in meta.items() if k not in ("affine", "names", "loopfill")}, indent=1,
                     default=str)[:3000])
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
