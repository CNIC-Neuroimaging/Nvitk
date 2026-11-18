"""Precompute the qvtpy 4D-flow pipeline (stages 2-5) on the example subject for
``qvtpy_pipeline_manim.py``: from the TOF segmentation to the 4D-flow labels and their LOCs.

- **Stage 2** (registration): the FLIRT matrix ``tof_to_4dflow.mat`` as a world transform
  (:func:`nvitk.registration.world.flirt_matrix_to_world`, TOF_resampled -> Angiography_3D), applied
  to the TOF MIP and the post-processed eICAB labels.
- **Stage 3** (centerlines): its outputs, read as written: the eICAB labels in the 4D-flow grid,
  the complex-difference (CD) vessel mask from the sliding threshold, the venous mask and the
  rasterised centerlines (arteries + named sinuses).
- **Stage 4** (4D-flow segmentation): :func:`~nvitk.pipes.qvtpy.util.segmentation.vessel_cd_segmentation.build_seg_4dflow_local`
  re-run with the example run's own settings (``segmentation_meta.json``) and recorded at each step
  (local CD threshold, trunk region growing, vertebral split, the blood flood's Frangi tree and
  watershed, siphon correction). The result is checked against the shipped ``seg_4dflow``.
- **Stage 5** (LOCs): ``locs.csv`` (station, tangent) and the 4D-flow segmentation's centerline
  branches.

Every voxel goes out at its own grid (TOF 0.5 mm, 4D flow 0.78 × 0.78 × 0.8 mm), with the TOF,
the 4D-flow angiogram and the CD as 8-bit volumes for the GPU MIP.

Run (nvitk environment)::

    python scripts/anim/qvtpy_pipeline_precompute.py -o scripts/anim/qvtpy_pipeline_stages.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

#: A complete example (eicab/, 4DFlow/, qvtpy/stage2-6); the stenosis renders pass --root.
DEFAULT_ROOT = Path("/home/imarcoss/placeholder/animations/fetal example")
SEG_STAGES = ("thr", "rg", "vsplit", "preflood", "flood", "final")


def _u8(vol: np.ndarray, lo_pct: float = 40.0, hi_pct: float = 99.9) -> np.ndarray:
    v = np.asarray(vol, np.float32)
    lo, hi = np.percentile(v[v > 0], [lo_pct, hi_pct])
    return (np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


class SegRecorder:
    """Wraps the stage-4 steps for one ``build_seg_4dflow_local`` run."""

    def __init__(self, vcs, dvt):
        self.vcs, self.dvt = vcs, dvt
        self.snap: dict[str, np.ndarray] = {}
        self.tree = None
        self.markers = None
        self.labeled = None
        self._orig: dict = {}

    def _wrap(self, mod, name, fn):
        self._orig[(mod, name)] = getattr(mod, name)
        setattr(mod, name, fn)

    def __enter__(self):
        vcs, dvt, o = self.vcs, self.dvt, self._orig

        def region_grow(seg, *a, **k):
            self.snap.setdefault("thr", np.array(seg, np.int32))
            return o[(vcs, "_region_grow_trunk")](seg, *a, **k)

        def split_vertebral(seg, *a, **k):
            self.snap["rg"] = np.array(seg, np.int32)
            out = o[(vcs, "split_vertebral_from_basilar")](seg, *a, **k)
            self.snap["vsplit"] = np.array(out[0], np.int32)
            return out

        def flood(seg, *a, **k):
            self.snap["preflood"] = np.array(seg, np.int32)
            info = o[(vcs, "flood_expand_vessels")](seg, *a, **k)
            self.snap["flood"] = np.array(seg, np.int32)
            self.flood_info = info
            return info

        def watershed(tree, markers, *a, **k):
            self.tree, self.markers = np.array(tree, bool), np.array(markers, np.int32)
            out = o[(dvt, "watershed_labels_into_vessels")](tree, markers, *a, **k)
            self.labeled = np.array(out, np.int32)
            return out

        self._wrap(vcs, "_region_grow_trunk", region_grow)
        self._wrap(vcs, "split_vertebral_from_basilar", split_vertebral)
        self._wrap(vcs, "flood_expand_vessels", flood)
        self._wrap(dvt, "watershed_labels_into_vessels", watershed)
        return self

    def __exit__(self, *exc):
        for (mod, name), fn in self._orig.items():
            setattr(mod, name, fn)


def _bfs_order(tree: np.ndarray, seeds: np.ndarray, max_steps: int = 400) -> np.ndarray:
    """Steps of 26-connected growth from *seeds* inside *tree* (the order a flood reaches each
    tree voxel); -1 where never reached."""
    from scipy import ndimage as ndi

    order = np.full(tree.shape, -1, np.int32)
    front = seeds & tree
    order[front] = 0
    reached = front.copy()
    st = np.ones((3, 3, 3), bool)
    for k in range(1, max_steps):
        grown = ndi.binary_dilation(front, structure=st) & tree & ~reached
        if not grown.any():
            break
        order[grown] = k
        reached |= grown
        front = grown
    return order


def compute(root: Path) -> tuple[dict, dict]:
    from nvitk.core.backend import set_global_backend

    set_global_backend("numpy")
    from nvitk.pipes.qvtpy.labels import QVTPY_ARTERIAL_ID_TO_NAME, VENOUS_NAME_BY_LABEL
    from nvitk.pipes.qvtpy.stage4_4dflow_segmentation import load_stage3_eicab, load_stage3_venous_mask
    from nvitk.pipes.qvtpy.util.segmentation import distal_vessel_tree as dvt
    from nvitk.pipes.qvtpy.util.segmentation import vessel_cd_segmentation as vcs
    from nvitk.pipes.qvtpy.util.segmentation.flood_options import FLOOD_DEFAULTS
    from nvitk.registration.world import flirt_matrix_to_world

    q = root / "qvtpy"
    s2, s3, s4, s5 = (q / n for n in ("stage2_registration", "stage3_centerline", "stage4_4dflow_segmentation",
                                       "stage5_loc_generation"))
    tof_p = root / "eicab" / "TOF_resampled.nii.gz"
    angio_p = root / "4DFlow" / "Angiography_3D.nii.gz"
    cd_p = root / "4DFlow" / "ComplexDifference_3D.nii.gz"

    # ---- stage 2: the registration as a world transform
    W = flirt_matrix_to_world(s2 / "tof_to_4dflow.mat", tof_p, angio_p)
    tof_img = nib.load(tof_p)
    tof_aff = tof_img.affine
    pp = np.asanyarray(nib.load(root / "eicab" / "TOF_eICAB_WB_pp.nii.gz").dataobj).astype(np.int32)
    tidx = np.argwhere(pp > 0)
    tof = np.asanyarray(tof_img.dataobj).astype(np.float32)

    # ---- 4D-flow volumes
    angio_img = nib.load(angio_p)
    f_aff = angio_img.affine
    angio = np.asanyarray(angio_img.dataobj).astype(np.float32)
    cd_img = nib.load(cd_p)
    cd = np.asanyarray(cd_img.dataobj).astype(np.float64)
    spacing = tuple(float(z) for z in cd_img.header.get_zooms()[:3])

    def load(p: Path, dt=np.int32) -> np.ndarray:
        return np.asanyarray(nib.load(p).dataobj).astype(dt)

    e4d = load(s3 / "eicab_in_4dflow.nii.gz")
    cdbin = load(s3 / "cd_vessel_binary_qc.nii.gz") > 0
    venous = load(s3 / "venous_mask.nii.gz") > 0
    clm = load(s3 / "centerlines_mask.nii.gz")
    seg_ref = load(s4 / "seg_4dflow.nii.gz")

    # ---- stage 4 re-run, recorded
    sm = json.loads((s4 / "segmentation_meta.json").read_text())
    flood_kw = {k: sm[k] for k in FLOOD_DEFAULTS if k in sm}
    eq, en = load_stage3_eicab(s3, "example", "anim")
    with SegRecorder(vcs, dvt) as rec:
        res = vcs.build_seg_4dflow_local(
            cd, clm, eicab_qvtpy=eq, eicab_native=en, venous_mask=load_stage3_venous_mask(s3), spacing=spacing,
            crop_padding_bbox=int(sm.get("crop_padding_bbox", 3)), thr_algorithm=sm.get("thr_algorithm", "lsthr"),
            cl_barrier_radius=int(sm.get("cl_barrier_radius", 2)), segment_acomm=bool(sm.get("segment_acomm", False)),
            **flood_kw)
    final = np.asarray(res.segmentation, np.int32)
    rec.snap["final"] = final
    agree = float((final == seg_ref).mean())
    diff = int((final != seg_ref).sum())
    snaps = rec.snap
    tree = rec.tree if rec.tree is not None else np.zeros(cd.shape, bool)
    markers = rec.markers if rec.markers is not None else np.zeros(cd.shape, np.int32)
    order = _bfs_order(tree, markers > 0)

    # ---- voxel set on the 4D-flow grid
    union = (e4d > 0) | cdbin | venous | (clm > 0) | (tree & (snaps["preflood"] == 0))
    for k in SEG_STAGES:
        union |= snaps[k] > 0
    fidx = np.argwhere(union)
    sel = tuple(fidx.T)

    # ---- MIP volumes: the labels' region with a margin
    def crop_u8(vol, idx, pad, lo=40.0, hi=99.9):
        lo_i = np.maximum(idx.min(0) - pad, 0)
        hi_i = np.minimum(idx.max(0) + pad + 1, vol.shape)
        c = tuple(slice(int(a), int(b)) for a, b in zip(lo_i, hi_i))
        return _u8(vol[c], lo, hi), lo_i

    tof8, t_lo = crop_u8(tof, tidx, np.array([60, 60, 40]))
    tof8_aff = tof_aff.copy()
    tof8_aff[:3, 3] = tof_aff[:3, :3] @ t_lo + tof_aff[:3, 3]
    angio8, a_lo = crop_u8(angio, fidx, np.array([30, 30, 20]), 50.0, 99.9)
    cd8, _ = crop_u8(cd.astype(np.float32), fidx, np.array([30, 30, 20]), 50.0, 99.9)
    f8_aff = f_aff.copy()
    f8_aff[:3, 3] = f_aff[:3, :3] @ a_lo + f_aff[:3, 3]

    locs = pd.read_csv(s5 / "locs.csv")
    br = json.loads((s4 / "centerlines_seg_branches.json").read_text())["branches"]
    bpts, bmeta = [], []
    for k, b in enumerate(br):
        p = np.asarray(b["points"], float)
        if len(p) >= 2:
            bpts.append(np.c_[p, np.full(len(p), k)])
            bmeta.append({"name": b["branch_name"], "label": int(b["parent_label"]), "trunk": bool(b.get("is_trunk"))})
    names = {**{int(k): v for k, v in QVTPY_ARTERIAL_ID_TO_NAME.items() if k > 0},
             **{int(k): v for k, v in VENOUS_NAME_BY_LABEL.items()}}
    vox_mm3 = float(np.prod(spacing))

    arrays = {
        "tof_vox": tidx.astype(np.int16), "tof_lab": pp[tuple(tidx.T)].astype(np.int8),
        "tof_affine": tof_aff, "tof8": tof8, "tof8_affine": tof8_aff, "W": W,
        "f_vox": fidx.astype(np.int16), "f_affine": f_aff, "angio8": angio8, "cd8": cd8, "f8_affine": f8_aff,
        "e4d": e4d[sel].astype(np.int8), "cdbin": cdbin[sel], "venous": venous[sel], "clm": clm[sel].astype(np.int8),
        "tree": tree[sel] & (snaps["preflood"][sel] == 0), "markers": markers[sel].astype(np.int8),
        "order": order[sel].astype(np.int16),
        "branches": np.concatenate(bpts).astype(np.float32),
        "locs": locs[["vessel_id", "centerline_x", "centerline_y", "centerline_z", "tangent_x", "tangent_y",
                      "tangent_z"]].to_numpy(np.float32),
    }
    for k in SEG_STAGES:
        arrays[f"seg_{k}"] = snaps[k][sel].astype(np.int8)
    fi = rec.flood_info if hasattr(rec, "flood_info") else {}
    meta = {
        "spacing_4d": list(spacing), "names": {str(k): v for k, v in names.items()},
        "registration": json.loads((s2 / "registration_meta.json").read_text()) | {"subject": None},
        "loc_names": locs["vessel_name"].tolist(), "loc_roles": locs["loc_role"].tolist(),
        "branches": bmeta,
        "volumes_mm3": {k: round(float((snaps[k] > 0).sum()) * vox_mm3, 1) for k in SEG_STAGES},
        "n_cdbin": int(cdbin.sum()), "n_venous": int(venous.sum()), "n_tree": int(tree.sum()),
        "agreement_with_seg_4dflow": agree, "voxels_differing": diff,
        "thr_algorithm": sm.get("thr_algorithm"), "flood_tree": fi.get("tree"),
        "flood_labels": fi.get("labels"),
        "vertebral_split": None if res.vertebral_split is None else res.vertebral_split.as_dict(),
        "translation_mm": [round(float(v), 2) for v in W[:3, 3]],
        "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(W[:3, :3]) - 1) / 2, -1, 1)))), 2),
    }
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="Example folder with eicab/, 4DFlow/ and qvtpy/stage2..5.")
    args = ap.parse_args()
    arrays, meta = compute(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta, default=str)), **arrays)
    print(json.dumps({k: meta[k] for k in ("volumes_mm3", "n_cdbin", "n_venous", "n_tree", "agreement_with_seg_4dflow",
                                           "voxels_differing", "translation_mm", "rotation_deg")}, default=str))
    print({k: v.shape for k, v in arrays.items()})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
