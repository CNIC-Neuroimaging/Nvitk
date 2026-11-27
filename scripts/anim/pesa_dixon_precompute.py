"""Precompute for ``pesa_dixon_manim.py``: the PESA-Health Dixon v5 pipeline on one subject.

Runs in the nvitk environment. For the three Dixon stations (HEAD = neck and shoulders, THORAX =
abdomen, LEGS = thighs) it reads the four contrasts (FAT, WATER, fat fraction, T2*) and the
stage-1 TotalSegmentator outputs, then rebuilds stage 2 step by step with the module's own
functions (:mod:`nvitk.pipes.pesa_health.dixon_v5.stage2_postprocess`), recording every intermediate
mask:

- HEAD: the paravertebral muscles (``autochthon``), biggest component each;
- THORAX: the liver from the WATER-contrast run (and the FAT-contrast one it replaces), its
  vessels (T2* > ``cfg.LIVER_VESSEL_T2STAR_MS``), each kidney's convex hull and the renal pelvis
  it leaves (dilated twice, cleared), L3 / L4 raw and eroded to the bone marrow;
- LEGS: the quadriceps and the femur / hip skeleton (femurs dilated) subtracted from them;
- the measurement erosion (``MEASURE_ERODE``) against the ``*_VOL`` twins.

The final masks are checked voxel for voxel against ``_process`` (the stage-2 worker) run into a
temporary folder, and the stage-3 means against ``stage3_measure.process_subject``. Every voxel
is kept at the Dixon grid (2 × 2 × 3 mm).

Usage (nvitk env)::

    python scripts/anim/pesa_dixon_precompute.py -o scripts/anim/pesa_dixon_stages.npz \\
        --nifti-root <NIFTI> --results-root <RESULTS> --batch 202603_Week4 --subject <PESA id>
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

DEFAULT_NIFTI = Path("/BIOIT_IMAGE/PESA_Fat/DATA/Visit-5-DIXON_PET-CT/DATA/NIFTI")
DEFAULT_RESULTS = Path("/BIOIT_IMAGE/PESA_Fat/DATA/Visit-5-DIXON_PET-CT/RESULTS")
TEX = 384  # coronal slice texture height (px)


def _u8(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)


def _mesh(binary: np.ndarray, affine: np.ndarray, sigma: float = 1.2, step: int = 2):
    from skimage.measure import marching_cubes
    sm = ndi.gaussian_filter(np.pad(binary.astype(np.float32), 2), sigma)
    v, f, _n, _ = marching_cubes(sm, 0.5, step_size=step)
    v = v - 2
    w = v @ affine[:3, :3].T + affine[:3, 3]
    return w.astype(np.float32), f.astype(np.int32)


def compute(nifti_dir: Path, seg_dir: Path) -> tuple[dict, dict]:
    from nvitk.core.backend import set_global_backend
    set_global_backend("numpy")
    from nvitk.io import imread
    from nvitk.morphology import dilate, erode
    from nvitk.pipes.pesa_health.common.label_erosion import erode_labels
    from nvitk.pipes.pesa_health.dixon_v5 import config as cfg
    from nvitk.pipes.pesa_health.dixon_v5 import stage2_postprocess as s2
    from nvitk.pipes.pesa_health.dixon_v5 import stage3_measure as s3
    from nvitk.pipes.pesa_health.dixon_v5.labels import HEAD_LABELS, LEGS_LABELS, THORAX_LABELS
    from nvitk.segmentation.hull_edt import convex_hull_3d
    from nvitk.segmentation.labels import biggest_cc
    from nvitk.segmentation.total_segmentator.class_maps import get_class_id

    def rd(parent: Path, stem: str):
        return s2._imread_opt(parent, stem)

    def A(img) -> np.ndarray:
        return np.asarray(img.data)

    arrays: dict = {}
    meta: dict = {"stations": {}, "labels": {"HEAD": HEAD_LABELS, "THORAX": THORAX_LABELS, "LEGS": LEGS_LABELS},
                  "liver_vessel_t2star_ms": cfg.LIVER_VESSEL_T2STAR_MS, "measure_erode": s2.MEASURE_ERODE,
                  "legs_skeleton_dilate": cfg.LEGS_SKELETON_DILATE}

    for region in cfg.REGION_ORDER:
        rdir = seg_dir / f"{cfg.INPUT_PREFIX}_{region}"
        maps = {k: rd(nifti_dir, f"{cfg.INPUT_PREFIX}_{region}_{k}") for k in cfg.INPUT_SUFFIXES}
        ref = maps["FAT"]
        aff = np.asarray(ref.affine, float)
        shape = A(ref).shape
        fat, water = A(maps["FAT"]).astype(np.float32), A(maps["WATER"]).astype(np.float32)
        ff, t2 = A(maps["FAT_FRACTION"]).astype(np.float32), A(maps["T2STAR"]).astype(np.float32)
        sig = fat + water
        body = sig > np.percentile(sig[sig > 0], 35) * 0.5
        body = ndi.binary_opening(body, iterations=1)
        lab, n = ndi.label(body)
        if n:
            body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
        body = np.stack([ndi.binary_fill_holes(body[..., z]) for z in range(shape[2])], -1)

        st: dict[str, np.ndarray] = {}   # name -> full-grid array (labels or bool)
        raw: dict[str, np.ndarray] = {}  # stage-1 outputs shown in the stage-1 step (class ids)
        if region == "HEAD":
            tot = rd(rdir, "total_mr")
            raw["total_mr"] = A(tot)
            head_vol = s2.build_head_mask(tot)
            st["final_vol"] = A(head_vol)
            st["final"] = A(erode_labels(head_vol, HEAD_LABELS, s2.MEASURE_ERODE))
        elif region == "THORAX":
            tot, vert = rd(rdir, "total_mr"), rd(rdir, "vertebrae_mr")
            wat = rd(rdir, cfg.THORAX_WATER_STEM)
            raw["total_mr"], raw["vertebrae_mr"], raw["total_mr_water"] = A(tot), A(vert), A(wat)
            lid = get_class_id("liver", "total_mr")
            liver_fat = s2._biggest_cc_or_empty(tot, lid)
            liver = s2._biggest_cc_or_empty(wat, lid)
            st["liver_fat"], st["liver"] = A(liver_fat) > 0, A(liver) > 0
            st["vessels"] = st["liver"] & (t2 > cfg.LIVER_VESSEL_T2STAR_MS)
            for side in ("left", "right"):
                k = s2._biggest_cc_or_empty(tot, get_class_id(f"kidney_{side}", "total_mr"))
                kb = A(k) > 0
                hull = np.asarray(convex_hull_3d(k.with_data(kb.astype(np.uint8))).data) > 0
                st[f"kidney_{side}"], st[f"hull_{side}"] = kb, hull
                # the pelvis exactly as _kidney_remove_pelvis finds it (biggest CC of hull minus kidney)
                pel = biggest_cc(k.with_data((hull & ~kb).astype(np.uint8)))
                st[f"pelvis_{side}"] = A(pel) > 0
                st[f"pelvis2_{side}"] = A(dilate(pel, footprint=1, iterations=2, mode="binary")) > 0
            for v in ("L3", "L4"):
                vb = s2.get_label(vert, get_class_id(f"vertebrae_{v}", "vertebrae_mr"), missing="empty")
                st[f"vert_{v}"] = A(vb) > 0
                ero = erode(vb.with_data((A(vb) > 0).astype(np.uint8)), footprint=s2._BN_ERODE)
                st[f"vert_{v}_ero"] = A(ero) > 0
                st[f"bn_{v}"] = A(s2._vertebra_narrow(vb)) > 0
            thorax, thorax_vol = s2.build_thorax_mask(tot, vert, wat, maps["T2STAR"])
            st["pre_erode"] = A(thorax)
            st["final_vol"] = A(thorax_vol)
            st["final"] = A(erode_labels(thorax, THORAX_LABELS, s2.MEASURE_ERODE))
        else:
            mus, tot = rd(rdir, "thigh_shoulder_muscles_mr"), rd(rdir, "total_mr")
            raw["thigh_shoulder_muscles_mr"], raw["total_mr"] = A(mus), A(tot)
            skel_raw = s2.build_skeleton_mask(tot)
            skel = s2.build_skeleton_mask(tot, dilate_rois=cfg.LEGS_SKELETON_DILATE)
            st["skeleton"], st["skeleton_dil"] = A(skel_raw) > 0, A(skel) > 0
            st["quads_raw"] = A(s2.build_legs_mask(mus, skeleton=None))
            st["pre_erode"] = A(s2.build_legs_mask(mus, skeleton=skel))
            st["final_vol"] = A(s2.build_legs_mask(mus, skeleton=skel_raw))
            st["final"] = A(erode_labels(s2.build_legs_mask(mus, skeleton=skel), LEGS_LABELS, s2.MEASURE_ERODE))

        # the voxels this station ever shows: every stage-1 class and every intermediate mask
        uni = np.zeros(shape, bool)
        for a in list(raw.values()) + list(st.values()):
            uni |= np.asarray(a) > 0
        idx = np.argwhere(uni)
        sel = tuple(idx.T)
        P = f"{region.lower()}_"
        arrays[P + "vox"] = idx.astype(np.int16)
        arrays[P + "affine"] = aff
        for k, a in raw.items():
            arrays[P + "s1_" + k] = np.asarray(a)[sel].astype(np.int16)
        for k, a in st.items():
            a = np.asarray(a)
            arrays[P + k] = a[sel].astype(np.int8 if a.dtype != bool else bool)
        arrays[P + "ff"] = ff[sel].astype(np.float16)
        arrays[P + "t2"] = t2[sel].astype(np.float16)

        # a coronal slice through the station (the structures' centre front-to-back), four contrasts
        key = np.asarray(st["final_vol"]) > 0
        yc = int(np.round(np.argwhere(key)[:, 1].mean())) if key.any() else shape[1] // 2
        if region == "THORAX":
            ks = st["kidney_left"] | st["kidney_right"]
            yc = int(np.round(np.argwhere(ks)[:, 1].mean()))
        alpha = body[:, yc, :].astype(np.float32)
        alpha = ndi.gaussian_filter(alpha, 0.8)
        tex = {}
        for name, a, lo, hi in (("FAT", fat, 0, np.percentile(fat[body], 99.5)),
                                ("WATER", water, 0, np.percentile(water[body], 99.5)),
                                ("FF", ff, 0, 100.0), ("T2", t2, 0, 80.0)):
            g = _u8(a[:, yc, :], lo, hi)
            tex[name] = np.stack([g, (alpha * 255).astype(np.uint8)], -1)  # (X, Z, 2): gray, alpha
        for k, v in tex.items():
            arrays[P + "tex_" + k] = v
        arrays[P + "body_v"], arrays[P + "body_f"] = _mesh(body, aff, sigma=1.5, step=2)
        meta["stations"][region] = {"shape": list(shape), "spacing": [float(x) for x in np.abs(np.diag(aff[:3, :3]))],
                                    "slice_y": yc, "n_vox": int(len(idx)), "states": sorted(st), "raw": sorted(raw)}

    # ---- verification: the stage-2 worker and stage 3
    with tempfile.TemporaryDirectory() as td:
        s2._process(seg_dir, nifti_dir, Path(td))
        diff = {}
        for region in cfg.REGION_ORDER:
            for suf, key in (("", "final"), ("_VOL", "final_vol")):
                ref = np.asarray(imread(str(Path(td) / f"{region}{suf}.nii.gz"), axes="XYZ").data)
                P = f"{region.lower()}_"
                full = np.zeros(ref.shape, np.int16)
                full[tuple(arrays[P + "vox"].T.astype(int))] = arrays[P + key]
                diff[f"{region}{suf}"] = int((full != ref).sum())
        row = s3.process_subject(nifti_dir, Path(td))
    meta["stage2_voxels_differ"] = diff
    meta["stage3"] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and v is not None else v)
                      for k, v in row.items()}
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--nifti-root", type=Path, default=DEFAULT_NIFTI)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--batch", default="202603_Week4")
    ap.add_argument("--subject", default="PESA1993744")
    args = ap.parse_args()
    from nvitk.pipes.pesa_health.dixon_v5 import config as cfg
    nifti_dir = args.nifti_root / args.batch / args.subject
    seg_dir = args.results_root / args.batch / cfg.STAGE1_DIR / args.subject
    arrays, meta = compute(nifti_dir, seg_dir)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta, default=str)), **arrays)
    print(json.dumps({k: meta[k] for k in ("stage2_voxels_differ",)}, indent=1))
    print(json.dumps(meta["stations"], indent=1))
    print({k: arrays[k].shape for k in arrays if k.endswith("_vox")})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
