"""Precompute for ``pesa_ctpet_manim.py``: the PESA-Health CT-PET v5 pipeline on one subject.

Runs in the nvitk environment. Stage 2 is the real worker
(:func:`nvitk.pipes.pesa_health.ct_pet_v5.stage2_postprocess._process`) run into a temporary folder,
with recorders wrapped around the module's own steps: the organ interiors cleared from the fat and
their safety dilation, the extremities, the urinary bladder grown along the PET uptake, the
ureter from the PET, the legs / trunk split of the intermuscular fat before and after the
neighbour vote, the deltoid's left / right split, the skeleton subtraction from the muscles and the
last-bed trim of the quadriceps (with the PET bed geometry, the noise model and the measured noise
of the lowest PET slices, and the quadriceps SUVs with and without the trim).
Its outputs (MO, FAT, FAT_BATCH, FAT_IM, ORGANS, MUSCLES, their ``_VOL`` twins, BODY, SKELETON)
are the masks shown; stage 3 (:func:`stage3_measure.process_subject`) gives the SUVs and volumes.

The whole-body CT grid (1.17 × 1.17 × 2 mm, 512 × 512 × ~520) is pooled 3 × 3 × 2 for the voxel
display (a block takes its most frequent non-zero label, so thin fat layers stay); the L3 / L4
bone marrow is kept at the CT grid. The SUV of every shown voxel is sampled from the PET (bw).

The last-bed trim needs the PET bed geometry: ``PT_beds.json`` in the subject's NIfTI folder,
else the PET DICOM under ``--dicom-root`` (stage 2 then caches ``PT_beds.json`` in the NIfTI
folder, so point the roots at a copy when the NIfTI folder must stay untouched).

Usage (nvitk env)::

    python scripts/anim/pesa_ctpet_precompute.py -o scripts/anim/pesa_ctpet_stages.npz \\
        --nifti-root <NIFTI> --results-root <RESULTS> --dicom-root <DICOM> --batch 202603_Week4 \\
        --subject <PESA id>
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
DEFAULT_DICOM = Path("/BIOIT_IMAGE/PESA_Fat/DATA/Visit-5-DIXON_PET-CT/DATA/DICOM")
POOL = (3, 3, 2)
EDGE_SLICES = 110   # lowest PET slices whose noise is measured


def H(a) -> np.ndarray:
    """Host numpy array of an Image or a (CuPy / NumPy) array."""
    from nvitk.core.backend import to_numpy
    from nvitk.types import Image
    if isinstance(a, Image):
        a = a.data
    return np.asarray(to_numpy(a))


def pool_labels(a: np.ndarray, f=POOL) -> np.ndarray:
    """Block-pool a label volume: each block takes its most frequent non-zero label (0 if none)."""
    a = np.asarray(a)
    sx, sy, sz = (s // k * k for s, k in zip(a.shape, f))
    a = a[:sx, :sy, :sz]
    ids = np.unique(a)
    ids = ids[ids != 0]
    shp = (sx // f[0], f[0], sy // f[1], f[1], sz // f[2], f[2])
    best = np.zeros((sx // f[0], sy // f[1], sz // f[2]), np.int16)
    bestc = np.zeros(best.shape, np.int16)
    for i in ids:
        c = (a == i).reshape(shp).sum((1, 3, 5), dtype=np.int16)
        m = c > bestc
        best[m] = i
        bestc[m] = c[m]
    return best


def sample_centre(a: np.ndarray, f=POOL) -> np.ndarray:
    """Block-sample (the block's centre voxel): for many-class label maps shown once."""
    sx, sy, sz = (s // k * k for s, k in zip(a.shape, f))
    return np.asarray(a)[f[0] // 2:sx:f[0], f[1] // 2:sy:f[1], f[2] // 2:sz:f[2]].astype(np.int16)


def pooled_affine(aff: np.ndarray, f=POOL) -> np.ndarray:
    out = aff.copy()
    out[:3, :3] = aff[:3, :3] * np.asarray(f, float)
    out[:3, 3] = aff[:3, :3] @ ((np.asarray(f, float) - 1) / 2) + aff[:3, 3]
    return out


def _mesh(binary: np.ndarray, affine: np.ndarray, sigma: float = 1.0, step: int = 1):
    from skimage.measure import marching_cubes
    sm = ndi.gaussian_filter(np.pad(binary.astype(np.float32), 2), sigma)
    v, f, _n, _ = marching_cubes(sm, 0.5, step_size=step)
    v = v - 2
    return (v @ affine[:3, :3].T + affine[:3, 3]).astype(np.float32), f.astype(np.int32)


def slice_noise(pet: np.ndarray, n: int) -> np.ndarray:
    """Noise of the *n* lowest slices of an XYZ PET (z up): std of (slice - 5x5 median) / mean
    inside the eroded body."""
    out = np.full(n, np.nan)
    for k in range(n):
        s = pet[:, :, k]
        sm = ndi.gaussian_filter(s, 3)
        body = ndi.binary_erosion(sm > 0.4 * np.percentile(sm, 95), iterations=4)
        if body.sum() < 300:
            continue
        res = s - ndi.median_filter(s, 5)
        out[k] = res[body].std() / s[body].mean()
    return out


def quad_stats(pet, labels) -> dict:
    """Stage-3 SUV statistics of the quadriceps (L, R, both) in *labels*."""
    from nvitk.measure import Measurer
    from nvitk.pipes.pesa_health.ct_pet_v5 import stage3_measure as s3
    out = {}
    for name, ids in (("L", (1,)), ("R", (2,)), ("LR", (1, 2))):
        b = s3._build_binary_mask(labels, ids)
        r = Measurer(pet, b).align("mask_to_raw").suv(kinds=("bw",), stats=("max", "mean", "p95", "p99"),
                                                      philips=False, revert_scaling=False)
        out[name] = {k: float(r[f"SUVbw_{key}"]) for k, key in (("max", "max"), ("mean", "mean"),
                                                                 ("p95", "95percentile"), ("p99", "99percentile"))}
    return out


def compute(nifti_dir: Path, seg_dir: Path, dicom_dir: Path | None = None) -> tuple[dict, dict]:
    from nvitk.core.backend import set_global_backend
    set_global_backend("cupy")
    from nvitk.io import imread
    from nvitk.measure.suv import suv_image
    from nvitk.morphology import erode
    from nvitk.pipes.pesa_health.ct_pet_v5 import config as cfg
    from nvitk.pipes.pesa_health.ct_pet_v5 import stage2_postprocess as s2
    from nvitk.pipes.pesa_health.ct_pet_v5 import stage3_measure as s3
    from nvitk.segmentation.total_segmentator.class_maps import get_class_id

    rec: dict[str, list] = {}

    def wrap(name: str, keep_args=()):
        fn = getattr(s2, name)

        def w(*a, **k):
            out = fn(*a, **k)
            rec.setdefault(name, []).append({"args": [H(a[i]) for i in keep_args], "out": out,
                                             "args_full": a[0] if name == "trim_last_pet_bed" else None})
            return out

        setattr(s2, name, w)
        return fn

    originals = {n: wrap(n, ka) for n, ka in (("_organs_mask", ()), ("_with_dilated_organs", ()),
                                              ("_remove_extremities", (0,)), ("_process_bladder", (0,)),
                                              ("_neighbour_majority", (0,)), ("split_lr_by_cc", (0,)),
                                              ("_subtract_skeleton", (0,)), ("trim_last_pet_bed", ()))}
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        s2._process(seg_dir, nifti_dir, out, dicom_dir=dicom_dir)
        for n, f in originals.items():
            setattr(s2, n, f)
        ct = imread(str(s2.resolve_nii(nifti_dir, cfg.INPUT_STEM)), axes="XYZ")
        pet = imread(str(s2.resolve_nii(nifti_dir, cfg.PET_STEM)), axes="XYZ")
        masks = {p.name.replace(".nii.gz", ""): H(imread(str(p), axes="XYZ")) for p in sorted(out.glob("*.nii.gz"))}
        row = s3.process_subject(pet, out)
        trim_in = rec["trim_last_pet_bed"][0]["args_full"]
        trim_report = rec["trim_last_pet_bed"][0]["out"][1]
        quad_full = quad_stats(pet, trim_in)
        quad_trim = quad_stats(pet, imread(str(out / "MUSCLES.nii.gz"), axes="XYZ"))
    total = imread(str(s2.resolve_nii(seg_dir, "total")), axes="XYZ")
    tissue = imread(str(s2.resolve_nii(seg_dir, cfg.FAT_TASK)), axes="XYZ")
    muscles = imread(str(s2.resolve_nii(seg_dir, "thigh_shoulder_muscles")), axes="XYZ")
    aff = np.asarray(ct.affine, float)
    T, TI, MU = H(total), H(tissue), H(muscles)
    hu = H(ct).astype(np.float32)
    paff = pooled_affine(aff)

    arrays: dict = {}
    meta: dict = {"ct_shape": list(T.shape), "ct_spacing": [float(x) for x in np.abs(np.diag(aff)[:3])],
                  "pool": list(POOL), "stage3": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                                 for k, v in row.items()}}

    # ---- the fat steps (full grid)
    tid = {n: get_class_id(n, cfg.FAT_TASK) for n in ("torso_fat", "subcutaneous_fat", "intermuscular_fat",
                                                      "skeletal_muscle")}
    fat_v0 = TI == tid["torso_fat"]
    fat_s0 = TI == tid["subcutaneous_fat"]
    ext_in = rec["_remove_extremities"][0]["args"][0] > 0      # subcutaneous fat inside the body
    fat_s1 = H(rec["_remove_extremities"][0]["out"]) > 0         # ... without the extremities
    organs_raw = H(rec["_organs_mask"][0]["out"]) > 0
    organs_dil = H(rec["_with_dilated_organs"][0]["out"]) > 0
    bl_in = rec["_process_bladder"][0]["args"][0] > 0
    bl_out = H(rec["_process_bladder"][0]["out"]) > 0
    ureter = masks.get("_URETER", np.zeros_like(T)) > 0
    l3l4 = np.isin(T, [get_class_id("vertebrae_L3", "total"), get_class_id("vertebrae_L4", "total")])
    zs = np.where(l3l4.any((0, 1)))[0]
    meta["l3l4_z_world"] = [float(aff[2, 2] * zs.min() + aff[2, 3]), float(aff[2, 2] * zs.max() + aff[2, 3])]
    meta["fat_counts"] = {"v_raw": int(fat_v0.sum()), "s_raw": int(fat_s0.sum()), "s_in_body": int(ext_in.sum()),
                          "s_no_ext": int(fat_s1.sum()), "organs_raw": int(organs_raw.sum()),
                          "organs_dil": int(organs_dil.sum()), "bladder_ts": int(bl_in.sum()),
                          "bladder_pet": int(bl_out.sum()), "ureter": int(ureter.sum())}

    # ---- intermuscular fat
    im0 = TI == tid["intermuscular_fat"]
    nb_in = rec["_neighbour_majority"][0]["args"][0]
    nb_out = H(rec["_neighbour_majority"][0]["out"])
    quads = np.isin(masks["MUSCLES_VOL"], [1, 2])
    qz = np.where(quads.any((0, 1)))[0]
    meta["quads_z_world"] = [float(aff[2, 2] * qz.min() + aff[2, 3]), float(aff[2, 2] * qz.max() + aff[2, 3])]
    meta["fat_im_switched"] = int((nb_in != nb_out).sum())

    # ---- muscles
    sk_in = H(rec["_subtract_skeleton"][0]["args"][0])
    sk_out = H(rec["_subtract_skeleton"][0]["out"])
    mus_pretrim = H(trim_in)

    # ---- the last PET bed: geometry, noise model and measured noise at the inferior end
    from nvitk.pipes.pesa_health.common.pet_beds import load_pet_beds
    beds = load_pet_beds(nifti_dir)
    if beds is None or not trim_report.get("applied"):
        raise SystemExit(f"no last-bed trim for this subject: {trim_report.get('reason')}")
    pa_ = np.asarray(pet.affine, float)
    assert pa_[2, 2] > 0 and np.allclose(pa_[2, :2], 0), "PET z axis must point up"
    pet_h = H(pet).astype(np.float32)
    zk = pa_[2, 2] * np.arange(EDGE_SLICES) + pa_[2, 3]
    sens = beds.sensitivity(zk)
    ref = beds.overlap_sensitivity()
    noise = slice_noise(pet_h, EDGE_SLICES)
    ov = np.where(np.abs(sens - ref) < 0.006)[0]
    arrays["edge_z"] = zk.astype(np.float32)
    arrays["edge_rel"] = (noise / np.nanmedian(noise[ov])).astype(np.float32)
    arrays["edge_model"] = np.sqrt(ref / np.clip(sens, 1e-3, None)).astype(np.float32)
    meta["pet_edge"] = {**trim_report, "group_starts": beds.group_starts, "step": beds.step,
                        "axial_fov": beds.axial_fov, "z_first": beds.z_first, "n_beds": len(beds.group_starts),
                        "overlap_sensitivity": ref, "quad_full": quad_full, "quad_trim": quad_trim}
    delt = rec["split_lr_by_cc"][0]["args"][0] > 0

    # ---- everything pooled onto the display grid
    P = {
        "s1_total": sample_centre(T), "s1_tissue": pool_labels(TI), "s1_muscles": sample_centre(MU),
        "fat_raw": pool_labels(np.where(fat_v0, 1, np.where(fat_s0, 2, 0))),
        "fat_noext": pool_labels(np.where(fat_v0, 1, np.where(fat_s1, 2, 0))),
        "extremity_fat": pool_labels((fat_s0 & ~fat_s1).astype(np.uint8)),
        "organs_raw": pool_labels(organs_raw.astype(np.uint8)), "organs_dil": pool_labels(organs_dil.astype(np.uint8)),
        "bladder_ts": pool_labels(bl_in.astype(np.uint8)), "bladder_pet": pool_labels(bl_out.astype(np.uint8)),
        "ureter": pool_labels(ureter.astype(np.uint8)),
        "fat_vol": pool_labels(masks["FAT_VOL"]), "fat": pool_labels(masks["FAT"]),
        "fat_batch": pool_labels(masks["FAT_BATCH"]), "fat_batch_vol": pool_labels(masks["FAT_BATCH_VOL"]),
        "im_raw": pool_labels(im0.astype(np.uint8)), "im_pre": pool_labels(nb_in), "fat_im": pool_labels(masks["FAT_IM"]),
        "organs_vol": pool_labels(masks["ORGANS_VOL"]), "organs": pool_labels(masks["ORGANS"]),
        "mus_raw": pool_labels(sk_in), "mus_vol": pool_labels(masks["MUSCLES_VOL"]),
        "muscles": pool_labels(masks["MUSCLES"]), "muscles_pretrim": pool_labels(mus_pretrim), "deltoid": pool_labels(delt.astype(np.uint8)),
        "skeleton": pool_labels(masks["SKELETON"]), "mo_p": pool_labels(masks["MO"]),
    }
    assert int((sk_out != masks["MUSCLES_VOL"]).sum()) == 0
    uni = np.zeros(P["fat"].shape, bool)
    for k, a in P.items():
        if k not in ("s1_total", "s1_muscles"):
            uni |= a > 0
    uni |= P["s1_total"] > 0
    uni |= P["s1_muscles"] > 0
    idx = np.argwhere(uni)
    sel = tuple(idx.T)
    arrays["vox"] = idx.astype(np.int16)
    arrays["affine"] = paff
    for k, a in P.items():
        arrays[k] = a[sel].astype(np.int16 if k.startswith("s1_") else np.int8)

    # SUV (bw) at every shown voxel, from the PET grid
    suv = H(suv_image(pet, pet.metadata, philips=False)).astype(np.float32)
    pa = np.asarray(pet.affine, float)
    w = idx @ paff[:3, :3].T + paff[:3, 3]
    pi = (w - pa[:3, 3]) @ np.linalg.inv(pa[:3, :3]).T
    arrays["suv"] = ndi.map_coordinates(suv, pi.T, order=1, mode="nearest").astype(np.float16)

    # ---- the bone marrow at the CT grid (a crop around L3 / L4)
    lo = np.maximum(np.argwhere(l3l4).min(0) - 2, 0)
    hi = np.minimum(np.argwhere(l3l4).max(0) + 3, T.shape)
    cr = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    vl = np.zeros(T[cr].shape, np.int8)
    for name, lab in (("L4", 1), ("L3", 2)):
        vb = T[cr] == get_class_id(f"vertebrae_{name}", "total")
        vl[vb] = lab
    iso = np.zeros_like(vl)
    for lab in (1, 2):
        m = total.with_data((T == (get_class_id("vertebrae_L4", "total") if lab == 1 else
                                   get_class_id("vertebrae_L3", "total"))).astype(np.uint8))
        e = H(erode(m, footprint=s2._BN_ERODE))[cr] > 0
        iso[e] = lab
    midx = np.argwhere(vl > 0)
    msel = tuple(midx.T)
    arrays["mo_vox"] = (midx + lo).astype(np.int16)
    arrays["mo_raw"] = vl[msel]
    arrays["mo_iso"] = iso[msel]
    arrays["mo_final"] = masks["MO"][cr][msel].astype(np.int8)
    arrays["mo_vol"] = masks["MO_VOL"][cr][msel].astype(np.int8)
    arrays["mo_hu"] = hu[cr][msel].astype(np.float16)
    arrays["mo_suv"] = ndi.map_coordinates(
        suv, (((midx + lo) @ aff[:3, :3].T + aff[:3, 3] - pa[:3, 3]) @ np.linalg.inv(pa[:3, :3]).T).T,
        order=1, mode="nearest").astype(np.float16)
    arrays["ct_affine"] = aff
    meta["mo_vol_max_hu"] = cfg.MO_VOL_MAX_HU
    meta["mo_xy_erode"] = cfg.MO_XY_ERODE
    meta["bn_erode"] = s2._BN_ERODE
    meta["suv_erode"] = s2.SUV_ERODE

    # ---- context: bone MIP, PET MIP, body and skeleton surfaces
    b8 = np.clip((hu - 150.0) / 1100.0, 0, 1)
    b8 = (ndi.zoom(b8, (0.5, 0.5, 0.5), order=1) * 255).astype(np.uint8)
    arrays["bone8"] = b8
    a8 = aff.copy()
    a8[:3, :3] = aff[:3, :3] * 2.0
    a8[:3, 3] = aff[:3, :3] @ np.full(3, 0.5) + aff[:3, 3]
    arrays["bone8_affine"] = a8
    s8 = (np.clip(ndi.zoom(suv, (0.5, 0.5, 0.5), order=1) / 6.0, 0, 1) * 255).astype(np.uint8)
    arrays["pet8"] = s8
    p8 = pa.copy()
    p8[:3, :3] = pa[:3, :3] * 2.0
    p8[:3, 3] = pa[:3, :3] @ np.full(3, 0.5) + pa[:3, 3]
    arrays["pet8_affine"] = p8
    bod = pool_labels((masks["BODY"] > 0).astype(np.uint8), (4, 4, 3)) > 0
    arrays["body_v"], arrays["body_f"] = _mesh(bod, pooled_affine(aff, (4, 4, 3)), sigma=1.2, step=1)
    sk = pool_labels((masks["SKELETON"] > 0).astype(np.uint8), (2, 2, 1)) > 0
    arrays["skel_v"], arrays["skel_f"] = _mesh(sk, pooled_affine(aff, (2, 2, 1)), sigma=0.8, step=2)

    # the shown SUV means agree with stage 3's (a check of the sampling, not a measurement)
    lab_o = arrays["organs"]
    meta["suv_check"] = {"liver_sampled": float(arrays["suv"][lab_o == 1].astype(float).mean()),
                         "liver_stage3": float(row["HIGADO_SUVmean"])}
    meta["n_vox"] = int(len(idx))
    meta["n_mo_vox"] = int(len(midx))
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--nifti-root", type=Path, default=DEFAULT_NIFTI)
    ap.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--dicom-root", type=Path, default=DEFAULT_DICOM)
    ap.add_argument("--batch", default="202603_Week4")
    ap.add_argument("--subject", default="PESA1993744")
    args = ap.parse_args()
    from nvitk.pipes.pesa_health.ct_pet_v5 import config as cfg
    nifti_dir = args.nifti_root / args.batch / args.subject
    seg_dir = args.results_root / args.batch / cfg.STAGE1_DIR / args.subject / "CT"
    arrays, meta = compute(nifti_dir, seg_dir, args.dicom_root / args.batch / args.subject)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta, default=str)), **arrays)
    print(json.dumps({k: meta[k] for k in meta if k != "stage3"}, indent=1, default=str))
    print({k: getattr(v, "shape", None) for k, v in arrays.items()})
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
