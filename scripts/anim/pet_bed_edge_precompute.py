"""Precompute for ``pet_bed_edge_manim.py``: the noisy inferior edge of a whole-body PET and the
proposed per-subject trim of the quadriceps SUV mask (PESA-Fat CT-PET).

Reads one subject's PET DICOMs, CT and stage-2 masks and writes everything the scene draws:

- the PET in Bq/ml (``pixel x RescaleSlope + RescaleIntercept`` per slice: this scanner rescales
  every slice to its own maximum), its coronal MIP, a coronal slab through the thighs and axial
  slices from the inferior end;
- the beds: slices grouped by ``FrameReferenceTime (0054,1300)``, bed centres from the constant runs
  of the Philips per-bed value ``(7053,10D5)`` (21 mm above the start of each group; the leg bed's
  centre is extrapolated one 100 mm step below the next one);
- the triangular sensitivity model ``sum_b max(0, 1 - |z - c_b| / 82)`` (164 mm axial field) and the
  noise measured per slice (std of the 5x5-median residual / mean inside the eroded body),
  relative to the overlap level (sens = 0.78);
- the CT and the quadriceps masks (``MUSCLES`` = measure, ``MUSCLES_VOL`` = volumetry; labels 1 left,
  2 right) resampled onto the PET grid (nearest neighbour for masks);
- the proposed cut (slices more than 41 mm below the leg-bed centre, i.e. noise > x1.25 the overlap
  level), and its effect on the measure-mask statistics, in percent (the SUV factor is one constant
  per series, so percentages do not depend on it);
- for every PET subject under ``<root>/PET``: where its inferior end sits in its leg bed and how many
  slices the rule removes (DICOM headers only).

Nothing here is in the pipeline: it documents an exploration (see
``placeholder/Auxiliary/informe_bordes_pet.md``). Run with the nvitk environment's Python (needs
pydicom, nibabel, scipy)::

    python scripts/anim/pet_bed_edge_precompute.py --root ~/placeholder --subject PESA1256641 \
        -o scripts/anim/pet_bed_edge_stages.npz
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pydicom
from scipy import ndimage as ndi

HALF_FOV = 82.0          # mm, half of the 164 mm axial field (FieldOfViewDimensions)
OVERLAP_SENS = 0.78      # sensitivity midway between two bed centres 100 mm apart
NOISE_LIMIT = 1.25       # tolerated noise relative to the overlap level
CENTRE_ABOVE_GROUP = 21.0  # mm, bed centre above the first slice of its FrameReferenceTime group
LEG_SLICES = 180         # slices of the inferior end kept for the leg views
AXIAL_SLICES = (0, 5, 10, 16, 28, 45)


def cut_distance() -> float:
    """Distance below the leg-bed centre beyond which noise exceeds ``NOISE_LIMIT`` (41 mm)."""
    sens_min = OVERLAP_SENS / NOISE_LIMIT**2
    return HALF_FOV * (1.0 - sens_min)


def read_headers(folder: str) -> list:
    ds = [pydicom.dcmread(f, stop_before_pixels=True) for f in glob.glob(os.path.join(folder, "*"))]
    ds.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    return ds


def bed_geometry(ds: list) -> dict:
    """Bed groups, plateau runs and bed centres (z ascending = legs first)."""
    z = np.array([float(d.ImagePositionPatient[2]) for d in ds])
    frt = np.array([float(d.FrameReferenceTime) for d in ds])
    d5 = np.array([float(d[(0x7053, 0x10D5)].value) for d in ds])
    groups, b0 = [], 0
    for i in range(1, len(ds) + 1):
        if i == len(ds) or frt[i] != frt[b0]:
            groups.append((b0, i - 1))
            b0 = i
    runs, i = [], 0
    while i < len(d5):
        j = i
        while j + 1 < len(d5) and d5[j + 1] == d5[i]:
            j += 1
        if j > i:
            runs.append((i, j))
        i = j + 1
    interior = [0.5 * (z[a] + z[b]) for a, b in runs[1:-1]]
    step = float(np.mean(np.diff(interior)))
    centres = np.array([interior[0] - step, *interior, interior[-1] + step])
    assert len(centres) == len(groups), (len(centres), len(groups))
    sens = np.clip(1.0 - np.abs(z[:, None] - centres[None]) / HALF_FOV, 0.0, None).sum(1)
    # the rule as it would be computed from the standard tag alone
    z_second = z[groups[1][0]]
    z_cut = z_second + CENTRE_ABOVE_GROUP - step - cut_distance()
    return dict(z=z, frt=frt, d5=d5, groups=np.array(groups), runs=np.array(runs), centres=centres,
                step=step, sens=sens, z_cut=z_cut, z_second=z_second,
                start=[str(ds[a][(0x7053, 0x10D2)].value) for a, _ in groups])


def pet_volume(folder: str) -> tuple[np.ndarray, np.ndarray, list]:
    """PET in Bq/ml as [k, row, col] (k ascending z) and the RAS affine of the [col, row, k] array."""
    ds = [pydicom.dcmread(f) for f in glob.glob(os.path.join(folder, "*"))]
    ds.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    vol = np.stack([d.pixel_array.astype(np.float32) * float(d.RescaleSlope) + float(d.RescaleIntercept) for d in ds])
    z = [float(d.ImagePositionPatient[2]) for d in ds]
    ipp = [float(v) for v in ds[0].ImagePositionPatient]
    ps = [float(v) for v in ds[0].PixelSpacing]
    aff = np.array([[-ps[1], 0, 0, -ipp[0]], [0, -ps[0], 0, -ipp[1]], [0, 0, z[1] - z[0], z[0]], [0, 0, 0, 1.0]])
    return vol, aff, ds


def to_pet_grid(img: nib.Nifti1Image, aff_pet: np.ndarray, shape_ijk: tuple, order: int, data=None) -> np.ndarray:
    """Resample *img* onto the PET [col, row, k] grid, returned as [k, row, col]."""
    m = np.linalg.inv(img.affine) @ aff_pet
    src = np.asanyarray(img.dataobj) if data is None else data
    out = ndi.affine_transform(src, m[:3, :3], m[:3, 3], output_shape=shape_ijk, order=order, cval=-1024 if order else 0)
    return np.transpose(out, (2, 1, 0))


def slice_noise(vol: np.ndarray, n: int) -> np.ndarray:
    """Per-slice noise: std of (image - 5x5 median) / mean inside the eroded body."""
    noise = np.full(n, np.nan)
    for i in range(n):
        s = vol[i]
        sm = ndi.gaussian_filter(s, 3)
        body = ndi.binary_erosion(sm > 0.4 * np.percentile(sm, 95), iterations=4)
        if body.sum() < 300:
            continue
        res = s - ndi.median_filter(s, 5)
        noise[i] = res[body].std() / s[body].mean()
    return noise


def stats(v: np.ndarray) -> np.ndarray:
    return np.array([v.max(), v.mean(), np.percentile(v, 95), np.percentile(v, 99), v.size], float)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.expanduser("~/placeholder"))
    ap.add_argument("--subject", default="PESA1256641")
    ap.add_argument("-o", "--output", default=str(Path(__file__).with_name("pet_bed_edge_stages.npz")))
    a = ap.parse_args()
    root, subj = Path(a.root), a.subject

    vol, aff, ds = pet_volume(str(root / "PET" / subj))
    geo = bed_geometry(ds)
    z, sens, centres = geo["z"], geo["sens"], geo["centres"]
    nk, nr, nc = vol.shape
    shape_ijk = (nc, nr, nk)
    print(f"PET {vol.shape}, z {z[0]:.0f}..{z[-1]:.0f}, {len(centres)} beds, groups {[b - a_ + 1 for a_, b in geo['groups']]}")
    print(f"leg-bed centre {centres[0]:.0f} (slice 0 at {centres[0] - z[0]:.0f} mm), cut z {geo['z_cut']:.0f}")

    # noise relative to the overlap level of the leg bed
    n_noise = 100
    noise = slice_noise(vol, n_noise)
    ov = np.where(np.abs(sens[:n_noise] - OVERLAP_SENS) < 0.006)[0]
    ov = ov[ov < 75]
    rel = noise / np.nanmedian(noise[ov])
    pred = np.sqrt(OVERLAP_SENS / np.clip(sens[:n_noise], 1e-3, None))

    # CT and masks on the PET grid
    ct = nib.load(str(root / "CT" / subj / "CT.nii.gz"))
    ct_pet = to_pet_grid(ct, aff, shape_ijk, order=1, data=np.asanyarray(ct.dataobj).astype(np.float32))
    meas = to_pet_grid(nib.load(str(root / "Masks" / subj / "MUSCLES.nii.gz")), aff, shape_ijk, 0)
    volm = to_pet_grid(nib.load(str(root / "Masks" / subj / "MUSCLES_VOL.nii.gz")), aff, shape_ijk, 0)
    quad_meas = np.where(np.isin(meas, (1, 2)), meas, 0).astype(np.uint8)
    quad_vol = np.where(np.isin(volm, (1, 2)), volm, 0).astype(np.uint8)

    # coronal plane through the thighs: the row holding most quadriceps (measure) voxels
    L = LEG_SLICES
    row = int(np.argmax((quad_meas[:L] > 0).sum((0, 2))))
    slab = slice(row - 1, row + 2)
    keep = z >= geo["z_cut"]
    n_cut = int((~keep).sum())

    out = dict(
        z=z, frt=geo["frt"], d5=geo["d5"], groups=geo["groups"], runs=geo["runs"], centres=centres,
        step=geo["step"], sens=sens, z_cut=geo["z_cut"], z_second=geo["z_second"], n_cut=n_cut,
        cut_distance=cut_distance(), noise=noise, rel=rel, pred=pred, overlap_slices=ov,
        pet_mip=vol.max(axis=1).astype(np.float32),                    # [k, col]
        pet_cor=vol[:L, slab].mean(axis=1).astype(np.float32),          # [k, col]
        ct_cor=ct_pet[:L, slab].mean(axis=1).astype(np.float32),
        meas_cor=quad_meas[:L, row], vol_cor=quad_vol[:L, row],
        vol_mip=(quad_vol[:L] > 0).any(axis=1), meas_mip=(quad_meas[:L] > 0).any(axis=1),
        cor_row=row, pixel_mm=float(abs(aff[0, 0])),
        axial_k=np.array(AXIAL_SLICES),
    )

    # axial crops around the thighs
    rows = np.where((quad_vol[:60] > 0).any((0, 2)))[0]
    cols = np.where((quad_vol[:60] > 0).any((0, 1)))[0]
    r0, r1 = max(rows.min() - 22, 0), min(rows.max() + 22, nr)
    c0, c1 = max(cols.min() - 22, 0), min(cols.max() + 22, nc)
    out["axial_pet"] = np.stack([vol[k, r0:r1, c0:c1] for k in AXIAL_SLICES]).astype(np.float32)
    out["axial_meas"] = np.stack([quad_meas[k, r0:r1, c0:c1] for k in AXIAL_SLICES])
    out["axial_vol"] = np.stack([quad_vol[k, r0:r1, c0:c1] for k in AXIAL_SLICES])
    out["axial_ref"] = np.percentile(vol[30:60, r0:r1, c0:c1], 99.7)

    # measure-mask statistics, full and trimmed; activity MIP inside the measure mask
    act_mip = np.zeros((L, nc), np.float32)
    for side, lab in (("L", 1), ("R", 2)):
        m = quad_meas == lab
        full = stats(vol[m])
        mt = m & keep[:, None, None]
        trim = stats(vol[mt])
        kmax = np.unravel_index(np.argmax(np.where(m, vol, -1)), vol.shape)
        kmax_t = np.unravel_index(np.argmax(np.where(mt, vol, -1)), vol.shape)
        per_slice_max = np.array([vol[k][m[k]].max() if m[k].any() else np.nan for k in range(L)])
        out[f"stats_{side}"] = np.stack([full, trim])
        out[f"max_{side}"] = np.array([kmax, kmax_t])
        out[f"slicemax_{side}"] = per_slice_max / full[1]
        out[f"first_meas_{side}"] = int(np.where(m.any((1, 2)))[0].min())
        out[f"first_vol_{side}"] = int(np.where((quad_vol == lab).any((1, 2)))[0].min())
        act_mip = np.maximum(act_mip, np.where(m[:L], vol[:L], 0).max(axis=1))
        pct = 100 * (trim[:4] / full[:4] - 1)
        print(f"{side}: first measure slice {out[f'first_meas_{side}']}, max at slice {kmax[0]} -> {kmax_t[0]}; "
              f"max {pct[0]:+.1f}% mean {pct[1]:+.2f}% p95 {pct[2]:+.1f}% p99 {pct[3]:+.1f}% vox {100 * (trim[4] / full[4] - 1):+.1f}%")
    out["act_mip"] = act_mip

    # the other subjects: where their inferior end sits in the leg bed
    cohort = []
    for p in sorted((root / "PET").iterdir()):
        if not p.is_dir():
            continue
        g = bed_geometry(read_headers(str(p)))
        d0 = g["centres"][0] - g["z"][0]
        cohort.append((d0, int((g["z"] < g["z_cut"]).sum()), len(g["centres"]), p.name == subj))
        print(f"  {p.name}: {len(g['centres'])} beds, slice 0 at {d0:.0f} mm below the leg-bed centre, removes {cohort[-1][1]}")
    out["cohort"] = np.array(cohort, float)
    np.savez_compressed(a.output, **out)
    print("wrote", a.output, f"{os.path.getsize(a.output) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
