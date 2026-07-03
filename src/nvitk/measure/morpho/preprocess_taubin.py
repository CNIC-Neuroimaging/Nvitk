#!/usr/bin/env python3
"""Taubin smoothing of multi-label segmentations.

All processing functions are importable; edit the CONFIG block below only for
direct single-case runs.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


# ============================================================================
# CONFIG FOR DIRECT RUN (paths must be supplied at runtime; no host defaults)
# ============================================================================
INPUT_PATH = None
OUTPUT_DIR = None   # None → same folder as INPUT_PATH
OUTPUT_NAME = None  # None → <input_stem>_taubin.nii.gz
KEEP_LARGEST_COMPONENT = False
TAUBIN_ITERS = 20
TAUBIN_LAMBDA = 0.65
TAUBIN_MU = -0.65
# ============================================================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_segmentation(path: str):
    img = nib.load(path)
    data = img.get_fdata().astype(np.int32)
    return data, img.affine, img.header


def save_segmentation(path: str, data: np.ndarray, affine, header) -> None:
    out = nib.Nifti1Image(data.astype(np.int16), affine=affine, header=header)
    out.set_data_dtype(np.int16)
    nib.save(out, path)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndi.label(mask.astype(bool), structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n <= 1:
        return mask.astype(bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = ndi.distance_transform_edt(mask)
    outside = ndi.distance_transform_edt(~mask)
    return inside - outside


def crop_with_padding(mask: np.ndarray, pad: int) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    coords = np.where(mask)
    if coords[0].size == 0:
        empty = (slice(0, 0), slice(0, 0), slice(0, 0))
        return mask[:0, :0, :0], empty
    slices = []
    for axis_coords, size in zip(coords, mask.shape):
        start = max(int(axis_coords.min()) - pad, 0)
        stop = min(int(axis_coords.max()) + pad + 1, size)
        slices.append(slice(start, stop))
    bbox = (slices[0], slices[1], slices[2])
    return mask[bbox], bbox


def _laplacian(field: np.ndarray) -> np.ndarray:
    lap = np.zeros_like(field, dtype=np.float32)
    for axis in range(field.ndim):
        lap += ndi.convolve1d(field, [1.0, -2.0, 1.0], axis=axis, mode="nearest")
    return lap / (2 * field.ndim)


def taubin_smooth_field(field: np.ndarray, iterations: int, lam: float, mu: float) -> np.ndarray:
    out = field.astype(np.float32, copy=True)
    for _ in range(int(iterations)):
        out = out + float(lam) * _laplacian(out)
        out = out + float(mu) * _laplacian(out)
    return out


def build_taubin_scores(
    seg: np.ndarray,
    labels: Iterable[int],
    keep_largest: bool,
    taubin_iters: int,
    taubin_lambda: float,
    taubin_mu: float,
    skip_smoothing: Iterable[int] = (),
) -> Dict[int, np.ndarray]:
    scores: Dict[int, np.ndarray] = {}
    skip_set = set(skip_smoothing)
    pad = max(2, int(np.ceil(max(taubin_iters, 1) / 2)))
    for label in labels:
        mask = seg == label
        if not mask.any():
            continue
        if keep_largest:
            mask = keep_largest_component(mask)
        cropped_mask, bbox = crop_with_padding(mask, pad=pad)
        sdf = signed_distance(cropped_mask).astype(np.float32)
        if int(label) not in skip_set:
            sdf = taubin_smooth_field(sdf, iterations=taubin_iters, lam=taubin_lambda, mu=taubin_mu)
        score = np.full(seg.shape, -1e6, dtype=np.float32)
        score[bbox] = sdf
        scores[int(label)] = score
    return scores


def fuse_from_scores(score_maps: Dict[int, np.ndarray], shape: Tuple[int, int, int]) -> np.ndarray:
    if not score_maps:
        return np.zeros(shape, dtype=np.int32)
    labels = sorted(score_maps)
    score_stack = np.stack([score_maps[label] for label in labels], axis=0)
    best_index = np.argmax(score_stack, axis=0)
    best_score = np.max(score_stack, axis=0)
    out = np.array([labels[i] for i in best_index.ravel()], dtype=np.int32).reshape(shape)
    out[best_score <= 0] = 0
    return out


def preprocess_segmentation(
    seg: np.ndarray,
    keep_largest: bool,
    taubin_iters: int,
    taubin_lambda: float,
    taubin_mu: float,
    skip_smoothing: Iterable[int] = (17, 18),
) -> np.ndarray:
    labels = [int(x) for x in np.unique(seg) if x != 0]
    scores = build_taubin_scores(
        seg=seg, labels=labels, keep_largest=keep_largest,
        taubin_iters=taubin_iters, taubin_lambda=taubin_lambda, taubin_mu=taubin_mu,
        skip_smoothing=skip_smoothing,
    )
    return fuse_from_scores(scores, seg.shape)


def main() -> None:
    if not INPUT_PATH:
        raise SystemExit("Set INPUT_PATH or call preprocess_segmentation() from stage7 / run_morphometrics_case.")
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Input segmentation not found: {INPUT_PATH}")

    out_dir = OUTPUT_DIR or os.path.dirname(os.path.abspath(INPUT_PATH))
    ensure_dir(out_dir)

    seg, affine, header = load_segmentation(INPUT_PATH)
    spacing = header.get_zooms()[:3]
    print(f"Loaded : {INPUT_PATH}")
    print(f"Spacing: {tuple(round(float(x), 3) for x in spacing)}")

    cleaned = preprocess_segmentation(
        seg=seg, keep_largest=KEEP_LARGEST_COMPONENT,
        taubin_iters=TAUBIN_ITERS, taubin_lambda=TAUBIN_LAMBDA, taubin_mu=TAUBIN_MU,
    )

    input_stem = os.path.basename(INPUT_PATH).replace(".nii.gz", "").replace(".nii", "")
    output_name = OUTPUT_NAME or f"{input_stem}_taubin.nii.gz"
    out_path = os.path.join(out_dir, output_name)
    save_segmentation(out_path, cleaned, affine, header)

    raw_labels = [int(x) for x in np.unique(seg) if x != 0]
    report = {
        "input": INPUT_PATH,
        "output": out_path,
        "spacing_mm": [float(x) for x in spacing],
        "smoothing_method": "taubin",
        "taubin_iters": int(TAUBIN_ITERS),
        "taubin_lambda": float(TAUBIN_LAMBDA),
        "taubin_mu": float(TAUBIN_MU),
        "keep_largest_component": bool(KEEP_LARGEST_COMPONENT),
        "raw_label_count": int(len(raw_labels)),
        "raw_labels": raw_labels,
        "raw_volume_voxels": {str(l): int(np.count_nonzero(seg == l)) for l in raw_labels},
        "output_volume_voxels": {str(l): int(np.count_nonzero(cleaned == l)) for l in raw_labels},
    }
    report_path = os.path.join(out_dir, f"{input_stem}_taubin_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved : {out_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
