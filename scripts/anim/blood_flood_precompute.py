"""Precompute the real ``blood_flood`` stages on a synthetic vessel phantom.

The qvtpy distal-expansion animation (``blood_flood_manim.py``) must show what the
algorithm actually does, not a hand-drawn cartoon. So this script builds a small
synthetic 4D-flow-like CD volume containing a vessel tree, then runs the *real*
:mod:`nvitk.segmentation.blood_flood` primitives over it stage by stage and dumps
every intermediate array to an ``.npz``.

The Manim scene only loads that ``.npz`` — it never re-implements the algorithm.
Keeping the two apart also means the renderer needs nothing but numpy, so it can
live in a separate (manim) environment from nvitk.

Run::

    python scripts/anim/blood_flood_precompute.py -o /tmp/blood_flood_stages.npz
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

# Phantom geometry is authored in voxel units, then placed at OFFSET inside a
# larger grid. The padding matters: the GMM in `hysteresis_vessel_tree` needs a
# realistic background population, otherwise the vessel itself dominates the fit
# and the hysteresis thresholds land so high that nothing distal survives.
GRID: tuple[int, int, int] = (58, 36, 26)
OFFSET: tuple[float, float, float] = (6.0, 5.0, 4.0)

# Tube wall sharpness (voxels of blur at the lumen edge) and background texture.
WALL_BLUR: float = 0.34
TEXTURE_SIGMA: float = 2.2
TEXTURE_AMP: float = 0.06
NOISE_SIGMA: float = 0.9
NOISE_AMP: float = 0.05

# Voxels drawn as the faint "CD volume" ghost in the animation.
DISPLAY_THRESHOLD: float = 0.40

# Distal-expansion parameters as qvtpy stage-4 runs them.
FRANGI_SIGMAS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
HYST_LOW_FACTOR: float = 3.5
HYST_HIGH_FACTOR: float = 0.5
THIN_KEEP_PERCENTILE: float = 55.0
BARRIER_DILATE_RADIUS: int = 1

LABEL_MCA: int = 1
LABEL_ACA: int = 2


def _polyline(points: list[tuple[float, float, float]], n: int = 160) -> np.ndarray:
    """Resample a control polygon into ``n`` points along a smooth Catmull-Rom-ish path."""
    ctrl = np.asarray(points, dtype=np.float64)
    if len(ctrl) < 3:
        t_ctrl = np.linspace(0.0, 1.0, len(ctrl))
        t = np.linspace(0.0, 1.0, n)
        return np.stack([np.interp(t, t_ctrl, ctrl[:, k]) for k in range(3)], axis=1)
    # Chord-length parameterization + cubic interpolation per axis (smooth curvature,
    # which is what makes Frangi respond as a tube rather than a chain of blobs).
    seg = np.linalg.norm(np.diff(ctrl, axis=0), axis=1)
    t_ctrl = np.concatenate([[0.0], np.cumsum(seg)])
    t_ctrl /= t_ctrl[-1]
    t = np.linspace(0.0, 1.0, n)
    from scipy.interpolate import CubicSpline

    cs = CubicSpline(t_ctrl, ctrl, axis=0)
    return np.asarray(cs(t), dtype=np.float64)


def _stamp_tube(
    field: np.ndarray,
    points: list[tuple[float, float, float]],
    r0: float,
    r1: float,
    *,
    amplitude: float = 1.0,
    n: int = 160,
) -> None:
    """Max-accumulate a smooth-walled tube of tapering radius into ``field``."""
    path = _polyline(points, n=n)
    radii = np.linspace(float(r0), float(r1), len(path))
    nx, ny, nz = field.shape
    for (cx, cy, cz), r in zip(path, radii):
        pad = int(np.ceil(r)) + 2
        x0, x1 = max(0, int(cx) - pad), min(nx, int(cx) + pad + 1)
        y0, y1 = max(0, int(cy) - pad), min(ny, int(cy) + pad + 1)
        z0, z1 = max(0, int(cz) - pad), min(nz, int(cz) + pad + 1)
        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue
        gx, gy, gz = np.meshgrid(
            np.arange(x0, x1), np.arange(y0, y1), np.arange(z0, z1), indexing="ij"
        )
        d = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2 + (gz - cz) ** 2)
        # Sigmoid wall: ~1 in the lumen, ~0 outside, one voxel of blur at the edge.
        val = amplitude / (1.0 + np.exp((d - r) / WALL_BLUR))
        np.maximum(field[x0:x1, y0:y1, z0:z1], val, out=field[x0:x1, y0:y1, z0:z1])


def _stamp_blob(
    field: np.ndarray,
    center: tuple[float, float, float],
    radius: float,
    *,
    amplitude: float = 1.0,
) -> None:
    """Max-accumulate an isotropic blob (bright in CD, but *not* tubular)."""
    _stamp_tube(field, [center, center], radius, radius, amplitude=amplitude, n=2)


# --- Phantom definition -------------------------------------------------------
# Named so the animation can caption each structure with what it stands for.

ICA_TRUNK = [(4.8, 13.0, 5.5), (5.4, 12.8, 7.5), (6.0, 12.5, 9.6)]
MCA_TRUNK = [(6.5, 12.5, 10.5), (12.0, 12.0, 11.5), (17.0, 12.5, 11.0), (22.0, 12.0, 11.0)]
MCA_SUP = [(22.0, 12.0, 11.0), (27.0, 15.0, 12.5), (32.0, 18.0, 13.5), (37.5, 20.5, 13.5)]
MCA_INF = [(22.0, 12.0, 11.0), (27.0, 9.0, 9.5), (32.0, 7.0, 8.5), (38.0, 5.5, 8.0)]
MCA_SUP_TWIG = [(32.0, 18.0, 13.5), (36.0, 21.0, 12.0), (40.5, 22.5, 10.0)]
MCA_INF_TWIG = [(31.0, 7.5, 8.8), (36.0, 4.5, 10.0), (41.0, 3.5, 12.0)]
ACA_TRUNK = [(43.0, 23.5, 4.0), (37.0, 22.0, 5.0), (32.0, 21.0, 6.5), (28.0, 19.5, 8.0)]
ACA_TWIG = [(28.0, 19.5, 8.0), (30.0, 20.0, 10.5), (33.0, 19.5, 12.5)]
VEIN = [(13.0, 22.0, 4.0), (20.0, 23.5, 3.0), (28.0, 24.0, 3.5)]
BLOB_A = (17.0, 5.5, 14.5)
BLOB_B = (33.0, 12.0, 3.0)

# Proximal extent (in authored x) of the stage-3 seeds each label starts from.
MCA_SEED_XMAX: float = 14.0
ACA_SEED_XMIN: float = 39.0


def _o(points):
    """Shift authored coordinates into the padded grid."""
    ox, oy, oz = OFFSET
    return [(x + ox, y + oy, z + oz) for (x, y, z) in points]


def _stamp_vessels(field: np.ndarray) -> None:
    """The MCA/ACA tree that distal expansion is meant to recover."""
    _stamp_tube(field, _o(MCA_TRUNK), 2.05, 1.80)
    _stamp_tube(field, _o(MCA_SUP), 1.60, 1.15)
    _stamp_tube(field, _o(MCA_INF), 1.55, 1.10)
    _stamp_tube(field, _o(MCA_SUP_TWIG), 1.10, 0.85)
    _stamp_tube(field, _o(MCA_INF_TWIG), 1.05, 0.82)
    _stamp_tube(field, _o(ACA_TRUNK), 1.50, 1.15)
    _stamp_tube(field, _o(ACA_TWIG), 1.05, 0.85)


def build_phantom(seed: int = 7) -> dict[str, np.ndarray]:
    """Synthetic bright-blood volume: vessel tree + barrier + distractors + noise."""
    rng = np.random.default_rng(seed)
    vessel = np.zeros(GRID, dtype=np.float64)
    _stamp_vessels(vessel)

    # ICA/basilar: bright and thick — the structure distal expansion must never claim.
    ica = np.zeros(GRID, dtype=np.float64)
    _stamp_tube(ica, _o(ICA_TRUNK), 2.15, 2.00)

    # A venous segment that is genuinely tubular (so Frangi + hysteresis keep it),
    # but touches no seed — this is what the marker-connected-CC step exists to drop.
    vein = np.zeros(GRID, dtype=np.float64)
    _stamp_tube(vein, _o(VEIN), 1.10, 0.95, amplitude=0.86)

    # Bright but non-tubular: killed by vesselness, not by intensity.
    blobs = np.zeros(GRID, dtype=np.float64)
    _stamp_blob(blobs, tuple(np.add(BLOB_A, OFFSET)), 2.35, amplitude=0.80)
    _stamp_blob(blobs, tuple(np.add(BLOB_B, OFFSET)), 2.25, amplitude=0.78)

    structure = np.maximum.reduce([vessel, ica, vein, blobs])

    def _smooth_field(sigma: float) -> np.ndarray:
        f = ndi.gaussian_filter(rng.normal(0.0, 1.0, GRID), sigma=sigma)
        return f / (np.abs(f).max() + 1e-9)

    texture = _smooth_field(TEXTURE_SIGMA)  # parenchyma: bright-ish, not tubular
    noise = _smooth_field(NOISE_SIGMA)
    intensity = np.clip(
        0.93 * structure + 0.06 + TEXTURE_AMP * texture + NOISE_AMP * noise, 0.0, 1.0
    )

    return {
        "intensity": intensity,
        "vessel": vessel,
        "ica": ica,
        "vein": vein,
        "blobs": blobs,
    }


def _markers(phantom: dict[str, np.ndarray]) -> np.ndarray:
    """Stage-3 proximal seeds: the short trunks that already carry a qvtpy label.

    Deliberately short — everything past them is what the distal pass must win back.
    """
    grid = phantom["intensity"].shape
    marks = np.zeros(grid, dtype=np.int32)
    xs = np.broadcast_to(np.arange(grid[0])[:, None, None], grid)
    ox = OFFSET[0]

    mca_seed = np.zeros(grid, dtype=np.float64)
    _stamp_tube(mca_seed, _o(MCA_TRUNK), 2.05, 1.80)
    marks[(mca_seed > 0.5) & (xs <= MCA_SEED_XMAX + ox)] = LABEL_MCA

    aca_seed = np.zeros(grid, dtype=np.float64)
    _stamp_tube(aca_seed, _o(ACA_TRUNK), 1.50, 1.15)
    marks[(aca_seed > 0.5) & (xs >= ACA_SEED_XMIN + ox)] = LABEL_ACA
    return marks


def _geodesic_order(tree: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """26-connected BFS depth from the seeds — the visual order of the flood front."""
    order = np.full(tree.shape, -1, dtype=np.int32)
    q: deque[tuple[int, int, int]] = deque()
    seeds = np.argwhere((markers != 0) & tree)
    for x, y, z in seeds:
        order[x, y, z] = 0
        q.append((int(x), int(y), int(z)))
    nx, ny, nz = tree.shape
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    while q:
        x, y, z = q.popleft()
        d = order[x, y, z] + 1
        for dx, dy, dz in offsets:
            u, v, w = x + dx, y + dy, z + dz
            if 0 <= u < nx and 0 <= v < ny and 0 <= w < nz:
                if tree[u, v, w] and order[u, v, w] < 0:
                    order[u, v, w] = d
                    q.append((u, v, w))
    return order


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    st = ndi.generate_binary_structure(3, 3)
    return ndi.binary_dilation(mask.astype(bool), structure=st, iterations=int(radius))


def _gmm_fit_for_plot(v: np.ndarray, fg: np.ndarray) -> dict[str, list[float]]:
    """Replicate the fit inside ``hysteresis_vessel_tree`` so the panel plot matches."""
    from sklearn.mixture import GaussianMixture

    samples_all = v[fg]
    samples_all = samples_all[np.isfinite(samples_all)]
    pos = samples_all[samples_all > 0]
    fit_floor = float(np.percentile(pos, 50.0))
    samples = pos[pos >= fit_floor]
    if samples.size < 50:
        samples = pos
    gmm = GaussianMixture(n_components=3, tol=1e-3, max_iter=100, n_init=1, random_state=0)
    gmm.fit(samples.reshape(-1, 1))
    order = np.argsort(gmm.means_.flatten())
    counts, edges = np.histogram(pos, bins=48)
    return {
        "gmm_means": [float(x) for x in gmm.means_.flatten()[order]],
        "gmm_vars": [float(x) for x in gmm.covariances_.flatten()[order]],
        "gmm_weights": [float(x) for x in gmm.weights_.flatten()[order]],
        "hist_counts": [float(x) for x in counts],
        "hist_edges": [float(x) for x in edges],
        "fit_floor": fit_floor,
    }


def compute_stages() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Run the real blood_flood stages over the phantom; return arrays + diagnostics."""
    from nvitk.segmentation.blood_flood import (
        hysteresis_vessel_tree,
        intensity_vesselness,
        keep_tree_components_touching_markers,
        thin_tree_by_vesselness,
        watershed_labels_into_vessels,
    )

    phantom = build_phantom()
    cd = phantom["intensity"]
    markers = _markers(phantom)

    # Foreground gate for the GMM fit, exactly as qvtpy stage-4 builds it.
    cd_pos = cd[cd > 0]
    fg_mask = cd > float(np.percentile(cd_pos, 25.0))

    # 1 - Frangi vesselness.
    vesselness, vmode = intensity_vesselness(cd, sigmas=FRANGI_SIGMAS)
    vesselness = np.asarray(vesselness, dtype=np.float64)

    # 2 - GMM + hysteresis vessel tree.
    tree_hyst, tree_meta = hysteresis_vessel_tree(
        vesselness, fg_mask, low_factor=HYST_LOW_FACTOR, high_factor=HYST_HIGH_FACTOR
    )
    tree_hyst = np.asarray(tree_hyst, dtype=bool)
    lowt = float(tree_meta["lowt"])
    hight = float(tree_meta["hight"])

    # 3 - keep only components a seed touches.
    tree_cc, cc_meta = keep_tree_components_touching_markers(tree_hyst, markers)
    tree_cc = np.asarray(tree_cc, dtype=bool)
    dropped_cc = tree_hyst & ~tree_cc

    # 4 - hard barrier: dilated ICA/basilar.
    ica_core = phantom["ica"] > 0.5
    barrier = _dilate(ica_core, BARRIER_DILATE_RADIUS)
    tree_barrier = (tree_cc & ~barrier) | (markers != 0)
    removed_by_barrier = tree_cc & ~tree_barrier

    # 5 - vesselness thinning of the weak Frangi shell (seeds protected).
    tree_thin, thin_meta = thin_tree_by_vesselness(
        tree_barrier,
        vesselness,
        keep_percentile=THIN_KEEP_PERCENTILE,
        protect=(markers != 0),
    )
    tree_thin = np.asarray(tree_thin, dtype=bool)
    tree_final = (tree_thin & ~barrier) | (markers != 0)
    removed_by_thin = tree_barrier & ~tree_final

    # 6 - watershed the seeds into the tree.
    labels = np.asarray(
        watershed_labels_into_vessels(
            tree_final, markers, connectivity=3, erode_markers=False
        ),
        dtype=np.int32,
    )

    order = _geodesic_order(tree_final, markers)
    dist = ndi.distance_transform_edt(tree_final)

    # Voxels the animation actually draws: the bright CD ghost plus anything any
    # stage ever touches, so no state has to materialize a cube out of nothing.
    display = (cd > DISPLAY_THRESHOLD) | tree_hyst | (markers != 0) | barrier

    arrays = {
        "intensity": cd.astype(np.float32),
        "vesselness": vesselness.astype(np.float32),
        "fg_mask": fg_mask,
        "mask_high": vesselness > hight,
        "mask_low": vesselness > lowt,
        "tree_hyst": tree_hyst,
        "tree_cc": tree_cc,
        "dropped_cc": dropped_cc,
        "barrier": barrier,
        "ica_core": ica_core,
        "tree_barrier": tree_barrier,
        "removed_by_barrier": removed_by_barrier,
        "tree_final": tree_final,
        "removed_by_thin": removed_by_thin,
        "markers": markers,
        "labels": labels,
        "order": order,
        "dist": dist.astype(np.float32),
        "display": display,
        "vessel_core": phantom["vessel"] > 0.5,
    }
    meta: dict[str, object] = {
        "grid": list(GRID),
        "frangi_sigmas": list(FRANGI_SIGMAS),
        "vesselness_mode": vmode,
        "hyst_low_factor": HYST_LOW_FACTOR,
        "hyst_high_factor": HYST_HIGH_FACTOR,
        "thin_keep_percentile": THIN_KEEP_PERCENTILE,
        "barrier_dilate_radius": BARRIER_DILATE_RADIUS,
        "display_threshold": DISPLAY_THRESHOLD,
        "lowt": lowt,
        "hight": hight,
        "gmm_means_sorted": [float(x) for x in tree_meta["means"]],
        "thin_threshold": float(thin_meta["threshold"]),
        "n_cc_total": int(cc_meta["n_cc_total"]),
        "n_cc_kept": int(cc_meta["n_cc_kept"]),
        "n_markers": int(np.count_nonzero(markers)),
        "n_tree_hyst": int(np.count_nonzero(tree_hyst)),
        "n_tree_final": int(np.count_nonzero(tree_final)),
        "n_labeled": int(np.count_nonzero(labels)),
        "n_removed_barrier": int(np.count_nonzero(removed_by_barrier)),
        "n_removed_thin": int(np.count_nonzero(removed_by_thin)),
        "n_dropped_cc": int(np.count_nonzero(dropped_cc)),
        "n_display": int(np.count_nonzero(display)),
        "max_order": int(order.max()),
        "label_mca": LABEL_MCA,
        "label_aca": LABEL_ACA,
    }
    meta.update(_gmm_fit_for_plot(vesselness, fg_mask))
    return arrays, meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    arrays, meta = compute_stages()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, meta=np.array(json.dumps(meta)), **arrays)

    summary = {k: v for k, v in meta.items() if not isinstance(v, list)}
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
