# blood_flood distal-expansion animation

A Manim animation of the qvtpy distal vessel expansion — the
`nvitk.segmentation.blood_flood` pipeline that grows stage-3 proximal MCA/ACA/PCA
labels out into the distal branches.

Two scripts, deliberately decoupled:

| script | environment | job |
| --- | --- | --- |
| `blood_flood_precompute.py` | the nvitk env | builds a synthetic vessel phantom, runs the **real** `blood_flood` primitives over it, dumps every intermediate array to `blood_flood_stages.npz` |
| `blood_flood_manim.py` | a manim env (numpy only) | draws those arrays as a voxel cloud |

The animation therefore shows what the algorithm actually produces — the Frangi
response, the GMM thresholds, which connected components survive, which voxels
the barrier and the thinning remove — rather than a hand-drawn approximation.
The renderer never imports nvitk, so it can run wherever manim is installed.

## Stages shown

1. **Bright-blood volume** `I(x)` — arteries, veins and parenchymal blobs all bright.
2. **Frangi vesselness** `V(x)` over σ ∈ {0.5, 1.0, 1.5, 2.0, 2.5}.
3. **GMM + hysteresis tree** — `low = μ₂ + 3.5·σ₂`, `high = μ₃ + 0.5·σ₃`; components of
   `{V > low}` that touch `{V > high}`.
4. **Marker-connected components** — components touching no stage-3 seed are dropped
   (the vein and the blobs).
5. **Hard barrier** — dilated ICA/basilar, `tree ← (tree ∧ ¬barrier) ∨ seeds`.
6. **Vesselness thinning** — drop the weak Frangi shell below p55, seeds protected.
7. **Watershed flood** — seeds flood the binary tree on `−EDT`; the front never
   leaves the tree, and the shared distal field is split between labels.

## Running

The manim environment was created with conda-forge (it needs native pango/cairo,
which the PyPI wheels do not supply for Python 3.13):

```bash
mamba create -y -n manim -c conda-forge python=3.12 manim numpy scipy
```

Regenerate the stage arrays (only needed if the phantom or the algorithm changes):

```bash
python scripts/anim/blood_flood_precompute.py -o scripts/anim/blood_flood_stages.npz
```

Fast preview — decimates voxels and shortens every beat, ~1 min:

```bash
BF_FAST=1 mamba run -n manim manim -ql --fps 15 scripts/anim/blood_flood_manim.py BloodFloodDistalExpansion
```

Full render (1080p30, ~45 min — manim re-rasterises ~1800 semi-transparent cubes
every frame, and the ambient camera rotation means no frame can be cached):

```bash
mamba run -n manim manim -qh --fps 30 -o blood_flood_distal_expansion scripts/anim/blood_flood_manim.py BloodFloodDistalExpansion
```

`BF_STAGES=/path/to/stages.npz` overrides the input arrays.

## Tuning the phantom

The phantom is not just decoration: `hysteresis_vessel_tree` fits a GMM, so the
background population drives where the thresholds land. An unrealistically clean
phantom pushes `low` up near the vessel core, the tree collapses onto the seeds,
and there is no distal growth left to animate. Hence the padded grid (`OFFSET`),
the smooth parenchymal `TEXTURE_AMP`, and the deliberately short seeds
(`MCA_SEED_XMAX` / `ACA_SEED_XMIN`). Check `n_markers` vs `n_labeled` and
`max_order` in the precompute summary after any change — those are what make the
flood read as growth.
