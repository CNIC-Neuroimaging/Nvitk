# nvitk algorithm animations (Manim)

Three animations, each split in two scripts:

| script | environment | job |
| --- | --- | --- |
| `*_precompute.py` | the nvitk env | runs the **real** nvitk code and dumps every intermediate to an `.npz` |
| `*_manim.py` | a manim env (numpy only) | draws those arrays; never imports nvitk |

The animations therefore show what the algorithms actually produce, not a hand-drawn
approximation. `anim_voxels.py` holds the shared voxel-cloud renderer (`VoxelField`: colour,
opacity, stroke and size per cube, interpolated between states) and the caption helpers used by
the two newer scenes.

| animation | scene | data |
| --- | --- | --- |
| distal vessel expansion (`blood_flood`, qvtpy stage 4) | `BloodFloodDistalExpansion` | synthetic phantom |
| ICA siphon correction (`nvitk.morphology.siphon_cut`) | `SiphonMinCut` | PESA13727025, right ICA |
| eICAB post-process (qvtpy stage 1) | `EicabPostProcess` | PESA10061584 |

The two newer `.npz` files hold display blocks and small voxel sets derived from those subjects'
eICAB masks (no image volume, no identifiers beyond the subject id).

## The manim environment

Created with conda-forge (it needs native pango/cairo, which the PyPI wheels do not supply for
Python 3.13):

```bash
mamba create -y -n manim -c conda-forge python=3.12 manim numpy scipy
```

Manim writes into `./media` unless given `--media_dir`.

## ICA siphon correction — `siphon_cut_*.py`

Steps shown, all from one `repair_loops` run on PESA13727025's right ICA (β₁ = 1):

1. **A loop in the mask**: the siphon's limbs touch, and a skeleton centerline shortcuts across
   the contact (69.6 mm).
2. **Pretess**: edge- and corner-only contacts removed (7 voxels; β₁ still 1).
3. **Bridge and curl**: the skeleton cycle (19 mm) split at its anchors: bridge 5.7 mm, curl 12.1 mm.
4. **The contact**: where `(I / I_curl) · (r / r_curl)` dips along the loop (0.23, on the bridge).
5. **The cost field** `w = clip(I / I_ref) · (0.25 + core)`: lumen expensive, wall and contact cheap.
6. **The ring opened** opposite the contact, with the source and sink beside the opening.
7. **The minimum vertex cut**: 11 voxels (1.4 mm³), cost 0.07 × a lumen slice.
8. **Accepted**: β₁ 1 → 0, the ICA otherwise untouched (−0.2 % volume).
9. **The centerline** now follows the curl (77.2 mm).

`siphon_cut_precompute.py` wraps `loop_candidates`, `contact_run`, `voxel_weights` and
`_min_vertex_cut` for the run, so the loop, the contact, the costs and the cut are what the
library saw. The vessel is drawn as 1 mm blocks (cube size = how much of the block the mask
fills); the cut and the pretess voxels at 0.5 mm.

```bash
python scripts/anim/siphon_cut_precompute.py -o scripts/anim/siphon_cut_stages.npz   # nvitk env
SC_FAST=1 mamba run -n manim manim -ql --fps 15 scripts/anim/siphon_cut_manim.py SiphonMinCut     # ~5 min
mamba run -n manim manim -qh --fps 30 -o siphon_mincut scripts/anim/siphon_cut_manim.py SiphonMinCut
```

`--eicab-dir` / `--label` pick another subject or ICA; `SC_STAGES` overrides the input arrays.

## eICAB post-process — `eicab_pp_*.py`

Steps shown, all from one `postprocess_eicab_labels` run on PESA10061584:

0. **eICAB's labels** (16 vessels inside its Circle-of-Willis box).
1. **Anterior choroidals dropped.**
2. **Local half-maximum re-segmentation**: cubes shrink (5148 → 3314 mm³); inset: an axial slice
   of the basilar with eICAB's and the half-maximum outlines, and the profile across it with
   core, background and `T = bg + ½(core − bg)`.
3. **Trunks to the image floor**: the minimum-cost paths and the lumen around them (LICA 46.6 mm,
   RICA 36.6 mm, basilar 25 mm); inset: coronal MIPs before / after.
4. **Vertebral split** (LVA / RVA).
5. **Distal growth**: the grown voxels appear in the order their label's front reached them; then
   the ACA midline plane (anchored at the AComm, ±5 mm).
6. **ICA siphon correction**: the right ICA's cut.

`eicab_pp_precompute.py` wraps the post-process's steps and, inside them, `_extension_box`,
`_paths_to_floor` and `_compete`, for the run. The labels are drawn as 3 mm blocks (the most
frequent label of each block, size by occupancy), about 1,600 cubes.

```bash
python scripts/anim/eicab_pp_precompute.py -o scripts/anim/eicab_pp_stages.npz   # nvitk env
EP_FAST=1 mamba run -n manim manim -ql --fps 15 scripts/anim/eicab_pp_manim.py EicabPostProcess   # ~5 min
mamba run -n manim manim -qh --fps 30 -o eicab_postprocess scripts/anim/eicab_pp_manim.py EicabPostProcess
```

`--eicab-dir` / `--block` pick another subject or block size; `EP_STAGES` overrides the input
arrays. A step the subject does not need is skipped (no ACA pair → no midline plane; no loop →
no siphon step).

## Distal vessel expansion — `blood_flood_*.py`

A walk through `nvitk.segmentation.blood_flood` on a synthetic phantom:

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

```bash
python scripts/anim/blood_flood_precompute.py -o scripts/anim/blood_flood_stages.npz
BF_FAST=1 mamba run -n manim manim -ql --fps 15 scripts/anim/blood_flood_manim.py BloodFloodDistalExpansion
mamba run -n manim manim -qh --fps 30 -o blood_flood_distal_expansion scripts/anim/blood_flood_manim.py BloodFloodDistalExpansion
```

`BF_STAGES=/path/to/stages.npz` overrides the input arrays.

### Tuning the phantom

The phantom is not just decoration: `hysteresis_vessel_tree` fits a GMM, so the
background population drives where the thresholds land. An unrealistically clean
phantom pushes `low` up near the vessel core, the tree collapses onto the seeds,
and there is no distal growth left to animate. Hence the padded grid (`OFFSET`),
the smooth parenchymal `TEXTURE_AMP`, and the deliberately short seeds
(`MCA_SEED_XMAX` / `ACA_SEED_XMIN`). Check `n_markers` vs `n_labeled` and
`max_order` in the precompute summary after any change — those are what make the
flood read as growth.

## Render times

Manim re-rasterises every cube each frame, and the slow camera rotation means no frame can
be cached. At 1080p30: blood flood (~1,800 cubes) ~45 min; siphon (~1,460 cubes, ~70 s) and
eICAB post-process (~1,600 cubes, ~95 s) about an hour each. The `*_FAST=1` previews decimate the
cubes and shorten every beat.
