# nvitk algorithm animations (Manim)

Seven animations, each split in two scripts:

| script | environment | job |
| --- | --- | --- |
| `*_precompute.py` | the nvitk env | runs the **real** nvitk code and dumps every intermediate to an `.npz` |
| `*_manim.py` | a manim env (numpy only) | draws those arrays; never imports nvitk |

The animations therefore show what the algorithms actually produce, not a hand-drawn
approximation. The four newer ones (below, "Dense GPU animations") draw their 3D content on the GPU
and need `moderngl` in the manim environment as well. `anim_voxels.py` holds the shared voxel-cloud renderer (`VoxelField`: colour,
opacity, stroke and size per cube, interpolated between states) and the caption helpers used by
the two newer scenes.

| animation | scene | data |
| --- | --- | --- |
| distal vessel expansion (`blood_flood`, qvtpy stage 4) | `BloodFloodDistalExpansion` | synthetic phantom |
| ICA siphon correction (`nvitk.morphology.siphon_cut`) | `SiphonMinCut` | PESA13727025, right ICA |
| eICAB post-process (qvtpy stage 1) | `EicabPostProcess` | PESA10061584 |

The two newer `.npz` files hold display blocks and small voxel sets derived from those subjects'
eICAB masks (no image volume, no identifiers beyond the subject id).

## Dense GPU animations (eICAB, morphometrics, qvtpy pipeline, hemodynamics)

Four outreach animations of the TOF morphometrics and 4D-flow hemodynamics pipelines, on the
**stenosis example** (`/home/imarcoss/placeholder/animations/stenosis example`, PESA1108809: a 26 %
stenosis of the left ICA). Two steps use the **fetal example** (`fetal example`, PESA2140369)
instead, because the stenosis subject cannot show them: the circle-of-Willis variants (absent right
A1, thread-like right P1, fetal right PCA) and the siphon min-cut (its siphons close no loop).
Unlike the Cairo scenes above (3 mm display blocks, ~1.1 s per frame for 1,600 cubes), they draw
every voxel at its own grid: 0.5 mm TOF voxels (~90,000 cubes), 0.78 mm 4D-flow voxels, surfaces,
wireframes, centerline tubes, ~16,000 velocity arrows, 5,000 particles with trails, live
cross-section maps and ray-marched MIPs of the TOF / 4D-flow angiograms, at a few ms per 1080p frame.

| animation | scene | what it shows |
| --- | --- | --- |
| `eicab_full` | `EicabFull` | TOF MIP → eICAB's circle-of-Willis labels in their box → the post-process: half-maximum re-segmentation (the mask peels from the outside in; the AChAs go with it), ICAs to the floor along the minimum-cost paths, the distal growth (Frangi tree, competing fronts in arrival order, written as A2 / M2), then quickly on one isolated ICA (fetal example) the siphon min-cut (loop, pretess, bridge / contact, cost field, opened ring, cut, β₁ 1 → 0), WB_pp vs CW_pp |
| `morpho` | `TofMorphometrics` | stage 7 in general: voxels → a surface per vessel as points + wireframe (close-up on the ICA) → VMTK centerlines → the two radii along the LICA (inscribed sphere rolling, six cross-sections with their 36 rays, live profile) → length / tortuosity / curvature / siphon mask → the stenosis on its path with the robust reference ±20 % band → circle-of-Willis variants (fetal example) → per-vessel table |
| `qvtpy_pipeline` | `QvtpyPipeline` | stages 2-5: TOF and 4D-flow angiograms misaligned → the rigid FLIRT transform applied gradually → the labels in the 4D-flow grid with their centerlines and the named sinuses → stage 4: local threshold, trunk region growing, the blood flood (tree + watershed wave), siphon cut → stage 5 LOC planes |
| `hemo` | `FlowHemodynamics` | stage 6 in general: velocity arrows pulsing through the cardiac cycle (cardiac clock) → particle pathlines → one LOC: live through-plane speed map + arrows, Q(t), PI / RI → all LOCs + mean-flow chart → dense stations coloured by PI, quality gate, PITC fit → the pulse wave travelling along the left ICA tree, beside its PWV plot: each station's cross-correlation delay vs distance (quality-excluded grey, MAD outliers open) and the area-weighted fit `pwv_fielding_m_s` |

The stenosis example ships only its `eicab/` outputs. The qvtpy stages 2-6 were run locally for
the animations (`python -m nvitk.pipes.qvtpy.run --stages stage2,stage3,stage4,stage5,stage6
--subjects PESA1108809 --nifti-root ~/DATA/PESA-Brain/Morphometrics/DATA --output-root <scratch>
--submit local --backend cpu`, with the example's `eicab/` copied under `<scratch>/PESA1108809/`),
and the pipeline / hemodynamics precomputes take a `--root` holding `eicab/`, `4DFlow/` and
`qvtpy/stage*` (symlinks are fine).

Shared modules: `gl_render.py` (headless moderngl/EGL renderer: instanced cubes, spheres and
arrows, meshes, tubes revealed along their length, textured quads, volume MIP; opaque first, then
translucent geometry sorted back-to-front), `gl_scene.py` (`GLScene`: a manim `Scene` whose
camera background is the GPU frame, orbit camera, `voxel_transition` with delayed waves and glow,
3D-anchored labels, the same captions and `--no-text` behaviour as `CaptionScene`), and
`vessel_palette.py` (eICAB and qvtpy label colours, ramps).

What each precompute runs (nvitk environment; `PYTHONPATH=src` if the env holds a packaged nvitk):

- `eicab_full_precompute.py`: `postprocess_eicab_variants` once (CW + WB masks), recording each
  step, the trunk paths, the distal tree / owner / arrival cost; and for the siphon step
  (`--siphon-eicab-dir`, the fetal example) its post-process up to the siphon correction, whose WB
  mask goes through `siphon_cut_precompute.compute` at 0.5 mm voxels (the same cut stage 1 made).
- `morpho_precompute.py`: `run_morphometrics_case` on `TOF_eICAB_WB_pp` (as stage 7; 5-15 min with
  VMTK, cached in `--morpho-dir`), surfaces, point tables, six LICA cross-sections cut as
  `compute_cross_section_radius` cuts them (their ray-median radii match the point table), the
  most severe stenosis; and the variants subject (`--variants-mask`, `--variants-morpho-dir`).
- `qvtpy_pipeline_precompute.py`: the FLIRT matrix as a world transform
  (`flirt_matrix_to_world`), stage 3's outputs, and `build_seg_4dflow_local` re-run with the
  example's own settings, recorded per step (it reproduces the shipped `seg_4dflow` exactly).
- `hemo_precompute.py`: velocities via `velocity_mm_s_from_phases`, `measure_loc_row` per LOC
  (matches the shipped `loc_measurements.csv`), `compute_vessel_hemodynamics(collect_plot_data=True)`
  (matches `vessel_hemodynamics.csv`; the cross-correlation PWV is refitted from its plot data with `pwv_fielding_xcor`'s filters and reproduces `pwv_fielding_m_s`), the through-plane speed on each LOC plane, and particle
  pathlines. The pathlines are a visualisation aid, not a stage-6 output: RK2 through the
  time-resolved field, with a gentle pull towards the lumen centre near the wall so particles
  survive the bends the 0.8 mm grid resolves coarsely (the precompute's docstring gives the numbers).

```bash
# nvitk environment
python scripts/anim/eicab_full_precompute.py -o scripts/anim/eicab_full_stages.npz        # ~1 min
python scripts/anim/morpho_precompute.py -o scripts/anim/morpho_stages.npz --morpho-dir /tmp/morpho_sten \
    --variants-morpho-dir /tmp/morpho_fetal
python scripts/anim/qvtpy_pipeline_precompute.py -o scripts/anim/qvtpy_pipeline_stages.npz --root <root>  # ~1 min
python scripts/anim/hemo_precompute.py -o scripts/anim/hemo_stages.npz --root <root>                     # ~1 min

# manim environment (needs moderngl and an EGL-capable GPU driver)
EF_FAST=1 python scripts/anim/eicab_full_manim.py -ql --fps 15      # preview, ~20 s
python scripts/anim/eicab_full_manim.py -qh --fps 30                # 1080p -> eicab_full.mp4
python scripts/anim/eicab_full_manim.py -qk --fps 30                # 4K
python scripts/anim/morpho_manim.py -qh --fps 30                    # -> tof_morphometrics.mp4
python scripts/anim/qvtpy_pipeline_manim.py -qh --fps 30            # -> qvtpy_pipeline.mp4
python scripts/anim/hemo_manim.py -qh --fps 30                      # -> flow_hemodynamics.mp4
```

Fast previews: `EF_FAST`, `MO_FAST`, `QP_FAST`, `HE_FAST`; other inputs: `EF_STAGES`,
`MO_STAGES`, `QP_STAGES`, `HE_STAGES`. `--no-text` drops the explanatory text (captions, title
cards, plot titles, 3D name tags, step notes, formulas) but, unlike the Cairo scenes, keeps what the
picture needs to be read: plots with their axes, colour keys and legends, the flow chart, the
per-vessel table, the re-segmentation inset and the siphon's β₁ counter (`GLScene.fixed(keep=True)`).
Both versions stay frame for frame in step. The four
`.npz` files are git-ignored: they hold image volumes (TOF, CD, velocities) and are 5-20 MB;
regenerate them with the precomputes. The example subject's identifier is shown nowhere.

## PESA-Health: Dixon MRI and CT-PET (`pesa_dixon_*.py`, `pesa_ctpet_*.py`)

Two outreach animations of the PESA-Health v5 pipelines (`nvitk.pipes.pesa_health`), stages 1-3, on one
Visit-5 subject (PESA1993744, batch 202603_Week4: both a Dixon and a CT-PET, segmented by the
current stage 1 with `tissue_4_types`, the WATER-contrast liver and the leg skeleton). Same GPU
renderer as above; red = the eroded measurement / SUV mask, blue = its uneroded volume twin (the QC
report's convention); colours in `pesa_palette.py`.

| animation | scene | what it shows |
| --- | --- | --- |
| `pesa_dixon` | `PesaDixon` | the three Dixon stations (neck and shoulders, abdomen, thighs) with their FAT / WATER / fat-fraction / T2* slices → stage 1 TotalSegmentator classes → the measured structures → the liver from the WATER run (gained / lost against the FAT run) → its vessels out (T2* > 50 ms) → the renal pelvis (convex hull minus kidney, grown twice) out → L3 / L4 eroded to the bone marrow → the femur (grown) out of the quadriceps, on a coronal cut → the one-voxel measurement erosion → voxels coloured by fat fraction → FF / T2* / volume chart, liver water fraction |
| `pesa_ctpet` | `PesaCtPet` | CT bone MIP and PET MIP → stage 1 classes → visceral and subcutaneous fat, the arms and legs out → the organs cleared with their margins (coronal cut) → the ureter from the PET → the L3-L4 slab (FAT_BATCH) → the intermuscular fat split into legs / trunk at the quadriceps' slices, the neighbour vote → organs and hemisphere-split muscles, the skeleton → L3 / L4 bone marrow at the CT grid (isotropic then in-plane erosion; the < 400 HU volume twin) → the SUV erosion against motion → the last PET bed: the quadriceps cut 41 mm below the leg-bed centre (noise inset: model and measured, the cut slices staying in MUSCLES_VOL, the SUV changes) → voxels coloured by SUV → SUVmean / SUVmax / volume chart |

- `pesa_dixon_precompute.py` rebuilds stage 2 with `dixon_v5.stage2_postprocess`'s own functions
  (NumPy backend), recording each step at the Dixon grid (2 × 2 × 3 mm, ~770,000 voxels), and
  checks the result against `_process` run into a temporary folder (0 voxels differ on all six
  masks) and the means against `stage3_measure.process_subject` (matches the subject's Excel row).
- `pesa_ctpet_precompute.py` runs the real `ct_pet_v5.stage2_postprocess._process` (CuPy) with
  recorders wrapped around its steps (`_organs_mask`, `_with_dilated_organs`,
  `_remove_extremities`, `_process_bladder`, `_neighbour_majority`, `split_lr_by_cc`,
  `_subtract_skeleton`, `trim_last_pet_bed`), then `stage3_measure.process_subject` for the chart
  and the quadriceps SUVs with and without the last-bed trim. The trim needs the PET bed geometry
  (`PT_beds.json` in the NIfTI folder, else the PET DICOM under `--dicom-root`; stage 2 then writes
  `PT_beds.json` into the NIfTI folder, so use a copy when that folder must stay untouched). The CT grid is pooled
  3 × 3 × 2 for the voxels (~1.35 M; a block takes its most frequent label so thin fat layers
  stay); the bone marrow stays at the CT grid; the SUV of each voxel is sampled from the PET (bw).
  At that pooling the skeleton subtraction from the muscles changes 2 voxels, so the scene shows
  the skeleton and says it rather than animating it.

Both default to the cluster layout (`/BIOIT_IMAGE/PESA_Fat/DATA/Visit-5-DIXON_PET-CT`, its `NIFTI`
and `RESULTS`); a local copy of the subject's NIfTI and stage-1 folders (~1 GB) works with
`--nifti-root` / `--results-root`.

```bash
# nvitk environment (PYTHONPATH=src if the env holds a packaged nvitk)
python scripts/anim/pesa_dixon_precompute.py -o scripts/anim/pesa_dixon_stages.npz --nifti-root <NIFTI> --results-root <RESULTS>  # ~15 s
python scripts/anim/pesa_ctpet_precompute.py -o scripts/anim/pesa_ctpet_stages.npz --nifti-root <NIFTI> --results-root <RESULTS> --dicom-root <DICOM>  # ~75 s

# manim environment
PD_FAST=1 python scripts/anim/pesa_dixon_manim.py -ql --fps 15     # preview
python scripts/anim/pesa_dixon_manim.py -qh --fps 30               # -> pesa_dixon.mp4
python scripts/anim/pesa_ctpet_manim.py -qh --fps 30               # -> pesa_ctpet.mp4
python scripts/anim/pesa_ctpet_manim.py -qh --fps 30 --no-text     # -> pesa_ctpet_notext.mp4
```

`GLScene` turns manim's partial-movie cache off: it hashes the mobjects only, so after a change to
the 3D state alone a cached render would splice in stale clips.

## The manim environment

Created with conda-forge (it needs native pango/cairo, which the PyPI wheels do not supply for
Python 3.13):

```bash
mamba create -y -n manim -c conda-forge python=3.12 manim numpy scipy
```

Manim writes into `./media` unless given `--media_dir`.

## Commented and text-free versions

The two newer scenes render either way: with every step commented (captions, title card,
legends, the β₁ badge, the 2D insets), or as the bare animation with `--no-text`. In the text-free
version those elements are still there but fully transparent, so both versions have the same
length and stay frame for frame in step (one can be cut against the other, or voiced over).

Run the scene script itself with the manim environment's Python: it takes manim's own options
plus `--no-text`, and names the output `<name>.mp4` / `<name>_notext.mp4`:

```bash
mamba run -n manim python scripts/anim/siphon_cut_manim.py -qh --fps 30             # siphon_mincut.mp4
mamba run -n manim python scripts/anim/siphon_cut_manim.py -qh --fps 30 --no-text   # siphon_mincut_notext.mp4
```

With the plain `manim` command (which rejects unknown options), set `ANIM_NO_TEXT=1` instead.

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
SC_FAST=1 mamba run -n manim python scripts/anim/siphon_cut_manim.py -ql --fps 15    # preview, ~5 min
mamba run -n manim python scripts/anim/siphon_cut_manim.py -qh --fps 30 [--no-text]
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
EP_FAST=1 mamba run -n manim python scripts/anim/eicab_pp_manim.py -ql --fps 15    # preview, ~5 min
mamba run -n manim python scripts/anim/eicab_pp_manim.py -qh --fps 30 [--no-text]
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
be cached. At 1080p30: blood flood (~1,800 cubes) ~45 min; siphon (~1,460 cubes, 58 s of video)
and eICAB post-process (~1,600 cubes, 64 s) about 35 min each (both can render in parallel). The `*_FAST=1` previews decimate the
cubes and shorten every beat.

## PET bed edge (PESA-Health CT-PET) — `pet_bed_edge_*.py`

`PetBedEdge` (Cairo, 2D, ~1 min to render at 1080p): why the last slices of a whole-body PET cut
through the thighs are noisy, and the proposed (not implemented) per-subject trim of the
quadriceps SUV mask. Beds from `FrameReferenceTime`, bed centres from the Philips `(7053,10D5)`
plateaus, the triangular sensitivity model, measured noise vs `sqrt(0.78 / sens)`, the measure
(red) and volumetry (blue) masks, the 41 mm rule and its effect, and why the number of removed
slices changes between subjects. Spanish text. The `.npz` holds PET/CT slices (git-ignored).

```bash
python scripts/anim/pet_bed_edge_precompute.py --root ~/placeholder --subject PESA1256641   # nvitk env, ~15 s
python scripts/anim/pet_bed_edge_manim.py -qh --fps 30            # manim env -> pet_bed_edge.mp4
python scripts/anim/pet_bed_edge_manim.py -qh --fps 30 --no-text  # -> pet_bed_edge_notext.mp4
```

Render the two versions one after the other: in parallel they race on manim's text cache.
