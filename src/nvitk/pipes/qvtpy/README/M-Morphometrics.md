# Morphometric metrics — qvtpy cheatsheet

Everything the **TOF geometry** side of the pipeline measures: vessel shape, caliber, tortuosity,
lesions and branching.

Produced by **stage 7** ([stage7_morphometrics.py](src/nvitk/pipes/qvtpy/stage7_morphometrics.py)),
which drives [nvitk.measure.morphometrics](src/nvitk/measure/morphometrics.py) over the eICAB
multilabel mask. Every threshold named below lives in one file:
[morphometrics_config.py](src/nvitk/measure/morphometrics_config.py).

```
eICAB WB/CW mask
   │  Taubin smoothing (20 iters, λ=0.65, μ=−0.65)
   │  MST gap bridging (same-label fragments ≤ 12 vox apart)
   ▼
per (label, connected component) job
   │  marching-cubes surface ─► VMTK centerlines ─► resample @ 0.10 mm
   ▼
per root→terminal path ──► geometry · radii · lesions
   │
   ├─► case_metrics_donut_tree.xlsx   (summaries + per-point sheets)
   ├─► centerlines/*.vtp, surfaces/*.vtp
   ├─► centerlines/tortuosity_metrics.{csv,xlsx}
   └─► radius_histograms/
```

---

## 0 · The three radius signals

Everything caliber-related depends on **which radius** you are reading. Three exist, and they are
not interchangeable.

| Signal | Column / VTP array | How it is computed | Used for |
|---|---|---|---|
| **Cross-section radius** | `radius_mm` / `CrossSectionRadius` | Cut the surface with the local normal plane, take the largest closed loop's area `A`, then `r = √(A/π)` | **All reported radius statistics**, taper, and the main enlargement columns |
| **Maximum inscribed sphere** | `maximum_inscribed_sphere_radius_mm` / `MaximumInscribedSphereRadius` | VMTK's MIS radius along the centerline | **Stenosis detection** (default `RADIUS_SOURCE_FOR_CALIBER_DETECTION`) |
| **EDT radius** | — | Spacing-aware distance transform sampled at the path voxels | Fallback wherever the cross-section cut fails |

`stenosis_detection_radius_mm` is the MIS signal with cross-section values patched in wherever MIS
is missing. Switching `RADIUS_SOURCE_FOR_CALIBER_DETECTION` to `"cross_section"` makes detection and
reporting use one signal throughout.

---

## 1 · Path geometry — sheet `00_Path_Summary`

One row per root→terminal centerline path. Source:
[metrics.py](src/nvitk/measure/morpho/metrics.py) + [centerlines.py](src/nvitk/measure/morpho/centerlines.py).

### Length & tortuosity

| Column | Formula | Unit |
|---|---|---|
| `length_mm` | Σ ‖pᵢ₊₁ − pᵢ‖ — arc length | mm |
| `chord_length_mm` | ‖p_last − p_first‖ | mm |
| `tortuosity_dm` | **DM** = `length / chord` — ≥ 1, exactly 1 when straight | — |

### Curvature & bending

| Column | Formula | Unit |
|---|---|---|
| `curvature_mean/median/p95/max_1_per_mm` | VMTK `Curvature` where available, else **Menger curvature** `κᵢ = ‖ab × ac‖ / (‖ab‖·‖ac‖·‖bc‖)` at each interior point | 1/mm |
| `torsion_1_per_mm` (per-point) | Frenet form `(r′×r″)·r‴ / ‖r′×r″‖²` from arc-length gradients; NaN where ill-conditioned or < 5 points | 1/mm |
| `inflection_count` | Sign flips of the local binormal (relative to a median reference direction) at points with smoothed `κ ≥ 0.02` | count |
| `bend_peak_count` | Local maxima of smoothed κ at or above `0.05` | count |

Smoothing for both counts: 7-point reflect-padded moving average (`INFLECT_SMOOTH_WIN`).

### Radius statistics

Computed on the **cross-section radius** (`radius_stats`):

| Column | Formula |
|---|---|
| `radius_mean_mm`, `radius_std_mm` | mean / std over finite points |
| `radius_cv` | `std / mean` — a high CV means caliber is unstable along the path |
| `radius_min_mm`, `radius_p05/p50/p95_mm` | order statistics |
| `taper_slope_mm_per_mm` | Slope of a linear fit of the **detection** radius vs arc length. Negative = the vessel narrows distally, as it should |

---

## 2 · Caliber lesions — stenosis & enlargement

The hard part is not the threshold, it is the **reference**: what *should* the radius be here?

### 2.1 The taper reference (`_compute_vessel_taper`)

Not a least-squares baseline — a **healthy-caliber envelope**, so a long stenosis cannot drag its
own denominator down.

| Step | Rule | Constant |
|---|---|---|
| 1. Exclude end zones from the *fit* | first/last 10 mm are not allowed to shape the envelope (but stay eligible for detection) | `STENOSIS_TAPER_FIT_EXCLUDE_END_MM` / `ENLARGEMENT_…` = 10.0 mm |
| 2. Exclude the siphon | points with smoothed `κ ≥ 0.10 mm⁻¹`, dilated ±5 mm in arc length — the ICA cavernous loop is anatomy, not pathology | `SIPHON_KAPPA_THRESHOLD` = 0.10, `SIPHON_DILATION_MM` = 5.0 |
| 3. Local high-percentile envelope | 85th percentile of radius inside a 20 mm sliding window | `TAPER_REFERENCE_PERCENTILE` = 85, `TAPER_REFERENCE_WINDOW_MM` = 20 |
| 4. Smooth | 4 mm moving average | `TAPER_REFERENCE_SMOOTH_MM` = 4.0 |
| 5. Force non-increasing | PAVA isotonic fit — a vessel does not widen distally | `TAPER_FIT_ENFORCE_NONINCREASING` = True |
| 6. Outlier iterations | Points deviating more than `0.60 ×` the lesion threshold are dropped from the fit and the envelope is recomputed, up to 3× (never below 45 % of the original fit points) | `TAPER_FIT_OUTLIER_FRACTION`, `TAPER_FIT_MAX_ITERATIONS`, `TAPER_FIT_MIN_HEALTHY_FRACTION` |
| 7. Two-pass re-fit | Detected lesion points are excluded from the fit and detection re-runs until the lesion mask stops changing, ≤ 5 iterations | `TAPER_TWO_PASS`, `TAPER_TWO_PASS_MAX_ITERATIONS` |

### 2.2 Detection

```
stenosis     pct(i) = (1 − r(i)/r_ref(i)) · 100        on the MIS detection radius
enlargement  pct(i) = (r(i)/r_ref(i) − 1) · 100        on the cross-section radius
```

A point is a **core** hit at `pct ≥ threshold`, a **support** hit at `pct ≥ support_threshold`.
Support runs are grown into segments; only those containing at least one core point survive
(hysteresis). Each surviving segment is then **trimmed back to its first and last core point**
before anything is reported.

That split is the design: the support contour decides *whether* a lesion is real and bridges small
internal dips, the core contour defines *what gets measured*. Accepting on the halo without
trimming would inflate `*_length_total_mm` and make "stenosis length" mean something other than
"length narrowed by at least 10 %".

| Parameter | Stenosis | Enlargement |
|---|---|---|
| Core threshold | **10 %** (`STENOSIS_THRESHOLD_PCT`) | **10 %** (`ENLARGEMENT_THRESHOLD_PCT`) |
| Support threshold | **5 %** = core × `STENOSIS_SUPPORT_FRACTION` (0.5) | **5 %** = core × `ENLARGEMENT_SUPPORT_FRACTION` (0.5) |
| Min segment length (on the **support** extent) | 3.0 mm | 5.0 mm, *or* ≥ `ENLARGEMENT_MIN_SUPPORT_LENGTH_MM` = **2.0 mm** |
| Reported extent | trimmed to the core contour | trimmed to the core contour |
| End exclusion (detection) | 2.0 mm | 3.0 mm |
| End exclusion (candidate arrays) | 5.0 mm | 5.0 mm |
| Max internal gap merged | 1.5 mm | 1.5 mm |
| Siphon suppresses detection | n/a | **No** (`SIPHON_SUPPRESSES_ENLARGEMENT_DETECTION = False`) |

> **Support is a fraction, not an absolute.** The support tier must sit *below* the core threshold,
> or the `min(support, core)` guard collapses the two contours and hysteresis silently stops
> working — which is what an absolute `25.0` did against a 10 % core before this was corrected.
> Deriving it as `core × fraction` makes the invariant structural: retuning the core threshold
> moves the support tier with it.

> **Length floors are in mm, never in points.** `ENLARGEMENT_MIN_SUPPORT_LENGTH_MM` is converted to
> a point count against each path's own sampling step (`points_for_length`). The previous
> point-count form meant 0.3 mm at the default 0.1 mm resampling — below TOF voxel size, and enough
> to make `ENLARGEMENT_MIN_LEN_MM` dead config.

### 2.3 What the corrected hysteresis changes

Measured on synthetic vessels at the pipeline's own 0.1 mm resampling.

**Lesions that were previously invisible are now found.** Both of these produced *zero* segments
before:

| Case | Span ≥ 10 % | Span ≥ 5 % | Before | After |
|---|---|---|---|---|
| Short core, broad shoulders | 1.3 mm | 3.1 mm | **not detected** | 1 segment, 19.4–20.6 mm |
| Two humps, sub-core dip between | 3.8 mm | 9.7 mm | **not detected** | 1 segment, 38.6–45.7 mm |

The second is the important one: 3.8 mm of genuine ≥ 10 % narrowing was reported as *nothing at
all*, because the core contour split it into two sub-3 mm pieces that each failed the length rule
independently. The support tier bridges the dip, the merged lesion clears 3 mm, and the reported
extent is still pinned to the core contour.

**Lesions that were already correct are unchanged.** On the earlier three-lesion fixture the output
is identical (2 segments at 16.1–23.5 and 54.5–62.2 mm, 13.90 mm total, 23.46 % peak) — the fix is
a rescue, not a re-scoring. A focal lesion whose ≥ 5 % footprint is still under 3 mm stays
rejected, which is the length rule doing its configured job at the edge of TOF resolution.

**Enlargement no longer reports noise.** On a vessel carrying one broad widening and one pinpoint
blip:

| | Before | After |
|---|---|---|
| Segments accepted | 2 — spans **4.20 mm** and **0.60 mm** | 1 — span **4.40 mm** |
| Sub-millimetre blips reported | 1 | 0 |

**The candidate diagnostic arrays are now distinct.** `StenosisCoreCandidate` vs
`StenosisSupportCandidate` went from byte-identical (163 vs 163 points) to genuinely different
(163 vs 228) — the support array now shows the halo the segment was grown through.

> ⚠️ `STENOSIS_SUPPORT_FRACTION`, `ENLARGEMENT_SUPPORT_FRACTION` and
> `ENLARGEMENT_MIN_SUPPORT_LENGTH_MM` are **working values pending cohort calibration** — defensible
> defaults (half the core threshold; a 2 mm floor that keeps focal saccular aneurysms while
> rejecting sub-voxel blips), not validated ones. Re-run a cohort before trusting lesion counts.

### 2.4 Reported severity

The moving reference is right for *finding* lesions and confusing for *reporting* them (two points
in one lesion would be scored against different denominators). So each accepted segment gets one
reference: the **maximum** envelope value within the segment ±5 mm
(`STENOSIS_SEGMENT_REFERENCE_MARGIN_MM`).

| Column | Meaning |
|---|---|
| `stenosis_percent_max` / `degree_of_stenosis_pct` | Peak severity over accepted segments (identical values) |
| `stenosis_segments_n` | Number of accepted segments |
| `stenosis_length_total_mm` | Arc length covered by flagged points |
| `radius_min_stenotic_mm` | Smallest detection radius inside a flagged segment |
| `radius_ref_mm` | Mean of the taper reference over the path (context, not a denominator) |
| `stenosis_segments_point_idx` | JSON `[(start, end), …]` point-index ranges |
| `stenosis_segments_detail_json` | JSON per segment: index range, arc-length span, peak percent |
| `enlargement_percent_max`, `enlargement_segments_n`, `enlargement_length_total_mm`, `radius_max_enlarged_mm`, `enlargement_segments_*` | Mirror columns for enlargement |

### 2.5 Per-point arrays (vessel sheets + VTP)

Every path also exports a per-point table. Useful when a summary number looks wrong and you need to
see *where* it came from:

| Column / VTP array | Meaning |
|---|---|
| `s_mm` | arc length |
| `radius_mm`, `diameter_mm` | cross-section radius, ×2 |
| `maximum_inscribed_sphere_radius_mm`, `stenosis_detection_radius_mm` | the other two radius signals |
| `StenosisReferenceRadius` / `EnlargementReferenceRadius` | the taper envelope |
| `StenosisThresholdRadius` | `0.90 × reference` — the line the radius must fall below |
| `EnlargementThresholdRadius` | `1.10 × reference` |
| `stenosis_raw_percent_point` | raw deviation, **before** segment acceptance |
| `stenosis_core_candidate_point` / `…_support_candidate_point` | candidates at the 10 % and 5 % contours (5 mm end exclusion). The support array is the halo the segment was grown through before being trimmed back |
| `stenosis_percent_point`, `is_stenotic` | reported severity and final binary flag |
| `enlargement_percent_point`, `is_enlarged` | same for enlargement |
| `EnlargementBinaryCS` / `EnlargementBinaryMIS` | enlargement re-run on each radius signal (VTP only) |
| `SiphonMask` | which points were treated as siphon |
| `curvature_1_per_mm`, `torsion_1_per_mm` | pointwise geometry |

---

## 3 · Branchpoints — sheet `02_Branchpoints`

From `compute_branchpoint_metrics`. Parent/daughter roles are assigned by Dijkstra distance from the
tree root; radii come from the **EDT**, sampled one skeleton node away from the junction.

| Column | Formula | Reads as |
|---|---|---|
| `n_daughters` | count of neighbours further from the root | 2 = ordinary bifurcation |
| `parent_radius_edt_mm` | EDT radius at the parent-side probe | mm |
| `daughter_radii_edt_mm_json` | list of daughter EDT radii | mm |
| `daughter_parent_radius_ratios_json` | `r_d / r_p` per daughter | — |
| `daughter_area_ratio_sum_r2_over_parent_r2` | `Σ r_d² / r_p²` — the **area-conservation ratio** | 1.0 = area preserved; > 1 = daughters over-segmented or parent under-segmented |
| `daughter_pair_angles_deg_json`, `…_min_deg`, `…_max_deg` | Pairwise angles between daughter direction vectors, estimated 5 skeleton steps out | degrees |

---

## 4 · Tree structure — sheet `01_Tree_Summary`

One row per (label, connected component). This is the *skeleton bookkeeping* sheet — use it to tell
"this vessel is genuinely short" from "the pipeline threw its branches away".

| Column | Meaning |
|---|---|
| `n_skeleton_voxels`, `n_endpoints`, `n_branchpoints` | raw graph size |
| `n_terminals` | endpoints reachable as path targets |
| `unique_skeleton_graph_length_mm` | total skeleton length counting each edge **once** |
| `n_centerline_paths` | accepted root→terminal paths |
| `centerline_paths_total_length_with_shared_trunks_mm` | Σ path lengths — **larger than the graph length**, because sibling arms re-walk the shared trunk |
| `n_centerline_paths_discarded_short` | paths under `MIN_CENTERLINE_PATH_LENGTH_MM` = 7.5 mm |
| `n_centerline_paths_discarded_spurious_arm` | tree arms under `MIN_TREE_ARM_LENGTH_MM` = 4 mm (10 mm for LICA/RICA/BA) |
| `n_centerline_paths_discarded_overlap`, `…_overlap_trimmed` | duplicate-trunk pruning bookkeeping |

Terminal spurs shorter than `PRUNE_SPUR_LENGTH_MM` = 2.0 mm are pruned from the skeleton first.

---

## 5 · Standalone tortuosity workbook

`centerlines/tortuosity_metrics.{csv,xlsx}`, from
[compute_tortuosity_metrics.py](src/nvitk/measure/morpho/export_utils/compute_tortuosity_metrics.py).
This is the **literature-convention** set, computed independently of `00_Path_Summary` on the
exported VTPs (resampled to 0.2 mm first).

| Column | Formula | Reads as |
|---|---|---|
| `tortuosity_index` | `length / chord` | Same definition as `tortuosity_dm`, different resampling — expect close but not identical values |
| `bend_count` | RDP bend events (see below) | number of real bends |
| `inflection_count_metric` (**ICM**) | `bend_count × tortuosity_index` | penalises many bends *and* long detours |
| `sum_of_angles_metric_deg_per_mm` (**SOAM**) | `Σ bend deflection angles / length` | total turning per mm — separates a tight coil from a long gentle curve |
| `bend_angle_sum_deg` | Σ deflection angles at bend apices | degrees |
| `total_curvature_deg_per_mm` | `Σ all turn angles / length` | includes sampling noise; SOAM is the cleaner sibling |
| `bending_length_mm` | Largest bend lobe apex distance | mm |
| `max_bending_index` | Point index of maximum perpendicular distance from the path chord | — |

**Bend detection (RDP).** Recursive Ramer–Douglas–Peucker: at each split the point farthest from its
local chord is a bend if its deviation exceeds `max(0.50 mm, 0.07 × local chord length)`
(`BEND_MIN_TOLERANCE_MM`, `BEND_TOLERANCE_FRACTION`). Candidates closer than 10 % of the total path
length are clustered, keeping the most prominent.

**Inflection detection.** Sign changes of a signed 3-D curvature (stabilised by an SVD-chosen
reference plane normal, Gaussian-smoothed with σ = 2 mm), merging inflections closer than
`MIN_INFLECTION_LOBE_MM` = 2 mm.

---

## 6 · Aggregates

### `03_LR_Asymmetry` — one row per vessel pair

Left and right sides are aggregated separately (sum for `length_mm`, min for `radius_min_mm`, max
for the lesion percentages, **length-weighted mean** for everything else), then compared:

```
AI    = (L − R) / ((L + R)/2)        symmetric asymmetry index, 0 = symmetric
ratio = L / R
```

Applied to: `length_mm`, `tortuosity_dm`, `radius_mean_mm`, `radius_p50_mm`, `radius_min_mm`,
`curvature_mean_1_per_mm`, `curvature_median_1_per_mm`, `stenosis_percent_max`,
`enlargement_percent_max`.

### `05_Hemisphere` — one row per side

Rolls up **every** path on a side (not paired):

| Column | Reducer |
|---|---|
| `length_total_mm` | sum |
| `tortuosity_mean`, `tortuosity_p90` | length-weighted mean / weighted percentile |
| `radius_p10/p50/p90_mm` | length-weighted percentiles of `radius_p50_mm` |
| `stenosis_segments_n`, `stenosis_length_total_mm` | sum |
| `stenosis_degree_max_pct` | max |
| `stenosis_degree_p90_pct` | 90th percentile over **non-zero** stenoses only |
| `enlargement_*` | same four reducers |

---

## 7 · What reaches the database

[morpho_db_publish.py](src/nvitk/pipes/qvtpy/common/morpho_db_publish.py) aggregates
`00_Path_Summary` into **one row per `vessel_name`** and writes to `image_measurements` under
modality `tof`, pipeline `tof_morpho`.

| `variable_id` | Unit | Aggregation across the vessel's paths |
|---|---|---|
| `length_mm` | mm | length-weighted mean |
| `radius_mean_mm` | mm | length-weighted mean |
| `radius_max_mm` | mm | max of `radius_p95_mm` |
| `tortuosity_dm` | — | length-weighted mean |
| `curvature_mean_1_per_mm`, `curvature_p95_1_per_mm` | 1/mm | length-weighted mean |
| `stenosis_percent_max` | % | max |
| `stenosis_segments_n` | count | sum |
| `stenosis_length_total_mm` | mm | sum |
| `radius_min_stenotic_mm` | mm | min |
| `enlargement_percent_max` | % | max |
| `enlargement_segments_n` | count | sum |
| `enlargement_length_total_mm` | mm | sum |
| `radius_max_enlarged_mm` | mm | max |

Paths with a non-positive or missing `length_mm` are dropped before weighting.

---

## 8 · Knobs

Stage-7 CLI:

| Flag | Default | Effect |
|---|---|---|
| `--eicab-mask-preference` | `wb` | whole-brain vs circle-of-Willis eICAB mask |
| `--use-postprocessed-mask/--no-…` | on | prefer `*_eICAB_CW_pp.nii.gz` |
| `--input-already-smoothed` | off | skip Taubin smoothing |
| `--n-workers` | ≤4 | parallel (label, component) jobs — VMTK is memory-hungry |
| `--skip-existing` | off | skip when `case_metrics_donut_tree.xlsx` exists |

The metric-changing constants are **not** CLI flags; edit
[morphometrics_config.py](src/nvitk/measure/morphometrics_config.py) or pass a `MorphometricsConfig`.
The ones that move numbers the most:

| Constant | Default | Changes |
|---|---|---|
| `CENTERLINE_RESAMPLE_STEP_MM` | 0.10 | point density → curvature, torsion, every per-point array |
| `RADIUS_SOURCE_FOR_CALIBER_DETECTION` | `maximum_inscribed_sphere` | which radius stenosis is judged on |
| `STENOSIS_THRESHOLD_PCT` / `ENLARGEMENT_THRESHOLD_PCT` | 10 / 10 | lesion sensitivity — the support tier follows automatically |
| `STENOSIS_SUPPORT_FRACTION` / `ENLARGEMENT_SUPPORT_FRACTION` | 0.5 / 0.5 | how far the halo reaches below the core; **must stay < 1.0** |
| `STENOSIS_MIN_LEN_MM` / `ENLARGEMENT_MIN_LEN_MM` | 3.0 / 5.0 | shortest accepted lesion, on the support extent |
| `ENLARGEMENT_MIN_SUPPORT_LENGTH_MM` | 2.0 | alternative floor that keeps focal aneurysms; `None` disables |
| `TAPER_REFERENCE_PERCENTILE` / `_WINDOW_MM` | 85 / 20 | how generous the "healthy" reference is |
| `SIPHON_KAPPA_THRESHOLD` / `_DILATION_MM` | 0.10 / 5.0 | how much of the ICA loop is excused |
| `MIN_CENTERLINE_PATH_LENGTH_MM` | 7.5 | which paths exist at all |

---

## See also

- [M-Hemodynamics.md](M-Hemodynamics.md) — the 4D-flow side (stage 6)
- [M-AutoQC.md](M-AutoQC.md) — automatic quality control (stage 9)
