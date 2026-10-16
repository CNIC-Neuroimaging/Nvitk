# Morphometric metrics — qvtpy cheatsheet

Everything the **TOF geometry** side of the pipeline measures: vessel shape, caliber, tortuosity,
lesions and branching.

Produced by **stage 7** ([stage7_morphometrics.py](src/nvitk/pipes/qvtpy/stage7_morphometrics.py)),
which drives [nvitk.measure.morphometrics](src/nvitk/measure/morphometrics.py) over the eICAB
multilabel mask. The algorithm is a port of **cow_morpho**, whose parameters were tuned on the
PESA TOF cohort's eICAB **circle-of-Willis (CW)** masks. Every parameter lives in a named profile in
[morphometrics_config.py](src/nvitk/measure/morphometrics_config.py).

```
eICAB CW mask (stage-7 default; WB needs its own profile, not yet tuned)
   │  Taubin smoothing (20 iters, λ=0.65, μ=−0.65)
   │  largest connected component per label (no gap bridging — cow parity)
   ▼
per label job
   │  marching-cubes surface ─► centerlines (VMTK by default) ─► resample @ 0.10 mm
   ▼
per root→terminal path ──► geometry · radii · lesions
   │  split into non-overlapping connected segments
   ▼
├─► vessel_summary.csv · segments.csv · lesions.csv · subject_summary.csv   (compact tables)
├─► case_metrics_donut_tree.xlsx   (compact sheets + cow-style sheets + per-point sheets)
├─► points/<path_id>.csv           (per-point tables)
├─► centerlines/*.vtp, surfaces/*.vtp   (scanner RAS)
├─► centerlines/tortuosity_metrics.{csv,xlsx}, radius_histograms/
└─► morphometrics_provenance.json  (profile, backend, parameters, versions, git revision)
```

---

## 0 · Profiles and centerline backends

### Profiles

A run's parameters are a **profile**, a complete parameter set with a name. They are applied
for the duration of the run and in every worker process:

```python
from nvitk.measure.morphometrics import run_morphometrics_case
from nvitk.measure.morphometrics_config import morphometrics_config

cfg = morphometrics_config("eicab_cw", n_workers=8).with_params(SIPHON_LEGACY_MASK=True)
run_morphometrics_case(seg, out_dir, config=cfg)
```

| Profile | Masks | Status |
|---|---|---|
| `eicab_cw` (default) | eICAB circle-of-Willis crops | cow_morpho's tuned values, plus the calibrated siphon fix |
| *(WB)* | eICAB whole-brain masks | pending: needs its own tuning (longer vessels, more distal branches) |

`with_params(NAME=value)` validates names, so a typo fails instead of being silently ignored.
Every parameter listed in a profile is read by the algorithm; dead knobs were removed.

### Centerline backends (`CENTERLINE_BACKEND`, `--centerline-backend`)

| Backend | Centerline | Inscribed radius (MIS) |
|---|---|---|
| `vmtk` (default) | cow_morpho's exact call: `vmtkCenterlines` (point-list seeds, Voronoi) + `vmtkCenterlineGeometry` | VMTK's `MaximumInscribedSphereRadius` |
| `native` | voxel-skeleton path, **re-centred** on the lumen (below) | distance to the surface |
| `raycast` | EXPIGEO ray-cast skeleton (experimental) | ray-cast radius |

Requesting `vmtk` without VMTK installed is an error, never a silent fallback.

**Native centring.** Skeleton voxels sit off-centre: their inscribed radius was 12–16 % below
VMTK's. Each point is moved, iteratively, to the centroid of the lumen cross-section in the plane
normal to the path, then climbs the inscribed-radius ridge, with each move capped near junctions.
The path is then smoothed. On PESA17808400 this brings the native MIS within 0.7–3 % of VMTK's, and
lengths within −1…+3 %. The same centring runs when a GUI tool uses the `native` method.

---

## 1 · Parity with cow_morpho

| Aspect | Status |
|---|---|
| Centerline call, resampling, seeds, trimmed-seed retries | identical (VMTK backend) |
| Taper reference, two-pass refit, outlier iterations | identical |
| Stenosis / enlargement detection rules (§5.2) | identical — cow's code, reproduced to the index by a test against cow's own module |
| Overlap pruning of shared trunks before detection | identical (`PRUNE_OVERLAPPING_FINAL_CENTERLINE_PREFIXES`, no kept prefix) |
| Fragments | identical: largest component per label, no gap bridging (`PROCESS_ALL_CONNECTED_COMPONENTS = False`, `BRIDGE_LABEL_GAPS_BEFORE_CENTERLINES = False`) |
| Siphon mask | **fixed** by default (§4); `SIPHON_LEGACY_MASK = True` reproduces cow exactly |
| VMTK / VTK versions | nvitk: VMTK 1.5.2 + VTK 9.6.2; cow: VMTK 1.5.0 + VTK 9.2.6. Same algorithm; on PESA17808400 the median MIS is ~2.5 % lower and path lengths agree within 1–3 % |
| `00_Path_Summary` | **different by design** (§2) |
| VTP coordinates | scanner RAS instead of index × spacing (§8) |
| Compact tables, lesion severity / confidence | added (§6) |
| Point-sheet names | unique `P###_…` names + `Sheet_Index`; cow truncated ids to 31 characters and overwrote colliding sheets |

`scripts/morpho/compare_with_cow.py` runs both pipelines on a case and compares the centerlines.

> **Curvature units.** cow's `discrete_curvature` returns **κ/2** (half the Menger curvature), so
> its siphon threshold of 0.10 means κ ≥ 0.20 mm⁻¹. nvitk keeps that function (and the legacy
> columns built on it) unchanged, and uses `menger_curvature` / `smoothed_curvature` (true κ) for
> everything new.

---

## 2 · Segments vs paths — `00_Path_Summary` and `07_Root_To_Terminal_Paths`

Root→terminal paths overlap: every arm of a tree re-walks the shared trunk. cow_morpho reports
one `00_Path_Summary` row per path, so summing its lengths counts the trunk once per arm.

nvitk keeps this design:

- **`00_Path_Summary`** has one row per **non-overlapping connected segment**, the pieces between
  junctions and ends, so each piece of vessel is measured once. A segment's geometry is cut from
  the measured (VMTK) centerline between the projections of the segment's ends. It falls back to the
  smoothed skeleton only when no measured path runs within 3 voxels; `geometry_source` in
  `segments.csv` says which was used.
- **`07_Root_To_Terminal_Paths`** has the cow-style per-path rows (with `path_id`, which links to
  `Sheet_Index` and `points/`). Lesions are detected on these paths, exactly as in cow.

Compare cow's `00_Path_Summary` with nvitk's **`07`** sheet, not with nvitk's `00`.

---

## 3 · The three radius signals

| Signal | Column / VTP array | How it is computed | Used for |
|---|---|---|---|
| **Cross-section radius** | `radius_mm` / `CrossSectionRadius` | cut the surface with the local normal plane, take the largest closed loop's area `A`, `r = √(A/π)` | radius statistics, taper, enlargement columns |
| **Maximum inscribed sphere** | `maximum_inscribed_sphere_radius_mm` / `MaximumInscribedSphereRadius` | VMTK's MIS radius (or the native / ray-cast equivalent) | **stenosis detection** (`RADIUS_SOURCE_FOR_CALIBER_DETECTION`) |
| **EDT radius** | — | spacing-aware distance transform at the path voxels | fallback where the cross-section cut fails |

`stenosis_detection_radius_mm` is the MIS signal, with cross-section values filled in where MIS is
missing. On TOF the cross-section radius normally exceeds the MIS by 10–25 %, because the lumen is
not circular.

---

## 4 · The siphon mask (taper-fit exclusion)

High-curvature stretches inflate oblique cross-sections. The **siphon mask** marks them so they
don't shape the healthy taper reference.

**The cow bug.** cow measured curvature point-to-point on the 0.1 mm resampled centerline, where
sampling jitter dominates. Noise alone exceeded its threshold, so the mask covered **100 %** of
every vessel. The taper fit had no points left, and the reference collapsed to a flat median radius.

**The fix (default).** Curvature (true κ) is measured on the centerline after Gaussian smoothing
along its arc length (`SIPHON_CURVATURE_SMOOTH_MM`). The mask is the points with
κ ≥ `SIPHON_KAPPA_THRESHOLD`, dilated by ±`SIPHON_DILATION_MM`.

| | σ (mm) | κ threshold (mm⁻¹) | dilation (mm) | ICA coverage | other vessels |
|---|---|---|---|---|---|
| cow / `SIPHON_LEGACY_MASK = True` | — | 0.10 on κ/2 | 5 | 100 % | 100 % |
| `eicab_cw` | **3.0** | **0.20** | **5** | 40 % | 9 % |

**Calibration.** The `eicab_cw` values come from `scripts/morpho/calibrate_siphon.py`, run on 20
random PESA CW masks on 2026-10-16:
- cow's threshold and dilation were kept, and only the smoothing σ (which cow lacked) was tuned;
- σ was chosen to best separate ICA from non-ICA coverage while keeping non-ICA coverage ≤ 15 %.

Curvature alone cannot confine the mask to the ICA, because the PCAs and SCAs bend just as tightly.
So the mask marks tight bends wherever they occur, which is exactly what the taper fit must skip.
Letting κ and the dilation move too gives σ = 5, κ = 0.12 (ICA 48 %, other 13 %). That point sits at
the edge of the grid, and its heavy smoothing flattens short vessels, so it was not adopted. The
script writes `chosen.json`, `sweep.csv`, `lesion_changes.csv` and `report.md`.

**What the fix changes.** The taper reference is now the healthy envelope cow's code was written to
compute, no longer the flat median the bug left behind. On the 20 calibration subjects:

| | stenoses | enlargements |
|---|---|---|
| cow mask (`SIPHON_LEGACY_MASK = True`) | 131 | 411 |
| calibrated mask | 211 (+61 %) | 283 (−31 %) |

Against a flat median, the wide proximal stretch of a tapering vessel reads as "enlarged", and
narrowings are measured against a radius that is too low. The new stenoses are corroborated by the
independent proximal / distal reference of the lesion table: 88 % are ≥ 10 % narrower in diameter
(median 30 %). The 10 % detection thresholds were kept, not retuned to reproduce counts that came
from the broken reference.

---

## 5 · Caliber lesions — stenosis and enlargement

### 5.1 The taper reference (`_compute_vessel_taper`)

The reference is a **healthy-caliber envelope**, not a least-squares baseline, so a long stenosis
cannot drag its own denominator down.

| Step | Rule | Constant |
|---|---|---|
| 1. Exclude end zones from the *fit* | the first/last 10 mm don't shape the envelope (they stay eligible for detection) | `STENOSIS_/ENLARGEMENT_TAPER_FIT_EXCLUDE_END_MM` = 10 |
| 2. Exclude the siphon mask | §4 | `SIPHON_*` |
| 3. Local high-percentile envelope | 85th percentile in a 20 mm sliding window | `TAPER_REFERENCE_PERCENTILE`, `TAPER_REFERENCE_WINDOW_MM` |
| 4. Smooth | 4 mm moving average | `TAPER_REFERENCE_SMOOTH_MM` |
| 5. Force non-increasing | PAVA isotonic fit: a vessel does not widen distally | `TAPER_FIT_ENFORCE_NONINCREASING` |
| 6. Outlier iterations | points beyond 0.60 × the lesion threshold leave the fit, ≤ 3×, never below 45 % of the fit points | `TAPER_FIT_OUTLIER_FRACTION`, `TAPER_FIT_MAX_ITERATIONS`, `TAPER_FIT_MIN_HEALTHY_FRACTION` |
| 7. Two-pass refit | detected lesions leave the fit and detection re-runs until stable, ≤ 5× | `TAPER_TWO_PASS`, `TAPER_TWO_PASS_MAX_ITERATIONS` |

### 5.2 Detection (cow_morpho's rules)

```
stenosis     pct(i) = (1 − r(i)/r_ref(i)) · 100        on the detection (MIS) radius
enlargement  pct(i) = (r(i)/r_ref(i) − 1) · 100        on the cross-section radius
```

Flagged runs are merged across gaps ≤ 1.5 mm and kept when long enough:

| Parameter | Stenosis | Enlargement |
|---|---|---|
| Threshold | 10 % | 10 % |
| Support threshold | `min(25 %, threshold)` = 10 % | same |
| Minimum length | 3.0 mm | 5.0 mm **or** ≥ 3 consecutive points |
| End exclusion (detection) | 2.0 mm | 3.0 mm |
| End exclusion (candidate arrays) | 5.0 mm | 5.0 mm |

Two cow behaviours to be aware of:
- The support threshold (25 %) is **above** the core one, so the `min()` collapses the hysteresis
  into a single 10 % contour.
- The enlargement "3 consecutive points" alternative accepts runs of 0.2 mm at the 0.1 mm step.
  The lesion table flags such runs as `short` (§6.3) and does not remove them.

Enlargements near the start of another centerline are suppressed
(`SUPPRESS_ENLARGEMENT_NEAR_CENTERLINE_STARTS`). Shared trunks are pruned from arms before the
final detection.

### 5.3 cow-style severity columns (`00` / `07` sheets)

Each accepted segment is scored against one reference: the maximum envelope value within ±5 mm
of the segment (`STENOSIS_SEGMENT_REFERENCE_MARGIN_MM`).

| Column | Meaning |
|---|---|
| `stenosis_percent_max` / `degree_of_stenosis_pct` | peak severity over accepted segments |
| `stenosis_segments_n`, `stenosis_length_total_mm` | count, flagged arc length |
| `radius_min_stenotic_mm` | smallest detection radius inside a flagged segment |
| `stenosis_segments_point_idx` / `_detail_json` | JSON index ranges / per-segment detail |
| `enlargement_*` | the mirror columns |

---

## 6 · Compact tables

The primary output, written as CSVs and as the first workbook sheets. Definitions are in
[vessel_features.py](src/nvitk/measure/morpho/vessel_features.py) and
[vessel_tables.py](src/nvitk/measure/morpho/export_utils/vessel_tables.py).

### 6.1 `vessel_summary.csv` / `Vessel_Summary` — one row per labelled vessel

Absent vessels of the topology get a row with `present = False`. The "main path" is the longest
root→terminal path of the vessel's largest component.

| Group | Columns |
|---|---|
| identity | `case_id` (stage 7: the subject id), `label`, `vessel_name`, `full_name`, `side`, `pair`, `territory`, `present`, `n_components` |
| size | `centerline_length_mm` (non-overlapping total), `main_length_mm`, `main_chord_length_mm`, `volume_mm3`, `surface_area_mm2`, `n_segments`, `n_bifurcations` |
| calibre | `diameter_{mean,median,std,min,max,p05,p95}_mm` (MIS, cross-section where MIS is missing) and the `cs_diameter_*` cross-section set; `diameter_proximal_mm` / `diameter_distal_mm` (first / last 5 mm); `diameter_last_5mm_mm`, `diameter_last_10mm_mm`; `diameter_min_max_ratio`; `cross_section_area_mean_mm2`; `taper_mm_per_mm` |
| blanking | `diameter_mean_blanked_mm`, `diameter_min_blanked_mm`, `blanked_fraction`: points within max(2 mm, 1.5 × radius) of a junction are left out |
| shape | `tortuosity_dm`, `tortuosity_first_10mm`, `tortuosity_last_10mm`, `tortuosity_last_20mm`, `bending_length_mm`, `curvature_{mean,p95,max}_1_per_mm`, `total_curvature_rad` (∫κ ds), `curvature_energy_1_per_mm` (∫κ² ds), `sum_of_angles_deg_per_mm`, `torsion_mean_abs_1_per_mm`, `n_turns` |
| direction | `chord_polar_angle_deg`, `chord_azimuth_angle_deg`, `polar_angle_min_deg`, `polar_angle_accumulated_change_deg` (in anatomical axes) |
| branching | `branching_angle_min_deg`, `branching_angle_max_deg`, `murray_area_ratio_median` |
| lesions | `n_stenosis`, `n_stenosis_high_confidence`, `stenosis_taper_percent_max`, `stenosis_diameter_percent_max`, `stenosis_length_total_mm`, `stenosis_worst_segment`, `stenosis_worst_from_origin_mm`, `stenosis_worst_confidence`, and the `enlargement_*` mirror |

Notes on the definitions:
- **Calibre statistics skip the tips**, the first and last max(`FEATURE_TIP_EXCLUDE_MM` = 1 mm,
  local radius). VMTK appends each seed point on the wall, where the inscribed radius is ~0, and a
  section there cuts the cap.
- **Curvature** is measured on the centerline smoothed at `FEATURE_CURVATURE_SMOOTH_MM` = 1 mm, in
  true 1/mm.
- **The integrals are over arc length**, so they don't depend on the sampling step.

The FLOWCAT `arterial` feature ideas (end-window diameters, min/max ratio, first/last-N-mm
tortuosity, bending length, curvature integrals, direction angles, bifurcation blanking) were
re-implemented here from their definitions. No code was copied (that project is PolyForm
Noncommercial).

### 6.2 `segments.csv` / `Segments` — one row per non-overlapping segment

The same calibre and shape columns per segment, plus `segment`, `path_id`, `is_terminal`,
`geometry_source`, `n_stenoses` and `n_enlargements`.

### 6.3 `lesions.csv` / `Lesions` — one row per lesion

Lesions are the detection segments of the measured paths (§5.2). A lesion that overlapping paths
measure twice is counted once, and is named after the segment containing its midpoint.

| Column | Meaning |
|---|---|
| `kind` | `stenosis` / `enlargement` |
| `segment`, `path_id`, `start_s_from_vessel_origin_mm`, `length_mm`, `x/y/z_mm` | location |
| `taper_percent` | peak severity against the taper reference (the detection measure) |
| `reference_radius_mm`, `reference` | the healthy reference: median detection radius 2–10 mm **upstream** (`proximal`), else downstream (`distal`), else `none` |
| `extreme_radius_mm` | minimum (stenosis) / maximum (enlargement) radius in the lesion |
| `diameter_percent`, `area_percent` | NASCET-like severity against that reference (positive = narrower for stenoses, wider for enlargements) |
| `confidence`, `confidence_class`, `confidence_flags` | below |

`confidence` starts at 1 and is multiplied by a factor for each condition that makes a calibre
change unreliable:

| Flag | Factor | Condition |
|---|---|---|
| `reference_distal` / `reference_none` | 0.8 | no proximal reference window |
| `in_siphon` | 0.7 | overlaps the siphon mask |
| `near_vessel_end` | 0.6 | within 5 mm of a path end (`LESION_NEAR_END_MM`) |
| `short` | 0.8 | shorter than 1.5 × the detection minimum length |
| `near_bifurcation` | 0.5 | within 3 mm of a junction (`LESION_NEAR_BIFURCATION_MM`) |
| `radius_disagreement` | 0.7 | the cross-section/MIS ratio inside the lesion departs > 20 % from the path's own median ratio: one estimator sees the lesion, the other doesn't |
| `cross_section_gaps` | 0.8 | > 20 % of the lesion has no valid cross-section |

Classes: **high** ≥ 0.7 (at most one mild penalty), **moderate** ≥ 0.4, **low** below. The class is
a reliability grade for the geometric measurement, not a clinical grade.

### 6.4 `subject_summary.csv` / `Subject` — one row per subject

| Columns | Meaning |
|---|---|
| `{left_A1, right_A1, AComm, left_PComm, right_PComm, left_P1, right_P1}_status` | `absent` / `hypoplastic` (median diameter < `HYPOPLASIA_DIAMETER_MM` = 1 mm) / `present` |
| `…_diameter_median_mm` | that vessel's median diameter |
| `cow_variant_segments_present`, `cow_complete` | completeness of the circle |
| `{pair}_{metric}_asymmetry_index` | 2·|L − R| / (L + R) for `main_length_mm`, `diameter_median_mm`, `tortuosity_dm`, `curvature_mean_1_per_mm` |
| `n_vessels_present`, `total_centerline_length_mm`, `total_n_stenosis[_high_confidence]`, `total_n_enlargement[_high_confidence]` | totals |

---

## 7 · Other workbook sheets

### Per-point tables (`P###_…` sheets, `points/<path_id>.csv`)

One table per measured path. `Sheet_Index` maps each sheet to its `path_id`, `label`,
`vessel_name` and CSV.

| Column | Meaning |
|---|---|
| `label`, `vessel_name`, `point_index`, `s_mm` | identity, arc length |
| `x_mm, y_mm, z_mm` / `x_ras, y_ras, z_ras` | computation frame (index × spacing) / scanner RAS |
| `radius_mm`, `diameter_mm`, `maximum_inscribed_sphere_radius_mm`, `stenosis_detection_radius_mm` | the radius signals |
| `curvature_1_per_mm`, `torsion_1_per_mm` | cow-style pointwise geometry |
| `stenosis_raw_percent_point`, `stenosis_core/support_candidate_point`, `stenosis_percent_point`, `is_stenotic` | stenosis detection, raw → final |
| `enlargement_percent_point`, `is_enlarged` | enlargement |
| `siphon_mask_point` | the siphon mask (§4) |

### `02_Branchpoints`

Parent and daughter roles come from the Dijkstra distance to the tree root. Radii are EDT values
one skeleton node away from the junction.
- Columns: `n_daughters`, `parent_radius_edt_mm`, `daughter_radii_edt_mm_json`,
  `daughter_parent_radius_ratios_json`, and `daughter_area_ratio_sum_r2_over_parent_r2`.
- The last one is Σ r_d² / r_p², the area-conservation (Murray) ratio: 1 means the area is
  preserved.
- `daughter_pair_angles_deg_json` and its `_min` / `_max` give the angles between daughter
  directions, 5 skeleton steps out.

### `01_Tree_Summary`

Skeleton bookkeeping per (label, component): the graph size, the unique graph length, the
accepted paths, and the paths discarded as short (< 7.5 mm), as spurious arms (< 4 mm; 10 mm for
the ICAs and basilar) or as overlaps. Terminal spurs < 2 mm are pruned first.

### `03_LR_Asymmetry`, `05_Hemisphere`, `04_Tree_Segments`, `06_Volumetry`

- **`03_LR_Asymmetry`** compares left and right per vessel pair (`AI = (L − R)/((L + R)/2)`).
- **`05_Hemisphere`** rolls each side up.
- **`04_Tree_Segments`** is the recursive anatomic segment tree.
- **`06_Volumetry`** has the label volumes.

### `centerlines/tortuosity_metrics.{csv,xlsx}`

Literature-convention tortuosity from the exported VTPs (resampled to 0.2 mm): `tortuosity_index`,
`bend_count` (RDP), `inflection_count_metric` (ICM), `sum_of_angles_metric_deg_per_mm` (SOAM),
`total_curvature_deg_per_mm` and `bending_length_mm`.

---

## 8 · Coordinates

Computation runs in **index × spacing** millimetres (cow's frame), so parity holds, and the frames
differ only rigidly.
- Every written VTP (centerlines and surfaces) is transformed to **scanner RAS** through the NIfTI
  affine and tagged with field data `NVITK_COORDINATE_FRAME = "RAS"` and `NVITK_AFFINE`. The VTPs
  therefore overlay the NIfTI in any viewer.
- The nvitk GUI and the Slicer module read both tagged and older untagged files.
- Point tables carry both frames.

---

## 9 · What reaches the database

[morpho_db_publish.py](src/nvitk/pipes/qvtpy/common/morpho_db_publish.py) writes
`image_measurements` rows (modality `tof`, pipeline `tof_morpho`):
- every numeric column of `vessel_summary.csv`, with `region_id` = the vessel name, for present
  vessels only;
- every numeric column of `subject_summary.csv`, with `region_id = "subject"`.

Units follow the column suffix (`_mm`, `_mm2`, `_mm3`, `_deg`, `_1_per_mm`, `_percent`). Older runs
without the compact tables fall back to the previous aggregation of `00_Path_Summary`.

---

## 10 · Knobs

Stage-7 CLI:

| Flag | Default | Effect |
|---|---|---|
| `--eicab-mask-preference` | `cw` | `cw` / `wb` eICAB mask (the profile is tuned on CW) |
| `--profile` | `eicab_cw` | morphometrics profile |
| `--centerline-backend` | profile (`vmtk`) | `vmtk` / `native` / `raycast` |
| `--use-postprocessed-mask/--no-…` | on | prefer `*_eICAB_CW_pp.nii.gz` |
| `--input-already-smoothed` | off | skip Taubin smoothing |
| `--n-workers` | ≤ 4 | parallel label jobs (VMTK is memory-hungry) |
| `--skip-existing` | off | skip when `case_metrics_donut_tree.xlsx` exists |

The parameters that move the numbers most (change them with `with_params`):

| Parameter | `eicab_cw` | Changes |
|---|---|---|
| `CENTERLINE_RESAMPLE_STEP_MM` | 0.10 | point density → every per-point array |
| `RADIUS_SOURCE_FOR_CALIBER_DETECTION` | `maximum_inscribed_sphere` | which radius stenosis is judged on |
| `STENOSIS_THRESHOLD_PCT` / `ENLARGEMENT_THRESHOLD_PCT` | 10 / 10 | lesion sensitivity |
| `STENOSIS_MIN_LEN_MM` / `ENLARGEMENT_MIN_LEN_MM` | 3 / 5 | shortest accepted lesion |
| `TAPER_REFERENCE_PERCENTILE` / `_WINDOW_MM` | 85 / 20 | how generous the healthy reference is |
| `SIPHON_LEGACY_MASK` | False | True = cow's (100 %-coverage) siphon mask |
| `SIPHON_CURVATURE_SMOOTH_MM` / `SIPHON_KAPPA_THRESHOLD` / `SIPHON_DILATION_MM` | 3 / 0.20 / 5 | the siphon mask (§4) |
| `MIN_CENTERLINE_PATH_LENGTH_MM` | 7.5 | which paths exist at all |
| `PROCESS_ALL_CONNECTED_COMPONENTS` / `BRIDGE_LABEL_GAPS_BEFORE_CENTERLINES` | False / False | fragment handling (cow parity) |

The GUI's *TOF morphometrics* tool exposes the profile, the backend and an optional same-label gap
bridge (`bridge_gaps_vox`, default 0 = off).

---

## See also

- [M-Hemodynamics.md](M-Hemodynamics.md) — the 4D-flow side (stage 6)
- [M-AutoQC.md](M-AutoQC.md) — automatic quality control (stage 9)
