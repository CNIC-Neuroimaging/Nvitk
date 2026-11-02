# Morphometric metrics — qvtpy cheatsheet

Everything the **TOF geometry** side of the pipeline measures: vessel shape, caliber, tortuosity,
lesions and branching.

Produced by **stage 7** ([stage7_morphometrics.py](src/nvitk/pipes/qvtpy/stage7_morphometrics.py)),
which drives [nvitk.measure.morphometrics](src/nvitk/measure/morphometrics.py) over the eICAB
multilabel mask. The algorithm is a port of **cow_morpho**, whose parameters were tuned on the
PESA TOF cohort's eICAB **circle-of-Willis (CW)** masks. Every parameter lives in a named profile in
[morphometrics_config.py](src/nvitk/measure/morphometrics_config.py).

```
stage 1's *_eICAB_WB_pp (A1/A2, M1/M2, P1/P2 split), else *_eICAB_CW_pp (ACA/MCA named A1/M1)
   │  Taubin smoothing: off by default since 2026-10-28 (--taubin: 20 iters, λ=0.65, μ=−0.65);
   │  the pp masks are already drawn at the local half maximum
   │  largest connected component per label (no gap bridging — cow parity)
   ▼
per label job
   │  marching-cubes surface ─► centerlines (VMTK by default) ─► resample @ 0.10 mm
   ▼
per root→terminal path ──► geometry · radii · lesions
   │  split into non-overlapping connected segments
   ▼
├─► morphometrics_report.csv       (the final report: selected features + high-confidence lesions)
├─► vessel_summary.csv · segments.csv · lesions.csv · subject_summary.csv   (compact tables)
├─► case_metrics_donut_tree.xlsx   (Report + compact sheets + cow-style sheets + per-point sheets)
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
| `eicab_cw` (default) | eICAB circle-of-Willis crops | cow_morpho's tuned values, with the calibrated siphon fix, 20 % lesion thresholds (cow: 10 %) and an enlargement minimum length in mm (§5.2) |
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
| Stenosis / enlargement detection rules (§5.2) | cow's code. With cow's values it reproduces cow's segments to the index (checked against cow's own module). `eicab_cw` now differs in two values: 20 % thresholds (cow: 10 %) and a 1.5 mm enlargement floor instead of cow's "3 consecutive points" |
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
| **Cross-section radius** | `radius_mm` / `CrossSectionRadius` | cut the surface with the local normal plane; `ray_median` (default): the median distance from the centerline point to the wall over 36 in-plane rays; `largest_loop` (cow): the largest closed loop's area `A`, `r = √(A/π)` (`CROSS_SECTION_RADIUS_METHOD`, below) | radius statistics, taper, enlargement columns |
| **Maximum inscribed sphere** | `maximum_inscribed_sphere_radius_mm` / `MaximumInscribedSphereRadius` | VMTK's MIS radius (or the native / ray-cast equivalent) | **stenosis detection** (`RADIUS_SOURCE_FOR_CALIBER_DETECTION`) |
| **EDT radius** | — | spacing-aware distance transform at the path voxels | fallback where the cross-section cut fails |

`stenosis_detection_radius_mm` is the MIS signal, with cross-section values filled in where MIS is
missing. On TOF the cross-section radius normally exceeds the MIS by 10–25 %, because the lumen is
not circular.

The cut plane's normal is the chord between the centerline points
`CROSS_SECTION_TANGENT_HALF_WINDOW_MM` upstream and downstream, by arc length. It always uses at
least the neighbouring point on each side, and is one-sided at the path ends. `eicab_cw` uses
**1.0 mm**. cow uses 0.1 mm: the immediate neighbours (±1 point) at the 0.1 mm resampling step, a
0.2 mm baseline. Centerline jitter over that short baseline tilts the plane, and an oblique cut has
a larger area; a wider window steadies it. On PESA17808400 (4 vessels, 1587 points), the radius jump between
consecutive points behaves as follows:

| Half-window | Step change p95 | p99 | CS / MIS median | Enlargements |
|---|---|---|---|---|
| 0.1 mm (cow) | 13.9 % | 38.9 % | 1.156 | 5 |
| 0.5 mm | 8.5 % | 29.2 % | 1.153 | 2 |
| **1.0 mm** (`eicab_cw`) | 8.1 % | 31.4 % | 1.155 | 2 |

The median calibre does not move; the spikes shrink. A window much wider than the vessel's radius
turns late in tight bends.

**From the cut to a radius** (`CROSS_SECTION_RADIUS_METHOD`). cow takes the **largest** closed loop
of the cut (`largest_loop`). The plane is infinite, so near the siphon or a bifurcation it also
cuts the vessel's other limb, a neighbouring branch or the merged carina, and the largest loop is
often not the one around the centerline point. The radius then spikes, and it climbs along the
vessel while the inscribed radius falls.

The default, `ray_median`, casts `CROSS_SECTION_RAYS` (36) rays in the plane from the centerline
point. Each ray stops at its first wall crossing, and the radius is the median distance: a circle
between the nearest and the farthest wall.
- A branch opening only lengthens the few rays that run into it, so it does not move the median
  unless it spans half the directions.
- Other vessels in the plane lie behind the vessel's own wall, so the rays never reach them.
- A point outside the lumen (an even number of wall crossings on most rays) gets NaN.
- The cross-section area becomes that circle's, `π r²`.

| Case | `largest_loop` | `ray_median` |
|---|---|---|
| straight tube, r = 1.5 mm | 1.499 | 1.495 |
| the same with a side branch (r = 1 mm), max near the opening | 2.254 | 1.504 |
| U-shaped vessel (r = 1.5 mm limb, r = 2.5 mm limb in the same plane) | 2.499 everywhere | 1.495 |

On PESA17808400 (17 paths, full CW mask):

| | `largest_loop` | `ray_median` |
|---|---|---|
| Paths whose cross-section radius rises while the inscribed one falls | 5 | 0 |
| Step change p95 / p99 | 3.7 % / 9.9 % | 3.4 % / 5.7 % |
| Cross-section / inscribed ratio, median / p95 | 1.149 / 2.065 | 1.116 / 1.249 |
| Points without a value | 0 % | 0 % |
| Stenoses / enlargements | 4 / 2 | 4 / 0 |

On the RICA, where the inscribed radius narrows to 1.05 mm, `largest_loop` reported 4.7 mm, the
siphon's other limb; `ray_median` follows the narrowing down to 1.15 mm.

On the 20 calibration subjects (306 paths):

| | `largest_loop` | `ray_median` |
|---|---|---|
| Paths whose cross-section radius rises while the inscribed one falls | 60 | 9 |
| Step change p95 / p99 | 3.7 % / 10.0 % | 3.4 % / 6.6 % |
| Cross-section / inscribed ratio, median / p95 | 1.161 / 2.125 | 1.115 / 1.285 |
| Stenoses (detected on the inscribed radius) | 99 | 99 |
| Enlargements | 73 | 28 |
| Confidence high / moderate / low | 14 / 78 / 80 | 22 / 58 / 47 |

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
(median 30 %). These counts were measured with the 10 % detection thresholds in use at the time.
The thresholds were not retuned then to reproduce counts that came from the broken reference;
they have since been raised to 20 % (§5.2).

---

## 5 · Caliber lesions — stenosis and enlargement

### 5.0 Two detection methods (`LESION_DETECTION_METHOD`, `--lesion-detection-method`)

- **`robust_reference`** (default since 2026-10-28; `measure/morpho/robust_reference.py`). One
  reference and one signed deviation per path, for both kinds:
  1. *Signal*: the cross-section radius (`ray_median`; MIS where missing; `ROBUST_RADIUS_SOURCE`),
     running median over `ROBUST_RADIUS_MEDIAN_MM` (1 mm).
  2. *Reference*: per junction-to-junction stretch, a robust LOWESS of log r (tricube window
     `ROBUST_REFERENCE_BANDWIDTH_MM` = 20 mm, 4 Tukey-bisquare passes; bumps and dips are
     down-weighted alike). Blanking zones and the path ends carry no weight and are bridged, never
     extrapolated into. The siphon is fitted like the rest: bridging its 25-30 mm would read the
     whole cavernous ICA as an enlargement and hide a C7 stenosis behind its step down. A weighted PAVA projection then lets the reference rise by
     at most `ROBUST_MAX_RISE_PER_MM` (1 %/mm): the petrous → cavernous widening is followed, a
     bulge cannot lift its own reference.
  3. *Detection*: `d = r / ref − 1`. Core `|d| ≥ max(20 %, 3 σ_path)` (σ_path: the path's robust
     scatter around the unconstrained fit), support half of it, the kind's minimum length. A point
     has one sign, so it is never both kinds. No core lies in a bifurcation's blanking zone, in a
     tip's wall ramp (two radii) or where the reference radius is under
     `ROBUST_MIN_REFERENCE_RADIUS_MM` (0.75 mm: a lumen 1–2 voxels across).
  4. *Lesion table*: severity against the robust reference at the extreme point
     (`reference = robust`); the windows (§6.3) only cross-check it. `in_siphon` counts only
     together with `radius_disagreement`; `noisy_path` (σ_path > 10 %) ×0.8.
- **`taper`**: cow_morpho's separate q85 stenosis / enlargement fits, below (§5.1–5.2), unchanged.

**Bifurcations and vessel ends, both methods.** A run whose extreme point lies in a bifurcation's
blanking zone or within two local radii (at least the kind's end exclusion) of a path end is
dropped, not just penalised. For a split vessel the bifurcations it shares with its other half
count too: the branch points of the **merged** M1 + M2 (A1 + A2, P1 + P2) mask near the
component (`labels_util.split_partner_junctions`). Measured alone, an M1 ends in the flare where
its M2s leave, which otherwise reads as an enlargement; likewise a branch leaving mid-M1 that the
distal growth labelled M2. The merged mask takes every piece of the other half, including those
`PROCESS_ALL_CONNECTED_COMPONENTS = False` leaves unmeasured: such a mid-M1 M2 branch is often a
piece of its own (PESA2140369's right M1: a 23 % "enlargement" at its bifurcation until 2026-10-29).

### 5.1 The taper reference (`_compute_vessel_taper`)

The taper reference is the expected healthy radius at every point of the path, the denominator
of detection: a stenosis is a run below `reference × (1 − 20 %)`, an enlargement a run above
`reference × (1 + 20 %)`. "Taper" is the vessel's natural narrowing along its length. It is a
**healthy-calibre envelope** rather than a least-squares baseline, so a long stenosis cannot drag
its own denominator down. Two methods (`TAPER_REFERENCE_METHOD`):

**`stretch` (default).** The path is cut at the **junctions** it passes through (the component's
skeleton branch points, the `02_Branchpoints` rows), and each junction-to-junction stretch gets its
own reference:
- **Natural taper, locally:** the envelope below (steps 3–7) made **local-linear**. Every 0.5 mm,
  it fits the 85th-percentile *line* of the stretch's radius within ±10 mm, its decline capped at
  `TAPER_STRETCH_MAX_DECLINE_PCT_PER_MM` (2 % of the local calibre per mm; natural taper is
  ≈ 1 %/mm, and the cap keeps a long narrowing from passing as "taper"). A local percentile on a
  sloping vessel leads the taper by ~7 mm; the local line does not, and it follows a vessel that is
  flat and then declines.
- **Junctions:** the bifurcation blanking zones (`BLANKING_MM`, `BLANKING_RADIUS_FACTOR`) stay out
  of the fit, and the reference may **step** at a junction, where a branch leaving lowers the
  calibre (Murray). A stretch shorter than `TAPER_STRETCH_MIN_LENGTH_MM` (10 mm) joins a
  neighbour, because two branch points close together would otherwise give a few millimetres
  (often exactly a narrowing) a reference of their own. A whole vessel shorter than
  `TAPER_STRETCH_SHORT_VESSEL_MM` (20 mm: A1s, PComms, AChAs, which ramp near their origin) gets
  its median, as with the `envelope` method, whose 10 mm end exclusions leave it nothing to fit.
- **Path ends and gaps:** the fit uses the whole stretch but its tips (no 10 mm end exclusion).
  Distally the reference continues the local decline into the vessel's end instead of freezing at
  the calibre 10 mm upstream. Proximally it is held, never extrapolated upwards past any measured
  healthy calibre. Gaps inside a stretch that the fit skips (siphon, detected lesions, junction
  zones) are bridged linearly from both sides.
- **Inspection:** the point tables carry the stretch of every point (`taper_stretch`), and the
  lesion inspector draws a dashed line where the reference may step.

**`envelope` (cow_morpho).** One envelope along the whole path (steps 1–7), blind to junctions. It
smears a calibre step at a bifurcation over ±10 mm and flags the daughter as a stenosis, and over
the last 10 mm it stays at the calibre measured further upstream.

On a synthetic vessel with a calibre step at a junction (2.0 → 1.5 mm), a 2 %/mm natural taper
after it and a real 4 mm, 50 % stenosis before it:
- `envelope` stays at 2.0 mm across the junction and flags the whole tapering daughter as one
  23 mm stenosis.
- `stretch` follows the true healthy calibre (1.42 / 1.22 / 0.96 mm against 1.44 / 1.20 / 0.90)
  and finds only the real stenosis.

On the 20 calibration subjects:

| | `envelope` | `stretch` |
|---|---|---|
| Stenoses | 99 | 62 (40 removed, 3 added) |
| … ending within 10 mm of the path end | 67 | 32 |
| … within 3 mm of a junction | 34 | 19 |
| … confidence high / moderate / low | 14 / 44 / 41 | 12 / 23 / 27 |
| Enlargements | 28 | 25 |

Of the 40 stenoses no longer called, 32 were distal fade-outs that the frozen end zone counted
against the calibre upstream, and 14 sat at junctions: a bifurcation flare taken as healthy, or a
daughter measured against its parent. The 3 new ones are borderline (20–30 % against a reference
within a few percent of the old one).

| Step | Rule | Constant |
|---|---|---|
| 1. Exclude end zones from the *fit* | the first/last 10 mm don't shape the envelope (they stay eligible for detection) | `STENOSIS_/ENLARGEMENT_TAPER_FIT_EXCLUDE_END_MM` = 10 |
| 2. Exclude the siphon mask | §4 | `SIPHON_*` |
| 3. Local high-percentile envelope | 85th percentile in a 20 mm sliding window | `TAPER_REFERENCE_PERCENTILE`, `TAPER_REFERENCE_WINDOW_MM` |
| 4. Smooth | 4 mm moving average | `TAPER_REFERENCE_SMOOTH_MM` |
| 5. Force non-increasing | PAVA isotonic fit: a vessel does not widen distally | `TAPER_FIT_ENFORCE_NONINCREASING` |
| 6. Outlier iterations | points beyond 0.60 × the lesion threshold leave the fit, ≤ 3×, never below 45 % of the fit points | `TAPER_FIT_OUTLIER_FRACTION`, `TAPER_FIT_MAX_ITERATIONS`, `TAPER_FIT_MIN_HEALTHY_FRACTION` |
| 7. Two-pass refit | detected lesions leave the fit and detection re-runs until stable, ≤ 5× | `TAPER_TWO_PASS`, `TAPER_TWO_PASS_MAX_ITERATIONS` |

### 5.2 Detection (cow_morpho's rules, `eicab_cw` values)

```
stenosis     pct(i) = (1 − r(i)/r_ref(i)) · 100        on the detection (inscribed) radius
enlargement  pct(i) = (r(i)/r_ref(i) − 1) · 100        on the cross-section radius (see below)
```

Flagged runs are merged across gaps ≤ 1.5 mm and kept when long enough:

| Parameter | Stenosis | Enlargement |
|---|---|---|
| Threshold | **20 %** (`STENOSIS_THRESHOLD_PCT`; cow: 10 %) | **20 %** (`ENLARGEMENT_THRESHOLD_PCT`; cow: 10 %) |
| Support threshold | `min(25 %, threshold)` = 20 % | same |
| Minimum length | 3.0 mm (`STENOSIS_MIN_LEN_MM`) | 5.0 mm (`ENLARGEMENT_MIN_LEN_MM`) **or** ≥ **1.5 mm** (`ENLARGEMENT_MIN_SUPPORT_LENGTH_MM`), i.e. 1.5 mm |
| End exclusion (detection) | 2.0 mm | 3.0 mm |
| End exclusion (candidate arrays) | 5.0 mm | 5.0 mm |
| Max internal gap merged | 1.5 mm | 1.5 mm |
| Radius tested | detection (`RADIUS_SOURCE_FOR_CALIBER_DETECTION`) | `cow` (`ENLARGEMENT_RADIUS_SOURCE`): cross-section, or detection on re-measured paths |
| Cross-section smoothing | — | off (`ENLARGEMENT_SMOOTH_CROSS_SECTION = False`; window `ENLARGEMENT_CROSS_SECTION_MEDIAN_MM` = 1.0 mm) |

Notes on these rules:
- **The support tier is inactive.** The support threshold (25 %) is **above** the core one, so the
  `min()` collapses the hysteresis into a single 20 % contour. A support below the core (say 5 %)
  would grow each lesion into its shoulders and bridge dips between them.
- **Enlargement floor in mm.** cow's alternative acceptance was "≥ 3 consecutive points", which
  at the 0.1 mm step accepted 0.2 mm runs: single cross-section spikes, far below TOF resolution.
  It is replaced by a floor in arc length, `ENLARGEMENT_MIN_SUPPORT_LENGTH_MM` = 1.5 mm, which is 3
  voxels at 0.5 mm. As the shorter of the two rules, it is the effective minimum. On the 342 paths
  of the 20 calibration subjects at 20 %:

  | Rule | Enlargements | Per subject |
  |---|---|---|
  | cow's 3-point rule | 205 | 10.3 |
  | 1.0 mm floor | 127 | 6.3 |
  | **1.5 mm floor** | **98** | **4.9** |
  | 2.0 mm floor | 79 | 4.0 |
  | 5.0 mm only | 27 | 1.4 |

- **Which radius enlargements are tested on** (`ENLARGEMENT_RADIUS_SOURCE`). The default,
  `"cow"`, keeps cow's choice, which depends on the code path:
  - a path's first measurement uses the **cross-section** radius;
  - a path re-measured after its shared trunk was pruned, and a donut arm, use the **detection**
    (inscribed) radius.

  The same bulge can therefore pass on one path and fail on another. `"cross_section"` or
  `"detection"` tests every path on that one radius; the default stays `"cow"` until they are
  compared. Tested alone on each radius, the 342 calibration paths give (20 %, 1.5 mm floor):

  | Radius | Enlargements | Per subject | Median peak |
  |---|---|---|---|
  | cross-section | 98 | 4.9 | +65 % |
  | cross-section, 1 mm median | 89 | 4.5 | +50 % |
  | detection (inscribed) | 29 | 1.4 | +40 % |

  `enlargement_radius_mm` (point tables, VTPs) records the radius each path used, and the lesion
  table measures each enlargement's severity on it. The VTP arrays `EnlargementBinaryCS` /
  `EnlargementBinaryMIS` are diagnostic flags from each radius alone.
- **Cross-section smoothing (off by default).** The cross-section radius spikes for one or two
  0.1 mm samples wherever a cut catches a branch opening or sits obliquely in a bend. Its
  sample-to-sample change reaches 11 % (95th percentile), against 5 % for the inscribed radius.
  `ENLARGEMENT_SMOOTH_CROSS_SECTION = True` tests a running median over
  `ENLARGEMENT_CROSS_SECTION_MEDIAN_MM` instead, so a bulge longer than half the window survives.
  With the 1.5 mm floor it changes 98 → 89 enlargements on the calibration paths. It applies
  wherever an enlargement test reads the cross-section radius: under `"cow"`, on re-measured
  paths only when `RADIUS_SOURCE_FOR_CALIBER_DETECTION` is `"cross_section"`.

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
| lesions | `n_stenosis`, `n_stenosis_high_confidence`, `stenosis_taper_percent_max`, `stenosis_diameter_percent_max`, `stenosis_length_total_mm`, `stenosis_worst_segment`, `stenosis_worst_from_origin_mm`, `stenosis_worst_confidence`, and the `enlargement_*` mirror; `hypoplasia`, `hypoplasia_confidence_class` (main vessels, §6.3) |

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
| `kind` | `stenosis` / `enlargement` / `hypoplasia` (an absent main vessel, below) |
| `segment`, `path_id`, `point_start` / `point_end` (indices on that path's point table), `start_s_from_vessel_origin_mm`, `length_mm`, `x/y/z_mm` | location |
| `taper_percent` | peak severity against the taper reference (the detection measure) |
| `reference_radius_mm`, `reference` | the healthy reference radius at the lesion (below) and how it was obtained: `interpolated`, `proximal`, `distal` or `none` |
| `reference_proximal_radius_mm`, `reference_distal_radius_mm` | the median radius of each reference window (NaN when a window has too few points) |
| `reference_{proximal,distal}_{start,end}_point` | each window's first / last point on the path's point table (what the lesion inspector shades) |
| `extreme_radius_mm` | minimum (stenosis) / maximum (enlargement) radius in the lesion |
| `diameter_percent`, `area_percent` | severity against the reference: `1 − r/ref` and `1 − (r/ref)²`, positive = narrower for stenoses, wider for enlargements |
| `confidence`, `confidence_class`, `confidence_flags` | below |

Each kind is measured on the radius its test used. Stenoses use the detection (inscribed) radius.
Enlargements use `enlargement_radius_mm`: cow tests them on the cross-section radius on a path's
first measurement, and on the detection radius on paths re-measured after trunk pruning. Measuring
an enlargement's severity on the other radius understated it.

**The healthy reference** (`LESION_REFERENCE_METHOD`). Detection uses the moving taper reference
(§5.1); the lesion table measures severity against windows of healthy vessel next to the lesion:

- **`interpolated`** (default; QCA's interpolated reference diameter):
  - A proximal window upstream of the lesion and a distal one downstream: since 2026-10-28 each is
    the nearest `LESION_REFERENCE_WINDOW_LENGTH_MM` (10 mm) of eligible points from
    `LESION_REFERENCE_GAP_MM` (2 mm) up to `LESION_REFERENCE_SEARCH_MM` (30 mm) away, so it reaches
    past a neighbouring lesion or a short eroded stretch (before: 2–10 mm fixed). A side with no
    room in the lesion's stretch crosses the junction (`reference_crosses_junction`, ×0.9). The
    median radius of each is interpolated linearly in
    arc length to the lesion's narrowest (widest) point. The vessel's natural taper between the two
    windows therefore is not counted as stenosis.
  - The windows keep to **healthy points of the lesion's own junction-to-junction stretch**. They
    leave out other lesions and the bifurcation blanking zones (`BLANKING_MM` /
    `BLANKING_RADIUS_FACTOR`), and they do not cross a junction, where a branch leaving steps the
    calibre down.
  - A distal median above the proximal one (post-stenotic dilatation) is capped at the proximal:
    a healthy vessel does not widen downstream. The lesion inspector draws each window's own median
    and notes the cap. With cow's `largest_loop` cross-section, whose radius climbs downstream, the
    cap applied to 18 of 41 interpolated enlargements on the calibration subjects; with
    `ray_median`, to 6 of 11. When the two medians differ by more than
    `LESION_REFERENCE_DISAGREEMENT_FRACTION` (25 %), more than taper explains (distal collapse, an
    unseen branch), the lesion is flagged `reference_inconsistent`.
  - With one window only, that window's median (`proximal` / `distal`). With neither (a lesion
    hemmed in by junctions, path ends and other lesions), the `proximal` method's plain windows,
    flagged `reference_fallback`.
- **`proximal`** (the method before 2026-10-21): the median radius 2–10 mm upstream, else
  downstream, else none, with no healthy-point or junction rules. A lesion's narrowest point often
  lies several mm downstream of that window, so all the taper in between counted as stenosis.

Neither method's windows reach into the first or last 3 mm of the path
(`LESION_REFERENCE_EXCLUDE_END_MM`), where the seed point and the end taper sit, and each window
needs ≥ 3 points.

On the 20 calibration subjects (172 lesions; `eicab_cw` of 2026-10-21), the old method was
recomputed from the same runs' point tables:

| | `proximal` | `interpolated` |
|---|---|---|
| References used | 154 proximal, 18 distal | 61 interpolated, 90 proximal only, 21 distal only (25 fallbacks), 11 inconsistent |
| Stenoses (99): median diameter % | 37.0 | 36.9; −2.9 points on the interpolated ones (per lesion, p10–p90 −5.3 … +5.2) |
| Enlargements (73): median diameter % | 59.5 | 78.1 |
| Confidence classes | | 5 lesions moderate → low, otherwise unchanged |

- **Stenoses:** where both windows exist, the distal one is a median 12 % narrower than the
  proximal one. That is the natural taper, and the interpolation no longer counts it as stenosis.
- **Enlargements:** the old windows often overlapped a neighbouring enlargement, typically
  cross-section spikes in the proximal ICA. Their radius was 4–5 mm against the 2.5–3 mm of the
  healthy windows, which understated the enlargement and once even gave −4 %.

`confidence` starts at 1 and is multiplied by a factor for each condition that makes a calibre
change unreliable:

| Flag | Factor | Condition |
|---|---|---|
| `reference_distal` / `reference_none` | 0.8 | no proximal reference window |
| `reference_inconsistent` | 0.8 | `interpolated`: the proximal and distal medians differ by > 25 % (`LESION_REFERENCE_DISAGREEMENT_FRACTION`) |
| `reference_fallback` | 0.8 | `interpolated`: no healthy window in the lesion's stretch, the plain windows were used |
| `in_siphon` | 0.7 | overlaps the siphon mask |
| `near_vessel_end` | 0.6 | within 5 mm of a path end (`LESION_NEAR_END_MM`) |
| `short` | 0.8 | shorter than 1.5 × the detection minimum length (stenoses 3 mm → < 4.5 mm; enlargements 1.5 mm → < 2.25 mm) |
| `near_bifurcation` | 0.5 | within 3 mm of a junction (`LESION_NEAR_BIFURCATION_MM`) |
| `radius_disagreement` | 0.7 | the cross-section/MIS ratio inside the lesion departs > 20 % from the path's own median ratio: one estimator sees the lesion, the other doesn't |
| `cross_section_gaps` | 0.8 | > 20 % of the lesion has no valid cross-section |

Classes: **high** ≥ 0.7 (at most one mild penalty), **moderate** ≥ 0.4, **low** below. The class is
a reliability grade for the geometric measurement, not a clinical grade.

**Hypoplasia rows.** Each **absent main vessel** gets one `kind = hypoplasia` row. The main vessels
are `HYPOPLASIA_MAIN_VESSELS`: both ICAs, the basilar, both MCAs, both ACAs, and each PCA's P1 and
P2 as separate vessels. Other absent vessels (PComms, AComm, SCAs, …) are normal variants and get
no row. The rows have no location (`path_id`, `x/y/z_mm` and the reference columns are empty).

On TOF, *absent* means **no flow signal in the eICAB mask**: aplasia, a hypoplasia below the
resolution, or an occlusion look the same. The calibre-based statuses of §6.4 (`hypoplastic` below
1 mm median diameter) are a separate, unchanged measure.

The confidence comes from the vessel's topological neighbours: its `flow_from` / `flow_to` vessels
and the vessels naming it in theirs.

| Flag | Factor | Condition |
|---|---|---|
| `absent` / `too_small` | — | no voxels of the label / voxels, but too few for a centerline (one of the two is always set) |
| `neighbours_absent` | 0.3 | no neighbour is present either: more likely the field of view or a segmentation dropout than a missing vessel |
| `fetal_pca` | — | a P1 absent while the same side's PComm and P2 are present (fetal PCA variant) |
| `a1_aplasia` | — | an ACA absent while the AComm and the other ACA are present |

So an absent vessel next to present ones is **high** (1.0), and a whole territory missing is
**low** (0.3). A mask with fewer than half of the main vessels present is taken for a part of the
circle (a label selection, a crop), not a whole one, and gets no hypoplasia rows at all
(`HYPOPLASIA_MIN_PRESENT_FRACTION`). With the profile's `PROCESS_SELECTED_TAGS_ONLY`, only the
selected tags are checked. In the GUI the rows are listed in the lesions table and shown in the lesion
inspector (card and flags, no profile); they have no point or extent in the viewer.

**Variants (2026-10-28).**
- A main vessel in the mask but without a measured centerline (a few voxels, no length or
  diameter) is a `hypoplasia` row with `too_small`, and its subject status is `hypoplastic`.
- An **A1 hypoplasia** gets a position where the missing A1 would run: midway between its ICA's end
  and the AComm (without one, the other A1's end), so the GUI can show and centre on it.
- **`kind = fetal`**, one row per side, on the P2: the P1 absent, too small or hypoplastic, and in
  the label volume the PComm touches the ICA and the P2 (directly, or through the P1 stump:
  `via_p1`). Placed at the PComm's centroid; `left_fetal_pca` / `right_fetal_pca` in the subject
  table and `has_fetal_pca` in the report and the database.

### 6.4 `subject_summary.csv` / `Subject` — one row per subject

| Columns | Meaning |
|---|---|
| `{left_A1, right_A1, AComm, left_PComm, right_PComm, left_P1, right_P1}_status` | `absent` / `hypoplastic` (median diameter < `HYPOPLASIA_DIAMETER_MM` = 1 mm) / `present` |
| `…_diameter_median_mm` | that vessel's median diameter |
| `cow_variant_segments_present`, `cow_complete` | completeness of the circle |
| `{pair}_{metric}_asymmetry_index` | 2·|L − R| / (L + R) for `main_length_mm`, `diameter_median_mm`, `tortuosity_dm`, `curvature_mean_1_per_mm` |
| `n_vessels_present`, `total_centerline_length_mm`, `total_n_stenosis[_high_confidence]`, `total_n_enlargement[_high_confidence]`, `total_n_hypoplasia[_high_confidence]` | totals |

### 6.5 `morphometrics_report.csv` / `Report` — the final report, one row per vessel

The final per-vessel output of stage 7, and what reaches the database (§9). It is written by
[report.py](src/nvitk/measure/morpho/export_utils/report.py) at the end of every run, as a CSV and as
the **first** workbook sheet. Every topology vessel has a row; absent ones have `present = False`
and empty features.

| Group | Columns |
|---|---|
| identity | `case_id`, `label`, `vessel_name`, `side`, `present` |
| features | `centerline_length_mm`, `diameter_median_mm`, `diameter_mean_mm`, `volume_mm3`, `surface_area_mm2`, `radius_mean_mm`, `tortuosity_dm`, `curvature_mean_1_per_mm`, `total_curvature_rad`, `inflection_count_metric`, `chord_azimuth_angle_deg`, `diameter_mean_blanked_mm`, `chord_polar_angle_deg`, `polar_angle_accumulated_change_deg` |
| lesions (**high confidence only**) | `has_stenosis`, `stenosis_diameter_percent_max`, `has_enlargement`, `enlargement_diameter_percent_max`, `has_hypoplasia` |

- The features are the GUI's vessel table columns of the same name: `vessel_summary.csv` (§6.1),
  `volume_mm3` from the label volumetry, `inflection_count_metric` from the tortuosity table.
- **`radius_mean_mm` is not half of `diameter_mean_mm`.** It is cow_morpho's mean cross-section
  radius (`00_Path_Summary`, length-weighted over the vessel's paths), while the diameters are the
  inscribed-sphere (MIS) ones.
- **The lesion columns count only `high` lesions** (§6.3). A vessel whose lesions are all moderate or
  low has `has_stenosis = False`, and a `*_percent_max` is empty without a high-confidence lesion.
  The full lesion list stays in `lesions.csv`.
- `has_hypoplasia` is set for the main vessels only.
- Older run folders get a report built on the fly from their tables, by the GUI and by the DB
  publish; hypoplasia then comes from the topology's absent main vessels, with the same rule.

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
| `stenosis_reference_radius_mm`, `enlargement_reference_radius_mm` | the healthy taper references the percentages are measured against (§5.1) |
| `enlargement_radius_mm` | the radius the enlargement test used (cross-section or detection radius) |

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
`image_measurements` rows (modality `tof`, pipeline `tof_morpho`) **from the report only** (§6.5),
with `region_id` = the vessel name:
- the 14 report features, for present vessels;
- `has_stenosis` / `has_enlargement` (unit `boolean`, 1/0) and the two `*_diameter_percent_max`
  (`%`, only when there is a high-confidence lesion), for present vessels;
- `has_hypoplasia` (1/0) for every main vessel, present or not.

A run without `morphometrics_report.csv` gets one built on the fly. Runs older than
`vessel_summary.csv` fall back to the previous aggregation of `00_Path_Summary`.

**Re-publishing replaces.** Publishing a subject first deletes all of that subject's `tof_morpho`
rows (`DataRepo.replace_rows`), so variables from earlier publishes that the report no longer has
(the full `vessel_summary` / `subject_summary` sets of before 2026-10-22) do not linger. Other
subjects and pipelines are untouched.

Units follow the column suffix (`_mm`, `_mm2`, `_mm3`, `_deg`, `_1_per_mm`, `_percent`).

---

## 10 · Knobs

Stage-7 CLI:

| Flag | Default | Effect |
|---|---|---|
| `--eicab-mask-preference` | `wb` | `wb`: stage 1's distal-grown `*_eICAB_WB_pp` (A1/A2, M1/M2, P1/P2 measured as separate vessels), else `*_CW_pp`; `cw`: `*_CW_pp` |
| `--profile` | `eicab_cw` | morphometrics profile |
| `--centerline-backend` | profile (`vmtk`) | `vmtk` / `native` / `raycast` |
| `--use-postprocessed-mask/--no-…` | on | prefer `*_eICAB_CW_pp.nii.gz`, stage 1's post-process (docs/pipelines/qvtpy.md): no anterior choroidals; labels re-drawn at their local half maximum; ICAs to the image floor (the basilar not extended, no vertebrals) and their loops below the siphon filled where the inner wall came out dark; ACAs / MCAs / PCAs / SCAs grown distally along the Frangi tree; siphon loops cut. The `eicab_cw` thresholds were calibrated on eICAB's own, thicker masks |
| `--taubin/--no-taubin` | off | Taubin-smooth the label mask first (`nvitk-qvtpy --stage7-taubin`) |
| `--lesion-detection-method` | profile (`robust_reference`) | `robust_reference` / `taper` (§5.0; `nvitk-qvtpy --lesion-detection-method`) |
| `--n-workers N` | min(4, CPUs/2) | vessels measured in parallel: one process per label/component job, 1 = one at a time (VMTK is memory-hungry). On SGE, N > 1 requests `-pe smp N` |
| `--skip-existing` | off | skip when `case_metrics_donut_tree.xlsx` exists |

The batch runner passes it as `nvitk-qvtpy --stage7-n-workers N`, locally and on SGE. On SGE, the
subject's array job then requests `-pe smp N`. Results do not depend on N. On PESA17808400
(CW mask, 17 vessel jobs), 1 worker took 143 s and 6 workers took 61 s, with identical tables.

The parameters that move the numbers most (change them with `with_params`):

| Parameter | `eicab_cw` | Changes |
|---|---|---|
| `CENTERLINE_RESAMPLE_STEP_MM` | 0.10 | point density → every per-point array |
| `CROSS_SECTION_TANGENT_HALF_WINDOW_MM` | 1.0 | the cross-section plane's normal: chord over ± this arc length (§3; cow: 0.1 = ±1 point) |
| `CROSS_SECTION_RADIUS_METHOD` / `CROSS_SECTION_RAYS` | `ray_median` / 36 | how the cut becomes a radius: median in-plane wall distance, or cow's `largest_loop` (§3) |
| `RADIUS_SOURCE_FOR_CALIBER_DETECTION` | `maximum_inscribed_sphere` | which radius stenosis is judged on |
| `STENOSIS_THRESHOLD_PCT` / `ENLARGEMENT_THRESHOLD_PCT` | 20 / 20 | lesion sensitivity (cow: 10 / 10) |
| `STENOSIS_SUPPORT_THRESHOLD_PCT` / `ENLARGEMENT_SUPPORT_THRESHOLD_PCT` | 25 / 25 | hysteresis tier; inactive at or above the threshold (§5.2) |
| `STENOSIS_MIN_LEN_MM` / `ENLARGEMENT_MIN_LEN_MM` | 3 / 5 | shortest accepted lesion |
| `ENLARGEMENT_MIN_SUPPORT_LENGTH_MM` | 1.5 | shorter enlargement floor; the effective minimum (cow: "3 consecutive points") |
| `ENLARGEMENT_RADIUS_SOURCE` | `cow` | which radius enlargements are tested on: `cow` (path-dependent), `cross_section` or `detection` (§5.2) |
| `ENLARGEMENT_SMOOTH_CROSS_SECTION` / `ENLARGEMENT_CROSS_SECTION_MEDIAN_MM` | False / 1.0 | running median of the cross-section radius enlargements are tested on |
| `TAPER_REFERENCE_METHOD` | `stretch` | detection's healthy reference: junction-to-junction stretches with a capped natural taper, or cow's whole-path `envelope` (§5.1) |
| `TAPER_STRETCH_MAX_DECLINE_PCT_PER_MM` / `TAPER_STRETCH_MIN_LENGTH_MM` / `TAPER_STRETCH_SHORT_VESSEL_MM` | 2 / 10 / 20 | steepest natural taper a stretch may follow; shortest stretch with a reference of its own; shorter vessels use their median |
| `TAPER_REFERENCE_PERCENTILE` / `_WINDOW_MM` | 85 / 20 | how generous the healthy reference is |
| `SIPHON_LEGACY_MASK` | False | True = cow's (100 %-coverage) siphon mask |
| `LESION_REFERENCE_METHOD` | `interpolated` | the lesion table's healthy reference: proximal + distal windows interpolated, or `proximal` (§6.3) |
| `LESION_REFERENCE_GAP_MM` / `_WINDOW_LENGTH_MM` / `_SEARCH_MM` / `LESION_REFERENCE_EXCLUDE_END_MM` | 2 / 10 / 30 / 3 | the reference windows: gap from the lesion, length, how far they may search, kept clear of the path ends |
| `LESION_DETECTION_METHOD` | `robust_reference` | §5.0; `taper` = cow's fits |
| `ROBUST_*` | see §5.0 | the robust reference: bandwidth 20 mm, max rise 1 %/mm, k σ = 3, support ½, min reference radius 0.75 mm |
| `LESION_REFERENCE_DISAGREEMENT_FRACTION` | 0.25 | proximal vs distal median gap that flags `reference_inconsistent` |
| `HYPOPLASIA_MAIN_VESSELS` | ICAs, BASILAR, M1s, A1s, PCA P1/P2 | the vessels whose absence (or a mask without a measured centerline, `too_small`) is a `hypoplasia` lesion (§6.3); an A2 / M2 never is |
| `SIPHON_CURVATURE_SMOOTH_MM` / `SIPHON_KAPPA_THRESHOLD` / `SIPHON_DILATION_MM` | 3 / 0.20 / 5 | the siphon mask (§4) |
| `MIN_CENTERLINE_PATH_LENGTH_MM` | 7.5 | which paths exist at all |
| `PROCESS_ALL_CONNECTED_COMPONENTS` / `BRIDGE_LABEL_GAPS_BEFORE_CENTERLINES` | False / False | fragment handling (cow parity) |

The GUI's *TOF morphometrics* tool exposes the profile, the backend, an optional same-label gap
bridge (`bridge_gaps_vox`, default 0 = off) and, for comparison, overrides of the enlargement
radius and the cross-section median.

---

## 11 · In the GUI

Measure → *Morphometrics* runs the same pipeline on a Labels layer, with the same profile,
backend, the legacy-siphon switch and the enlargement-radius / cross-section-median overrides.
*Parallel vessel jobs* (default 1) measures that many vessels at once, as `--n-workers` does. The case id defaults to the layer name. It shows the
centerlines, the vessel surfaces (one Mesh-panel surface whose vertices carry their `label`) and
the lesions as points (orange stenoses, blue enlargements, opaque by confidence, sized by severity).

The results window gets four tables:
- **vessels**: every column of §6.1, then the per-vessel aggregates of the earlier measures:
  - `00_Path_Summary` (§5.3);
  - volumetry;
  - the tortuosity workbook (§7).

  A clashing name gets `_legacy`; e.g. `tortuosity_dm_legacy` is cow's length-weighted mean over
  segments.
- **lesions** (§6.3): clicking a row shows that lesion and opens it in the *Lesion inspector*.
- **segments** and **subject**.

On the vessel table, *Colour labels by* colours the measured labels and the surfaces by any
column. Visualization → *TOF morphometrics* does the same for an existing stage-7 folder.

**Analysing lesions** (`nvitk.gui.viz.lesion_inspector`):
- **Hover** a lesion point for its card: vessel and segment, diameter / area % against the
  reference (with both radii), taper %, length, position, and the confidence with each penalty in
  words.
- **Click** it, or its table row, to open the **Lesion inspector** dock. It plots the calibre
  along the measured path:
  - the radius the lesion was tested on and the other radius;
  - the dashed taper reference, and the dotted threshold the radius had to cross;
  - the lesion's run and the green reference window its severity is measured against;
  - the siphon mask and the path's other lesions.

  It shows why the lesion was called, and whether it is a real calibre change or a local artefact
  (a cross-section spike at a branch opening, say). ◀ / ▶ step through the lesions.
- **Morpho lesion extents** draws each lesion's run along its centerline.
- The lesion table's rows are tinted by confidence class (green high, yellow moderate, red low).
  A filter above it hides the less confident lesions in the viewer (points and extents): all,
  moderate and high, high only, or any minimum confidence.

Outputs older than the reference columns get the stenosis reference back from the stored
percentages.

---

## See also

- [M-Hemodynamics.md](M-Hemodynamics.md) — the 4D-flow side (stage 6)
- [M-AutoQC.md](M-AutoQC.md) — automatic quality control (stage 9)
