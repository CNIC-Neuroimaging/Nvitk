# Hemodynamic metrics — qvtpy cheatsheet

Everything the **4D-flow** side of the pipeline measures: what it is, how it is computed, in
which unit, and where it lands.

All of it is produced by **stage 6** ([stage6_measure.py](src/nvitk/pipes/qvtpy/stage6_measure.py)),
from three inputs: the stage-5 LOC table (`locs.csv`), the stage-4 multilabel mask
(`seg_4dflow`), and the AP/RL/FH phase + magnitude NIfTIs.

```
locs.csv ─┐
seg_4dflow├─► stage 6 ─┬─► loc_measurements.csv      (one row per LOC)
phases   ─┘            ├─► pitc_profile.csv          (one row per sampled station)
                       ├─► vessel_hemodynamics.csv   (one row per root / per branch)
                       └─► plots/ + pitc_masks/
```

Two independent measurement modes run in the same stage:

| Mode | Sampling | Answers | Output |
|---|---|---|---|
| **LOC-wise** | A handful of curated stations chosen by stage 5 | "What is the flow *here*?" | `loc_measurements.csv` |
| **Dense profile** | Every station along every named branch of a root tree | "How does pulsatility *evolve* along the tree?" | `pitc_profile.csv` + `vessel_hemodynamics.csv` |

---

## 0 · The primitives everything is built on

Every metric below reduces to one oblique cross-section and one time series.
Source: [cross_section.py](src/nvitk/measure/cross_section.py).

| Step | What happens | Key detail |
|---|---|---|
| **Plane** | A plane normal to the centerline tangent is cut through the volume at the station | Grid size ≈ `2·radius_vox·4 + 1` when supersampling (default `radius_vox = 10` vox) |
| **In-plane mask** | Either the stage-4 label is re-sliced (`label_constrain=True`, default) or the CD/magnitude slice is re-thresholded (`measure_resegment`) | Only the connected component nearest the plane centre is kept |
| **Area** `A` | `count(mask) × pixel_area`, with tilt-corrected pixel spacing | mm² — see `_cross_section_area_mm2` |
| **Circularity** | `(R_in / R_out)²` from the in-plane distance transform | 0–1; 1 = perfect disk. Used as a quality term, not reported per LOC in the DB |
| **Velocity** `v(t)` | Masked mean of `(vx,vy,vz)·t̂` on the plane, per cardiac frame | mm/s |
| **Flow** `Q(t)` | `v(t) · A / 1000` | mL/s |

> **Sign convention.** `v(t)` and `Q(t)` are taken as **magnitudes** (`abs`) before any index is
> computed, so a flipped centerline tangent cannot invert a mean flow or a PI.
> Consequence: communicating arteries carry no directional information in this pipeline.

---

## 1 · LOC-wise metrics — `loc_measurements.csv`

Computed by [loc_measure.py](src/nvitk/pipes/qvtpy/util/loc/loc_measure.py), one row per stage-5 LOC.
`nt` = number of cardiac frames.

| Column | Formula | Unit |
|---|---|---|
| `loc_cross_section_area_mm2` | in-plane mask area (§0) | mm² |
| `loc_mean_velocity_mm_s` | `mean_t abs(v(t))` | mm/s |
| `loc_mean_flow_ml_s` | `mean_t abs(Q(t))` | mL/s |
| `loc_pi` | **PI** = `(max_t Q − min_t Q) / mean_t Q` | — |
| `loc_ri` | **RI** = `(max_t Q − min_t Q) / max_t abs(Q)` | — |
| `loc_velocity_mm_s_t{0..nt-1}` | per-frame `abs(v(t))` | mm/s |
| `loc_flow_ml_s_t{0..nt-1}` | per-frame `abs(Q(t))` | mL/s |
| `loc_role` | `init` / `fin` / `mid` — which station of the vessel this is | — |

**PI vs RI.** Both use the same numerator (pulse amplitude). PI normalises by the *mean* of the
cycle, RI by its *peak*. Because `Q(t)` is already non-negative here, `RI = (max−min)/max` is the
classical Pourcelot index, and `PI ≥ RI` always.

### Where the LOCs sit

Placement is stage-5's job ([loc_selection.py](src/nvitk/pipes/qvtpy/util/loc/loc_selection.py)); it
matters because it defines what "the ICA flow" means:

| Vessel group | Rule |
|---|---|
| L/R ICA, basilar | one LOC each, preferably on a shared axial plane at maximum axial alignment |
| MCA / ACA / PCA | **dual** `init` + `fin` from trunk ↔ side-branch intersections (M1/A1/P1 then M2/A2/P2) |
| ACA fallback | AComm / circle-of-Willis junction stations when named side branches are missing |
| Venous sinuses | arc midpoint **inside the venous slab mask**; the SSS/STR pair is then separated away from the confluence, and a transverse sinus falls back to its longest in-mask segment |

The DB import keeps **one row per vessel**, preferring the `init` role (see
[db_publish.py](src/nvitk/pipes/qvtpy/common/db_publish.py)).

---

## 2 · Dense station profile — `pitc_profile.csv`

[vessel_hemodynamics.py](src/nvitk/pipes/qvtpy/util/hemodynamics/vessel_hemodynamics.py) re-derives
the centerlines from the mask, decomposes each vessel into its **named bifurcation branches**
(e.g. `LMCA-M1`, `LMCA-M2a`) and samples a cross-section every `stride` points.

Each row is one station:

| Column | Meaning |
|---|---|
| `vessel_id` / `vessel_name` | parent label id / named branch |
| `root_region_id` | which tree it feeds — `L_ICA`, `R_ICA` or `Basilar` |
| `station_index` | index along the branch polyline |
| `distance_mm` | **arc length from the tree root**, including branch offsets |
| `pi` | PI of this station's `Q(t)` |
| `quality` / `quality_metric` | station quality score, 0–4, and which metric produced it (§3) |
| `area_mm2`, `circularity` | cross-section geometry |
| `flow_mean_ml_s` | `mean_t Q(t)` |

### The three trees

| `region_id` | Root | Branches admitted |
|---|---|---|
| `L_ICA` | LICA | LACA, LMCA |
| `R_ICA` | RICA | RACA, RMCA |
| `Basilar` | BA | LPCA, RPCA (+ LVA, RVA as proximal starts) |

A hard allow-list (`PITC_GROUP_ALLOWED_IDS`) rejects any station whose label does not belong to its
root, and **communicating arteries are always excluded** — their unsigned flow would corrupt the
transmission fit.

### Distance bookkeeping

```
root station        distance = arc(root, station)
branch station      distance = arc_total(root) + gap(root_end → branch_start) + arc(branch, station)
basilar (dual VA)   distance = mean(arc_total(LVA), arc_total(RVA)) + arc(BA, station)
```

When both vertebrals are present the posterior tree starts at the **two VA tips** rather than the
basilar's inferior tip, and each VA is sampled from its own origin at offset 0.

---

## 3 · Station quality — the gate for everything downstream

Nothing enters a PITC or PWV fit without passing quality. Two metrics are available
(`--pitc-quality-metric`), both on a **0–4** scale:

### `stdv_from_mean` (default) — QVTplus `StdvFromMean` port

Evaluated over a sliding window of neighbouring stations on the same branch
(`branch_window_slices`, the 0-based port of the MATLAB indexing). Four terms, each clipped, summed
and clamped to `[0, 4]`:

| Term | Formula | Rewards |
|---|---|---|
| `qv_meanflow` | `1 − std(Q̄)/floor_f`, clipped to `[-1, 1]` | flow that is stable station-to-station |
| `qv_area` | `1 − std(A)/floor_a`, clipped to `[-1, 1]` | caliber that is stable station-to-station |
| `qv_circ` | `mean(circularity)`, clipped to `[0, 1]` | round lumens (not partial-volume smears) |
| `qv_tight` | `1 − mean_t(max_i Q − min_i Q)/floor_f`, clipped to `[-1, 1]` | waveforms that agree across the window |

`floor_f` / `floor_a` are robust denominators (`max(abs(mean), 5 % of peak, 1e-9)`) so a near-zero
mean flow cannot blow the score up.

### `waveform` — single-station alternative

`Q = 4 / (1 + roughness/amplitude)` where `roughness = std(Δ²Q)` and `amplitude = max Q − min Q`.
Scores one station in isolation; useful when branch context is unreliable.

### Turning quality into weights

```
w = clip( (Q − thresh) / (4 − thresh), 0, 1 )        thresh = QUALITY_THRESH_DEFAULT = 2.5
```

A station at Q ≤ 2.5 gets weight 0 and is excluded outright; Q = 4 gets weight 1.
Source: `quality_weights` in [hemodynamics.py](src/nvitk/measure/hemodynamics.py).

---

## 4 · PITC — Pulsatility Index Transmission Coefficient

**Question it answers:** how fast does pulsatility decay as blood moves distally down one tree?

```
PI(d) = pitc_slope · d + pitc_intercept
```

Weighted least squares of every admitted station's PI against its distance from the root, using the
quality weights above. Per root, written to `vessel_hemodynamics.csv` on the `row_kind = root` row.

| Column | Meaning | Unit |
|---|---|---|
| `pitc_slope` | PI change per mm of travel — **negative = pulsatility is damped distally** | 1/mm |
| `pitc_intercept` | Extrapolated PI at the root (d = 0) | — |
| `pitc_r2` | Weighted R² of the fit | — |
| `pitc_n` | Stations that survived `w > 0` | count |
| `global_pi` | Quality-weighted mean PI over the whole tree | — |

Returns all-NaN when fewer than 2 stations carry positive weight.

---

## 5 · PWV — Pulse Wave Velocity

**Question it answers:** how fast does the pressure wave itself travel?

Eligibility for **both** estimators: `quality > 2.5`, **≥ 3 stations**, sorted by distance, and a
valid `temporal_resolution_s` (derived from the phase NIfTI header). Otherwise both are written
empty.

### 5.1 Björnfot optimizer — the DB variable (`pwv`)

Port of QVTplus `enc_PWV_WO` / `PWVest3_share`.

1. **Normalise** each station: `Q/A` → velocity, subtract mean, divide by std.
2. **Weight** each station by `A / scaling²`, normalised by the max (`weight_mode="area"`).
3. **Jointly fit** one shared velocity template *and* one PWV by weighted least squares
   (`scipy.optimize.least_squares`, method `lm`, initial PWV = 10 m/s). The model shifts the shared
   template by `d / PWV` at every station and minimises the weighted residual.

| Column | Meaning |
|---|---|
| `pwv_bjornfoot_m_s` | Accepted PWV, else empty string |
| `pwv_n_stations` | Stations entering the fit |

Diagnostics kept for the QC figure (not persisted to CSV): per-station weighted residual RMS,
expected delay, and template-vs-observed waveform correlation.

### 5.2 Fielding cross-correlation — the QC estimator

Port of QVTplus `enc_PWV_XCor`.

1. Spline-upsample each cycle to **500 samples** (tripled for periodic boundaries).
2. Circular cross-correlation against the **most proximal** station → transit delay `τᵢ` in seconds.
3. Reject outlier delays with a robust MAD z-score (`abs(z) > 3.5`, only when ≥ 4 points remain).
4. Weighted linear fit `τ = d / PWV` reusing the Björnfot area weights → `PWV = 1 / slope`.

| Column | Meaning |
|---|---|
| `pwv_fielding_m_s` | Accepted PWV, else empty string |
| `pwv_r_fielding` | Mean absolute cross-correlation of the stations kept — the confidence in the delays |

### 5.3 Acceptance gate

```
accept  ⇔  0 < PWV < 30 m/s          (PWV_MIN_M_S / PWV_MAX_M_S)
```

Applied identically to both estimators. A rejected value is written as an **empty cell**, never as a
number — so a downstream mean is never contaminated by a failed fit.

---

## 6 · Damping index

Per **named branch**, on the `row_kind = branch` rows of `vessel_hemodynamics.csv`:

```
damping_index = (PI_root − PI_branch) / PI_root
```

where `PI_root` is the mean PI over the root vessel's own stations and `PI_branch` the mean over
that branch's stations. Positive = the branch is less pulsatile than its parent (normal);
negative = pulsatility *rose* distally, which is worth a look.

---

## 7 · What reaches the database

[db_publish.py](src/nvitk/pipes/qvtpy/common/db_publish.py) imports into `image_measurements`,
keyed by `(subject_uid, pipeline_id, variable_id, region_id, frame_index)`.

| `variable_id` | Source column | Unit | Note |
|---|---|---|---|
| `flow_mean` | `loc_mean_flow_ml_s` | **mL/min** | ×60 on import — the AutoQC bands assume this |
| `flow_tseries` | `loc_flow_ml_s_t{n}` | **mL/min** | one row per `frame_index` |
| `cross_section_area` | `loc_cross_section_area_mm2` | mm2 | Stage 9's hypoplasia gate reads this |
| `velocity_mean` | `loc_mean_velocity_mm_s` | mm/s | magnitude — `mean_t abs(v(t))`, not signed |
| `pi` | `loc_pi` | — | |
| `ri` | `loc_ri` | — | |
| `pitc_slope` | `pitc_slope` | 1/mm | region = `L_ICA` / `R_ICA` / `Basilar` |
| `pitc_intercept` | `pitc_intercept` | — | |
| `pwv` | `pwv_bjornfoot_m_s` | m/s | |
| `pwv_fielding_xcor` | `pwv_fielding_m_s` | m/s | |
| `damping_index` | `damping_index` | — | region = branch name |

Rows with a missing or non-numeric value are skipped rather than written as zero — a vessel with no
measurable cross-section is absent from `cross_section_area`, not recorded as 0 mm2 (which stage 9
would read as hypoplastic).

---

## 8 · Knobs

Stage-6 CLI flags that change the numbers above:

| Flag | Default | Effect |
|---|---|---|
| `--pitc-stride` | 1 | Sample every *n*-th centerline point. Higher = faster, coarser fits |
| `--pitc-quality-thresh` | 2.5 | Quality gate for PITC **and** PWV. Raising it shrinks `pitc_n` / `pwv_n_stations` |
| `--pitc-quality-metric` | `stdv_from_mean` | or `waveform` |
| `--pitc-measure-resegment` | off | Re-threshold the plane instead of re-slicing the label |
| `--pitc-label-constrain` | on | Restrict the in-plane mask to the station's own label |
| `--skip-pitc` | off | LOC measurements only |
| `NVITK_PITC_BRANCH_WORKERS` | ≤4 | Per-branch sampling threads |

Cross-section knobs (`cross_section_radius_vox`, `cross_section_res`, `plane_interp_order`,
`cs_supersampling`, `thr_algorithm`) apply to both modes and are documented inline in
[cross_section.py](src/nvitk/measure/cross_section.py).

---

## See also

- [M-Morphometrics.md](M-Morphometrics.md) — the TOF geometry side (stage 7)
- [M-AutoQC.md](M-AutoQC.md) — how these numbers are checked (stage 9)
