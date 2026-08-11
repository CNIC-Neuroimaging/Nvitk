# AutoQC metrics — qvtpy cheatsheet

Stage 6 answers *"what is the flow?"*. **Stage 9** answers *"should we believe it?"*.

Produced by [stage9_autoqc.py](src/nvitk/pipes/qvtpy/stage9_autoqc.py) on top of the literature-grounded
checks in [hemodynamics.py](src/nvitk/measure/hemodynamics.py). It reads **published measurements**,
not pipeline files, so it is safe to re-run after any re-import and never touches stage outputs.

```
image_measurements
  flow_mean            ─┐
  cross_section_area   ─┤
                        ├─► stage 9 ─┬─► image_measurements     (per-vessel qc_*)
pitc_profile.csv       ─┘            └─► clinical_measurements  (per-subject qc_*)
  (station flows, optional)
```

> **Soft by design.** A low score marks a measurement *worth looking at*, never one to delete.
> Real pathology — high-grade stenosis, AVM, moyamoya collaterals — legitimately falls outside a
> healthy-cohort band with nothing wrong with the data. The pipeline flags and never drops.

---

## 0 · At a glance

| Metric | Level | Range | Fails when | Tolerance constant |
|---|---|---|---|---|
| `qc_hypoplastic` | vessel | 0 / 1 | caliber < 0.8 mm — **not a failure**, a gate | `HYPOPLASIA_DIAM_MM` = 0.8 mm |
| `qc_flow_plausible` | vessel | 0–1 | flow outside the vessel's literature band | `FLOW_BAND_TOLERANCE` = 0.15 |
| `qc_conservation` | vessel | signed ratio | junction inflow ≠ outflow | 0.20 / 0.30 / 0.20 by class ⚠️ |
| `qc_segment_cv` | vessel | ≥ 0 | flow drifts along a non-branching segment | `SEGMENT_CV_TOL` = 0.25 ⚠️ |
| `qc_score` | vessel | 0–1 | combined — see §5 | — |
| `qc_flag` | vessel | 0 / 1 / NaN | `qc_score ≤ 0.5` | `QC_SCORE_FLAG_BELOW` = 0.5 |
| `qc_ap_share` | subject | % | — (reported) | — |
| `qc_ap_flag` | subject | 0 / 1 | anterior share outside 72 ± 10 % | `ANTERIOR_SHARE_PCT` / `_TOL_PCT` |
| `qc_subject_flag` | subject | 0 / 1 | any vessel flag **or** the AP flag | — |

**NaN ≠ pass.** A check that could not be evaluated writes NaN and is skipped in the mean. A vessel
where *no* check applied gets `qc_flag = NaN`, not 0.

⚠️ **Rows marked with a warning sign carry provisional tolerances** — they are still being assessed
and will be finalised. See §4.

---

## 1 · Unit inference — the first thing that runs

The literature bands are in **mL/min**. Stage 6 publishes flow in mL/s (×60 on import), but other
importers publish per-minute tables directly. Assuming either one silently mis-scales the whole
cohort by 60×, so the scale is **inferred from the magnitude** and logged:

```
median |flow| ≥ 20  →  already mL/min  (×1)
median |flow| <  20  →  mL/s            (×60)
```

`FLOW_SCALE_BOUNDARY = 20.0`. A healthy ICA is ~257 mL/min ≈ 4.3 mL/s — the two scales are two
orders of magnitude apart, so nothing plausible sits near the boundary. Override with `--flow-scale`.

The stage also warns when one `(subject, vessel)` cell carries **more than one region spelling**
(`LICA` from one importer, `left_ica` from another) — that means two pipelines are mixed in the
dataset and every conservation check would silently average them. Restrict with `--pipeline`.

---

## 2 · `qc_hypoplastic` — the gate, evaluated first

```
equivalent_diameter = 2 · √(area / π)
hypoplastic  ⇔  equivalent_diameter < 0.8 mm      (or area ≤ 0 / non-finite)
```

Threshold from Krabbe-Hartkamp et al. 1998 (< 0.8 mm on 3D TOF MRA).

**Why it runs first:** a hypoplastic vessel carries almost no flow *by anatomy*. Scoring it against a
patent-vessel band would report a normal circle of Willis as a data-quality failure. When
`qc_hypoplastic == 1`, `qc_flow_plausible` is set to NaN and skipped.

`cross_section_area` is published by the stage-6 DB importer from `loc_cross_section_area_mm2`
(unit `mm2`). If it is absent — a cohort imported before that was added — every vessel is scored as
patent, and the stage warns loudly rather than passing over it, because that systematically
over-flags the anterior circle.

---

## 3 · `qc_flow_plausible` — literature band, 0–1

### The bands

Time-averaged **|flow| in mL/min**, per vessel:

| Key | Low | High | | Key | Low | High |
|---|---|---|---|---|---|---|
| `ICA` | 56.5 | 521.3 | | `MCA` | 26.5 | 310.7 |
| `VA` | 17.0 | 215.8 | | `MCAdist` | 1.0 | 110.5 |
| `BA` | 11.0 | 348.4 | | `ACA` | 14.0 | 176.8 |
| `PCA` | 9.0 | 117.0 | | `ACAdist` | 1.0 | 78.0 |

Derived from **Zarrinkoob et al. 2015** (*J Cereb Blood Flow Metab* 35:648–654, 94 healthy subjects,
2D PC-MRI), but **not used verbatim**: 4D flow reads 20–46 % lower than 2D PC-MRI at matched levels,
so the band is widened to `0.5·(μ − 3σ) … 1.3·(μ + 3σ)`. The point is to catch gross failures — flow
5–10× physiological, or ~0 in a patent vessel — not to grade normal variation.

**Exempt (always NaN):** `ACOMM`, `PCOMM`, `LPCOMM`, `RPCOMM`. A communicating artery carrying
near-zero or reversed flow is a normal circle of Willis — Krabbe-Hartkamp et al. found the CoW
anatomically complete in only ~51 % of healthy subjects.

**Also NaN:** anything unrecognised, including the venous sinuses (`SSS`, `STRS`, `LTS`, `RTS`) —
they have no band and are silently skipped in the plausibility term.

**Name resolution.** Side prefixes are stripped (`LICA`, `Left_ICA`, `L-ICA` → `ICA`), and aliases
map `BASI`/`BASILAR` → `BA`, `VERT`/`VERTEBRAL` → `VA`, `M1` → `MCA`, `A1` → `ACA`, `P2` → `PCA`,
`MCAdist` / `ACAdist` for distal segments.

### The score

```
inside the band                     →  1.0
outside, crossing bound B           →  max(0, 1 − excess / 0.15)
                                       excess = |value − B| / |B|
```

Score reaches **0 at 15 % beyond the bound** (`FLOW_BAND_TOLERANCE`). The decay is deliberately
steep, because the band is already far wider than any healthy cohort — a value outside it is not an
unusual subject, it is a number a correctly segmented vessel could not have produced. (A gentler
decay let an ICA at 31 mL/min — an eighth of a healthy one — score 0.55 and pass a 0.5 gate.)

Symmetric on both sides: the low end already carries extra room for the 4D-vs-2D underestimation.

---

## 4 · `qc_conservation` — junction mass balance

```
residual     = Σ inflow − Σ outflow
qc_conservation = residual / Σ inflow          (signed, dimensionless)
```

Positive = flow is being *lost* between parent and daughters (unmeasured side branch, or an
underestimated daughter). Zero = perfect conservation.

### The rules

Terms come from [`CONSERVATION_RULES`](src/nvitk/stats/vessel_network.py). Communicating-artery terms
are **dropped** — LOC flows are unsigned, so a PComm term cannot be given a direction — which
reduces each rule to the classical parent → daughters check.

| Rule | Balance evaluated | Anchors (rows that carry the residual) | Tolerance |
|---|---|---|---|
| `left_carotid_split` | `lica − laca − lmca` | `lica` | **0.20** arterial |
| `right_carotid_split` | `rica − raca − rmca` | `rica` | **0.20** arterial |
| `basilar_inflow` | `lva + rva − basi` | `basi`, `lva`, `rva` | **0.20** arterial |
| `posterior_split` | `basi − lpca − rpca` | `basi`, `lpca`, `rpca` | **0.30** distal |
| `venous_drainage` | `sss + strs − lts − rts` | `sss`, `strs`, `lts`, `rts` | **0.20** venous |

A junction with any vessel missing is **skipped**, not failed. When a vessel participates in two
rules (the basilar is both the VA confluence and the PCA origin), the **worst residual by magnitude**
is kept, together with that rule's tolerance.

### Why three tolerances

There is no single published cranial threshold. ISMRM 2017 (Roberts et al.) observed ~1–10 %
residuals at well-conserved proximal junctions and 11–55 % where flow was clearly broken; venous
confluence imbalances of ~4–9 % are typical even in good data.

| Class | Tolerance | Reason it is looser / tighter |
|---|---|---|
| Arterial proximal | 0.20 | Top of the "good" band, plus room for the unmeasured anterior choroidal / ophthalmic |
| Distal (BA → PCA) | 0.30 | AICA and SCA leave the basilar before the terminal bifurcation and are never measured |
| Venous | 0.20 | Cortical, petrosal and emissary tributaries join the transverse sinuses directly — a zero residual is anatomically unreachable |

> ⚠️ **These tolerances are provisional — under assessment, to be finalised.** The table above is
> what the code applies today: `CONSERVATION_TOL_ARTERIAL = 0.20`, `_DISTAL = 0.30`, `_VENOUS = 0.20`,
> with `CONSERVATION_TOL = 0.15` as the *single-number* default used by the CLI flag and by filters.
> Some inline comments and the `--conservation-tol` help text still quote an older 10 / 15 / 20 % set.
> Treat every number in §4 and `SEGMENT_CV_TOL` in §5 as a working value, and re-read the constants in
> [hemodynamics.py](src/nvitk/measure/hemodynamics.py) before quoting them anywhere that matters.

`--conservation-tol X` retunes the **arterial** gate and scales the other two by `X / 0.20`, so the
relative spacing survives an override.

**Ambiguity to keep in mind:** a failing junction is genuinely undecidable between "the flow
measurement is wrong" and "the vessel tree is incomplete" — a missed side branch removes real outflow
and looks identical.

---

## 5 · `qc_segment_cv` — along-segment consistency

Mass is conserved along a segment with no branches, so station-to-station flow should barely move.

The input is each station's **time-averaged (mean) flow**, and the metric is **normalized by the
segment's own mean** — so it is dimensionless and a 700 mL/min sinus and a 60 mL/min vertebral are
on one scale:

```
Q̄ᵢ            = station i's mean flow over the cardiac cycle   (flow_mean_ml_s)
qc_segment_cv = std(Q̄ᵢ) / |mean(Q̄ᵢ)|        over the vessel's main-path stations
```

Equivalently: normalize each station to the segment mean, then take the spread. Because the CV is
scale-invariant, normalizing the stations first and taking the SD afterwards gives the identical
number — the metric is already expressed "by mean and std along the segment", and there is nothing
further to normalize.

The same quantity ×100 is what `percent_variation_from_mean` produces per station for the
cohort-level report in §8, so the per-scan CV and that report's SD are the same statistic in
different units (0.03 here ⇔ 3 % there).

| | |
|---|---|
| **Source** | station flows (`flow_mean_ml_s`) from stage 6's `pitc_profile.csv` |
| **Grouping** | one CV **per individual vessel**, not per tree |
| **Stations used** | that vessel's **main path only** — bifurcation arms excluded |
| **Vessels** | every vessel the profile carries: L/R ICA, L/R VA, basilar, and the L/R ACA / MCA / PCA trunks. `SEGMENT_CV_EXCLUDED_NODES` (empty by default) holds anything to skip |
| **Minimum** | 3 finite stations; otherwise NaN |
| **Tolerance** | `SEGMENT_CV_TOL = 0.25` ⚠️ |

A low CV says the centerline and segmentation track the vessel; a high one says the centerline
drifted, partial volume ate the lumen, or a side branch was missed. The QVT validation paper reports
along-segment percent variation with SD ≈ 3 %, so 0.25 is a soft review gate (≈ 8× that scatter), not
a physiological limit.

### Per vessel, and main path only — why

**Per vessel, not per tree.** Every station in `pitc_profile.csv` is tagged with the root tree it
feeds (`L_ICA`, `R_ICA`, `Basilar`), but a tree spans several vessels *and the bifurcation between
them*, where flow is supposed to change. Grouping by tree folds the carotid terminus into the ICA's
own consistency and measures anatomy instead of data quality. Each vessel is scored on its own.

**Main path only.** Stage 4/6 decompose a branched territory into a trunk plus bifurcation arms and
name them with `qvtpy_branch_names`:

| Vessel kind | Trunk (**used**) | Arms / extras (**dropped**) |
|---|---|---|
| ACA / MCA / PCA | `LACA-A1`, `LMCA-M1`, `LPCA-P1` | `LMCA-M2a`, `LMCA-M2b`, … |
| ICA / basilar / VA | `LICA`, `BASILAR`, `LVA` | `LICA-b2`, `BASILAR-b3`, … |

Pooling a trunk with its arms breaks the premise the metric rests on: flow legitimately steps down
past each bifurcation, so the CV would measure the branching, not the segmentation.
`is_main_path_station` accepts either the canonical trunk name or the bare vessel name — so a
profile written before branch naming still resolves — and never accepts an `-M2a` / `-b2` suffix.

`region_id` is published as the plain vessel name (`LMCA`, not `LMCA-M1`), so it joins the flow rows
directly.

> **What changed.** This check previously resolved only `LICA`, `RICA` and `BASILAR` — the three
> tree roots. The ACA/MCA/PCA trunks were silently dropped because `LMCA-M1` does not resolve to a
> canonical node, and the vertebrals were never in the allow-list. `sss` was in the list but
> unreachable: `pitc_profile.csv` is written for the arterial root groups only and carries no venous
> stations. Coverage goes from 3 vessels to as many as 11.

Requires `--results-root` / a results source. Without one, `qc_segment_cv` stays NaN and drops out of
the score.

---

## 6 · `qc_score` and `qc_flag` — combining

Residuals and CVs are **errors**, not scores, so each is mapped onto `[0, 1]` by its own tolerance
before averaging:

```
conservation_score = clip( 1 − |qc_conservation| / tol_for_that_junction , 0, 1 )
segment_score      = clip( 1 − |qc_segment_cv|   / SEGMENT_CV_TOL        , 0, 1 )

qc_score = mean( qc_flow_plausible, conservation_score, segment_score )    skipping NaN

qc_flag  = 1  if qc_score ≤ 0.5
         = 0  if qc_score >  0.5
         = NaN if no check applied
```

So a metric exactly *at* its tolerance scores 0 on that term, and a vessel with only one applicable
check is judged entirely by it.

---

## 7 · Subject level

### `qc_ap_share` / `qc_ap_flag`

```
anterior  = lica + rica
posterior = basi          (or lva + rva when the basilar is absent)

qc_ap_share = 100 · anterior / (anterior + posterior)
qc_ap_flag  = 1  ⇔  |qc_ap_share − 72| > 10
```

Zarrinkoob et al. report a **72 / 28 %** split with SD ~4–5 %, stable across age, sex and brain
volume — the cheapest subject-level screen available, needing no reference scan. The ±10 %
tolerance is deliberately wider than the population SD because anatomic variants (fetal PCA,
hypoplastic A1) shift the ratio without any measurement being wrong.

Computed only when **both** carotids *and* either the basilar or both vertebrals are present —
never from half the inflow. A non-finite share counts as outside the band.

### `qc_subject_flag`

```
qc_subject_flag = 1  ⇔  (any vessel qc_flag > 0)  or  (qc_ap_flag > 0)
```

---

## 8 · Cohort-level checks (reports, not gates)

The metrics above answer *"should this scan be reviewed?"*. These answer the other question the 4D
Flow consensus statement poses — *"does this pipeline conserve flow at all?"* — which is a property
of a **cohort**, not a subject. Neither can gate an individual scan; both belong in a dataset QC
report. Available as functions, not published by the stage.

### `consensus_junction_report`

Regresses junction outflow on inflow **across subjects** — the internal-consistency check the
QVT/CPS validation work reports as slope, intercept, 95 % CI and Pearson *r*.

| Output | Meaning |
|---|---|
| `slope`, `slope_ci_low/high` | **Read this one.** Conservation ⇔ slope ≈ 1 |
| `slope_includes_one` | the actual pass/fail of the check |
| `intercept` + CI | a systematic offset a relative residual would hide in its denominator |
| `r`, `p_value` | correlation — *not* evidence of conservation on its own |
| `mean_rel_residual` | the quantity the per-scan gate thresholds |
| `n`, `n_trimmed` | pairs used, pairs removed by the robust fences |

> A pipeline that loses a fixed fraction of outflow still correlates near-perfectly: *r* stays above
> 0.99 while the slope sits at 0.88. **`r` alone certifies nothing.**

Junctions checked (`CONSENSUS_JUNCTIONS`): both carotid splits and the venous confluence — the three
the validation paper analyses — plus `basilar_inflow` and `posterior_split`.

`robust=True` drops pairs outside **Tukey fences** `[Q1 − 3·IQR, Q3 + 3·IQR]` on either axis before
fitting. One leaked segmentation reporting 10⁶ mL/min sits at enormous leverage and can set the slope
by itself, so the fenced fit is the honest sensitivity check — report it *beside* the full one, never
instead of it.

### `consensus_segment_report`

Each segment is centred on its own mean and expressed in percent (which puts a 700 mL/min sinus and a
60 mL/min vertebral on one scale), all stations are pooled, and a **minimum-variance unbiased
Gaussian** is fitted (sample SD corrected by the `c4(n)` factor; χ² intervals).

- The **SD** is the result — the QVT validation work reports roughly **3 %**.
- A `mean_pct` far from zero means the *pooling* is lopsided, not that flow is drifting.
- Segments: `lica`, `rica`, `sss` (`CONSENSUS_SEGMENTS`), plus a pooled `all` row.

> **Not the same segment set as §5.** `CONSENSUS_SEGMENTS` deliberately mirrors the segments the
> validation paper reports, so the SD here is comparable to its ≈ 3 %. The per-scan `qc_segment_cv`
> runs on every vessel instead. Note also that this function groups by `canonical_node`, so it
> expects region labels that already resolve — feeding it raw `pitc_profile.csv` rows would drop
> every branch-named station (`LMCA-M1` → no node), and `sss` never appears in that file at all.

---

## 9 · What is published

All QC metrics land as **ordinary measurements** — same schema, provenance and catalog registration
as any importer-produced variable — so the Statmodels filter and the GUI colour picker can use them
without knowing this stage exists.

| `variable_id` | Table | Label |
|---|---|---|
| `qc_flow_plausible` | `image_measurements` | Flow plausibility (literature band, 0–1) |
| `qc_hypoplastic` | `image_measurements` | Plausibly hypoplastic (< 0.8 mm) |
| `qc_conservation` | `image_measurements` | Junction mass-conservation residual |
| `qc_segment_cv` | `image_measurements` | Along-segment flow CV |
| `qc_score` | `image_measurements` | Combined per-vessel QC score (0–1) |
| `qc_flag` | `image_measurements` | Vessel QC flag (1 = review) |
| `qc_ap_share` | `clinical_measurements` | Anterior share of cerebral inflow (%) |
| `qc_ap_flag` | `clinical_measurements` | Anterior/posterior split flag (1 = outside 72 ± 10 %) |
| `qc_subject_flag` | `clinical_measurements` | Subject QC flag |

Upsert keys: `(subject_uid, region_id, variable_id, frame_index)` for image rows,
`(subject_uid, visit_id, variable_id)` for clinical rows. The SQLite index is rebuilt **once** after
every table is written — without it the Parquet holds the new metrics while every SQLite-backed read
(the GUI) still returns the old ones.

### Re-running: existing rows are cleared first

An upsert overwrites any row whose key it re-emits, but that is **not enough on its own**. A metric
that was computable last run and is not this run simply produces no row, and its old value would
survive — so the dataset would keep reporting, say, a `qc_conservation` residual for a junction
whose vessels are no longer all present.

Stage 9 therefore **purges before it writes** (`purge_subject_qc`): every `qc_*` row belonging to a
subject being scored is deleted, then only what this run actually computed is written back. Absence
is the honest representation of "not evaluated".

| Scope | Behaviour |
|---|---|
| Variables touched | only those in `QC_VARIABLES` — `flow_mean`, `pi`, and every other measurement are untouched |
| Subjects touched | only those in this run; other subjects' QC is left alone |
| Pipelines touched | matched on `variable_id` regardless of `pipeline_id` — this stage is the sole producer of `qc_*`, so a row under another pipeline id is a leftover, not a parallel result |
| SQLite | a purge that removes rows counts as a table change, so the index is rebuilt even if no variable produced a row to write |

Disable with `--no-purge-existing` to add QC for new subjects without touching what is stored.
Under `--dry-run` the purge is reported but nothing is removed.

---

## 10 · Running it

```bash
python -m nvitk.pipes.qvtpy.stage9_autoqc --dataset /path/to/dataset --dry-run
```

| Flag | Default | Effect |
|---|---|---|
| `--dataset` | settings | Dataset root; omit to use `.nvitk/settings.json` |
| `--pipeline` | `latest` | Which `image_measurements` pipeline to read |
| `--flow-variable` | `flow_mean` | The measurement being scored |
| `--area-variable` | `cross_section_area` | Needed to excuse hypoplastic vessels |
| `--flow-scale` | inferred | Force the mL/min conversion factor |
| `--conservation-tol` | 0.15 | Retunes the arterial gate; distal/venous scale with it |
| `--segment-cv-tol` | 0.25 | Along-segment CV gate |
| `--score-flag-below` | 0.5 | Combined-score flag threshold |
| `--submit` | `local` | Where to recover missing measurements from: `local` / `sge` (SFTP) / `xnat` |
| `--results-root` | configured | Results tree for recovery **and** for `pitc_profile.csv` segment CV |
| `--subjects` | all | Comma/space-separated subject ids |
| `--no-recover` | off | Score only what the dataset carries |
| `--purge-existing/--no-purge-existing` | purge | Clear each scored subject's previous `qc_*` rows before writing |
| `--dry-run/--write` | `--write` | Compute and report without writing |
| `--report PATH` | — | Also dump the per-vessel scores to CSV |

**Recovery.** Stage 6 writes its CSVs to disk before anything imports them, so a dataset whose import
has not run — or has run only for the flows, leaving no areas — can still be scored by pointing
`--results-root` at the tree ([autoqc_sources.py](src/nvitk/pipes/qvtpy/autoqc_sources.py)). Remote
modes stage into a temporary directory that is removed afterwards.

Two files are read, and only a missing variable is recovered:

| Dataset variable | Recovered from `loc_measurements.csv` column |
|---|---|
| `flow_mean` | `loc_mean_flow_ml_s` |
| `cross_section_area` | `loc_cross_section_area_mm2` |
| `velocity_mean` | `loc_mean_velocity_mm_s` |
| `pi` / `ri` | `loc_pi` / `loc_ri` |

`pitc_profile.csv` is read unconditionally (when a source is given) — it is the **only** input for
`qc_segment_cv`, so that check always needs a results source.

`cross_section_area` **is** published by the stage-6 DB importer, so the hypoplasia gate works
without recovery on any cohort imported after that change. Datasets imported before it have no area
rows; re-run the stage-6 publish, or pass `--results-root` for that run.

### Scoring a frame without running the stage

`compute_qc_columns(frame, flow_column=…, region_column=…, area_column=…)` adds the `qc_*` columns to
an already-assembled analysis frame (what the GUI does on a dataset where the stage has not run).
Results are **columns on that frame only** — nothing is published, and `qc_segment_cv` is always NaN
because no station profile is available. Running the stage remains the way to make the metrics
available to every later session.

### Plotting

[scripts/pesa_brain/plotter/autoqc_summary.py](scripts/pesa_brain/plotter/autoqc_summary.py) draws
per-vessel pass rates, score distributions, conservation residuals with the tolerance overlaid, and
the subject-level AP split against its 72 % reference.

---

## See also

- [M-Hemodynamics.md](M-Hemodynamics.md) — the measurements being checked (stage 6)
- [M-Morphometrics.md](M-Morphometrics.md) — TOF geometry (stage 7)
