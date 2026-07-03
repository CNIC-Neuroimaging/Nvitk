# qvtpy 4D-Flow MRI Pipeline — Quality Assessment

**Document type:** Internal engineering assessment (consulting / advisory)
**Subject:** `nvitk.pipes.qvtpy` — Python reimplementation and evolution of the MATLAB QVTPlus tool
**Date:** 3 July 2026
**Status:** Advisory only. No pipeline code was modified in the preparation of this report.

---

## 1. Executive summary

The `qvtpy` pipeline is a well-structured, modular reimplementation of QVTPlus that processes 4D-Flow MRI of the cerebral vasculature end to end, from XNAT retrieval through per-location hemodynamic measurement. Its defining architectural decision — carrying a **named multilabel anatomy** (arterial labels 1–14, venous labels 31–34) through every stage instead of QVTPlus's binary contrast-difference (CD) graph plus manual per-branch curation — is a genuine advance. It removes a major manual bottleneck and makes per-vessel, cross-section-based measurement natural.

That same decision, however, **relocates the pipeline's principal risk** from manual labeling (QVTPlus) to automated **segmentation quality** (qvtpy). Every downstream metric — velocity, flow, PI/RI, and the newer vessel-level PITC/PWV work — inherits the fidelity of the stage-4 multilabel segmentation, which is produced by a per-vessel local threshold followed by an **unbounded, single-gate 6-connected region-growing (RG)** step. The RG primitive has no volume ceiling and no rollback, so a single leak into adjacent bright tissue can silently and catastrophically over-segment a vessel. Several anatomically hard structures — the communicating arteries, the vertebral arteries, and the venous sinuses — are resolved by **heuristics applied after segmentation** (centerline-only seeds, skeleton bifurcation detection, greedy geometric assignment) that are sensitive to registration accuracy and upstream segmentation quality.

The engineering response to these risks is, encouragingly, **already underway**: RG volume caps with rollback, mandatory native-eICAB SCA barrier walls (including on the posterior communicating arteries), conservative venous RG, an ACA left/right split that no longer depends on segmenting the AComm, improved vertebral detection, temporal-resolution propagation, and PITC/PWV metrics with quality gating are all in the current development cycle. This report is intended to (a) document the current state and its failure modes precisely, and (b) confirm that the in-progress mitigations target the right problems, while flagging what still needs hardening — most importantly **automated per-subject QC that gates rather than warns**, and porting QVTPlus's `StdvFromMean` quality metric into the new PITC/PWV fits.

**Bottom line:** the architecture is sound and the trajectory is correct. The pipeline is not yet safe to run fully unattended at population scale because its dominant failure mode (RG over-segmentation) is silent and its QC surface is advisory, not enforcing. Closing that gap is the highest-value work.

---

## 2. Pipeline strengths

**Multilabel anatomy versus the QVTPlus binary-CD graph.** QVTPlus segments a single binary CD vasculature, builds a centerline graph, and relies on a manually curated `LabelsQVT.csv` to associate graph branches with named vessels. `qvtpy` instead maintains an explicit multilabel volume (`seg_4dflow`) with fixed, anatomically meaningful label ids for 14 arterial and 4 venous territories. This is structurally superior: vessel identity is a first-class property of the data rather than a spreadsheet lookup applied post hoc, and it enables per-vessel cross-section sampling without graph-to-name reconciliation.

**Modular stages with QC artifacts.** The pipeline is cleanly decomposed (stage0 download/convert → stage1 eICAB → stage2 registration → stage3 centerlines → stage4 segmentation → stage5 LOC generation → stage6 measurement). Each stage persists intermediate outputs and metadata: `segmentation_meta.json` records every RG parameter, per-vessel voxel counts at each processing step (`n_voxels_after_threshold`, `n_voxels_after_island_clean`, `n_voxels_after_region_growing`), thresholds, and warnings; `vertebral_split.json` and the ACA sequential-grow metadata capture the outcome of the hard heuristics; stage6 emits five-panel cross-section QC PNGs. This instrumentation is a real asset — most of the QC signals a gating layer would need are **already being computed and written to disk**.

**eICAB priors.** Using eICAB arterial segmentation on the TOF as an anatomical prior (warped into 4D-flow space in stage 2) gives the pipeline a principled starting point for label identity and for RG barriers. The use of eICAB **native** labels that qvtpy otherwise drops — LSCA/RSCA (15/16) — purely as region-growing barrier walls is a clever reuse of anatomy the pipeline does not itself measure.

**Reusable `nvitk.measure` primitives.** Hemodynamic indices (`pulsatility_index`, `resistivity_index`, `mean_velocity_mm_s`), phase-to-velocity conversion, cross-section extraction, and masked-plane flow series live in shared, backend-agnostic `nvitk` modules rather than being embedded in the pipeline. This makes the measurement layer testable in isolation and reusable across the sibling QVTPlus-style code paths.

---

## 3. Critical weak points

The table below summarizes the highest-impact risks. Severity is rated **High / Medium / Low** by the combination of likelihood and the magnitude of the effect on reported metrics. Each row is expanded in the notes that follow.

| # | Area | Risk | Severity |
|---|------|------|----------|
| 1 | **Region growing** (`region_growing.py`) | Unbounded 6-connected BFS with a single mean-seed intensity gate and no volume cap or rollback. Once a leak crosses into adjacent bright tissue, growth continues until the intensity gate or a barrier stops it — with no per-vessel size sanity check. Produces catastrophic, silent over-segmentation. | **High** |
| 2 | **Communicating arteries** (LPComm/RPComm/AComm) | Segmented centerline-seed + RG only, with *undilated* other-label barriers and (historically) **no native-eICAB SCA wall** on the posterior communicating arteries. Small, low-signal vessels with no volume caps are highly leak-prone into ICA/PCA/MCA. | **High** |
| 3 | **Venous labeling** (`venous_heuristics.py`) | Greedy, first-come geometric assignment of skeleton branches to SSSV/STRV/LTSV/RTSV; historically no venous RG; scoring depends on midline (`nx/2`) and direction priors that are sensitive to registration and head positioning. Mislabeling is a topology error, not a graceful degradation. | **High** |
| 4 | **eICAB dependency & registration** | Label identity, ACA junction inference, and RG barriers all depend on eICAB warped into 4D-flow space (stage 2). A registration failure propagates silently into every downstream label. Missing native-eICAB file degrades barriers to a `log.warning`, not a stop. | **High** |
| 5 | **Temporal resolution propagation** | `mean_flow_ml_s` accepts `temporal_resolution_s` but discards it (`del temporal_resolution_s`); stage-6 flow is per-frame with no time base. PWV (which is intrinsically a timing measurement) cannot be reliable without `temporal_resolution_s` threaded end to end. | **High** (for PWV) |
| 6 | **Vertebral split** (`vertebral_split.py`) | LVA/RVA are recovered by a *post-hoc* skeleton Y-junction heuristic on the basilar label, with L/R assigned by the y-coordinate mean of inferior branches. No stage-3 VA centerlines to anchor identity → susceptible to L/R swap and to missing the split entirely on a merged or noisy basilar. | **Medium–High** |
| 7 | **PITC sampling** | Dense per-station intra-vessel sampling compounds any per-station segmentation error; with no quality weighting, noisy stations dilute or bias the vessel-level fit. | **Medium–High** |
| 8 | **QC coverage** | Failures are **warnings, not gates**: missing eICAB, empty centerline masks, QC-PNG exceptions, and implausible voxel counts all log and continue. No automated per-subject accept/reject. | **High** (process risk) |
| 9 | **Label paste order effects** | `build_seg_4dflow_local` iterates labels in ascending id order and `_paste_crop_mask`/RG only write into `seg == 0`. The first-labeled vessel wins contested voxels; ACA/comm/vertebral results depend on processing order. | **Medium** |

### Notes

**(1) Region growing — the central risk.** `_bfs_intensity_grow` implements a plain 6-connected flood: from the seed voxels it visits neighbors, admits any voxel that (a) is not forbidden, (b) passes `_intensity_passes_gate` (`value >= max(mean_seed * frac, abs_floor)` for hyperintense CD), and (c) is unlabeled, and enqueues it — until the queue empties. There is **no maximum voxel count, no expected-volume prior, and no rollback** if the grown region is implausibly large. The only checks against leakage are the intensity gate and the `forbidden` barrier mask. Because the gate is a single global threshold derived from the seed mean, any contiguous bright structure above that threshold (adjacent vessel, venous sinus, imaging artifact) will be absorbed wholesale. This is the mechanism behind Type-A over-growth (§4) and it is **silent**: `seg_4dflow` will simply contain a much larger label, and nothing downstream questions it.

**(2) Communicating arteries.** Comm vessels (LPComm/RPComm/AComm) are handled by `_segment_communicating_rg_only`: seed from the stage-3 centerline, then RG with an *undilated* "other segmentation" barrier. The native-eICAB SCA barrier (`QVTPY_RG_PCA_BASILAR_EICAB_BARRIER_IDS`) is applied to **PCA and basilar**, not to the communicating arteries. So historically the PComms grew without SCA walls, and comm vessels — which are thin, dim, and anatomically wedged between larger arteries — inherit the RG primitive's lack of volume caps. This is a high-probability leak site.

**(3) Venous labeling.** `assign_venous_branches` skeletonizes the venous foreground, splits it at junctions, and greedily assigns each of SSSV/STRV/LTSV/RTSV to its best-scoring unused branch. Scores are geometric priors: SSSV favors midsagittal position (`cx` near `nx/2`) and vertical direction; LTSV/RTSV favor a hemisphere (`cx < nx/2` vs `>`) and transverse direction; STRV favors alignment with a fixed reference vector. The assignment is greedy and order-dependent (`MATLAB_QVT_VENOUS_VESSEL_NAMES` order), gated only by a low `min_assign_score = 0.05`, and has **no connected-component count check** to detect the wrong number of sinuses. Because hemisphere is decided by `nx/2`, any residual left/right registration offset or head tilt biases LTSV/RTSV assignment. A venous mislabel is a topology error that corrupts the affected vessel's entire measurement.

**(4) eICAB dependency.** eICAB is the anatomical backbone: it seeds label identity, provides the AComm junction voxel for the ACA split, and supplies SCA barrier walls. All of this is only as good as the stage-2 registration of eICAB/TOF into 4D-flow space. A misregistration does not raise — it produces a plausible-looking but anatomically shifted prior, and the error propagates through stages 3–6 with no checkpoint. When the native-eICAB ids file is missing, stage 4 logs a warning and proceeds **without** the SCA barriers, quietly increasing PCA/basilar leak risk.

**(5) Temporal resolution.** `velocity_mm_s_from_phases` and the stage-6 series are computed per acquired frame; `mean_flow_ml_s` takes a `temporal_resolution_s` argument only to immediately `del` it. PI and RI are ratios and survive this (they are time-base invariant), but **PWV is a wave-speed measurement** — it fundamentally requires the frame time base. Any PWV computed on an implicit unit time step is dimensionally wrong and cannot be compared across acquisitions with different temporal resolution.

**(6) Vertebral split.** `split_vertebral_from_basilar` runs *after* the basilar has been segmented and grown: it skeletonizes the basilar label, looks for the lowest-`z` branch node of degree ≥ 3, walks the two inferior branches, and flood-fills them into LVA/RVA, assigning left/right by the mean y-coordinate of each branch. There are no independent stage-3 VA centerlines to anchor VA identity, so the split is entirely contingent on (a) the basilar RG having captured both VAs as one connected tube, and (b) a clean Y-junction being found. Failure modes are quiet: no split (VAs remain labeled basilar) or an L/R swap.

**(7) PITC sampling.** PITC-style analysis samples many stations along each vessel. Per-station cross-sections inherit the local segmentation quality; with dense sampling and **no per-station quality weight**, a run of poorly segmented stations biases the vessel-level fit rather than being down-weighted. This is exactly the failure QVTPlus mitigates with `StdvFromMean` gating (§6).

**(8) QC coverage.** The pipeline computes rich per-vessel statistics but treats anomalies as advisory. There is no automated rule that says, e.g., "LICA volume fraction is 6× the cohort median → reject." Empty centerline masks, missing priors, and QC rendering exceptions all log-and-continue. At population scale this means silent-bad segmentations enter the results table indistinguishable from good ones.

**(9) Paste order.** Peripheral labels are thresholded and pasted in ascending id order, and both pasting and RG only claim `seg == 0` voxels. Contested voxels therefore go to whichever vessel is processed first (lower id). This introduces a deterministic but anatomically arbitrary bias at vessel junctions and interacts with the ACA and comm handling.

---

## 4. Failure mode taxonomy

The following taxonomy is intended as shared vocabulary for triage, QC rule design, and holdout review labeling.

**Type A — Over-grow (RG leak).** A label expands far beyond its true vessel because the unbounded BFS crosses the intensity gate into adjacent bright tissue with no volume cap or rollback (§3.1). *Signature:* vessel voxel count and cross-section area dramatically above cohort norms; label bleeds into a neighboring territory or a sinus. *Most exposed:* comm arteries, basilar/PCA, any vessel adjacent to a venous sinus.

**Type B — Neighbor bleed (boundary mis-attribution).** Voxels genuinely near a bifurcation are assigned to the wrong adjacent vessel because of paste order (§3.9), barrier under-dilation, or an RG that reaches a shared boundary first. *Signature:* modest but systematic mislabeling at junctions (ICA/MCA/ACA trifurcation, vertebro-basilar junction, AComm). *Distinct from Type A* in that total volume can look normal.

**Type C — Under-segment (missed or truncated vessel).** A vessel is partially or entirely missing because the local threshold rejected dim signal, the centerline seed was empty, the largest-connected-component cleanup removed a disconnected true segment, or RG was skipped (all venous labels are RG-skipped). *Signature:* zero/low voxel count, empty-mask warnings, missing LOC rows.

**Type D — Topology error.** The anatomy is mislabeled at the identity level: venous sinus mis-assignment (§3.3), vertebral-artery L/R swap or un-split basilar (§3.6), or an uncorrected LACA/RACA overlap at the AComm junction. *Signature:* a label maps to the wrong physical vessel; downstream measurements are precise but attributed to the wrong territory — the most dangerous class because the numbers look plausible.

**Type E — Measurement drift.** Segmentation is acceptable but the reported hemodynamics are biased: temporal-resolution loss corrupts PWV (§3.5), cross-section plane obliquity/interpolation biases area and flow, LOC placement lands on a poor station, or PITC aggregates noisy stations without quality weighting (§3.7). *Signature:* metrics within plausible range but inconsistent with paired vessels or with test–retest.

---

## 5. Prioritized recommendations

Ordered by expected risk reduction per unit of engineering effort.

1. **Bound region growing with volume caps and rollback (P0).** Add a per-label maximum voxel budget (derived from the eICAB prior volume and/or a cohort percentile) to `region_grow_*`, and roll back the grow — reverting to the pre-RG mask — if the final region exceeds the budget or if the added fraction per iteration spikes (a leak signature). This directly neutralizes the dominant Type-A failure and is the single highest-value change.

2. **Make the native-eICAB SCA barrier mandatory and extend it to the PComms (P0).** Treat a missing native-eICAB ids file as a **hard error** for any subject where PCA/basilar/PComm RG will run, rather than a warning. Add SCA walls to the communicating-artery RG path, not only PCA/basilar.

3. **Ship an automated per-subject QC dashboard that gates (P0).** Convert the already-persisted statistics into accept/reject rules with an escalation tier for borderline cases. Minimum signals: per-vessel **volume fraction** vs cohort distribution (Type A/C), venous **connected-component count** vs expected (Type D), ACA overlap-correction magnitude and vertebral-split outcome (Type B/D), **PITC R²** per vessel and **PWV acceptance** (Type E). The point is to change the default from "warn and continue" to "fail closed."

4. **Thread `temporal_resolution_s` end to end (P0 for PWV).** Read the frame time base from the acquisition metadata and propagate it through stage 6 into every time-dependent metric. Stop discarding it in `mean_flow_ml_s`. Block PWV output when the time base is unavailable.

5. **Port QVTPlus `StdvFromMean`-style quality gating into PITC/PWV (P1).** QVTPlus (`enc_HQVesselFlows`) keeps only branch points whose `StdvFromMean` quality exceeds a threshold before averaging flows. Replicate this: attach a per-station quality/residual metric and gate stations before fitting, so dense PITC sampling does not fit noise (§6).

6. **Add conservative, capped venous RG with CC validation (P1).** Replace the historically RG-free venous path with a bounded RG (mirroring recommendation 1) and validate the sinus count/topology before labeling, so venous assignment fails loudly rather than mislabeling.

7. **Establish a manual-review holdout (P1).** Maintain a fixed, expert-labeled holdout set and report per-label Dice/volume error and topology-correctness on every pipeline change. This is the ground truth against which the QC rules and the in-progress heuristics are calibrated.

8. **Ensemble / multi-threshold segmentation for stability (P2).** Combine several threshold operating points (lsthr/lthr/otsu, or perturbed intensity fractions) and keep the consensus, using disagreement as an additional QC signal. Reduces sensitivity to any single threshold choice.

---

## 6. Comparison to QVTPlus

**What qvtpy eliminates.** QVTPlus requires manual curation of `LabelsQVT.csv` to map centerline-graph branches to named vessels — a per-subject bottleneck and a source of operator variability. `qvtpy` removes this entirely by carrying named multilabel anatomy from eICAB priors through segmentation. This is a real gain in throughput and reproducibility.

**What the shift costs.** The manual-labeling burden does not disappear; it **transforms into a segmentation-quality burden**. In QVTPlus, a human guaranteed vessel identity; in qvtpy, an unbounded RG plus geometric heuristics do, and they can be confidently wrong (Type D). The pipeline therefore needs automated QC to occupy the reliability role the human curator used to fill.

**The quality gate that must be ported.** QVTPlus does not average all branch points blindly — `enc_HQVesselFlows` filters branches by `data_struct.StdvFromMean` against `params.thresh`, keeping only high-quality flow points before computing mean vessel flows. `qvtpy`'s new dense PITC/PWV work has no equivalent gate today. **Without porting `StdvFromMean`-style per-point quality selection, dense PITC fits will regress toward noise**, and PWV cross-correlation will be driven by low-quality stations. This is not optional polish; it is a load-bearing part of what made QVTPlus's flow numbers trustworthy.

**The multilabel advantage is real but conditional.** Named per-vessel masks are structurally superior to a binary graph — but that advantage is only *realized* when measurements are taken as **per-vessel cross-sections sampled along the correct centerline**. If segmentation identity is wrong (Type D) or cross-section placement/obliquity is poor (Type E), the multilabel structure produces precise numbers for the wrong thing. The masks are a necessary condition for better measurement, not a sufficient one.

---

## 7. In-progress mitigations (current development cycle)

Several of the weak points above are **actively being addressed**, and the direction is consistent with this assessment's priorities. These should be framed as in-flight mitigations rather than open gaps:

- **RG volume caps + rollback** — directly targets the P0 Type-A over-segmentation risk (§3.1, rec. 1).
- **SCA walls on the posterior communicating arteries** — extends the native-eICAB barrier to the comm RG path (§3.2, rec. 2).
- **Conservative venous region growing** — replaces the historically RG-free venous handling with a bounded grow (§3.3, rec. 6).
- **ACA left/right split without segmenting the AComm** — junction-plane split with stray-island pruning, reducing dependence on AComm segmentation for the L/R decision (§3.9, Type B/D).
- **Improved vertebral detection** — hardening the basilar→LVA/RVA split (§3.6, Type D).
- **Temporal-resolution propagation** — threading `temporal_resolution_s` for correct PWV (§3.5, rec. 4).
- **PITC/PWV with quality gating** — the vessel-level metrics with a QVTPlus-style quality selection (§6, rec. 5).

**Recommended emphasis for the remainder of the cycle.** The heuristic and metric improvements above are necessary but individually insufficient without the **enforcing QC layer (rec. 3)**. The most cost-effective way to convert this work into population-scale reliability is to land the volume caps/rollback and the automated gating dashboard together, validated against a manual-review holdout (rec. 7), so that the residual failures of the heuristics are *caught* rather than silently propagated. Until the QC surface gates rather than warns, the pipeline should be treated as supervised, not unattended.

---

*Prepared as an internal engineering assessment. Findings are grounded in a read-only review of the `nvitk.pipes.qvtpy` sources (region growing, stage-4 segmentation and its heuristics, LOC selection, stage-6 measurement) and the reference `qvtplus` MATLAB post-processing. No pipeline code was modified.*
