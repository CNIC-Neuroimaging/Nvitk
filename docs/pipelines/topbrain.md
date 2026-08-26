# TopBrain

Multi-class whole-brain vessel segmentation for the
[ToPBrain 2026](https://topbrain2026.grand-challenge.org/) and
[ToPAneu 2026](https://topaneu-26.grand-challenge.org/) challenges, from raw imaging through to a
submittable Grand Challenge container.

The training release is **50 volumes from 25 patients** (25 CTA + 25 MRA, paired) — far too few
to train a 36-class 3D network from scratch. The encoder is therefore bootstrapped from
self-supervised pre-training and only then fine-tuned on the labelled data.

```bash
nvitk-topbrain --stages stage0,stage1,stage2 \
  --pretrain-source openmind --checkpoint-name primusm-mae \
  --loss dice_ce_cldice --folds 0,1,2,3,4
```

## The six stages

| Stage | Does | Consumes | Produces |
|---|---|---|---|
| `stage0` | data preparation | challenge release, your cohorts | nnU-Net dataset; optional nnssl corpus |
| `stage1` | pre-training | a published checkpoint, or the corpus | a **bundle** |
| `stage2` | transfer training | bundle + stage-0 dataset | trained folds |
| `stage3` | evaluation | predictions + references | the six challenge metrics |
| `stage4` | inference | a trained model, new cases | masks |
| `stage5` | packaging | a trained model | Grand Challenge container |

`--stages` takes ids or aliases (`dataprep`, `pretrain`, `train`, `evaluate`, `infer`,
`package`) in any order; they are re-sorted into pipeline order.

## In-tree frameworks

Two frameworks live inside the pipeline rather than being installed:

`pipes/topbrain/nnssl`
  Self-supervised pre-training. Not installable (no packaging metadata), so it is used off
  `PYTHONPATH`. It targets Python 3.12 and imports `typing.override`; a shim backfills that on
  3.11, without which **all** nnssl trainer discovery fails.

`pipes/topbrain/nnunet`
  An nnU-Net build carrying the nnssl fine-tuning support — `nnUNetv2_preprocess_like_nnssl`,
  `PretrainedTrainer`, `PretrainedTrainer_Primus` — that released `nnunetv2` does not have. It is
  **deliberately not installed**: the rest of nvitk (TotalSegmentator especially) depends on the
  released `nnunetv2`, and shadowing that globally would break it. Stage 2 invokes it in a
  subprocess whose `PYTHONPATH` puts it first, so only the child sees it.

## stage0 — data preparation

Writes both dataset formats the pipeline consumes, applying the same modality-aware
harmonisation to each so a model never sees the two on different scales.

**Why images are rewritten rather than referenced.** nnU-Net picks an intensity-normalisation
scheme per *channel*, not per case; nnssl emits one scheme per collection and records only
spacings in its fingerprint, so it has no CT normalisation at all. CTA is calibrated (HU, air at
−1000) and TOF-MRA is not (arbitrary units, floor at 0) — neither framework can normalise a mixed
set. Mapping both to `[0, 1]` here is what makes one modality-agnostic model legitimate.

**Folds are grouped by patient.** Each patient contributes a CTA *and* an MRA; a case-level split
would put one in training and the other in validation and leak the subject.

| Option | Default | Notes |
|---|---|---|
| `--dataprep-target` | `train` | `train`, `corpus`, or `both` |
| `--label-set` | `ta36` | `ta36` (36 classes, modality-agnostic — the scored track), `v1_ct` (40), `v1_mr` (42) |
| `--modality` | `both` | Restrict to one modality; for per-modality ablations |
| `--extra-train` | — | `name=/images:/labels:modality`, repeatable. Your own annotated cohorts, merged with the challenge cases |
| `--no-challenge` | off | Train only on `--extra-train` cohorts |
| `--corpus-source` | — | `name:modality=/path[:glob]`, repeatable. Unlabeled pre-training data |
| `--num-folds` / `--seed` | `5` / `12345` | Patient-grouped, reproducible |
| `--ct-window` | `-100 1500` HU | The upper bound is deliberately high: the TA36 infraclinoid ICA classes run through the carotid canal *inside bone* |
| `--mr-percentiles` | `0.5 99.5` | Over non-zero voxels — TOF stores air as exactly 0 over much of the field |
| `--overwrite` / `--workers` | off / `1` | |

The training data is **not** a valid `--corpus-source`: pre-training on the volumes you fine-tune
on buys little and muddies the evaluation. The stage refuses `--dataprep-target corpus` without
an explicit source.

## stage1 — pre-training

Produces a **bundle**, never touching the labelled data.

```
<results_root>/stage1_pretrain/<name>/
├── checkpoint_final.pth      # nnssl-format weights; what stage 2 consumes
├── adaptation_plan.json      # architecture + encoder/stem key layout
├── segmentation_model.pth    # a materialised end-to-end segmentation network
└── bundle.json               # provenance
```

`segmentation_model.pth` is the pre-trained encoder with a segmentation decoder and head
attached, at the plan's recommended patch size. It is portable and inspectable on its own, and
building it proves the weights instantiate before any training is queued. Stage 2 does not read
it — the training build regenerates the head for the actual class count and target spacing of the
downstream dataset, which stage 1 cannot know.

| Option | Default | Notes |
|---|---|---|
| `--pretrain-source` | `openmind` | `openmind` (download a published checkpoint) or `scratch` (train with nnssl) |
| `--checkpoint-name` | — | See [Published checkpoints](#published-checkpoints) |
| `--bundle-name` | checkpoint/trainer name | Bundle directory name |
| `--init-checkpoint-name` | — | Seed `scratch` from a published checkpoint — *domain-adaptive* pre-training |
| `--ssl-trainer` | `SparkMAETrainer` | Any nnssl trainer class |
| `--ssl-config` | `median` | **Not** `onemmiso`: vessels here are 0.3–0.6 mm and would not survive 1 mm isotropic |
| `--ssl-loss` | trainer's own | `mse`, `mse_masked`, `l1`, `ssim`, `ms_ssim`, `spark`, or a dotted path |
| `--ssl-patch-size` / `--ssl-batch-size` / `--ssl-epochs` / `--ssl-lr` | trainer defaults | nnssl keeps these on the trainer, not in the plan |
| `--no-export-model` | off | Skip `segmentation_model.pth` |

```{warning}
nnssl's self-supervised losses are **not** interchangeable across trainer families: `SparkTrainer`
calls its loss with `(prediction, groundtruth, mask)` and the MAE trainers with
`(model_output, target, mask)`. The stage compares signatures and refuses an incompatible pair
upfront rather than failing inside a dataloader worker minutes later.
```

When continuing from a checkpoint, use a low `--ssl-lr` (~1/10 of the from-scratch value) and few
`--ssl-epochs`. At the from-scratch rate the published features are overwritten within a few
epochs and the result is worse than the checkpoint you started from.

## Published checkpoints

`nvitk-topbrain --list-checkpoints`. Fetched into the configured model root on first use,
together with the `adaptation_plan.json` stage 2 needs. All are trained on
[OpenMind](https://huggingface.co/datasets/AnonRes/OpenMind) (~114 k brain MR volumes).

**ResEnc-L (convolutional)** — `resencl-mae`, `resencl-voco`, `resencl-vf`, `resencl-mg`,
`resencl-s3d`, `resencl-simclr`, `resencl-swinunetr`

**Primus-M (transformer)** — `primusm-mae`, `primusm-simmim`, `primusm-voco`, `primusm-vf`,
`primusm-mg`, `primusm-simclr`, `primusm-swinunetr`

Both families are usable: the in-tree build fine-tunes ResEnc-L through `PretrainedTrainer` and
Primus-M through `PretrainedTrainer_Primus`, and the trainer family is selected automatically
from the bundle's architecture.

Worth knowing when choosing: `*-mae` is the best-studied general default; `*-vf` (VolumeFusion)
has a dense pseudo-segmentation pretext, closest in shape to this task; `*-voco` learns global
position and context rather than local texture.

## stage2 — transfer training

`nnUNetv2_preprocess_like_nnssl` reads the bundle's adaptation plan, derives spacing and
normalisation, and writes a `ptPlans__<name>____Spacing…` file recording where the weights are
and how to load them. `PretrainedTrainer` then builds the network, loads the encoder, and
fine-tunes with a warm-up schedule. **This is where the encoder becomes a task-specific
segmentation model.**

| Option | Default | Notes |
|---|---|---|
| `--loss` | `dice_ce` | See [Loss selection](#loss-selection) |
| `--loss-config` | — | JSON object, or path to a JSON file, of loss kwargs |
| `--folds` | `0` | Comma list, or `all` |
| `--adaptation-mode` | `default_nnunet` | How spacing/normalisation are derived. `default_nnunet` lets nnU-Net plan (recommended — it picks a sub-millimetre spacing suited to the vessels); `like_pretrained` copies the pre-training spacing; `no_resample` keeps native; `fixed` uses `--target-spacing` |
| `--patch-size` / `--batch-size` | from the plan | The plan mandates 160³, sized for the A100s the checkpoints were trained on. Must be a multiple of 32 |
| `--num-epochs` | build default | Smoke tests and short ablations |
| `--from-scratch` | off | Same architecture, random init — the control run |

The patch override edits the generated plans **in place**, keeping `data_identifier` so the
preprocessed data is reused rather than regenerated. Primus interpolates its positional embedding
to the configured patch size, so a smaller patch stays compatible with the pre-trained weights.

Stage 0's patient-grouped splits are installed before training; left alone, nnU-Net invents a
case-level split that leaks patients across folds.

## Loss selection

Four of the six scored metrics are topology or detection metrics, and vessels occupy 0.2–0.5 % of
a head volume — so the objective is a swappable axis, not a constant.

```bash
nvitk-topbrain --list-losses
nvitk-topbrain --loss dice_ce_cldice --loss-config '{"weight_cldice": 0.5, "iters": 3}'
nvitk-topbrain --loss my_pkg.my_module:MyLoss
```

| Name | Objective |
|---|---|
| `dice_ce` | Dice + cross-entropy *(default)* |
| `dice_ce_nosmooth`, `dice`, `ce`, `dice_topk10`, `topk10` | Stock variants |
| `focal`, `dice_focal` | Down-weight the easy background |
| `dice_ce_cldice` | Adds soft centerline Dice — targets the clDice and β0 metrics |
| `dice_ce_skelrec` | Adds skeleton recall — weights thin side-road vessels by length, not calibre |
| `dice_ce_cldice_focal` | Topology and imbalance together |

`dice_ce` is the default deliberately: a topology loss unmeasured against a plain baseline tells
you nothing.

Each loss becomes a generated trainer class in
`nnunet/nnunetv2/training/nnUNetTrainer/topbrain/`, in two families
(`nnUNetTrainerTopBrain_*` and `nnUNetTrainerTopBrainPrimus_*`). One class per loss, so each
objective trains into its own results directory. A custom dotted-path loss reaches its trainer
through `TOPBRAIN_LOSS_SPEC`, since the CLI can only pass a class name.

```{note}
Every ToPBrain trainer disables mirroring, in training **and** at inference. The labels are
lateralised (`R-ICA` vs `L-ICA`), so a flip produces an image whose correct labels are the
mirrored *class ids*, not the mirrored mask.
```

Topology losses default to a merged "any vessel" foreground. `per_class` is available and
memory-bounded (class chunking plus gradient checkpointing) but roughly 25× slower.

## stage3 — evaluation

Scores **held-out** predictions. By default (`--predictions-from cv`) it gathers each fold's
`fold_N/validation/` output — the cases that fold held out — giving one prediction per case from
a model that never trained on it. Folds whose validation output is missing are reported, so a
partial cross-validation is visible rather than silently scored as complete; overlapping folds
are refused outright.

```{warning}
Do not point this stage at inference run over the training set (`--predictions-from folder` with
stage 4's output over `imagesTr`). That grades the model partly on its own training data and
inflates every metric, with no visible symptom.
```

Class-average Dice, centerline Dice, connected-component (β0) error, HD95 (with the challenge's
290 mm missing-class penalty), invalid-neighbour error, and side-road detection F1 at IoU 0.25 —
plus a per-modality breakdown, because one modality-agnostic model can hide a large CT/MRA gap.

Classes absent from **both** masks are excluded from the averages rather than scored perfect:
most classes are absent from most cases, and scoring them 1.0 would let a model inflate its
average by predicting nothing.

```{warning}
These are nvitk's implementations, for ranking runs against each other. Use
[TopBrain_Eval_Metrics](https://github.com/CoWBenchmark/TopBrain_Eval_Metrics) for
leaderboard-comparable numbers. The TA36 adjacency table is **derived, not published** — the
challenge ships tables only for the v1 label sets, and TA36 renumbers everything above 34.
```

## stage4 — inference

Predicts over a folder, then cleans up per class. Raw predictions are retained beside the
post-processed ones, so a threshold change can be re-scored without re-running inference.

`--min-volume-mm3` (default 5.0) drops small components per class. `--largest-only` is **off** by
default: a few classes are genuinely multi-component in a field of view, and deleting the second
piece trades a fragmentation error for a missing-structure error, which scores worse.

## stage5 — packaging

Assembles a Grand Challenge build context: Dockerfile, entry point, nvitk (minus the in-tree
frameworks) and the model. Left on disk whether or not Docker is present, since the analysis host
and the machine with a Docker daemon are usually different. `--build` runs `docker build`;
`--save` writes an upload-ready `tar.gz`.

The container reads one `.mha` from `/input/images/head-{ct,mr}-angio/` and writes a mask of
identical shape to `/output/images/head-{ct,mr}-angio-segmentation/`. TA36 is modality-agnostic,
so one image serves both sockets.

## Configuration

`pipelines.topbrain` holds SGE settings; `pipelines.topbrain_paths` holds `local_*` / `cluster_*`
twins for `challenge_root`, `nnssl_raw`, `nnssl_preprocessed`, `nnssl_results`, `nnunet_raw`,
`nnunet_preprocessed`, `nnunet_results`, `results_root`, `model_root`, `corpus_root`.

Usual precedence inversion: under `--submit local` a CLI flag beats configuration; under
`--submit sge` the `cluster_*` value wins.

## Cluster execution

```bash
nvitk-topbrain --stages stage0,stage1,stage2 --submit sge --emit-script run_topbrain.sh
```

One SGE job per stage, chained with `-hold_jid`. Stages are cohort-scoped rather than
per-subject: nnU-Net and nnssl own their own parallelism. Worker commands are built against
**container-side** paths only; the framework roots are mounted at `/nnunet/*`, `/nnssl/*` and
`/corpus`.

Always `--emit-script` and read the generated blocks, then `--dry-run`, before submitting.

## GUI

The three label vocabularies are registered in the {doc}`Main GUI <../gui/index>` catalog as
`topbrain-ta36`, `topbrain-v1-ct` and `topbrain-v1-mr`, and auto-detected on load. Detection
matters: the release stores all three under the *same* filenames and their values collide above
34 — label 35 is `R-ICA-C1-C5`, `VoG` or `R-ECA` depending on the set. Where the path does not
name a label set, the catalog falls back to the modality in the filename and the largest label
value present.
