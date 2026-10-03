# CViT — Convolutional Vision Transformer

Dense 2D/3D segmentation with a Vision Transformer whose tokens come from **convolutions**
instead of a linear projection of flattened patches. The pipeline pre-trains (self-supervised),
trains, evaluates, probes, predicts and exports CViT models on top of the vendored
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet) and [nnssl](https://github.com/MIC-DKFZ/nnssl)
engines, so planning, preprocessing, augmentation, deep supervision and sliding-window inference
are nnU-Net's, unchanged.

```bash
nvitk-cvit --stages dataprep,train,evaluate,probe \
  --data-root /data/MyTask --dataset-name MyTask --channel-names CT \
  --arch CViTB --folds 0,1,2,3,4
```

## Why convolutional tokens

A standard ViT cuts the volume into patches and feeds each one through a single linear layer. In
3D that is a `Conv3d` with kernel = stride = patch size. Inside a patch the model then behaves
like a plain MLP on raw voxels, so the geometric priors that make convolutions good at shape
(locality, weight sharing, translation equivariance) are missing at exactly the scale where local
shape lives. Attention adds global context *between* patches but never sees structure *within*
one.

CViT keeps the transformer and replaces the tokenizer with a small CNN:

- **attention runs over geometry-aware tokens**, so global context is mixed on top of local shape
  features rather than on top of raw intensities;
- **the same multi-scale conv features feed the decoder as skips**, recovering fine structures
  (thin vessels, small lesions) that a 1/8-resolution token grid cannot represent.

The idea has a long history: hybrid ViT (Dosovitskiy et al., 2020), early-convolution ViTs
(Xiao et al., 2021), CvT and CeiT, TransUNet and TransBTS for medical images, and Primus with its
V2/V3 conv stems (Wald et al., 2025). This pipeline is a controlled setting to *measure* the
idea: three tokenizers behind one interface, skip controls, and an
{doc}`attention-usage probe <cvit-attention-usage>` that tells you whether the transformer is
actually being used. The point of that probe is a known failure mode, the one behind the Primus
paper: a hybrid network can learn to route everything through its CNN skips and leave the
transformer idle.

```{toctree}
:maxdepth: 1

cvit-architecture
cvit-ssl
cvit-attention-usage
```

## Stages

| Stage | Aliases | Does | Consumes | Produces |
|---|---|---|---|---|
| `stage0` | `dataprep`, `data` | Validation and conversion to nnU-Net raw, folds, optional nnssl corpus | your images + labels | `nnUNet_raw/Dataset<ID>_<Name>`, `splits_final.json`, `pretrain_data.json` |
| `stage1` | `pretrain`, `ssl` | SimMIM / MAE pre-training on the corpus | the corpus | an encoder **bundle** |
| `stage2` | `train`, `finetune` | CViT plans, preprocessing, per-fold training | the dataset (+ bundle) | trained folds, `cvit_stage2.json` |
| `stage3` | `evaluate`, `eval` | Cross-validation metrics | held-out predictions | `metrics.csv`, `metrics.json` |
| `stage3b` | `probe`, `attention` | Attention-usage probe | a trained run | `probe.csv`, `usage.json`, `layers.png` |
| `stage4` | `infer`, `predict` | Prediction on new cases | a trained run or export | masks on the input grids |
| `stage5` | `export`, `package` | Portable model bundle | a trained run | folder with `predict.py` |

`--stages` takes ids or aliases in any order; they are re-sorted into pipeline order. The
default is `dataprep,train`. Stages 3–5 act on the **latest** stage-2 run of the dataset unless
`--run-name` names another (the run name is nnU-Net's results folder,
`<trainer>__<plans>__<configuration>`).

## Input data

Either an nnU-Net-style folder or a plain pair of folders under `--data-root`:

```text
<data_root>/imagesTr/<case>_0000.nii.gz   (one file per channel: _0000, _0001, …)
<data_root>/labelsTr/<case>.nii.gz
# or
<data_root>/images/<case>.nii.gz          (single channel)
<data_root>/labels/<case>.nii.gz
```

`--images-dir` / `--labels-dir` override the locations. NIfTI, MHA/MHD and NRRD are read. Files
already in `.nii.gz` are copied byte-for-byte; anything else is converted with `nvitk.io.imsave`,
which writes voxels and geometry.

**Stage 0 fails loudly on:**

- an image/label pair on different grids (shape, or affine beyond 1e-3 mm);
- NaN/Inf voxels;
- non-integer label values;
- undeclared label values;
- missing labels;
- empty labels (`--allow-empty` downgrades this one to a warning).

Each of these otherwise trains silently on wrong data. A float label whose values are integral is
rewritten as an integer volume, with a warning.

`--channel-names` matters. nnU-Net normalises a channel named `CT` with dataset-wide intensity
statistics and every other name with per-case z-scores. `--labels "background=0,liver=1,tumour=2"`
names the classes; without it every value found becomes `label_<v>`.

**Folds** are grouped by patient when `--group-regex` is given, e.g.
`'^(?P<group>sub-\d+)'` keeps every session of a subject in one fold. Groups are shuffled with
`--seed` and assigned to the fold with the fewest cases.

## Running

### Locally

Stages run in-process under a pipeline tracker. nnU-Net and nnssl run in subprocesses, so an
out-of-memory error during training cannot take down the orchestrator. The torch device follows
`--backend` (`gpu` → `cuda` when a CUDA device is visible) unless you pass `--device`.

```bash
# tokenizer ablation, 5-fold each, then the probe
for t in hierarchical intra_patch linear; do
  nvitk-cvit --stages train,evaluate,probe --dataset-name MyTask --tokenizer $t --folds 0,1,2,3,4
done
```

Every architectural variant gets its own plans identifier
(`cvitPlans_<preset>_<tokenizer>_<decoder>_s<skips>…_<hash>`), hence its own results folder, so
ablations never overwrite each other. They all share **one** preprocessed copy of the dataset.

### On the SGE cluster

```bash
nvitk-cvit --stages dataprep,pretrain,train,evaluate,probe --submit sge \
  --ssl simmim --corpus-from-train --folds 0,1,2,3,4 --parallel-folds \
  --emit-script run_cvit.sh --dry-run
```

- **Job chaining.** Each stage is one job in a `-hold_jid` chain. `--parallel-folds` splits
  training into one planning/preprocessing job plus one job per fold, all holding on the
  preparation job, and the next stage waits for every fold.
- **Driver script.** The whole chain is written into a single driver script. It runs where `qsub`
  exists: locally on a submit host, otherwise on the login node over SSH. With `--no-remote` it is
  only written.
- **Container paths.** Worker commands use container-side paths only:
  - `/nvitk/data` is `data_root`;
  - `/nvitk/output` is `results_root`;
  - `/models`, `/nnunet/{raw,preprocessed,results}`, `/nnssl/{raw,preprocessed,results}` and
    `/corpus` hold the other roots;
  - user paths outside those roots are bound at the same path.

  A command naming an absolute path that nothing mounts is refused at submission, not after the
  queue wait.

Always `--emit-script` + `--dry-run` and read the blocks first.

## Configuration

`sge.json` → `pipelines.cvit` (SGE project/account/memory/queue, container, script and log
directories) and `pipelines.cvit_paths` (`local_*` / `cluster_*` twins of `data_root`,
`nnunet_raw`, `nnunet_preprocessed`, `nnunet_results`, `nnssl_raw`, `nnssl_preprocessed`,
`nnssl_results`, `corpus_root`, `results_root`, `model_root`). Every root can also be passed as
`--<root>` (e.g. `--nnunet-results`). The usual precedence applies: under `--submit local` the
flag wins; under `--submit sge` the `cluster_*` value wins. See {doc}`../configuration`.

## Training options (stage 2)

| Option | Default | Notes |
|---|---|---|
| `--arch` | `CViTB` | `CViTS` / `CViTB` / `CViTM` / `CViTL` (`nvitk-cvit --list presets`) |
| `--tokenizer` | `hierarchical` | `hierarchical`, `intra_patch`, `linear` — see {doc}`cvit-architecture` |
| `--decoder` | `unet` | `unet` (skips + deep supervision) or `patch` (no skips) |
| `--token-stride` | `8` | Power of two; per-axis strides follow nnU-Net's pooling schedule |
| `--skips` / `--skip-drop` / `--skip-schedule` / `--skip-gate` | `all` / `0` / `constant` / `none` | See {doc}`cvit-attention-usage` |
| `--arch-json` | — | Any `CViTConfig` field as JSON, e.g. `'{"num_registers": 4}'` |
| `--configuration` | `3d_fullres` | `2d` for 2D data |
| `--baseline-planner` | `nnUNetPlannerResEncM` | The nnU-Net planner that chooses spacing, patch and pooling |
| `--patch-size` / `--batch-size` | baseline's | Patch is rounded down to the token stride |
| `--loss` / `--loss-config` | `dice_ce` | `nvitk-cvit --list losses`, or `pkg.module:Callable` |
| `--epochs` / `--iterations-per-epoch` | 1000 / 250 | nnU-Net's schedule |
| `--lr` / `--warmup-epochs` | `3e-4` / `50` | AdamW, linear warm-up then poly decay; warm-up is capped at a tenth of short runs |
| `--from-bundle` / `--llrd` | — / `1.0` | Fine-tune a stage-1 encoder; layer-wise LR decay (e.g. `0.75`) |
| `--no-mirror` | off | Disable mirroring (lateralised labels) in training and test-time augmentation |
| `--probe-every` | `25` | Attention statistics logged during training (`0` = off) |
| `--folds` / `--parallel-folds` | `0` | Comma list; one SGE job per fold with `--parallel-folds` |

Run-time settings travel to the trainer through `NVITK_CVIT_*` environment variables
(`nvitk.pipes.cvit.util.trainer_env`), because nnU-Net builds trainers by class name only. The
architecture itself lives in the plans file (`arch_kwargs`), so every checkpoint rebuilds the
exact network it was trained with.

## Outputs

```text
<results_root>/
├── stage0_dataprep/<dataset>/cvit_stage0.json      # cases, labels, spacings, folds
├── stage1_pretrain/<bundle>/                       # see cvit-ssl
├── stage2_train/<dataset>/<run>/cvit_stage2.json   # plans, trainer, CViT config, settings
├── stage2_train/<dataset>/latest.json
├── stage3_evaluate/<dataset>/<run>/metrics.{csv,json}
├── stage3b_probe/<dataset>/<run>/{probe.csv,usage.json,layers.png}
├── stage4_infer/<run>/<case>.nii.gz + cvit_stage4.json
└── stage5_export/<name>/
<nnunet_results>/<dataset>/<run>/fold_<k>/
    checkpoint_final.pth, validation/, cvit_probe.jsonl, tensorboard/
```

**Stage 3** scores each case against `labelsTr` with
{mod}`nvitk.measure.segmentation_metrics`:

- the metrics are Dice, clDice, β₀ error and HD95, with HD95 in **mm** from the reference spacing;
- a label absent from both masks is skipped rather than scored as 1;
- results are reported per label, per fold and as a cohort mean.

**Stage 4** stages plain `<case>.nii.gz` inputs under nnU-Net's names and converts other formats.
After prediction it re-reads every output and **fails** if its shape or affine differs from the
input's. `--largest-component` keeps the largest 26-connected component per label, and
`cvit_stage4.json` reports the foreground volume in ml.

**Stage 5** writes weights-only checkpoints, the plans, `cvit_config.json`, checksums, a verbatim
copy of {mod}`nvitk.nn` (`cvit_nn/`) and a `predict.py` that runs with torch and the released
`nnunetv2` alone. The network is rebuilt from `cvit_nn` and handed to nnU-Net's predictor, so the
export runs on machines without nvitk.

```bash
nvitk-cvit-infer --model-dir <export or results run> -i scans/ -o masks/   # with nvitk
python <export>/predict.py -i scans_nnunet_named/ -o masks/                 # without
```

## Troubleshooting

`No trained CViT run recorded`
: Stages 3–5 need a finished stage 2 for that `--dataset-id`/`--dataset-name`, or an explicit `--run-name`.

`The baseline plan has only N stages`
: The baseline network is too shallow for the token stride (very small images). Lower `--token-stride`.

`input_shape … must be divisible by the token stride`
: An explicit `--patch-size` that the stride schedule cannot tile; the derived plans round automatically.

`--from-bundle fixes [...]`
: The bundle decides tokenizer, transformer size and stem; drop the conflicting architecture flags.

`nnssl preprocessing produced no volumes`
: Every corpus volume failed in nnssl (its errors are printed above the message). Fix the corpus and rerun stage 1 with `--overwrite`.

Probe indices are NaN
: The intervention was not run for that model (e.g. `skips_off` with the patch decoder) or the full model's Dice is 0.

## Command reference

```{eval-rst}
.. click:: nvitk.pipes.cvit.run:main
   :prog: nvitk-cvit

.. click:: nvitk.pipes.cvit.stage4_infer:main
   :prog: nvitk-cvit-infer
```
