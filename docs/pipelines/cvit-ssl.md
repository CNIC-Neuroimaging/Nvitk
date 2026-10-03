# CViT self-supervised pre-training

Stage 1 pre-trains the CViT **tokenizer and transformer encoder** on unlabelled volumes with
masked image modelling, on the vendored [nnssl](https://github.com/MIC-DKFZ/nnssl) engine. nnssl
handles corpus parsing, planning, blosc2 preprocessing, augmentation, logging and checkpointing;
the CViT trainers ({mod}`nvitk.pipes.cvit.ssl_trainers`) replace the network, the masking, the
objective and the optimiser.

```bash
# corpus = training images (channel 0) + an external unlabelled cohort, then pre-train and fine-tune
nvitk-cvit --stages dataprep,pretrain,train \
  --corpus-from-train --corpus-source ext:mr=/data/unlabelled_tof \
  --ssl simmim --ssl-epochs 300 --ssl-patch-size 128,128,128 --mask-ratio 0.6 \
  --arch CViTB --tokenizer hierarchical --llrd 0.75 --folds 0,1,2,3,4
```

## The corpus (stage 0)

`--corpus-source name:modality=/path[:glob]` (repeatable) and `--corpus-from-train` write an nnssl
`pretrain_data.json` under `<nnssl_raw>/Dataset<corpus-id>_<Name>Corpus/`. The source syntax and
its cohort presets are the topbrain ones (`nvitk.pipes.topbrain.util.collection`). By default
volumes are referenced as they are; nnssl z-scores each one. `--corpus-harmonize` maps CT to `[0, 1]`
with an HU window and MR with robust percentiles first, which is needed when a corpus mixes CT and MR.

nnssl pre-trains single-channel, so `--corpus-from-train` uses channel 0. A multi-channel
downstream model still loads the encoder: the input convolutions are adapted (see below).

## Methods

`--ssl simmim` (default)
: **Every token is encoded.** Masked tokens are replaced by a learned `mask_token` after the tokenizer.
  Works with every tokenizer.

`--ssl mae`
: **Masked tokens are dropped from the encoder** (cheaper per step); a `mask_token` fills them back
  before the decoder. Rotary and absolute position embeddings are gathered at the kept positions.
  Intended for `linear` / `intra_patch`.

In both, a token is masked as a whole token-stride block, and every sample has exactly
`round(mask_ratio · N)` masked tokens. Reconstruction goes through the skip-free patch decoder and
the loss is MSE on **masked voxels only**.

```{important}
**Masked voxels are zeroed at the input, not only replaced at the token level.** With a conv
tokenizer, a visible token's receptive field reaches into its masked neighbours; replacing the
masked *token* alone would let the network copy the answer through the convolution. Zeroing the
input closes that path. For `intra_patch` this makes masking exact, since no receptive field crosses
a patch. For `hierarchical` it leaves only a thin border leak, and it is why SimMIM rather than MAE
is the recommended method there (MAE warns).
```

The decoder has no skips during pre-training, which forces the reconstruction signal through
attention. This is also why the downstream decoder starts fresh.

## Training details

| Setting | Value |
|---|---|
| Optimiser | AdamW (β 0.9/0.98), weight decay 0.05 on matrices only |
| Schedule | linear warm-up (`--warmup-epochs`, capped at a tenth of the run), then poly decay |
| Gradient clipping | 1.0 |
| Patch / batch | `--ssl-patch-size` (default 128³) / `--ssl-batch-size` |
| Spacing | `--ssl-configuration median` (default), `noresample`, or `onemmiso` (discouraged for sub-mm structures) |
| Resume | `--continue-training` picks up `checkpoint_latest.pth` (written every epoch) |

Before the first step, nnssl's adaptation check builds the **downstream** segmentation CViT and
strictly loads the pre-trained encoder into it. An encoder that could not be fine-tuned fails at
epoch 0, not after days of pre-training.

Planning and preprocessing are reused when already complete. A corpus where nnssl failed every
volume stops the stage with a clear error instead of crashing inside the first batch.

## The bundle

```text
<results_root>/stage1_pretrain/<name>/        # name: --ssl-name, default <method>_<preset>
├── checkpoint_final.pth    # nnssl checkpoint: network_weights + cvit_config + adaptation plan
├── encoder.pth             # tokenizer.* + encoder.* tensors only (what stage 2 loads)
├── cvit_config.json        # the encoder architecture
├── adaptation_plan.json    # nnssl adaptation plan (CViT key mapping)
└── bundle.json             # provenance: corpus, method, patch, mask ratio, epochs
```

## Fine-tuning from a bundle (stage 2)

When `pretrain` and `train` run together, stage 2 uses the bundle just produced; otherwise pass
`--from-bundle <dir>`.

- **Adopted from the bundle.** The tokenizer, the transformer size, the stem widths, blocks and
  strides, and position handling are taken from `cvit_config.json`, because parameter shapes and
  names must match. Decoder and skip controls stay free. Conflicting flags are rejected.
- **Loaded by the trainer.** `encoder.pth` is loaded with
  {func}`~nvitk.nn.cvit.load_pretrained_encoder`. The positional embedding is resampled to the
  fine-tuning patch, and input convolutions are adapted to the dataset's channel count (`repeat`,
  response-preserving).
- **Fresh parameters.** The decoder, and with `--tokenizer linear` also the skip-only conv stem,
  start fresh. The log line reports exactly how many tensors were loaded, resized, adapted and left
  fresh.
- **Layer-wise LR decay** (`--llrd 0.75`, BEiT-style). The decoder trains at the base rate,
  transformer block *i* at `lr · d^(depth − i)`, and the tokenizer and embeddings at
  `lr · d^(depth + 1)`. The CViT LR schedule multiplies each parameter group's `lr_scale`;
  nnU-Net's own schedulers would overwrite it.

Evaluate a pre-trained run against a from-scratch control with the same plans (same
`--arch`/`--tokenizer`), ideally with {doc}`cvit-attention-usage` on both. Pre-training that only
sharpens the conv stem shows up there as higher skip reliance, not attention reliance.
