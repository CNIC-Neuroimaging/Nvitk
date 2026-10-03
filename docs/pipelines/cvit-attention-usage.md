# Skip control and attention usage

A hybrid CNN-transformer segmenter can learn to route almost everything through its convolutional
skips, leaving the transformer to contribute little. This is the critique behind *Primus*
(Wald et al., 2025), and it makes "convolution + attention" claims hard to verify. CViT therefore
ships two tools:

1. **Skip controls**, which constrain the shortcut during training.
2. **An attention-usage probe**, which measures afterwards what the trained model actually relies
   on.

## Skip controls

All of them are `CViTConfig` fields, set on the CLI (or with `--arch-json`) and stored in the
plans. Every combination therefore gets its own results folder.

| Flag | Field | Effect | Why it is a real control |
|---|---|---|---|
| `--skips all\|none\|0011` | `skips` | Enables skips per level F0…F3 (level 0 first). A disabled level is built **without** its concatenation channels, not zero-filled. | `none` gives a pure-ViT decoder. Deep-only skips (`0011`) keep coarse context but remove the fine-detail shortcut. |
| `--skip-drop p` | `skip_drop_prob` | Training only. Each skip level is zeroed per sample with probability *p*; the kept ones are scaled by `1/(1-p)`. Inactive at inference. | The decoder must also segment from the tokens alone. The convolution after the concatenation cannot undo a random dropout. |
| `--skip-schedule warmup --skip-warmup-epochs N` | `skip_schedule` | The skip scale ramps 0 → 1 over the first *N* epochs (`set_skip_scale`, applied each epoch). | The transformer path is learned first and the skips refine it afterwards. The scale changes over time, so the network cannot pre-compensate for it. |
| `--skip-gate learned --skip-gate-l1 λ` | `skip_gate`, `skip_gate_init`, `skip_gate_l1` | A learned per-level gate `σ(g_i)` multiplies the skip; the trainer adds `λ · Σ σ(g_i)` to the loss. Gate values are logged every probe epoch. | Without the L1 term the following convolution could rescale the gate away. With it, the gate becomes a readable estimate of how much each skip is needed. |

```{note}
There is deliberately **no fixed per-skip multiplier**. The skip is concatenated and passed
through conv → InstanceNorm, which absorbs any constant factor within a few updates: a fixed
weight would be a control that does nothing. The only internal scale is `set_skip_scale`, driven
by the warm-up schedule and by the probe's `skips_off` intervention.
```

## Attention-usage probe (stage 3b)

```bash
nvitk-cvit --stages probe --dataset-name MyTask --layer-sweep --local-radius-mm 10,30,60
```

For each trained fold, the probe takes the cases that fold **held out** (`splits_final.json`) and:

1. reads them with the plan's own reader and preprocesses them once;
2. predicts each one under every intervention below, with the same sliding window, Gaussian
   weighting and mirroring as inference, resampled back to the original grid;
3. scores each prediction against the reference (Dice, clDice, β₀ error, HD95 in mm).

Interventions are applied through {func}`nvitk.nn.cvit.intervene`. It flips module attributes,
restores them exactly on exit, and is unit-tested to do so.

| Intervention | What changes | Question it answers |
|---|---|---|
| `full` | nothing | Reference score |
| `attn_off` | the attention residual branch is removed in every block (the MLPs stay) | Does mixing between tokens matter at all? |
| `attn_off@Lnn` (`--layer-sweep`) | the same, one layer at a time | Which layers carry it? |
| `attn_local@Rmm` | each token may only attend within *R* **millimetres** (token spacing = plan spacing × token stride) | Does the gain come from long-range context or only from the neighbourhood? |
| `attn_uniform` | attention weights replaced by a uniform average | Do the learned attention patterns matter, or just global pooling? |
| `transformer_off` | the token grid bypasses the encoder | What does the whole transformer contribute? |
| `tokens_off` | the token path into the decoder is zeroed, so it sees the skips only | How much does the segmentation rely on the transformer? |
| `skips_off` | the skips are zeroed, so the decoder sees the tokens only (U-Net decoder) | How much does it rely on the CNN shortcut? |

The probe also computes **passive statistics** on a centre crop of the first held-out case of each
fold. They use an explicit softmax on 256 random query tokens:

- mean attention distance in mm, per layer and per head;
- normalised attention entropy;
- attention mass on register tokens;
- the residual ratio `‖LS(Attn(LN x))‖ / ‖x‖`, i.e. how much each layer's attention changes its
  input;
- the learned gate values.

During training the same statistics are logged every `--probe-every` epochs on one fixed validation
batch, to `<fold>/cvit_probe.jsonl` and TensorBoard (`<fold>/tensorboard/`, `cvit_probe/*`).

### Reliance indices (`usage.json`)

Computed per case on the class-averaged Dice (and per label), then reported as **median and IQR
across cases**, so a single outlier cannot drive them:

| Index | Definition | Reads as |
|---|---|---|
| **AR** attention reliance | `(D_full − D_tokens_off) / D_full` | Fraction of the performance that needs the transformer output |
| **SR** skip reliance | `(D_full − D_skips_off) / D_full` | Fraction that needs the CNN skips |
| **TR** transformer reliance | `(D_full − D_transformer_off) / D_full` | Contribution of the encoder on top of the raw conv tokens |
| **LR@R** long-range gain | `D_full − D_attn_local@R` | Dice lost when attention is restricted to *R* mm |
| **PR** pattern reliance | `D_full − D_attn_uniform` | Dice lost when the learned attention is replaced by averaging |

Plus `delta_dice` for every intervention, including each `attn_off@Lnn`.

**Reading them:**

- **AR ≈ 0, SR high.** The CNN with an idle transformer. Try `--skips 0011`, `--skip-drop 0.3`
  or `--skip-schedule warmup`.
- **AR high, LR@R ≈ 0 for small R.** Attention is used, but only locally. The transformer behaves
  like a large-kernel convolution, and the "global" part of the hypothesis is not supported for
  this task.
- **AR high, LR@R large.** Long-range context genuinely contributes. This is the result that
  supports the hybrid design.
- **SR ≈ 0 with skips enabled.** The skips are redundant. Compare boundary metrics (HD95, clDice)
  before removing them, since Dice is insensitive to thin structures.
- **Learned gates** shrinking towards 0 under L1 give the same verdict, measured during training.
- Negative values are possible on small or barely-trained models (an intervention can help by
  chance). Look at the IQR, not only the median.

### Outputs

`probe.csv`
: One row per case × intervention × label. Columns: `case`, `fold`, `intervention`, `label`
  (`all` = class average), `dice`, `cl_dice`, `b0_error`, `hd95`.

`usage.json`
: `summary.overall` and `summary.per_label` hold the indices above as `{median, q25, q75, n}`.
  `attention_stats` holds the per-fold passive statistics, and the run's `cvit_config` is
  included.

`layers.png`
: Median ΔDice per removed layer (layer sweep) and mean attention distance per layer.

## A recommended ablation

Hold everything else fixed (dataset, folds, `--arch`, epochs, loss). Train every cell below, and
score each one with stages 3 and 3b:

```bash
for tok in hierarchical intra_patch linear; do
  for skip in "--skips all" "--skips none" "--skips 0011" \
              "--skip-drop 0.3" "--skip-schedule warmup --skip-warmup-epochs 100"; do
    nvitk-cvit --stages train,evaluate,probe --dataset-name MyTask --folds 0,1,2,3,4 \
      --tokenizer $tok $skip --local-radius-mm 10,30,60
  done
done
```

The tokenizer axis answers the original question: do convolutional tokens beat linear ones, and
does strictly intra-patch convolution suffice? The skip axis says whether the answer survives once
the CNN shortcut is constrained. Report Dice/HD95 from stage 3 alongside AR, SR and LR from stage
3b. A better Dice that comes with AR ≈ 0 is a better CNN, not evidence for attention.
