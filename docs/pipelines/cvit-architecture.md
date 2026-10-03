# CViT architecture

```{image} ../_static/cvit_architecture.svg
:alt: CViT architecture — conv tokenizer, transformer encoder, U-Net decoder with gated skips, SSL head
:width: 720px
:align: center
```

The network is {class}`nvitk.nn.cvit.CViT`:

```text
x (B, C, *S) ─► tokenizer ─► token grid (B, E, *g)  +  skips [F0 … F_{L-1}]
                               │
                               ▼
                transformer encoder (L blocks)
                               │
                               ▼
              decoder(tokens, skips) ─► logits (B, K, *S)  [+ deep-supervision list]
```

`g = S / token_stride`. Everything is dimension-agnostic, so 2D configurations use the same code.

## Tokenizers

All three return the same `(grid, skips)` pair, so the transformer and the decoder are shared and
an ablation changes only where the tokens come from.

| `tokenizer` | Tokens from | Receptive field of a token | Skips from |
|---|---|---|---|
| `hierarchical` (default) | strided residual conv stages over the whole volume → 1×1 projection | crosses patch borders | the same stages |
| `intra_patch` | the same CNN run **independently inside each patch** | exactly one patch | the per-patch maps folded back |
| `linear` | `Conv(k = s = token_stride)` on raw voxels (ViT / Primus) | one patch, linear | a separate conv stem (built only if a skip needs it) |

**`hierarchical`** is the early-convolution ViT: stage *i* is a residual block group with
per-axis stride `stem_strides[i]` (level 0 has stride 1), InstanceNorm + LeakyReLU, as in
nnU-Net's residual encoders.

**`intra_patch`** follows the original idea literally:

- **Unfold.** The volume is cut into non-overlapping token patches, `(B·N, C, *p)`.
- **Per-patch CNN.** The same weight-shared CNN runs on every patch, zero-padded at the patch
  border. It uses LayerNorm over channels, because InstanceNorm is undefined on the final 1-voxel
  map and LayerNorm uses no spatial statistics. Its stride schedule ends at one voxel, which is
  one token.
- **Fold back.** Intermediate maps are folded back into full-volume skips. This is exact, because
  patches do not overlap.

A unit test checks the guarantee: the gradient of a token with respect to any voxel outside its
own patch is exactly zero. Information therefore crosses patches only through attention.

**`linear`** is the control. Its skip stem exists so that `linear` and `hierarchical` differ
*only* in token origin, not in what the decoder receives.

### Anisotropy and token stride

`stem_strides` holds one per-axis tuple per level, and their product is the token stride. In the
pipeline the strides are the first `log2(--token-stride) + 1` stages of the nnU-Net baseline's
pooling schedule. Anisotropic data (e.g. 3 × 0.8 × 0.8 mm) therefore gets strides like
`(1,1,1),(1,2,2),(2,2,2),(2,2,2)`, with token stride `(4, 8, 8)`. The patch size is rounded down to
a multiple of the token stride.

## Transformer encoder

{class}`nvitk.nn.cvit.transformer.TransformerEncoder`: pre-norm blocks
`x + LS(Attn(LN x))`, `x + LS(SwiGLU(LN x))` with:

- `F.scaled_dot_product_attention` (flash / memory-efficient kernels when available);
- **axial 3D rotary embeddings**: one frequency band per axis, with a `head_dim` remainder passed
  through unrotated so any head size works;
- a learned **absolute embedding** on the token grid, trilinearly resampled when the input size
  changes (other patch sizes, sliding-window borders);
- optional **register tokens** (`num_registers`): positionless tokens that absorb global
  information;
- LayerScale (`layer_scale_init`, 0.1) and stochastic depth (`drop_path_rate`, linear over depth).

For pre-training the encoder also accepts a SimMIM `token_mask` (masked tokens replaced by a
learned `mask_token`) or MAE `keep_idx` (masked tokens dropped; rotary and absolute embeddings
gathered at the kept positions).

## Decoders

**`unet`** ({class}`nvitk.nn.cvit.decoders.UNetDecoder`):

- **Token level.** At level `L-1` (token resolution) the projected transformer output `T` is
  concatenated with `F_{L-1}`.
- **Shallower levels.** Each shallower level upsamples with a transposed conv (the inverse of the
  tokenizer's stride), concatenates its skip, and applies `decoder_convs` conv blocks.
- **Deep supervision.** Heads sit on levels `0 … L-2`, highest resolution first. That is exactly
  the list nnU-Net's deep-supervision wrapper expects, because the plans carry
  `pool_op_kernel_sizes = stem_strides` and nnU-Net derives its scales as
  `1 / cumprod(pool_op_kernel_sizes)[:-1]`.

**`patch`** ({class}`nvitk.nn.cvit.decoders.PatchDecoder`): transposed convolutions from the token
grid alone, one per strided level, with LayerNorm + GELU and halving widths. It has no skips. It is
the masked-image-modelling head and the skip-free ablation.

Skip connections can be disabled per level, dropped, warmed up or gated. See
{doc}`cvit-attention-usage`.

## Presets

Sizes follow Primus, so `head_dim` is a multiple of 6 and every channel of every head carries a
3D rotary frequency.

| Preset | `embed_dim` | `depth` | `num_heads` | Parameters (1 channel, 3 classes, default stem) |
|---|---|---|---|---|
| `CViTS` | 396 | 12 | 6 | 39.9 M |
| `CViTB` | 792 | 12 | 12 | 109.5 M |
| `CViTM` | 864 | 16 | 12 | 162.9 M |
| `CViTL` | 1056 | 24 | 16 | 341.8 M |

Default stem: channels `32·2^i` (capped at 320), blocks `(1, 1, 2, 2, …)`, token stride 8, which
at 128³ means 16³ = 4096 tokens.

## `CViTConfig` reference

{class}`nvitk.nn.cvit.CViTConfig` is the serialisable architecture plan stored in a plans file's
`arch_kwargs`, a bundle's `cvit_config.json` and every pre-training checkpoint.

| Field | Default | Meaning |
|---|---|---|
| `input_channels`, `num_classes` | 1, 2 | Set by nnU-Net from the dataset |
| `input_shape` | (128,128,128) | Training patch; its length sets 2D/3D |
| `tokenizer` | `hierarchical` | `hierarchical` / `intra_patch` / `linear` |
| `stem_channels`, `stem_blocks`, `stem_strides` | (32,64,128,256), (1,1,2,2), isotropic 1,2,2,2 | Per level |
| `embed_dim`, `depth`, `num_heads`, `mlp_ratio` | 792, 12, 12, 8/3 | Transformer size |
| `use_rope`, `use_abs_pos_embed`, `num_registers` | True, True, 0 | Position handling |
| `drop_path_rate`, `attn_drop`, `proj_drop`, `layer_scale_init` | 0.1, 0, 0, 0.1 | Regularisation |
| `decoder`, `decoder_convs`, `deep_supervision` | `unet`, 2, True | Decoder |
| `skips` | `all` | `all`, `none`, bits (`0011`) or bools per level |
| `skip_drop_prob` | 0.0 | Per-sample skip dropout |
| `skip_schedule`, `skip_warmup_epochs` | `constant`, 50 | Skip warm-up |
| `skip_gate`, `skip_gate_init`, `skip_gate_l1` | `none`, 0.5, 0 | Learned gates + L1 |
| `preset` | None | Provenance (`CViTB` …) |

`strides` (nnU-Net's name) is accepted as an alias of `stem_strides`. Unknown fields raise, so a
typo never passes silently.

## Weight transfer

{func}`nvitk.nn.cvit.load_pretrained_encoder` loads the `tokenizer.*` and `encoder.*` tensors from
a stage-1 bundle, another CViT checkpoint, or any nnU-Net/nnssl checkpoint (`network_weights`,
with `module.` and `_orig_mod.` prefixes stripped). It handles three cases:

- **Positional embedding**: resampled when the patch size changes.
- **Input channels**: every conv consuming the raw input (`model.keys_to_in_proj`) is adapted.
  `repeat` tiles the kernels and divides by the expansion factor, which leaves the response to a
  channel-replicated input unchanged; `mean` averages the kernels instead.
- **Strict mode**: any other shape mismatch raises, as does a checkpoint without encoder tensors.
  A silent partial load is how a "pretrained" run quietly trains from scratch.

The key mapping follows nnssl's adaptation-plan contract: `key_to_encoder = "encoder"`,
`key_to_stem = "tokenizer"` and `key_to_lpe = "encoder.pos_embed"`.

## Using `nvitk.nn.cvit` directly

The module imports only torch, so it is usable in any PyTorch code:

```python
import torch
from nvitk.nn.cvit import CViT, CViTMIM, load_pretrained_encoder, intervene

net = CViT(input_channels=1, num_classes=3, preset="CViTS", input_shape=(96, 96, 96),
           tokenizer="intra_patch", skips="0011", skip_drop_prob=0.2)
logits = net(torch.randn(2, 1, 96, 96, 96))          # list: 96³, 48³, 24³ (deep supervision)

mim = CViTMIM(1, method="simmim", preset="CViTS", input_shape=(96, 96, 96), tokenizer="intra_patch")
mask = CViTMIM.random_token_mask(2, mim.config.grid_shape, ratio=0.6)
recon, vox_mask = mim(torch.randn(2, 1, 96, 96, 96), mask)
loss = CViTMIM.loss(recon, torch.randn(2, 1, 96, 96, 96), vox_mask)

load_pretrained_encoder(net, mim.state_dict())         # transfers tokenizer + encoder

with intervene(net.eval(), "tokens_off"):               # decoder sees the conv skips only
    skips_only = net(torch.randn(1, 1, 96, 96, 96))
```
