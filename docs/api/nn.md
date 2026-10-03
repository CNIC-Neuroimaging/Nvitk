# Deep-learning building blocks (`nvitk.nn`)

`nvitk.nn` holds PyTorch network components used by nvitk's learning pipelines. It requires
`torch` and is **not** imported by `import nvitk`, so the rest of the toolkit never needs torch
(the same rule as `nvitk.segmentation.losses`).

```{important}
**Self-containment contract.** Modules under `nvitk.nn` import only `torch` and their own
siblings, through relative imports, never other `nvitk` modules. That is what lets the CViT
export ({doc}`../pipelines/cvit`, stage 5) copy the folder verbatim as `cvit_nn/` and run it with
torch and the released `nnunetv2` alone. It also lets the in-tree nnU-Net import it inside its
training subprocess.
```

## `nvitk.nn.blocks`

Dimension-agnostic (1D–3D) components: `ConvNormAct`, `ResBlock` / `ResStage` (nnU-Net-style
residual encoder blocks), `ConvBlock`, `LayerNormNd` (channel LayerNorm per location),
`DropPath`, `LayerScale`, `SwiGLU`, and `init_weights` (He-normal for convolutions, truncated
normal for linear layers; safe with `Module.apply`).

## `nvitk.nn.cvit` — Convolutional Vision Transformer

| Module | Contents |
|---|---|
| `config` | `CViTConfig` (serialisable architecture plan, validation, presets `CViTS/B/M/L`), `parse_skips` |
| `tokenizers` | `HierarchicalConvTokenizer`, `IntraPatchConvTokenizer`, `LinearPatchTokenizer`, `build_tokenizer` |
| `transformer` | `RoPE`, `Attention` (SDPA), `Block`, `TransformerEncoder` (absolute + rotary positions, registers, SimMIM/MAE masking, local-attention masks) |
| `decoders` | `UNetDecoder` (skips, deep supervision, skip controls), `PatchDecoder` |
| `model` | `CViT` (segmentation; nnU-Net constructor contract), `CViTMIM` (masked image modelling), `build_cvit`, `config_from_kwargs` |
| `weights` | `load_pretrained_encoder` (pos-embed resize, input-channel adaptation, strict), `encoder_state_dict`, `read_state_dict`, `layerwise_lr_groups` |
| `probe` | `intervene` (attention-usage interventions), `attention_stats` (distance in mm, entropy, residual ratio), `skip_gate_values` |

Design, tokenizer semantics, presets and a usage example are in {doc}`../pipelines/cvit-architecture`.
The interventions and indices are in {doc}`../pipelines/cvit-attention-usage`.

## `nvitk.nn.schedulers`

`WarmupPolyLR`: linear warm-up then poly decay, stepped per epoch by the nnU-Net and nnssl
engines. It multiplies each parameter group's `lr_scale`, so layer-wise LR decay survives the
schedule. `decay_groups` splits parameters into weight-decay and no-decay groups (norms, biases,
embeddings, gates).

## Related

- {mod}`nvitk.segmentation.loss_registry`: the torch-free registry of segmentation and SSL losses
  shared by the CViT and TopBrain trainers. The implementations beyond nnU-Net's own are in
  {mod}`nvitk.segmentation.losses`.
- `nvitk.pipes._engines`: the vendored nnU-Net / nnssl builds and their environment helpers
  (`env.nnunet_env`, `env.apply_nnssl_env`, `nnunet_run`).
