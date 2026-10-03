"""
CViT — Convolutional Vision Transformer.

Description
-----------
A ViT whose tokens come from convolutions instead of a linear projection of flattened patches,
so attention operates on geometry-aware tokens and the conv features double as decoder skips.

:mod:`~nvitk.nn.cvit.config`        :class:`CViTConfig` (serialisable architecture plan) + presets
:mod:`~nvitk.nn.cvit.tokenizers`    ``hierarchical`` / ``intra_patch`` / ``linear`` tokenizers
:mod:`~nvitk.nn.cvit.transformer`   encoder with RoPE, registers, SimMIM / MAE masking
:mod:`~nvitk.nn.cvit.decoders`      U-Net decoder with skip controls; patch decoder
:mod:`~nvitk.nn.cvit.model`         :class:`CViT`, :class:`CViTMIM`, :func:`build_cvit`
:mod:`~nvitk.nn.cvit.weights`       pretrained-encoder transfer, layer-wise LR decay
:mod:`~nvitk.nn.cvit.probe`         attention-usage interventions and statistics

Examples
--------
>>> import torch
>>> from nvitk.nn.cvit import CViT
>>> net = CViT(1, 4, preset="CViTS", input_shape=(64, 64, 64))
>>> [o.shape[2:] for o in net(torch.zeros(1, 1, 64, 64, 64))]   # deep supervision
[torch.Size([64, 64, 64]), torch.Size([32, 32, 32]), torch.Size([16, 16, 16])]
"""

from __future__ import annotations

from .config import PRESETS, CViTConfig, parse_skips
from .model import CViT, CViTMIM, build_cvit, config_from_kwargs
from .probe import MODES as PROBE_MODES
from .probe import attention_stats, intervene, skip_gate_values
from .weights import encoder_state_dict, layerwise_lr_groups, load_pretrained_encoder, read_state_dict

__all__ = [
    "CViT",
    "CViTConfig",
    "CViTMIM",
    "PRESETS",
    "PROBE_MODES",
    "attention_stats",
    "build_cvit",
    "config_from_kwargs",
    "encoder_state_dict",
    "intervene",
    "layerwise_lr_groups",
    "load_pretrained_encoder",
    "parse_skips",
    "read_state_dict",
    "skip_gate_values",
]
