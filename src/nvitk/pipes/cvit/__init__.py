"""
CViT — Convolutional Vision Transformer segmentation pipeline.

A ViT tokenises a volume with a linear projection of flattened patches, which discards the
geometric priors of convolutions exactly where local shape lives. CViT tokenises with
convolutions instead (:mod:`nvitk.nn.cvit`): attention runs over geometry-aware tokens and the
multi-scale conv features double as U-Net decoder skips.

The pipeline trains, pre-trains, evaluates, probes and exports CViT models on top of the
vendored nnU-Net (planning, preprocessing, augmentation, deep supervision, sliding-window
inference) and nnssl (self-supervised masked image modelling) engines in
:mod:`nvitk.pipes._engines`.

Entry points
------------
``nvitk-cvit`` (:mod:`nvitk.pipes.cvit.run`) — all stages, local or SGE.
``nvitk-cvit-infer`` (:mod:`nvitk.pipes.cvit.stage4_infer`) — standalone inference.

Stages
------
``stage0``   dataset validation / conversion to nnU-Net raw (+ optional nnssl corpus)
``stage1``   self-supervised pre-training (SimMIM / MAE) → encoder bundle
``stage2``   CViT plans, preprocessing and per-fold training
``stage3``   cross-validation evaluation (Dice / HD95 / clDice per label)
``stage3b``  attention-usage probe (interventions + attention statistics)
``stage4``   inference on new cases
``stage5``   portable model export
"""

from __future__ import annotations

__all__: list[str] = []
