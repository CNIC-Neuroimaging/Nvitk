"""ToPBrain / ToPAneu whole-brain vessel segmentation pipeline.

Builds a 36-class (TA36) multiclass vessel segmentation model for the
`ToPBrain 2026 <https://topbrain2026.grand-challenge.org/>`_ and
`ToPAneu 2026 <https://topaneu-26.grand-challenge.org/>`_ challenges from the
25 CTA + 25 MRA volumes of the TopBrain data release.

Because 50 volumes cannot train a 36-class 3D network from scratch, the encoder is
bootstrapped from self-supervised pre-training: either a published checkpoint (OpenMind
ResEnc-L) or a run of the vendored `nnssl <https://github.com/MIC-DKFZ/nnssl>`_ framework
over an assembled angiographic corpus. Fine-tuning then runs on **stock nnU-Net v2** — the
nnssl ResEnc-L and nnU-Net's ``nnUNetPlannerResEncL`` reference network share a topology and
a parameter-name space, so a small adapter is enough (see
:mod:`nvitk.pipes.topbrain.util.weight_adapter`).

Entry point
-----------
``nvitk-topbrain`` (:mod:`nvitk.pipes.topbrain.run`) — stages 0-7, dispatched either locally
or as SGE jobs.

Stages
------
``stage0``
    Challenge release → nnU-Net raw dataset (harmonised intensities, remapped labels,
    grouped-by-patient folds).
``stage1``/``stage2``
    *(SSL route)* assemble an unlabeled corpus → nnssl pre-training.
``stage3``
    Pre-trained checkpoint → nnU-Net-shaped weights.
``stage4``
    nnU-Net planning, preprocessing and per-fold training.
``stage5``/``stage6``
    Inference with topology-aware post-processing → the six challenge metrics.
``stage7``
    Grand Challenge Docker submission.
"""

from __future__ import annotations

__all__: list[str] = []
