"""
Deep-learning building blocks (PyTorch).

Description
-----------
Reusable network components for nvitk's learning pipelines. Requires ``torch``; deliberately
**not** imported by ``nvitk/__init__`` so that importing the toolkit never needs it (the same
rule as :mod:`nvitk.segmentation.losses`).

Self-containment contract
-------------------------
Modules under ``nvitk.nn`` import only ``torch`` and **relative** siblings — never other
``nvitk`` modules. That lets a pipeline copy the folder verbatim into an inference container
(see :mod:`nvitk.pipes.cvit.stage5_export`) and lets the in-tree nnU-Net build import it inside
its training subprocess without dragging the GPU/array stack along.

Subpackages
-----------
:mod:`nvitk.nn.blocks`
    Dimension-agnostic conv / norm / regularisation blocks (1D–3D).
:mod:`nvitk.nn.cvit`
    Convolutional Vision Transformer: conv tokenizers, transformer encoder, U-Net / patch
    decoders, masked-image-modelling variant, weight transfer and attention-usage probes.
"""

from __future__ import annotations

__all__ = ["blocks", "cvit"]
