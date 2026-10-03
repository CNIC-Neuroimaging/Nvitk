"""
CViT self-supervised trainers on the nnssl engine (SimMIM and MAE).

Description
-----------
nnssl supplies the corpus handling — collection parsing, planning, preprocessing to blosc2,
data loading with spatial augmentation, logging and checkpointing. These trainers only replace
the network (:class:`nvitk.nn.cvit.CViTMIM`), the masking, the objective and the optimiser.

Import order
------------
nnssl binds its data roots **at import**, so this module must be imported only after
:func:`nvitk.pipes._engines.env.apply_nnssl_env` (stage 1 does this).

Masking
-------
Each step draws a token-grid mask with exactly ``round(mask_percentage · N)`` masked tokens per
sample (:meth:`CViTMIM.random_token_mask`). Masked voxels are zeroed at the input and masked
tokens replaced (SimMIM) or dropped (MAE); the loss is MSE on masked voxels only. A token is one
token-stride block, so the mask granularity follows the tokenizer exactly.

Adaptation plan
---------------
nnssl verifies before training that its pretrained weights will load downstream. The plan names
``nvitk.nn.cvit.CViT`` with the CViT key mapping; :meth:`verify_adaptation_plans` is overridden to
build that downstream network and load the encoder with :func:`load_pretrained_encoder`
(``strict=True``), so an incompatible configuration fails at epoch 0 instead of at fine-tuning.
The architecture itself travels as ``cvit_config.json`` beside the adaptation plan and as
``cvit_config`` inside every checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from torch import autocast
from torch._dynamo import OptimizedModule

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseMAETrainer import BaseMAETrainer
from nnssl.utilities.helpers import dummy_context

from nvitk.core.logger import Logger
from nvitk.nn.cvit import CViT, CViTMIM, load_pretrained_encoder
from nvitk.nn.cvit.weights import _unwrap
from nvitk.nn.schedulers import WarmupPolyLR, decay_groups

log = Logger()

#: ``arch_class_name`` written into the adaptation plan.
CVIT_ARCH_CLASS = "nvitk.nn.cvit.CViT"

#: Bundle / trainer-output file carrying the serialised :class:`CViTConfig` of the encoder.
CVIT_CONFIG_FILE = "cvit_config.json"


class CViTSimMIMTrainer(BaseMAETrainer):
    """SimMIM pre-training of a CViT tokenizer + encoder.

    Attributes set by stage 1 before :meth:`initialize`
    ---------------------------------------------------
    cvit_arch
        :class:`~nvitk.nn.cvit.CViTConfig` fields (``preset`` allowed). ``input_shape`` is
        always taken from ``config_plan.patch_size``.
    mask_percentage
        Fraction of tokens masked per sample.
    """

    method = "simmim"

    def __init__(self, plan, configuration_name: str, fold, pretrain_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (128, 128, 128)
        self.cvit_arch: dict[str, Any] = {"preset": "CViTB"}
        self.mask_percentage = 0.6
        self.initial_lr = 3e-4
        self.weight_decay = 5e-2
        self.grad_clip = 1.0
        self.warmup_epochs = 50
        self.enable_deep_supervision = False

    # ---- network + adaptation plan ---------------------------------------------------------
    def build_architecture_and_adaptation_plan(self, config_plan, num_input_channels, num_output_channels):
        arch = {k: v for k, v in dict(self.cvit_arch).items() if k != "input_shape"}
        net = CViTMIM(num_input_channels, method=self.method,
                      input_shape=tuple(int(p) for p in config_plan.patch_size), **arch)
        self.recommended_downstream_patchsize = tuple(int(p) for p in config_plan.patch_size)
        plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans(CVIT_ARCH_CLASS),
            pretrain_plan=self.plan,
            pretrain_num_input_channels=num_input_channels,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            key_to_encoder=net.key_to_encoder,
            key_to_stem=net.key_to_stem,
            keys_to_in_proj=tuple(net.keys_to_in_proj),
            key_to_lpe=net.key_to_lpe,
        )
        save_json(net.config.to_dict(), join(self.output_folder_base, CVIT_CONFIG_FILE), sort_keys=True)
        return net, plan

    def verify_adaptation_plans(self, adaptation_plan_dict: dict, configuration: str, state_dict: dict):
        """Build the downstream :class:`CViT` and load the encoder into it, strictly."""
        cfg = _unwrap(self.network).config
        downstream_kwargs = {
            k: v for k, v in cfg.to_dict().items()
            if k not in ("input_channels", "num_classes", "deep_supervision", "decoder", "skips")
        }
        downstream = CViT(cfg.input_channels, 2, deep_supervision=False, decoder="unet",
                          skips="all", **downstream_kwargs)
        # strict=True raises on any pre-trained tensor that does not fit the downstream network.
        # Downstream tensors absent from pre-training are by construction parts the MIM network
        # does not build — e.g. the linear tokenizer's skip stem, which only feeds decoder skips —
        # and legitimately start from scratch.
        report = load_pretrained_encoder(downstream, state_dict, strict=True)
        self.print_to_log_file(
            f"Adaptation verified: {len(report['loaded'])} encoder tensors load into a downstream CViT"
            + (f"; {len(report['missing'])} downstream-only tensor(s) start fresh "
               f"(e.g. {report['missing'][0]})" if report["missing"] else "")
        )

    def build_loss(self):
        return CViTMIM.loss

    def configure_optimizers(self):
        groups = decay_groups(_unwrap(self.network), self.weight_decay)
        optimizer = torch.optim.AdamW(groups, lr=self.initial_lr, weight_decay=self.weight_decay,
                                      betas=(0.9, 0.98), fused=self.device.type == "cuda")
        warmup = int(min(self.warmup_epochs, max(1, self.num_epochs // 10)))
        return optimizer, WarmupPolyLR(optimizer, self.initial_lr, self.num_epochs, warmup)

    # ---- steps ---------------------------------------------------------------------------------
    def _masked_step(self, batch: dict) -> torch.Tensor:
        data = batch["data"].to(self.device, non_blocking=True)
        cfg = _unwrap(self.network).config
        grid = tuple(s // t for s, t in zip(data.shape[2:], cfg.token_stride))
        token_mask = CViTMIM.random_token_mask(data.shape[0], grid, self.mask_percentage, device=self.device)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            recon, vox_mask = self.network(data, token_mask)
            return CViTMIM.loss(recon, data, vox_mask)

    def train_step(self, batch: dict) -> dict:
        self.optimizer.zero_grad(set_to_none=True)
        loss = self._masked_step(batch)
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        return {"loss": self._masked_step(batch).detach().cpu().numpy()}

    # ---- checkpoints carry the architecture ----------------------------------------------------
    def save_checkpoint(self, filename: str, live_upload: bool = False) -> None:
        if self.local_rank != 0:
            return
        if self.disable_checkpointing:
            self.print_to_log_file("No checkpoint written, checkpointing is disabled")
            return
        mod = self.network.module if self.is_ddp else self.network
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod
        checkpoint = {
            "network_weights": mod.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
            "logging": self.logger.get_checkpoint(),
            "_best_ema": self._best_ema,
            "current_epoch": self.current_epoch + 1,
            "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "nnssl_adaptation_plan": self.adaptation_plan.serialize(),
            "cvit_config": mod.config.to_dict(),
            "cvit_method": self.method,
        }
        torch.save(self._convert_numpy(checkpoint), filename)


class CViTMAETrainer(CViTSimMIMTrainer):
    """MAE pre-training (masked tokens dropped from the encoder)."""

    method = "mae"

    def __init__(self, plan, configuration_name: str, fold, pretrain_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.mask_percentage = 0.75


#: Trainer by ``--ssl`` method.
SSL_TRAINERS: dict[str, type] = {"simmim": CViTSimMIMTrainer, "mae": CViTMAETrainer}


def read_bundle_config(checkpoint: Path | str) -> dict[str, Any]:
    """The :class:`CViTConfig` dict stored in a CViT pretraining checkpoint (or its sidecar)."""
    ckpt = Path(checkpoint)
    sidecar = ckpt.parent / CVIT_CONFIG_FILE
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if "cvit_config" not in data:
        raise ValueError(f"{ckpt} is not a CViT pretraining checkpoint (no 'cvit_config').")
    return dict(data["cvit_config"])


__all__ = [
    "CVIT_ARCH_CLASS",
    "CVIT_CONFIG_FILE",
    "CViTMAETrainer",
    "CViTSimMIMTrainer",
    "SSL_TRAINERS",
    "read_bundle_config",
]
