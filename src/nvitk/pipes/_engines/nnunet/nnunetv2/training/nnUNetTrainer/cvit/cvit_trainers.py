"""CViT trainers: Convolutional Vision Transformers on nnU-Net's training engine.

The network is :class:`nvitk.nn.cvit.CViT`, rebuilt from the plans file's ``arch_kwargs`` (a
serialised ``CViTConfig``), so everything nnU-Net does around it — planning, preprocessing,
augmentation, deep supervision, validation, sliding-window prediction — is unchanged.

What these trainers change
--------------------------
Optimiser
    AdamW (betas 0.9/0.98, weight decay on matrices only) with a linear warm-up followed by
    poly decay, gradient clipping at 1 — the transformer recipe of nnU-Net's Primus trainers.
    nnU-Net's own warm-up schedulers write one LR into every parameter group, which would erase
    layer-wise LR decay, so :class:`nvitk.nn.schedulers.WarmupPolyLR` multiplies by each
    group's ``lr_scale``.
Pretrained encoder
    ``NVITK_CVIT_PRETRAINED`` names a checkpoint whose ``tokenizer.*`` / ``encoder.*`` weights are
    loaded in :meth:`initialize` (positional embedding resampled, input channels adapted), with
    optional layer-wise LR decay (``NVITK_CVIT_LLRD``).
Skip control
    ``skip_schedule="warmup"`` ramps the decoder's skip scale from 0 to 1 over
    ``skip_warmup_epochs``; ``skip_gate="learned"`` adds ``skip_gate_l1 · Σ gates`` to the loss.
Attention probe
    Every ``NVITK_CVIT_PROBE_EVERY`` epochs, attention statistics and gate values are computed on
    one fixed validation batch and appended to ``cvit_probe.jsonl`` (and TensorBoard events under
    ``<fold>/tensorboard`` when the package is installed).
Loss
    One class per entry of :data:`nvitk.segmentation.loss_registry.SEGMENTATION_LOSSES`
    (``nnUNetTrainerCViT_<loss>``) so each objective gets its own results folder, plus
    ``_custom`` reading ``NVITK_CVIT_LOSS_SPEC``.

All settings beyond the plans file arrive through :mod:`nvitk.pipes.cvit.util.trainer_env`.
"""

from __future__ import annotations

import json
from math import isfinite

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from torch import autocast

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import dummy_context

from nvitk.nn.cvit import (
    attention_stats,
    build_cvit,
    layerwise_lr_groups,
    load_pretrained_encoder,
)
from nvitk.nn.cvit.weights import _unwrap
from nvitk.nn.schedulers import WarmupPolyLR, decay_groups
from nvitk.pipes.cvit.util.trainer_env import TRAINER_PREFIX, read_settings
from nvitk.segmentation.loss_registry import (
    SEGMENTATION_LOSSES,
    LossContext,
    build_segmentation_loss,
)

#: Accepted ``network_class_name`` values in a plans file.
CVIT_CLASS_NAMES = ("nvitk.nn.cvit.CViT", "nvitk.nn.cvit.model.CViT", "CViT")


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────


class nnUNetTrainerCViT(nnUNetTrainer):
    """CViT on nnU-Net, Dice + CE (nnU-Net default objective)."""

    loss_name: str = "dice_ce"
    loss_config: dict = {}

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        # Explicit signature: nnUNetTrainer.__init__ rebuilds my_init_kwargs by introspecting it.
        super().__init__(plans, configuration, fold, dataset_json, device)
        s = read_settings()
        self.cvit_settings = s
        if s.num_epochs is not None:
            self.num_epochs = int(s.num_epochs)
        if s.iterations_per_epoch is not None:
            self.num_iterations_per_epoch = int(s.iterations_per_epoch)
            self.num_val_iterations_per_epoch = max(1, int(s.iterations_per_epoch) // 5)
        self.initial_lr = float(s.lr)
        self.weight_decay = float(s.weight_decay)
        # A 50-epoch warm-up would swallow a short run entirely.
        self.warmup_epochs = int(min(s.warmup_epochs, max(1, self.num_epochs // 10)))

        arch = dict(self.configuration_manager.network_arch_init_kwargs)
        self.cvit_arch = arch
        self.enable_deep_supervision = (
            arch.get("decoder", "unet") == "unet" and bool(arch.get("deep_supervision", True))
        )
        self.skip_schedule = str(arch.get("skip_schedule", "constant"))
        self.skip_warmup_epochs = int(arch.get("skip_warmup_epochs", 50))
        self.skip_gate_l1 = float(arch.get("skip_gate_l1", 0.0)) if arch.get("skip_gate") == "learned" else 0.0
        self._probe_batch: torch.Tensor | None = None
        self._tb_writer = None

    # ---- network ---------------------------------------------------------------------------
    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
                                   num_input_channels, num_output_channels,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """Rebuild :class:`~nvitk.nn.cvit.CViT` from the plans' ``arch_kwargs``.

        Raises
        ------
        ValueError
            If the plans file describes another architecture — a CViT trainer silently training
            a ResEnc U-Net would mislabel every result.
        """
        if architecture_class_name not in CVIT_CLASS_NAMES:
            raise ValueError(
                f"{architecture_class_name!r} is not a CViT plan. Generate plans with "
                f"nvitk-cvit (stage2) or set network_class_name to 'nvitk.nn.cvit.CViT'."
            )
        return build_cvit(
            dict(arch_init_kwargs),
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
        )

    def initialize(self):
        super().initialize()
        if self.cvit_settings.pretrained:
            report = load_pretrained_encoder(self.network, self.cvit_settings.pretrained)
            self.print_to_log_file(
                "Pretrained encoder "
                f"{self.cvit_settings.pretrained}: loaded={len(report['loaded'])} "
                f"resized={report['resized']} adapted={report['adapted']} "
                f"missing={len(report['missing'])} skipped={len(report['skipped'])}"
            )
        net = _unwrap(self.network)
        n_params = sum(p.numel() for p in net.parameters())
        self.print_to_log_file(f"CViT config: {json.dumps(net.config.to_dict(), sort_keys=True)}")
        self.print_to_log_file(f"CViT parameters: {n_params / 1e6:.2f} M; deep supervision "
                               f"{self.enable_deep_supervision}; warm-up {self.warmup_epochs} epochs")

    def configure_optimizers(self):
        net = _unwrap(self.network)
        llrd = float(self.cvit_settings.llrd)
        if llrd < 1.0:
            groups = layerwise_lr_groups(net, self.initial_lr, decay=llrd, weight_decay=self.weight_decay)
        else:
            groups = decay_groups(net, self.weight_decay)
        optimizer = torch.optim.AdamW(
            groups, lr=self.initial_lr, weight_decay=self.weight_decay, betas=(0.9, 0.98),
            fused=self.device.type == "cuda",
        )
        scheduler = WarmupPolyLR(optimizer, self.initial_lr, self.num_epochs, self.warmup_epochs)
        return optimizer, scheduler

    # ---- loss --------------------------------------------------------------------------------
    def _resolve_loss(self) -> tuple[str, dict]:
        return self.loss_name, dict(self.loss_config)

    def _build_loss(self):
        name, config = self._resolve_loss()
        ctx = LossContext(
            batch_dice=self.configuration_manager.batch_dice,
            has_regions=self.label_manager.has_regions,
            ignore_label=self.label_manager.ignore_label,
            is_ddp=self.is_ddp,
            num_classes=self.label_manager.num_segmentation_heads,
        )
        self.print_to_log_file(f"Building loss {name!r} with config {config}")
        loss = build_segmentation_loss(name, ctx, config)
        if not self.enable_deep_supervision:
            return loss
        scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(scales))])
        weights[-1] = 1e-6 if (self.is_ddp and not self._do_i_compile()) else 0
        return DeepSupervisionWrapper(loss, weights / weights.sum())

    # ---- mirroring -------------------------------------------------------------------------
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation, dummy_2d, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        if self.cvit_settings.no_mirror:
            self.inference_allowed_mirroring_axes = None
            mirror_axes = None
        return rotation, dummy_2d, initial_patch_size, mirror_axes

    # ---- epoch hooks -----------------------------------------------------------------------
    def set_deep_supervision_enabled(self, enabled: bool):
        _unwrap(self.network).decoder.deep_supervision = bool(enabled) and self.enable_deep_supervision

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        if self.skip_schedule == "warmup":
            scale = min(1.0, self.current_epoch / max(self.skip_warmup_epochs, 1))
            _unwrap(self.network).set_skip_scale(scale)
            self.print_to_log_file(f"Skip scale: {scale:.3f}")

    def train_step(self, batch: dict) -> dict:
        data = batch["data"].to(self.device, non_blocking=True)
        target = batch["target"]
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(data)
            loss = self.loss(output, target)
            if self.skip_gate_l1 > 0:
                loss = loss + self.skip_gate_l1 * _unwrap(self.network).gate_penalty()

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
            self.optimizer.step()
        return {"loss": loss.detach().cpu().numpy()}

    def on_epoch_end(self):
        every = int(self.cvit_settings.probe_every)
        last = self.current_epoch == self.num_epochs - 1
        if self.local_rank == 0 and every > 0 and (self.current_epoch % every == 0 or last):
            try:
                self._log_probe()
            except Exception as exc:  # monitoring must never kill a training run
                self.print_to_log_file(f"Attention probe failed (training continues): {exc!r}")
        super().on_epoch_end()

    # ---- attention probe -------------------------------------------------------------------
    def _log_probe(self) -> None:
        if self._probe_batch is None:
            batch = next(self.dataloader_val)
            self._probe_batch = batch["data"][:1].clone()
        x = self._probe_batch.to(self.device)
        net = _unwrap(self.network)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            stats = attention_stats(net, x, spacing_mm=self.configuration_manager.spacing, n_queries=256)
        stats["epoch"] = int(self.current_epoch)
        stats["skip_scale"] = float(getattr(net.decoder, "skip_scale", 1.0))
        with open(join(self.output_folder, "cvit_probe.jsonl"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(stats) + "\n")
        summary = ", ".join(
            f"L{l['layer']}:{l['mean_distance_mm']:.1f}mm/{l['residual_ratio']:.3f}"
            for l in stats["layers"]
        )
        self.print_to_log_file(f"Attention probe (distance / residual ratio): {summary}")
        if stats["skip_gates"]:
            self.print_to_log_file(f"Skip gates: {stats['skip_gates']}")
        self._write_tensorboard(stats)

    def _write_tensorboard(self, stats: dict) -> None:
        if self._tb_writer is None:
            try:
                from nvitk.core.tensorboard import summary_writer_class
                self._tb_writer = summary_writer_class()(log_dir=join(self.output_folder, "tensorboard"))
            except ImportError:
                self._tb_writer = False
        if not self._tb_writer:
            return
        epoch = stats["epoch"]
        for layer in stats["layers"]:
            i = layer["layer"]
            for key in ("mean_distance_mm", "entropy", "residual_ratio", "register_mass"):
                value = layer[key]
                if isfinite(value):
                    self._tb_writer.add_scalar(f"cvit_probe/{key}/layer_{i:02d}", value, epoch)
        for level, value in stats["skip_gates"].items():
            self._tb_writer.add_scalar(f"cvit_probe/skip_gate/level_{level}", value, epoch)
        self._tb_writer.add_scalar("cvit_probe/skip_scale", stats["skip_scale"], epoch)
        self._tb_writer.flush()

    def on_train_end(self):
        super().on_train_end()
        if self._tb_writer:
            self._tb_writer.close()


class _CustomLossMixin:
    """Reads ``{"loss": name, "config": {...}}`` from ``NVITK_CVIT_LOSS_SPEC``."""

    def _resolve_loss(self) -> tuple[str, dict]:
        spec = self.cvit_settings.loss_spec
        if not spec or not spec.get("loss"):
            raise RuntimeError(
                f"{type(self).__name__} needs NVITK_CVIT_LOSS_SPEC='{{\"loss\": ..., \"config\": ...}}'. "
                "Run through 'nvitk-cvit' rather than invoking nnUNetv2_train directly."
            )
        return str(spec["loss"]), dict(spec.get("config") or {})


# One trainer per registered loss, materialised where nnU-Net's class lookup finds them.
for _loss in SEGMENTATION_LOSSES:
    _cls = type(f"{TRAINER_PREFIX}_{_loss}", (nnUNetTrainerCViT,), {
        "__doc__": f"CViT on nnU-Net, loss {_loss!r}: {SEGMENTATION_LOSSES[_loss].description}",
        "loss_name": _loss,
        "loss_config": {},
    })
    globals()[_cls.__name__] = _cls
globals()[f"{TRAINER_PREFIX}_custom"] = type(
    f"{TRAINER_PREFIX}_custom", (_CustomLossMixin, nnUNetTrainerCViT),
    {"__doc__": "CViT on nnU-Net with a loss given by NVITK_CVIT_LOSS_SPEC."},
)
del _loss, _cls

__all__ = [
    "nnUNetTrainerCViT",
    *(f"{TRAINER_PREFIX}_{name}" for name in SEGMENTATION_LOSSES),
    f"{TRAINER_PREFIX}_custom",
]
