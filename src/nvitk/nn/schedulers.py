"""
Learning-rate schedules for transformer training (epoch-stepped, nnU-Net / nnssl compatible).

Description
-----------
:class:`WarmupPolyLR` is a linear warm-up followed by polynomial decay — the recipe nnU-Net's
Primus trainers use — with one difference that matters for fine-tuning: it multiplies the
scheduled rate by each parameter group's ``lr_scale`` instead of writing one value into every
group. nnU-Net's own warm-up schedulers do the latter, which silently erases layer-wise LR decay
(:func:`nvitk.nn.cvit.weights.layerwise_lr_groups`).

Both training engines call ``scheduler.step(current_epoch)`` once per epoch, so the schedule is a
pure function of the epoch and needs no state in the checkpoint.
"""

from __future__ import annotations

from torch.optim.lr_scheduler import _LRScheduler


class WarmupPolyLR(_LRScheduler):
    """Linear warm-up for ``warmup`` epochs, then ``(1 - t)^exponent`` decay to 0 at ``max_steps``.

    Parameters
    ----------
    initial_lr
        Peak learning rate (reached at the end of warm-up).
    max_steps
        Total epochs.
    warmup
        Warm-up epochs (``0`` disables warm-up).
    """

    def __init__(self, optimizer, initial_lr: float, max_steps: int, warmup: int,
                 exponent: float = 0.9, current_step: int | None = None):
        self.optimizer = optimizer
        self.initial_lr = float(initial_lr)
        self.max_steps = int(max_steps)
        self.warmup = int(max(warmup, 0))
        self.exponent = float(exponent)
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def lr_at(self, step: int) -> float:
        """Scheduled base rate at epoch *step* (before ``lr_scale``)."""
        if self.warmup and step < self.warmup:
            return self.initial_lr * (step + 1) / self.warmup
        span = max(self.max_steps - self.warmup, 1)
        frac = min(max(step - self.warmup, 0) / span, 1.0)
        return self.initial_lr * (1.0 - frac) ** self.exponent

    def step(self, current_step=None):  # noqa: D102 - signature fixed by the engines
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        lr = self.lr_at(int(current_step))
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)


def decay_groups(model, weight_decay: float) -> list[dict]:
    """Two AdamW groups: decay on matrices / kernels; none on norms, biases, embeddings, gates."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(("pos_embed", "registers", "mask_token", "gate_logits")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "lr_scale": 1.0},
        {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
    ]


__all__ = ["WarmupPolyLR", "decay_groups"]
