"""LR schedulers and weight-averaging utilities.

Framework-reusable; none of these know anything about a specific task.

The schedulers expose ``interval`` (and ``monitor`` where applicable) so
``AbstractModel._scheduler_config`` can build a Lightning scheduler config
without per-class branches. Only mutable state is round-tripped via
``state_dict`` / ``load_state_dict``; configuration (``factor``, ``min_lr``,
``warmup_steps``…) is owned by the constructor.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List

import numpy as np
import torch
import torch.nn as nn


class SmoothReduceLROnPlateau(torch.optim.lr_scheduler.LRScheduler):
    """Reduce LR when smoothed recent loss exceeds smoothed historical loss.

    Plateau detector that avoids fighting noise: compares a short-window
    recent mean against the longer-window historical mean, multiplied by
    ``reduction_threshold``.
    """

    interval = "epoch"
    monitor = "val_loss"

    def __init__(
        self,
        optimizer,
        smoothing_window: int = 10,
        historical_window: int = 100,
        reduction_threshold: float = 0.95,
        cooldown: int = 20,
        factor: float = 0.8,
        min_lr: float = 1e-6,
        verbose: bool = False,
    ):
        self.smoothing_window = smoothing_window
        self.historical_window = historical_window
        self.reduction_threshold = reduction_threshold
        self.cooldown = cooldown
        self.factor = factor
        self.min_lr = min_lr
        self.losses: List[float] = []
        self.verbose = verbose
        super().__init__(optimizer)

    def state_dict(self) -> Dict:
        state = super().state_dict()
        state["losses"] = list(self.losses)
        return state

    def load_state_dict(self, state_dict: Dict) -> None:
        state = dict(state_dict)
        self.losses = list(state.pop("losses", []))
        super().load_state_dict(state)

    def step(self, metrics: float | None = None) -> None:
        if metrics is None:
            self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
            return
        self.losses.append(metrics)
        while len(self.losses) > self.historical_window:
            self.losses.pop(0)
        if len(self.losses) < self.historical_window:
            return

        recent_avg = float(np.mean(self.losses[-self.smoothing_window:]))
        historical_avg = float(np.mean(self.losses[:-self.smoothing_window]))

        if recent_avg > historical_avg * self.reduction_threshold:
            if self.verbose:
                print(
                    f'Reducing learning rate: recent_avg={recent_avg:.4f} > '
                    f'historical_avg={historical_avg:.4f} * {self.reduction_threshold}'
                )
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = max(group['lr'] * self.factor, self.min_lr)
                if self.verbose:
                    print(f'Reducing learning rate of group {i} to {group["lr"]:.4e}.')
            if self.cooldown > 0:
                self.losses = self.losses[:-self.cooldown]

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self) -> List[float]:
        return [group['lr'] for group in self.optimizer.param_groups]


class CosineWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Linear warmup → cosine decay to ``min_lr_ratio * base_lr``.

    Step once per training batch. Each param group gets the same multiplier
    applied to its own ``base_lr`` (snapshotted at construction).
    """

    interval = "step"

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ):
        if not 0 <= warmup_steps < total_steps:
            raise ValueError("warmup_steps must satisfy 0 <= warmup_steps < total_steps")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer)

    def _multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        if step >= self.total_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def get_lr(self) -> List[float]:
        m = self._multiplier(max(0, self.last_epoch))
        return [base_lr * m for base_lr in self.base_lrs]


class EMAWeightTracker:
    """Exponential moving average of a parameter iterable.

    Holds a parallel shadow copy of each parameter. ``update(params)`` after
    each optimizer step refreshes the shadow. ``swap_into(params)`` is a
    context manager that temporarily replaces ``params`` with the shadow
    values for the duration of a ``with`` block (e.g. an eval call).

    Matches or exceeds SWA on test accuracy at zero training cost
    (arxiv:2411.18704). Typical decay 0.999.
    """

    def __init__(
        self,
        parameters: Iterable[nn.Parameter],
        *,
        decay: float = 0.999,
        warmup_steps: int = 1000,
    ):
        if not 0.0 < decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        self.decay = float(decay)
        self.warmup_steps = int(warmup_steps)
        self._num_updates = 0
        self._shadow: List[torch.Tensor] = [
            p.detach().clone().to(dtype=torch.float32) for p in parameters
        ]

    def _effective_decay(self) -> float:
        if self.warmup_steps <= 0:
            return self.decay
        return min(self.decay, (1.0 + self._num_updates) / (10.0 + self._num_updates))

    @torch.no_grad()
    def update(self, parameters: Iterable[nn.Parameter]) -> None:
        d = self._effective_decay()
        for shadow, param in zip(self._shadow, parameters):
            shadow.mul_(d).add_(param.detach().to(dtype=torch.float32), alpha=1.0 - d)
        self._num_updates += 1

    @contextmanager
    def swap_into(self, parameters: Iterable[nn.Parameter]) -> Iterator[None]:
        """Temporarily replace ``parameters`` with EMA copies for the duration
        of a ``with`` block, then restore them on exit."""
        params = list(parameters)
        backup = [p.detach().clone() for p in params]
        with torch.no_grad():
            for p, shadow in zip(params, self._shadow):
                p.data.copy_(shadow.to(dtype=p.dtype, device=p.device))
        try:
            yield
        finally:
            with torch.no_grad():
                for p, b in zip(params, backup):
                    p.data.copy_(b)

    def state_dict(self) -> Dict:
        return {
            "_num_updates": self._num_updates,
            "_shadow": [s.detach().cpu() for s in self._shadow],
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self._num_updates = int(state_dict["_num_updates"])
        self._shadow = [t.clone() for t in state_dict["_shadow"]]
