"""LR schedulers and weight-averaging utilities.

Framework-reusable; none of these know anything about geolocation.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List

import numpy as np
import torch
import torch.nn as nn


class SmoothReduceLROnPlateau:
    """Reduce LR when smoothed recent loss exceeds smoothed historical loss.

    Plateau detector that avoids fighting noise: compares a short-window
    recent mean against the longer-window historical mean, multiplied by
    `reduction_threshold`. Good general-purpose fallback.
    """

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
        self.optimizer = optimizer
        self.smoothing_window = smoothing_window
        self.historical_window = historical_window
        self.reduction_threshold = reduction_threshold
        self.cooldown = cooldown
        self.factor = factor
        self.min_lr = min_lr
        self.losses: List[float] = []
        self.verbose = verbose
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if k != 'optimizer'}

    def load_state_dict(self, state_dict: Dict) -> None:
        self.__dict__.update(state_dict)

    def step(self, metrics: float) -> None:
        self.losses.append(metrics)
        while len(self.losses) > self.historical_window:
            self.losses.pop(0)
        if len(self.losses) < self.historical_window:
            return

        recent_avg = np.mean(self.losses[-self.smoothing_window:])
        historical_avg = np.mean(self.losses[:-self.smoothing_window])

        if recent_avg > historical_avg * self.reduction_threshold:
            if self.verbose:
                print(
                    f'Reducing learning rate: recent_avg={recent_avg:.4f} > '
                    f'historical_avg={historical_avg:.4f} * {self.reduction_threshold}'
                )
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = max(group['lr'] * self.factor, self.min_lr)
                if self.verbose:
                    new_lr = group['lr']
                    print(f'Reducing learning rate of group {i} to {new_lr:.4e}.')
            if self.cooldown > 0:
                self.losses = self.losses[:-self.cooldown]

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class CosineWarmupScheduler:
    """Linear warmup → cosine decay to `min_lr_ratio * base_lr`.

    Step once per training batch. Each param group gets the same multiplier
    applied to its own `base_lr` (snapshotted at construction).
    """

    def __init__(
        self,
        optimizer,
        *,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.01,
    ):
        assert 0 <= warmup_steps < total_steps
        assert 0.0 <= min_lr_ratio <= 1.0
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self._step_count = 0
        self._base_lrs: List[float] = [float(g['lr']) for g in self.optimizer.param_groups]

    def _multiplier(self, step: int) -> float:
        if step < self.warmup_steps:
            return (step + 1) / max(1, self.warmup_steps)
        if step >= self.total_steps:
            return self.min_lr_ratio
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def step(self, metric=None) -> None:
        del metric  # accepted for protocol compatibility with plateau schedulers
        m = self._multiplier(self._step_count)
        for base, group in zip(self._base_lrs, self.optimizer.param_groups):
            group['lr'] = base * m
        self._step_count += 1

    def state_dict(self) -> Dict:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "_step_count": self._step_count,
            "_base_lrs": list(self._base_lrs),
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.warmup_steps = int(state_dict["warmup_steps"])
        self.total_steps = int(state_dict["total_steps"])
        self.min_lr_ratio = float(state_dict["min_lr_ratio"])
        self._step_count = int(state_dict["_step_count"])
        self._base_lrs = [float(x) for x in state_dict["_base_lrs"]]


class EMAWeightTracker:
    """Exponential moving average of a parameter iterable.

    Holds a parallel shadow copy of each parameter. `update(params)` after
    each optimizer step refreshes the shadow. `swap_into(params)` is a
    context manager that temporarily replaces `params` with the shadow
    values for the duration of a `with` block (e.g. an eval call).

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
        assert 0.0 < decay < 1.0
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
        """Call after optimizer.step() to refresh the shadow from live params."""
        d = self._effective_decay()
        for shadow, param in zip(self._shadow, parameters):
            shadow.mul_(d).add_(param.detach().to(dtype=torch.float32), alpha=1.0 - d)
        self._num_updates += 1

    @contextmanager
    def swap_into(self, parameters: Iterable[nn.Parameter]) -> Iterator[None]:
        """Temporarily replace `parameters` with EMA copies for the duration
        of a `with` block, then restore them on exit."""
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
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "_num_updates": self._num_updates,
            "_shadow": [s.detach().cpu() for s in self._shadow],
        }

    def load_state_dict(self, state_dict: Dict) -> None:
        self.decay = float(state_dict["decay"])
        self.warmup_steps = int(state_dict["warmup_steps"])
        self._num_updates = int(state_dict["_num_updates"])
        self._shadow = [t.clone() for t in state_dict["_shadow"]]
