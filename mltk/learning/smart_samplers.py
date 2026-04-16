"""Feedback-driven samplers for class-imbalanced training.

Extends `modules/samplers.py` (static weighting) with stateful samplers
that adapt during training:

    - `AdaptiveFrequencySampler` : target a class distribution and
                                    online-correct for actually-seen drift.
    - `LossAwareSampler`         : oversample classes with high recent loss.

Task-agnostic; they operate on integer class labels.

References:
    HAR — Duggal et al., https://poloclub.github.io/papers/21-bigdata-har.pdf
    Dynamic Curriculum Learning — Wang et al., ICCV 2019
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


_TargetFn = Callable[[np.ndarray], np.ndarray]


def _resolve_target_fn(name_or_callable) -> _TargetFn:
    """Map a target-distribution name to a class-weight function."""
    if callable(name_or_callable):
        return name_or_callable
    name = name_or_callable.lower()
    if name == "uniform":
        return lambda counts: np.ones_like(counts, dtype=np.float64)
    if name == "sqrt":
        return lambda counts: 1.0 / np.sqrt(counts.astype(np.float64))
    if name == "linear":
        return lambda counts: 1.0 / counts.astype(np.float64)
    if name == "log":
        return lambda counts: 1.0 / np.log1p(counts.astype(np.float64))
    raise ValueError(f"Unknown target_fn name: {name!r}")


class AdaptiveSampler(Sampler[int], ABC):
    """Stateful Sampler base class with feedback hooks.

    Protocol: the training loop constructs with `class_labels`, passes to
    a DataLoader, then after every batch calls whichever of
    `update_losses` / `update_frequencies` the concrete subclass consumes.

    Subclasses implement `_class_weights()` returning a [num_classes] array
    of per-class weights (in the `self.unique_classes` order). The base
    class handles distributing those to per-item weights and drawing
    indices.
    """

    def __init__(
        self,
        class_labels: Sequence[int],
        num_samples: Optional[int] = None,
        rng_seed: int = 0,
    ):
        super().__init__(None)
        self.class_labels = np.asarray(class_labels, dtype=np.int64)
        self.num_items = len(self.class_labels)
        assert self.num_items > 0, "class_labels is empty"
        self.num_samples = int(num_samples) if num_samples is not None else self.num_items
        self._rng = np.random.default_rng(rng_seed)

        self._indices_by_class: Dict[int, np.ndarray] = {}
        for cid in np.unique(self.class_labels):
            self._indices_by_class[int(cid)] = np.where(self.class_labels == cid)[0]

        self.unique_classes: List[int] = sorted(self._indices_by_class.keys())
        self._class_counts = np.array(
            [len(self._indices_by_class[c]) for c in self.unique_classes], dtype=np.float64
        )
        self.num_classes = len(self.unique_classes)

    @abstractmethod
    def _class_weights(self) -> np.ndarray:
        """Return [num_classes] weights for this epoch, ordered by `unique_classes`."""
        ...

    def update_losses(self, class_ids: Sequence[int], losses: Sequence[float]) -> None:
        """Feedback hook for loss-driven samplers. Default: no-op."""
        del class_ids, losses

    def update_frequencies(self, class_ids: Sequence[int]) -> None:
        """Feedback hook for frequency-tracking samplers. Default: no-op."""
        del class_ids

    def __iter__(self) -> Iterator[int]:
        class_weights = self._class_weights()
        assert class_weights.shape == (self.num_classes,)
        # Distribute each class's weight uniformly across its items.
        item_weights = np.empty(self.num_items, dtype=np.float64)
        for c, w in zip(self.unique_classes, class_weights):
            idxs = self._indices_by_class[c]
            item_weights[idxs] = (w / len(idxs)) if len(idxs) else 0.0
        total = item_weights.sum()
        if total <= 0:
            probs = np.full(self.num_items, 1.0 / self.num_items, dtype=np.float64)
        else:
            probs = item_weights / total
        draws = self._rng.choice(self.num_items, size=self.num_samples, replace=True, p=probs)
        return iter(int(i) for i in draws)

    def __len__(self) -> int:
        return self.num_samples


class AdaptiveFrequencySampler(AdaptiveSampler):
    """Target a distribution; correct for observed drift from that target.

    Closes the loop on a static inverse-frequency sampler: tracks how many
    times each class has actually been drawn and rebuilds weights each
    epoch to compensate for under/over-sampling. `correction_rate=0`
    degenerates to a static sampler.
    """

    def __init__(
        self,
        class_labels: Sequence[int],
        target_fn: "str | _TargetFn" = "sqrt",
        *,
        correction_rate: float = 0.5,
        num_samples: Optional[int] = None,
        rng_seed: int = 0,
    ):
        super().__init__(class_labels, num_samples=num_samples, rng_seed=rng_seed)
        assert 0.0 <= correction_rate <= 1.0
        self.correction_rate = float(correction_rate)

        target = _resolve_target_fn(target_fn)(self._class_counts)
        self._target = target / target.sum()  # [num_classes]
        self._seen_counts = np.zeros(self.num_classes, dtype=np.int64)

    def update_frequencies(self, class_ids: Sequence[int]) -> None:
        class_to_idx = {c: i for i, c in enumerate(self.unique_classes)}
        for cid in class_ids:
            i = class_to_idx.get(int(cid))
            if i is not None:
                self._seen_counts[i] += 1

    def _class_weights(self) -> np.ndarray:
        if self.correction_rate <= 0:
            return self._target
        total_seen = max(1, int(self._seen_counts.sum()))
        seen_frac = self._seen_counts / total_seen
        # ratio > 1 where we're under-sampled; < 1 where we're over-sampled.
        ratio = self._target / np.maximum(seen_frac, 1e-8)
        return self._target * (ratio ** self.correction_rate)


class LossAwareSampler(AdaptiveSampler):
    """Oversample classes with high recent loss (hardness-aware reweighting).

    Tracks per-class EMA of training loss and reweights sampling by
    `base_target * (class_ema / global_ema) ** alpha`, clamped to
    `[min_ratio, max_ratio]` so easy classes don't starve and hard classes
    don't get loop-stuck on outliers.
    """

    def __init__(
        self,
        class_labels: Sequence[int],
        *,
        base_target: "str | _TargetFn" = "sqrt",
        alpha: float = 0.5,
        ema_decay: float = 0.98,
        min_ratio: float = 0.25,
        max_ratio: float = 4.0,
        num_samples: Optional[int] = None,
        rng_seed: int = 0,
    ):
        super().__init__(class_labels, num_samples=num_samples, rng_seed=rng_seed)
        assert 0.0 <= ema_decay < 1.0
        self.alpha = float(alpha)
        self.ema_decay = float(ema_decay)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)

        base = _resolve_target_fn(base_target)(self._class_counts)
        self._base = base / base.sum()
        # NaN = "haven't seen this class yet"; treated as ratio=1 at weight time.
        self._ema_loss = np.full(self.num_classes, np.nan, dtype=np.float64)
        self._global_ema_loss: Optional[float] = None
        self._class_to_idx = {c: i for i, c in enumerate(self.unique_classes)}

    def update_losses(self, class_ids: Sequence[int], losses: Sequence[float]) -> None:
        assert len(class_ids) == len(losses)
        d = self.ema_decay
        for cid, loss in zip(class_ids, losses):
            if not math.isfinite(loss):
                continue
            i = self._class_to_idx.get(int(cid))
            if i is None:
                continue
            prev = self._ema_loss[i]
            self._ema_loss[i] = float(loss) if math.isnan(prev) else d * prev + (1.0 - d) * float(loss)
            self._global_ema_loss = (
                float(loss) if self._global_ema_loss is None
                else d * self._global_ema_loss + (1.0 - d) * float(loss)
            )

    def _class_weights(self) -> np.ndarray:
        global_mean = max(self._global_ema_loss or 1.0, 1e-8)
        # Unseen classes get ratio=1; seen classes get clamped loss/mean.
        ratios = np.where(
            np.isnan(self._ema_loss),
            1.0,
            np.clip(self._ema_loss / global_mean, self.min_ratio, self.max_ratio),
        )
        return self._base * (ratios ** self.alpha)
