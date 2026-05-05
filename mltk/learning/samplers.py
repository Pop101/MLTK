"""Stateless ``WeightedRandomSampler`` builders for class-imbalanced data.

For stateful (feedback-driven) variants see ``smart_samplers.py``.

Target distributions are passed as **callables** (``Callable[[np.ndarray],
np.ndarray]`` from class counts to per-class weights). Predefined choices
live in ``smart_samplers`` (``UNIFORM_TARGET``, ``SQRT_TARGET``,
``LINEAR_TARGET``, ``LOG_TARGET``).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from mltk.learning.smart_samplers import (
    LINEAR_TARGET,
    LOG_TARGET,
    SQRT_TARGET,
    TargetFn,
)


def make_weighted_sampler(
    labels: List[int],
    target_fn: TargetFn,
    num_samples: Optional[int] = None,
) -> WeightedRandomSampler:
    """Build a ``WeightedRandomSampler`` from per-class weights.

    ``target_fn`` receives the [num_classes] array of class counts (sorted
    by class id) and returns one weight per class. Per-item weights are the
    class weight divided uniformly across that class's items.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    unique_labels, counts = np.unique(labels_arr, return_counts=True)
    class_weights = target_fn(counts)
    label_to_weight = dict(zip(unique_labels, class_weights))
    sample_weights = np.array([label_to_weight[label] for label in labels_arr], dtype=np.float64)
    sample_weights = sample_weights / sample_weights.sum()
    n = num_samples if num_samples is not None else len(labels_arr)
    return WeightedRandomSampler(torch.DoubleTensor(sample_weights), n, replacement=True)


def create_sqrt_sampler(labels: List[int], num_samples: Optional[int] = None) -> WeightedRandomSampler:
    """Sample with probability ∝ 1/sqrt(class_freq). Moderate rebalancing."""
    return make_weighted_sampler(labels, SQRT_TARGET, num_samples)


def create_linear_sampler(labels: List[int], num_samples: Optional[int] = None) -> WeightedRandomSampler:
    """Sample with probability ∝ 1/class_freq. Full rebalancing."""
    return make_weighted_sampler(labels, LINEAR_TARGET, num_samples)


def create_log_sampler(labels: List[int], num_samples: Optional[int] = None) -> WeightedRandomSampler:
    """Sample with probability ∝ 1/log(class_freq + 1). Gentle rebalancing."""
    return make_weighted_sampler(labels, LOG_TARGET, num_samples)
