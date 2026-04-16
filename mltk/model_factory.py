"""Factory abstractions for constructing model components.

Post-migration to precomputed features, the only remaining factory is
`SkipAttentionMLPFactory`, which builds per-level heads for
`HierarchicGeoClassifier`. Each head is a `FeaturePerspective` →
`SkipAttentionMLP` trunk followed by a configurable *classifier tail*:

    `linear`   — vanilla `nn.Linear(hidden, num_classes)`. Baseline.
    `cosine`   — `CosineClassifier` from `modules.classifier_heads`, for
                 metric-learning-style separation. Recommended default.

Plugging in the margin-bearing tails (`arcface`, `subcenter_arcface`) at
the per-level head is possible but requires threading the target labels
through `head(features, targets=...)` — the hierarchical classifier's
grouped forward pass doesn't currently do that. Use margin tails in the
flat classifier instead, where target flow is straightforward.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch.nn as nn

from mltk.classifier_heads import CosineClassifier
from mltk.feature_perspective import FeaturePerspective
from mltk.skipattnmlp import SkipAttentionMLP


class ModelFactory(ABC):
    """Abstract factory for creating model components."""

    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def create_model(self, **kwargs) -> nn.Module: ...

    @abstractmethod
    def get_optimizer_param_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]: ...


class SkipAttentionMLPFactory(ModelFactory):
    """Factory for creating per-level SkipAttentionMLP classifier heads.

    Used by `HierarchicGeoClassifier._create_head` to build a fresh head for
    each internal hierarchy node on demand. The output dimensionality is
    overridden per-level at `create_model` time because it depends on the
    number of children (or leaf size) for the specific hierarchy node.

    Args:
        input_dim:       feature dim fed into the head.
        output_dim:      default num classes; overridden per-call.
        num_hidden_dims: hidden width of the SkipAttentionMLP trunk.
        heads:           FeaturePerspective attention heads.
        depth:           SkipAttentionMLP depth.
        tail_type:       `"linear"` or `"cosine"`. Default `"cosine"` — stronger
                         inter-class separation via metric-learning.
        cosine_scale:    temperature for the cosine classifier tail.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 0,
        num_hidden_dims: int = 2048,
        heads: int = 8,
        depth: int = 5,
        *,
        tail_type: str = "cosine",
        cosine_scale: float = 30.0,
    ):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.num_hidden_dims = num_hidden_dims
        self.heads = heads
        self.depth = depth
        self.tail_type = tail_type
        self.cosine_scale = float(cosine_scale)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _build_tail(self, hidden_dim: int, num_classes: int) -> nn.Module:
        if self.tail_type == "linear":
            return nn.Linear(hidden_dim, num_classes)
        if self.tail_type == "cosine":
            return CosineClassifier(
                feat_dim=hidden_dim,
                num_classes=num_classes,
                scale=self.cosine_scale,
                learnable_scale=False,
            )
        raise ValueError(f"Unknown tail_type: {self.tail_type!r}")

    def create_model(self, output_dim: Optional[int] = None, **kwargs) -> nn.Module:
        final_output_dim = output_dim if output_dim is not None else self.output_dim

        assert self.input_dim > 0, f"input_dim must be positive, got {self.input_dim}"
        assert final_output_dim > 0, f"output_dim must be positive, got {final_output_dim}"
        assert self.num_hidden_dims > 0, f"num_hidden_dims must be positive, got {self.num_hidden_dims}"

        return nn.Sequential(
            FeaturePerspective(self.input_dim, self.input_dim, num_heads=self.heads),
            SkipAttentionMLP(
                in_features=self.input_dim,
                out_features=self.num_hidden_dims,
                depth=self.depth,
            ),
            self._build_tail(self.num_hidden_dims, final_output_dim),
        )

    def get_optimizer_param_groups(self, model: nn.Module, base_lr: float) -> List[Dict[str, Any]]:
        return [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': base_lr}]
