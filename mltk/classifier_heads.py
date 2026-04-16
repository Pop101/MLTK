"""Classifier-head drop-ins for large-N classification.

Each head replaces `nn.Linear(feat_dim, num_classes)` and returns raw logits
suitable for a softmax/KL loss. Margin-bearing variants take an optional
`targets` kwarg that is only consulted during training.

References:
    ArcFace            — Deng et al., CVPR 2019 (arxiv:1801.07698)
    CosFace            — Wang et al., CVPR 2018 (arxiv:1801.09414)
    Sub-center ArcFace — Deng et al., ECCV 2020
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineClassifier(nn.Module):
    """`scale * cos(theta(features, class_centers))`.

    L2-normalizes weights and inputs, then scales. Stronger inter-class
    separation than vanilla softmax at equivalent parameter count.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        *,
        scale: float = 30.0,
        learnable_scale: bool = False,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.xavier_normal_(self.weight)
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))

    def _cosine(self, features: torch.Tensor) -> torch.Tensor:
        """Return [B, num_classes] raw cosine similarity."""
        f = F.normalize(features.float(), dim=-1, eps=1e-8)
        w = F.normalize(self.weight.float(), dim=-1, eps=1e-8)
        return (f @ w.t()).to(dtype=features.dtype)

    def forward(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        del targets  # accepted for protocol compatibility with margin variants
        return self._cosine(features) * self.scale


class ArcFaceHead(CosineClassifier):
    """CosineClassifier + additive angular margin (Deng et al., CVPR 2019).

    When `targets` is supplied during training, adds margin `m` to the target
    class's angle: `logit = scale * cos(theta + m)` for the true class,
    `scale * cos(theta)` elsewhere. At eval (or when `targets is None`) falls
    back to plain cosine.

    Targets may be a 1-D LongTensor of indices or a [B, num_classes] soft
    distribution (margin interpolated by target mass — matches the
    Haversine-weighted KL training target).
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        *,
        scale: float = 30.0,
        margin: float = 0.35,
        learnable_scale: bool = False,
        easy_margin: bool = True,
    ):
        super().__init__(feat_dim, num_classes, scale=scale, learnable_scale=learnable_scale)
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)
        self._cos_m = math.cos(self.margin)
        self._sin_m = math.sin(self.margin)
        self._th = math.cos(math.pi - self.margin)
        self._mm = math.sin(math.pi - self.margin) * self.margin

    def _apply_margin(self, cosine: torch.Tensor) -> torch.Tensor:
        """Return cos(theta + m) with easy/hard margin stabilization."""
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))
        phi = cosine * self._cos_m - sine * self._sin_m
        if self.easy_margin:
            return torch.where(cosine > 0, phi, cosine)
        return torch.where(cosine > self._th, phi, cosine - self._mm)

    def forward(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        cosine = self._cosine(features)
        if targets is None or not self.training:
            return cosine * self.scale

        phi = self._apply_margin(cosine)
        if targets.dim() == 1:
            one_hot = torch.zeros_like(cosine).scatter_(1, targets.long().view(-1, 1), 1.0)
        else:
            one_hot = targets.to(dtype=cosine.dtype)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale


class SubCenterArcFaceHead(ArcFaceHead):
    """K sub-centers per class, max-pooled (Deng et al., ECCV 2020).

    Clean samples collapse onto a dominant sub-center; noisy/atypical samples
    attach to secondary sub-centers. Essential for large-N classification
    with noisy labels

    `num_sub_centers=1` reduces to plain ArcFace.
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        *,
        num_sub_centers: int = 3,
        scale: float = 30.0,
        margin: float = 0.35,
        learnable_scale: bool = False,
        easy_margin: bool = True,
    ):
        # Skip CosineClassifier.__init__ (it would size the weight wrong);
        # call the grandparent and build our own [C, K, D] weight.
        nn.Module.__init__(self)
        assert num_sub_centers >= 1
        self.num_classes = num_classes
        self.num_sub_centers = int(num_sub_centers)
        self.weight = nn.Parameter(torch.empty(num_classes, num_sub_centers, feat_dim))
        nn.init.xavier_normal_(self.weight)
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(scale)))
        # Margin setup (mirror ArcFaceHead).
        self.margin = float(margin)
        self.easy_margin = bool(easy_margin)
        self._cos_m = math.cos(self.margin)
        self._sin_m = math.sin(self.margin)
        self._th = math.cos(math.pi - self.margin)
        self._mm = math.sin(math.pi - self.margin) * self.margin

    def _cosine(self, features: torch.Tensor) -> torch.Tensor:
        """Override: cosine to all K*C sub-centers, then max over K per class."""
        f = F.normalize(features.float(), dim=-1, eps=1e-8)  # [B, D]
        w = F.normalize(
            self.weight.float().view(self.num_classes * self.num_sub_centers, -1), dim=-1, eps=1e-8
        )
        all_cos = (f @ w.t()).view(-1, self.num_classes, self.num_sub_centers)
        return all_cos.max(dim=-1).values.to(dtype=features.dtype)
