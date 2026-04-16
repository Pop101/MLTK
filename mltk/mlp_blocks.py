"""Modern MLP building blocks for classifier heads.

Task-agnostic nn.Modules. Drop any of these into any project.

References:
    SwiGLU     — Shazeer 2020 (arxiv:2002.05202).
    DropPath   — Huang et al. 2016 (arxiv:1603.09382).
    LayerScale — Touvron et al. 2021 (arxiv:2103.17239).
    RMSNorm    — `torch.nn.RMSNorm` (PyTorch ≥ 2.4), we just re-export it here.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export so downstream code has a single import point for MLP components.
RMSNorm = nn.RMSNorm


class SwiGLU(nn.Module):
    """Swish-gated linear unit FFN: `down(silu(gate(x)) * up(x))`.

    Standard in modern LLMs (Llama/Mistral/PaLM/Apple). Default `hidden_dim`
    is `8/3 * dim` rounded up to a multiple of 8, which is flop-parity with
    a GELU FFN at `expansion=4`.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = ((int(dim * 8 / 3) + 7) // 8) * 8
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class DropPath(nn.Module):
    """Per-sample stochastic depth. Drops the block's output with probability
    `drop_prob`; at eval it's a no-op."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        assert 0.0 <= drop_prob < 1.0
        self.drop_prob = float(drop_prob)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class LayerScale(nn.Module):
    """Per-channel learnable residual scale (CaiT). Init near zero so the
    residual starts nearly silent and the skip path dominates."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class ModernMLPBlock(nn.Module):
    """Pre-norm residual block: RMSNorm → SwiGLU → LayerScale → DropPath → add."""

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: Optional[int] = None,
        drop_path: float = 0.0,
        layer_scale_init: Optional[float] = 1e-5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim=hidden_dim, dropout=dropout)
        self.layer_scale = LayerScale(dim, layer_scale_init) if layer_scale_init is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.layer_scale(self.ffn(self.norm(x))))
