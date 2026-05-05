"""Multi-activation feature transform with self- and cross-attention.

Pipeline (batched feature vector ``[B, in_dim]``):

    1. ``len(activations)`` parallel projections each ending in a different
       activation function (default: GELU, SiLU, Tanh).
    2. Self-attention across the activation perspectives.
    3. Mean-pool to ``[B, in_dim_aligned]``.
    4. Project to ``[B, out_dim_aligned]``.
    5. Cross-attention against a bank of ``num_queries`` learnable queries.
    6. Mean-pool over queries; final ``Linear(out_dim_aligned -> out_dim)``.

Both inner widths are rounded up to multiples of ``num_heads`` so the
multi-head attention modules don't have to carry padding logic.
"""
import torch
import torch.nn as nn


def _round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


class FeaturePerspective(nn.Module):
    """Multi-perspective feature mixer with learnable queries.

    Args:
        input_dim: width of the source feature vector.
        output_dim: width of the produced feature vector.
        num_heads: attention heads. Inner widths are rounded up to a
            multiple of this value. Must be >= 2.
        dropout: applied after each activation and inside the attention
            modules.
        activations: per-perspective activation modules. Default
            ``[GELU, SiLU, Tanh]``.
        num_queries: number of learnable cross-attention queries.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        num_heads: int = 16,
        dropout: float = 0.10,
        activations: list[nn.Module] | None = None,
        num_queries: int = 8,
    ):
        super().__init__()
        if num_heads < 2:
            raise ValueError("num_heads must be >= 2")
        if num_queries < 1:
            raise ValueError("num_queries must be >= 1")
        if activations is None:
            activations = [nn.GELU(), nn.SiLU(), nn.Tanh()]

        self.input_dim = input_dim
        self.output_dim = output_dim

        in_aligned = _round_up(input_dim, num_heads)
        out_aligned = _round_up(output_dim, num_heads)

        self.feature_extractors = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, in_aligned), activation, nn.Dropout(dropout))
            for activation in activations
        ])

        self.attention = nn.MultiheadAttention(
            embed_dim=in_aligned, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(in_aligned)
        self.attention_dropout = nn.Dropout(dropout)

        self.project_features = nn.Linear(in_aligned, out_aligned)

        self.queries = nn.Parameter(torch.randn(num_queries, out_aligned) * 0.02)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=out_aligned, num_heads=num_heads // 2, dropout=dropout, batch_first=True
        )

        self.refinement = nn.Sequential(nn.Linear(out_aligned, output_dim), nn.GELU())

        self._init_weights()

    def _init_weights(self) -> None:
        for extractor in self.feature_extractors:
            nn.init.kaiming_normal_(extractor[0].weight, nonlinearity='relu')
            nn.init.constant_(extractor[0].bias, 0)
        nn.init.kaiming_normal_(self.project_features.weight, nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``x: [B, input_dim]`` to ``[B, output_dim]``."""
        features = [extractor(x) for extractor in self.feature_extractors]
        features_stacked = torch.stack(features, dim=1)

        attn_output, _ = self.attention(features_stacked, features_stacked, features_stacked)
        attn_output = self.attention_norm(attn_output)
        attn_output = self.attention_dropout(attn_output)
        features = attn_output.mean(dim=1)

        features = self.project_features(features)
        features = features.unsqueeze(1)

        queries = self.queries.unsqueeze(0).expand(x.size(0), -1, -1)
        features, _ = self.cross_attention(queries, features, features)

        features = features.mean(dim=1)
        return self.refinement(features)
