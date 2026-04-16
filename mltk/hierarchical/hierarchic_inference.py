"""Hierarchical-tree specialization of `SuperModel`.

Adds tree-aware head construction (output dim derived from hierarchy
children/leaf count) and beam-search descent inference. Training signal
lives in subclasses (see `HierarchicGeoClassifier`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, List, Optional, Tuple

import torch
import torch.nn as nn

from mltk.hierarchical.hierarchic_dataset import (
    HierarchicDataset,
    HierarchyInformation,
    LevelPath,
)
from mltk.models.model_factory import ModelFactory
from mltk.models.supermodel import SuperModel


@dataclass
class BeamCandidate:
    """One candidate during beam-search descent."""
    level_path: LevelPath
    log_prob: float = 0.0
    leaf_logits: Optional[torch.Tensor] = None


def _coerce_path_to_tuple(path_like) -> Tuple:
    """Normalize a LevelPath / LeafPath / tensor / list into a plain tuple."""
    if torch.is_tensor(path_like):
        if path_like.dim() == 0:
            raise ValueError(f"Expected 1-D path tensor, got scalar: {path_like}")
        path_like = path_like.tolist()
    return tuple(None if v is None else int(v) for v in path_like)


class HierarchicInference(SuperModel):
    """Tree-structured classifier over precomputed features.

    Heads are instantiated lazily per visited `LevelPath`. The output
    dimensionality of each head comes from the hierarchy (leaf size or
    child count), so the caller only supplies a ModelFactory that knows
    how to build an MLP given a target `output_dim`.
    """

    def __init__(
        self,
        hierarchical_structure,
        *,
        head_input_dim: int,
        head_factory: ModelFactory,
        device=None,
        dtype=torch.float32,
    ):
        if hierarchical_structure is None:
            raise ValueError("hierarchical_structure must be provided")
        if isinstance(hierarchical_structure, HierarchicDataset):
            hierarchy_information = hierarchical_structure.to_hierarchy_information()
        elif isinstance(hierarchical_structure, HierarchyInformation):
            hierarchy_information = hierarchical_structure
        else:
            raise TypeError(
                f"hierarchical_structure must be HierarchicDataset or HierarchyInformation, "
                f"got {type(hierarchical_structure)}"
            )

        super().__init__(input_dim=head_input_dim, trunk=None, device=device, dtype=dtype)
        self.hierarchy_information: HierarchyInformation = hierarchy_information
        self._model_factory: ModelFactory = head_factory
        # SuperModel stashed `input_dim` in init_params, but our __init__
        # takes `head_input_dim`. Swap the key so `cls(**init_params)` works.
        self.init_params.pop("input_dim", None)
        self.init_params.update({
            "hierarchical_structure": self.hierarchy_information.clone(),
            "head_input_dim": head_input_dim,
        })

    def _canonical_key(self, key: Hashable) -> Tuple:
        return _coerce_path_to_tuple(key)

    def _build_head(self, key: Hashable) -> nn.Module:
        """Hierarchy-aware head construction. Output dim = leaf_size or num_children."""
        info = self.hierarchy_information
        if info.is_leaf(key):
            output_dim = int(info.get_leaf_size(key))
        else:
            output_dim = len(info.get_children(key))
        if output_dim <= 0:
            raise ValueError(f"Level {key!r} has output_dim={output_dim}; cannot build a head.")
        return self._model_factory.create_model(output_dim=output_dim)

    # ---- inference: beam-search descent ----

    def predict_leaf_from_feature(
        self,
        feature: torch.Tensor,
        beam_size: int = 1,
    ) -> List[BeamCandidate]:
        """Beam-search descent for a single [1, feat_dim] feature.

        Returns up to `beam_size` `BeamCandidate`s sorted by log-prob
        descending. Each terminal candidate has `leaf_logits` populated.
        `beam_size=1` is exact greedy descent.
        """
        assert feature.dim() == 2 and feature.size(0) == 1, (
            f"expects [1, feat_dim], got {tuple(feature.shape)}"
        )
        assert beam_size >= 1
        info = self.hierarchy_information

        beam: List[BeamCandidate] = [BeamCandidate(level_path=info.root_level())]
        for _ in range(info.key_length + 1):
            next_beam: List[BeamCandidate] = []
            any_internal = False
            for cand in beam:
                if info.is_leaf(cand.level_path) or not info.get_children(cand.level_path):
                    next_beam.append(cand)
                    continue
                any_internal = True
                children = info.get_children(cand.level_path)
                log_probs = torch.log_softmax(self.get_head(cand.level_path)(feature), dim=-1)[0]
                # Expand top-K children (K=beam_size capped at num_children).
                topk = torch.topk(log_probs, k=min(beam_size, len(children)))
                for lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                    next_beam.append(
                        BeamCandidate(
                            level_path=info.advance_by_child(cand.level_path, children[idx]),
                            log_prob=cand.log_prob + lp,
                        )
                    )
            next_beam.sort(key=lambda c: c.log_prob, reverse=True)
            beam = next_beam[:beam_size]
            if not any_internal:
                break

        # Populate leaf_logits for every terminal candidate.
        for cand in beam:
            if cand.leaf_logits is None:
                cand.leaf_logits = self.get_head(cand.level_path)(feature)
        return beam

    def predict(self, image: torch.Tensor, beam_size: int = 1) -> torch.Tensor:
        """Sparse [B, total_leaves] prediction via per-sample beam descent.

        Greedy mode drops each sample's leaf logits into its contiguous
        slice. Beam mode writes a softmax-weighted mixture across all
        terminal candidates.
        """
        for head in self._heads.values():
            head.eval()
        info = self.hierarchy_information
        if not info.leaf_offsets:
            raise RuntimeError("HierarchyInformation is missing leaf_offsets; cannot build flat prediction.")

        with torch.no_grad():
            features = self.extract_features(image)
            total_size = sum(info.leaf_counts.values())
            merged = torch.zeros(image.size(0), total_size, device=image.device, dtype=image.dtype)

            for b in range(image.size(0)):
                candidates = self.predict_leaf_from_feature(features[b : b + 1], beam_size=beam_size)
                if not candidates:
                    continue
                path_probs = torch.softmax(
                    torch.tensor([c.log_prob for c in candidates], dtype=torch.float32), dim=0
                )
                for cand, w in zip(candidates, path_probs):
                    if not info.is_leaf(cand.level_path):
                        continue
                    start, end = info.get_leaf_index_range(cand.level_path)
                    if beam_size == 1:
                        merged[b, start:end] = cand.leaf_logits[0].to(dtype=merged.dtype)
                    else:
                        leaf_probs = torch.softmax(cand.leaf_logits[0].float(), dim=-1)
                        merged[b, start:end] += (w * leaf_probs).to(dtype=merged.dtype)

            return merged
