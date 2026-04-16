"""MLTK — a small PyTorch toolkit for hierarchical / frozen-feature classifiers.

Layout:
    mltk.learning         — training lifecycle: classifier base, losses,
                            schedulers, samplers
    mltk.building_blocks  — reusable nn.Module primitives: MLP blocks,
                            attention-augmented MLP, classifier heads,
                            gradient-checkpointed Sequential
    mltk.models           — composed architectures: SuperModel, model
                            factories, frozen vision backbone loader
    mltk.hierarchical     — tree-structured dispatch: HierarchyInformation,
                            HierarchicDataset, beam-search inference

Everything important is also re-exported at the top level so short imports
work:

    from mltk import AbstractClassifier, SuperModel, SmoothReduceLROnPlateau
"""
from mltk.learning import (
    AbstractClassifier,
    KLDivLossWithSoftmax,
    SmoothReduceLROnPlateau,
    create_sqrt_sampler,
)
from mltk.building_blocks import (
    CheckpointedSequential,
    CosineClassifier,
    FeaturePerspective,
    SwiGLU,
    DropPath,
    LayerScale,
    ModernMLPBlock,
    SkipAttentionMLP,
)
from mltk.models import (
    SuperModel,
    ModelFactory,
    SkipAttentionMLPFactory,
    BackboneSpec,
    load_backbone,
)
from mltk.hierarchical import (
    HierarchyInformation,
    HierarchicDataset,
    PerLevelSampler,
    LevelPath,
    LeafPath,
    HierarchicInference,
    BeamCandidate,
)

__all__ = [
    # learning
    "AbstractClassifier",
    "KLDivLossWithSoftmax",
    "SmoothReduceLROnPlateau",
    "create_sqrt_sampler",
    # building blocks
    "CheckpointedSequential",
    "CosineClassifier",
    "FeaturePerspective",
    "SwiGLU",
    "DropPath",
    "LayerScale",
    "ModernMLPBlock",
    "SkipAttentionMLP",
    # models
    "SuperModel",
    "ModelFactory",
    "SkipAttentionMLPFactory",
    "BackboneSpec",
    "load_backbone",
    # hierarchical
    "HierarchyInformation",
    "HierarchicDataset",
    "PerLevelSampler",
    "LevelPath",
    "LeafPath",
    "HierarchicInference",
    "BeamCandidate",
]

__version__ = "0.2.0"
