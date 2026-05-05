"""MLTK — small toolkit of building blocks on top of PyTorch Lightning.

Layout:
    mltk.learning         — Lightning lifecycle base, losses, schedulers,
                            samplers, hyperparameter search
    mltk.building_blocks  — reusable nn.Module primitives: MLP blocks,
                            attention-augmented MLP, classifier heads,
                            gradient-checkpointed Sequential
    mltk.models           — SuperModel (shared trunk + lazy heads),
                            ModelFactory protocol
    mltk.data             — dataset bases (with train/eval mode toggle),
                            standard image pipeline, LRU + disk caches
    mltk.hierarchical     — tree-structured dispatch: HierarchyInformation,
                            HierarchicDataset, beam-search inference
    mltk.devices          — cross-platform accelerator selection

Transform paradigm: datasets own randomization (augmentation), models own
normalization. See ``mltk.data`` for the rationale.

Train with ``lightning.Trainer``. ``AbstractModel`` is a ``LightningModule``;
checkpoint with ``Trainer.save_checkpoint`` and restore with
``cls.load_from_checkpoint``.
"""
from mltk.learning import (
    AbstractModel,
    AdaptiveFrequencySampler,
    ChoiceParam,
    CosineWarmupScheduler,
    EMAWeightTracker,
    FloatRange,
    HyperparameterOptimizer,
    IntRange,
    LINEAR_TARGET,
    LOG_TARGET,
    KLDivLossWithSoftmax,
    LossAwareSampler,
    OptimizationMode,
    OptimizationResult,
    SearchStrategy,
    SmoothReduceLROnPlateau,
    SQRT_TARGET,
    TrialResult,
    UNIFORM_TARGET,
    create_linear_sampler,
    create_log_sampler,
    create_sqrt_sampler,
    make_weighted_sampler,
)
from mltk.building_blocks import (
    ArcFaceHead,
    CheckpointedSequential,
    CosineClassifier,
    DropPath,
    FeaturePerspective,
    LayerScale,
    ModernMLPBlock,
    SkipAttentionMLP,
    SubCenterArcFaceHead,
    SwiGLU,
)
from mltk.models import (
    ModelFactory,
    SuperModel,
)
from mltk.hierarchical import (
    BeamCandidate,
    HierarchicDataset,
    HierarchicInference,
    HierarchyInformation,
    LeafPath,
    LevelPath,
    PerLevelSampler,
)
from mltk.data import (
    AbstractDataset,
    AbstractImageDataset,
    DeviceLRUCache,
    DiskCachedDataset,
    LIGHT_AUGMENT,
    MapDataset,
    ReplicatedDataset,
    STANDARD_AUGMENT,
    STRONG_AUGMENT,
    pad_to_square,
)
from mltk.devices import (
    auto_device,
    device_label,
    empty_cache,
    is_rocm,
    synchronize,
)

__all__ = [
    # learning
    "AbstractModel",
    "AdaptiveFrequencySampler",
    "ChoiceParam",
    "CosineWarmupScheduler",
    "EMAWeightTracker",
    "FloatRange",
    "HyperparameterOptimizer",
    "IntRange",
    "KLDivLossWithSoftmax",
    "LINEAR_TARGET",
    "LOG_TARGET",
    "LossAwareSampler",
    "OptimizationMode",
    "OptimizationResult",
    "SearchStrategy",
    "SmoothReduceLROnPlateau",
    "SQRT_TARGET",
    "TrialResult",
    "UNIFORM_TARGET",
    "create_linear_sampler",
    "create_log_sampler",
    "create_sqrt_sampler",
    "make_weighted_sampler",
    # building blocks
    "ArcFaceHead",
    "CheckpointedSequential",
    "CosineClassifier",
    "DropPath",
    "FeaturePerspective",
    "LayerScale",
    "ModernMLPBlock",
    "SkipAttentionMLP",
    "SubCenterArcFaceHead",
    "SwiGLU",
    # models
    "ModelFactory",
    "SuperModel",
    # data
    "AbstractDataset",
    "AbstractImageDataset",
    "DeviceLRUCache",
    "DiskCachedDataset",
    "LIGHT_AUGMENT",
    "MapDataset",
    "ReplicatedDataset",
    "STANDARD_AUGMENT",
    "STRONG_AUGMENT",
    "pad_to_square",
    # hierarchical
    "BeamCandidate",
    "HierarchicDataset",
    "HierarchicInference",
    "HierarchyInformation",
    "LeafPath",
    "LevelPath",
    "PerLevelSampler",
    # devices
    "auto_device",
    "device_label",
    "empty_cache",
    "is_rocm",
    "synchronize",
]

__version__ = "0.3.0"
