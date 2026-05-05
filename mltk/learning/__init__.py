"""Training-time primitives: Lightning lifecycle, losses, schedulers, samplers,
hyperparameter search."""
from mltk.learning.abstract_model import AbstractModel
from mltk.learning.hyperparameter_optimizer import (
    ChoiceParam,
    FloatRange,
    HyperparameterOptimizer,
    IntRange,
    OptimizationMode,
    OptimizationResult,
    SearchStrategy,
    TrialResult,
)
from mltk.learning.kldivlosssoftmax import KLDivLossWithSoftmax
from mltk.learning.schedulers import (
    CosineWarmupScheduler,
    EMAWeightTracker,
    SmoothReduceLROnPlateau,
)
from mltk.learning.samplers import (
    create_linear_sampler,
    create_log_sampler,
    create_sqrt_sampler,
    make_weighted_sampler,
)
from mltk.learning.smart_samplers import (
    AdaptiveFrequencySampler,
    LossAwareSampler,
    LINEAR_TARGET,
    LOG_TARGET,
    SQRT_TARGET,
    UNIFORM_TARGET,
)

__all__ = [
    "AbstractModel",
    "ChoiceParam",
    "CosineWarmupScheduler",
    "EMAWeightTracker",
    "FloatRange",
    "HyperparameterOptimizer",
    "IntRange",
    "OptimizationMode",
    "KLDivLossWithSoftmax",
    "OptimizationResult",
    "SearchStrategy",
    "SmoothReduceLROnPlateau",
    "TrialResult",
    "AdaptiveFrequencySampler",
    "LossAwareSampler",
    "LINEAR_TARGET",
    "LOG_TARGET",
    "SQRT_TARGET",
    "UNIFORM_TARGET",
    "create_linear_sampler",
    "create_log_sampler",
    "create_sqrt_sampler",
    "make_weighted_sampler",
]
