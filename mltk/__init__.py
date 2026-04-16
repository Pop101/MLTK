"""MLTK — a small PyTorch toolkit for hierarchical / frozen-feature classifiers.

Everything that is not specific to any one domain (geolocation, NLP, …) lives
here. The downstream project imports these building blocks and layers its own
domain-specific training loss and dataset on top.

Namespaces are kept flat on purpose — one module per concept, so a grep for
``from mltk.foo import Bar`` finds everything there is.
"""
from mltk.base_classifier import AbstractClassifier
from mltk.supermodel import SuperModel
from mltk.hierarchic_dataset import (
    HierarchyInformation,
    HierarchicDataset,
    PerLevelSampler,
    LevelPath,
    LeafPath,
)
from mltk.hierarchic_inference import HierarchicInference, BeamCandidate
from mltk.model_factory import ModelFactory, SkipAttentionMLPFactory
from mltk.skipattnmlp import SkipAttentionMLP
from mltk.feature_perspective import FeaturePerspective
from mltk.mlp_blocks import SwiGLU, DropPath, LayerScale, ModernMLPBlock
from mltk.checkpointedsequential import CheckpointedSequential
from mltk.schedulers import SmoothReduceLROnPlateau
from mltk.samplers import create_sqrt_sampler
from mltk.kldivlosssoftmax import KLDivLossWithSoftmax
from mltk.classifier_heads import CosineClassifier
from mltk.backbones import BackboneSpec, load_backbone

__all__ = [
    "AbstractClassifier",
    "SuperModel",
    "HierarchyInformation",
    "HierarchicDataset",
    "HierarchicInference",
    "BeamCandidate",
    "PerLevelSampler",
    "LevelPath",
    "LeafPath",
    "ModelFactory",
    "SkipAttentionMLPFactory",
    "SkipAttentionMLP",
    "FeaturePerspective",
    "SwiGLU",
    "DropPath",
    "LayerScale",
    "ModernMLPBlock",
    "CheckpointedSequential",
    "SmoothReduceLROnPlateau",
    "create_sqrt_sampler",
    "KLDivLossWithSoftmax",
    "CosineClassifier",
    "BackboneSpec",
    "load_backbone",
]

__version__ = "0.1.0"
