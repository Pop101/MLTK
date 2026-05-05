"""Every name we export should actually be importable."""
import importlib
import mltk


def test_top_level_exports_are_real():
    for name in mltk.__all__:
        assert hasattr(mltk, name), f"mltk.__all__ promises {name} but it is not exported"


def test_subpackages_import():
    for pkg in ["learning", "building_blocks", "models", "data", "hierarchical"]:
        importlib.import_module(f"mltk.{pkg}")


def test_submodules_import():
    modnames = [
        "learning.abstract_model",
        "learning.hyperparameter_optimizer",
        "learning.kldivlosssoftmax",
        "learning.schedulers",
        "learning.samplers",
        "learning.smart_samplers",
        "building_blocks.checkpointedsequential",
        "building_blocks.classifier_heads",
        "building_blocks.feature_perspective",
        "building_blocks.mlp_blocks",
        "building_blocks.skipattnmlp",
        "models.supermodel",
        "models.model_factory",
        "data.abstract_dataset",
        "data.cache",
        "data.disk_cached_dataset",
        "data.map_dataset",
        "data.replicated_dataset",
        "hierarchical.hierarchic_dataset",
        "hierarchical.hierarchic_inference",
        "devices",
    ]
    for m in modnames:
        importlib.import_module(f"mltk.{m}")
