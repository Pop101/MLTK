"""Every name we export should actually be importable."""
import importlib
import mltk


def test_top_level_exports_are_real():
    for name in mltk.__all__:
        assert hasattr(mltk, name), f"mltk.__all__ promises {name} but it is not exported"


def test_submodules_import():
    modnames = [
        "base_classifier",
        "supermodel",
        "hierarchic_dataset",
        "hierarchic_inference",
        "model_factory",
        "skipattnmlp",
        "feature_perspective",
        "mlp_blocks",
        "checkpointedsequential",
        "schedulers",
        "samplers",
        "smart_samplers",
        "kldivlosssoftmax",
        "classifier_heads",
        "backbones",
    ]
    for m in modnames:
        importlib.import_module(f"mltk.{m}")
