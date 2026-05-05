"""Abstract head factory for ``SuperModel``.

A factory is anything callable: ``factory(key) -> nn.Module``. The
``ModelFactory`` ABC is offered for callers that want a typed protocol with
introspectable input dimension; ``SuperModel`` itself just needs the call
syntax. Concrete factories (geo per-level heads, per-task heads, etc.)
belong in the consuming project, not in MLTK.
"""
from abc import ABC, abstractmethod
from typing import Hashable

import torch.nn as nn


class ModelFactory(ABC):
    """Optional typed-protocol base for head factories.

    Subclasses implement ``__call__(key) -> nn.Module`` and expose an
    ``input_dim`` property so callers (e.g. ``SuperModel``) can size things
    upstream without hardcoding numbers.
    """

    @property
    @abstractmethod
    def input_dim(self) -> int: ...

    @abstractmethod
    def __call__(self, key: Hashable) -> nn.Module: ...
