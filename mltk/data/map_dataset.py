"""Lazy ``map(fn)`` over a Dataset.

The functional-programming sibling of ``DeviceLRUCache``: where the cache
wraps a Dataset to memoize its outputs, ``MapDataset`` wraps a Dataset to
*transform* its outputs lazily on every access. The most common composition
is "compute frozen-backbone features, then cache to disk":

    feats   = MapDataset(images, fn=lambda img, lbl: (backbone(img), lbl))
    cached  = DiskCachedDataset(feats, cache_dir="cache/dino_feats")

After the first epoch ``cached`` reads from mmap'd disk; ``MapDataset`` and
``backbone`` are bypassed entirely.

Naming note: we deliberately avoid "TransformDataset" because "transform"
already means torchvision ``Compose`` in this codebase; ``MapDataset``
matches the standard functional-programming term for "apply ``fn`` to
each element."
"""
from __future__ import annotations

from typing import Callable

from torch.utils.data import Dataset

from mltk.data.abstract_dataset import AbstractDataset


class MapDataset(AbstractDataset):
    """Lazy elementwise transform of a source Dataset.

    Args:
        dataset: source Dataset. Items can be anything ``fn`` knows how to
                 consume; the (item, label) tuple shape is the typical
                 case but not enforced.
        fn: callable applied per item. Two supported signatures
            (auto-detected by tuple-unpacking the source item):

                - ``fn(item)`` if the source returns a single value
                - ``fn(*item)`` if the source returns a tuple/list

            The latter lets you write ``lambda img, lbl: (backbone(img), lbl)``
            without re-packing.
        train: forwarded to ``AbstractDataset``.
        unpack: explicit override for the calling convention if auto-detect
                is wrong.

    ``train()`` / ``eval()`` propagate to the source if it is also an
    ``AbstractDataset``.
    """

    def __init__(
        self,
        dataset: Dataset,
        fn: Callable,
        train: bool = True,
        unpack: bool | None = None,
    ):
        super().__init__(train=train)
        if not callable(fn):
            raise TypeError(f"fn must be callable, got {type(fn).__name__}")
        self.dataset = dataset
        self.fn = fn
        self._unpack = unpack

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        unpack = self._unpack
        if unpack is None:
            unpack = isinstance(item, (tuple, list))
        return self.fn(*item) if unpack else self.fn(item)

    def train(self, mode: bool = True) -> "MapDataset":
        super().train(mode)
        if isinstance(self.dataset, AbstractDataset):
            self.dataset.train(mode)
        return self
