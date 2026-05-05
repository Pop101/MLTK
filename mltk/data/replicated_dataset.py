"""``ReplicatedDataset`` — repeat a source ``n`` times.

Designed for the "cache N augmented features per source image" pattern:

    img_ds   = ImageRatingDataset(df, train=True, augment=STANDARD_AUGMENT)
    rep      = ReplicatedDataset(img_ds, n=8)            # 8 × K items
    feats    = MapDataset(rep, fn=lambda img, lbl: (backbone(img), lbl))
    cached   = DiskCachedDataset(feats, cache_dir="cache/...")
    cached.precompute()

When the source applies stochastic augmentation, each replica index draws
its own augment independently. ``DiskCachedDataset`` writes-once-per-index,
so the FIRST visit to each replica is what gets frozen — leaving ``n``
distinct augmented versions cached per source item.

``train()`` / ``eval()`` propagate to the source if it is also an
``AbstractDataset``.
"""
from __future__ import annotations

from torch.utils.data import Dataset

from mltk.data.abstract_dataset import AbstractDataset


class ReplicatedDataset(AbstractDataset):
    """Repeat a source dataset ``n`` times.

    Args:
        dataset: source Dataset.
        n: replica count. ``len(self) == n * len(dataset)``. Must be ``>= 1``.
        train: forwarded to ``AbstractDataset``.
    """

    def __init__(self, dataset: Dataset, n: int, train: bool = True):
        super().__init__(train=train)
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.dataset = dataset
        self.n = int(n)
        self._source_len = len(dataset)

    def __len__(self) -> int:
        return self._source_len * self.n

    def __getitem__(self, idx: int):
        if not (0 <= idx < len(self)):
            raise IndexError(f"index {idx} out of range for len={len(self)}")
        return self.dataset[idx % self._source_len]

    def train(self, mode: bool = True) -> "ReplicatedDataset":
        super().train(mode)
        if isinstance(self.dataset, AbstractDataset):
            self.dataset.train(mode)
        return self
