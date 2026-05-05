"""Per-item LRU cache wrapper for any ``torch.utils.data.Dataset``.

Use case: image-loading datasets are I/O bound. The first epoch pays the
disk + decode cost; later epochs serve from a fixed-size cache. The cache
optionally lives on a target device (e.g. CPU when running on DirectML
where GPU memory is precious, or directly on CUDA when GPU RAM is cheap).

This is intentionally *not* device-aware in the dispatch sense: the caller
picks the cache device. ``mltk.devices.auto_device`` plus a single line
of glue (``"cpu" if device.type == "privateuseone" else device``) covers
the typical "DirectML iGPU OOMs at 1.5 GB cached tensors" failure mode.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional, Union

import torch
from torch.utils.data import Dataset


class DeviceLRUCache(Dataset):
    """Wraps another Dataset with a bounded LRU cache.

    On a hit, returns the cached ``(item, label)`` directly. On a miss,
    fetches from the underlying dataset, optionally moves to ``device``,
    inserts into the cache, and evicts the LRU entry if over capacity.

    Args:
        dataset: the wrapped dataset. Items must be ``(tensor, tensor)``
                 tuples (or anything where ``.to(device)`` is meaningful
                 for both halves).
        cache_size: maximum number of items to keep cached. ``0`` disables
                    caching entirely (always passes through).
        device: where cached tensors live. ``None`` keeps them on whatever
                device the wrapped dataset returns them on.
    """

    def __init__(
        self,
        dataset: Dataset,
        cache_size: int = 5000,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if cache_size < 0:
            raise ValueError(f"cache_size must be >= 0, got {cache_size}")
        self.dataset = dataset
        self.cache_size = int(cache_size)
        self.device = torch.device(device) if device is not None else None
        # idx -> (image, label)
        self._cache: "OrderedDict[int, Any]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        if self.cache_size > 0 and idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]

        image, label = self.dataset[idx]
        if self.device is not None:
            if torch.is_tensor(image):
                image = image.to(self.device)
            if torch.is_tensor(label):
                label = label.to(self.device)

        if self.cache_size > 0:
            self._cache[idx] = (image, label)
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

        return image, label

    def clear_cache(self) -> None:
        """Drop every cached item. Useful when the wrapped dataset's
        underlying contents change or when reclaiming memory."""
        self._cache.clear()

    @property
    def cache_load(self) -> int:
        """Current number of items in the cache."""
        return len(self._cache)
