"""DeviceLRUCache: hit/miss/eviction + optional device routing."""
import pytest
import torch
from torch.utils.data import Dataset

from mltk import DeviceLRUCache


class _CountingDataset(Dataset):
    """Counts how many times each idx was fetched from disk."""

    def __init__(self, n=8):
        self.n = n
        self.calls = [0] * n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self.calls[idx] += 1
        return torch.tensor([idx], dtype=torch.float32), torch.tensor(idx)


def test_cache_hit_avoids_underlying_fetch():
    ds = _CountingDataset()
    cached = DeviceLRUCache(ds, cache_size=8)
    _ = cached[0]
    _ = cached[0]
    _ = cached[0]
    assert ds.calls[0] == 1


def test_cache_miss_then_hit():
    ds = _CountingDataset()
    cached = DeviceLRUCache(ds, cache_size=4)
    for i in range(4):
        _ = cached[i]
    # all four primed; second pass must be 0 new fetches
    before = list(ds.calls)
    for i in range(4):
        _ = cached[i]
    assert ds.calls == before


def test_cache_eviction_is_lru():
    ds = _CountingDataset()
    cached = DeviceLRUCache(ds, cache_size=2)

    _ = cached[0]   # cache: [0]
    _ = cached[1]   # cache: [0, 1]
    _ = cached[0]   # touch 0 -> [1, 0]
    _ = cached[2]   # evict 1, cache: [0, 2]

    # At this point 0 and 2 should hit; 1 should miss.
    pre0, pre1, pre2 = ds.calls[0], ds.calls[1], ds.calls[2]
    _ = cached[0]
    _ = cached[2]
    assert ds.calls[0] == pre0  # hit, no fetch
    assert ds.calls[2] == pre2  # hit, no fetch

    _ = cached[1]
    assert ds.calls[1] == pre1 + 1  # was evicted, must miss


def test_cache_size_zero_passes_through():
    ds = _CountingDataset()
    cached = DeviceLRUCache(ds, cache_size=0)
    for _ in range(3):
        _ = cached[0]
    assert ds.calls[0] == 3
    assert cached.cache_load == 0


def test_cache_clear_drops_entries():
    ds = _CountingDataset()
    cached = DeviceLRUCache(ds, cache_size=4)
    for i in range(3):
        _ = cached[i]
    assert cached.cache_load == 3
    cached.clear_cache()
    assert cached.cache_load == 0


def test_cache_negative_size_raises():
    with pytest.raises(ValueError):
        DeviceLRUCache(_CountingDataset(), cache_size=-1)


def test_cache_device_routing():
    """Cache device routes both image and label tensors to that device.
    Tested with CPU since that's universally available on CI."""
    ds = _CountingDataset()
    cached = DeviceLRUCache(ds, cache_size=2, device="cpu")
    img, lbl = cached[0]
    assert img.device.type == "cpu"
    assert lbl.device.type == "cpu"


def test_cache_len_passes_through():
    cached = DeviceLRUCache(_CountingDataset(n=12), cache_size=4)
    assert len(cached) == 12
