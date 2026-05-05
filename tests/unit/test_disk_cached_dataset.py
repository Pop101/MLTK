"""DiskCachedDataset: schema lock, mmap round-trip, partial fill recovery,
chunk boundary correctness, MapDataset composition."""
import json

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from mltk import DiskCachedDataset, MapDataset


class _CountingDataset(Dataset):
    """Returns deterministic synthetic items; counts how many times the
    source was hit so we can assert cache hits."""

    def __init__(self, n=10, image_shape=(4,), label_shape=()):
        self.n = n
        self.image_shape = image_shape
        self.label_shape = label_shape
        self.calls = 0

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self.calls += 1
        # image[i] = i, i+1, i+2... label[i] = i (scalar)
        img = torch.arange(idx, idx + int(np.prod(self.image_shape)),
                           dtype=torch.float32).reshape(self.image_shape)
        if self.label_shape == ():
            lbl = torch.tensor(float(idx))
        else:
            lbl = torch.full(self.label_shape, float(idx))
        return img, lbl


# --------------------------------------------------------------------------
# Basic round-trip + cache hit
# --------------------------------------------------------------------------
def test_first_pass_fills_cache_and_second_pass_serves_from_disk(tmp_path):
    src = _CountingDataset(n=8)
    cache = DiskCachedDataset(src, cache_dir=tmp_path, chunk_size=4)

    # Pass 1: every access is a miss + fill.
    pass1 = [cache[i] for i in range(8)]
    assert src.calls == 8

    # Pass 2: every access is a hit; source not re-touched.
    pass2 = [cache[i] for i in range(8)]
    assert src.calls == 8  # unchanged

    for (a_img, a_lbl), (b_img, b_lbl) in zip(pass1, pass2):
        torch.testing.assert_close(a_img, b_img)
        torch.testing.assert_close(a_lbl, b_lbl)


def test_manifest_describes_schema(tmp_path):
    src = _CountingDataset(n=4, image_shape=(3, 5))
    cache = DiskCachedDataset(src, cache_dir=tmp_path, chunk_size=2)
    _ = cache[0]
    mf = json.loads((tmp_path / "manifest.json").read_text())
    assert mf["n_items"] == 4
    assert mf["chunk_size"] == 2
    assert mf["n_chunks"] == 2
    assert mf["image"]["shape"] == [3, 5]
    assert mf["image"]["dtype"] == "float32"
    assert mf["label"]["shape"] == []
    assert mf["label"]["dtype"] == "float32"


def test_chunk_boundary_correctness(tmp_path):
    """Item layout uses divmod(idx, chunk_size); chunks are short on the
    last one when n is not a multiple of chunk_size."""
    src = _CountingDataset(n=7)  # chunks: [0..2], [3..5], [6]
    cache = DiskCachedDataset(src, cache_dir=tmp_path, chunk_size=3)
    pass1 = [cache[i] for i in range(7)]
    pass2 = [cache[i] for i in range(7)]
    for a, b in zip(pass1, pass2):
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[1], b[1])


def test_random_access_o1(tmp_path):
    """Filling out of order must place items at the right slots."""
    src = _CountingDataset(n=6)
    cache = DiskCachedDataset(src, cache_dir=tmp_path, chunk_size=3)
    # Touch in scrambled order
    for i in [4, 0, 5, 2, 1, 3]:
        _ = cache[i]
    # Now read in order; values must reflect idx = i, not insertion order
    for i in range(6):
        img, lbl = cache[i]
        assert lbl.item() == float(i)
        assert img[0].item() == float(i)


# --------------------------------------------------------------------------
# Resilience: partial fill survives across instances
# --------------------------------------------------------------------------
def test_partial_fill_resumes_after_reopen(tmp_path):
    """Fill some items; close; reopen; only missing items refetch."""
    src = _CountingDataset(n=10)
    cache = DiskCachedDataset(src, cache_dir=tmp_path, chunk_size=4)
    for i in [0, 1, 2, 3, 4]:
        _ = cache[i]
    cache.flush()
    # Fresh instance, same dir.
    src2 = _CountingDataset(n=10)
    cache2 = DiskCachedDataset(src2, cache_dir=tmp_path, chunk_size=4)
    # Access same items: must be hits, no source calls.
    for i in [0, 1, 2, 3, 4]:
        _ = cache2[i]
    assert src2.calls == 0
    # Access an unfilled item: one source call.
    _ = cache2[7]
    assert src2.calls == 1


def test_fill_progress_and_is_fully_filled(tmp_path):
    src = _CountingDataset(n=5)
    cache = DiskCachedDataset(src, cache_dir=tmp_path, chunk_size=2)
    assert cache.fill_progress() == (0, 5)
    assert not cache.is_fully_filled()
    for i in range(5):
        _ = cache[i]
    assert cache.fill_progress() == (5, 5)
    assert cache.is_fully_filled()


def test_precompute_fills_missing(tmp_path):
    src = _CountingDataset(n=6)
    cache = DiskCachedDataset(src, cache_dir=tmp_path, chunk_size=3)
    cache.precompute()
    assert cache.is_fully_filled()


# --------------------------------------------------------------------------
# Schema lock
# --------------------------------------------------------------------------
def test_schema_mismatch_on_write_raises(tmp_path):
    """Once the manifest is written, items with different shape must raise."""
    src = _CountingDataset(n=4, image_shape=(4,))
    cache = DiskCachedDataset(src, cache_dir=tmp_path, chunk_size=2)
    _ = cache[0]  # locks schema as image.shape=[4]

    # Replace source with one that produces a different shape.
    bad = _CountingDataset(n=4, image_shape=(8,))
    cache.dataset = bad  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="shape"):
        _ = cache[1]


def test_n_items_mismatch_on_reopen_raises(tmp_path):
    """If the source's len() changes between runs, refuse to reuse the
    cache — addresses would be wrong."""
    src = _CountingDataset(n=5)
    c1 = DiskCachedDataset(src, cache_dir=tmp_path, chunk_size=2)
    _ = c1[0]
    c1.flush()
    src_smaller = _CountingDataset(n=4)
    with pytest.raises(RuntimeError, match="n_items mismatch"):
        DiskCachedDataset(src_smaller, cache_dir=tmp_path, chunk_size=2)


# --------------------------------------------------------------------------
# Composition with MapDataset
# --------------------------------------------------------------------------
def test_compose_with_map_dataset(tmp_path):
    """Typical usage: MapDataset applies an expensive transform; cache
    serves it on subsequent epochs without rerunning fn."""
    src = _CountingDataset(n=4, image_shape=(2,))
    fn_calls = {"n": 0}

    def expensive_fn(img, lbl):
        fn_calls["n"] += 1
        return img * 10, lbl

    mapped = MapDataset(src, fn=expensive_fn)
    cache = DiskCachedDataset(mapped, cache_dir=tmp_path, chunk_size=2)

    # First epoch: fn runs n times.
    [cache[i] for i in range(4)]
    assert fn_calls["n"] == 4

    # Second epoch: zero fn calls; cache serves from disk.
    [cache[i] for i in range(4)]
    assert fn_calls["n"] == 4

    # And values are the transformed values.
    img0, lbl0 = cache[0]
    torch.testing.assert_close(img0, torch.tensor([0.0, 10.0]))
    assert lbl0.item() == 0.0


# --------------------------------------------------------------------------
# Constructor validation
# --------------------------------------------------------------------------
def test_chunk_size_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        DiskCachedDataset(_CountingDataset(), cache_dir=tmp_path, chunk_size=0)


def test_oob_index_raises(tmp_path):
    cache = DiskCachedDataset(_CountingDataset(n=4), cache_dir=tmp_path, chunk_size=2)
    with pytest.raises(IndexError):
        _ = cache[10]
