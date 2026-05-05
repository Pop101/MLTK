"""MapDataset: lazy fn over a source Dataset, train/eval propagation."""
import pytest
import torch
from torch.utils.data import Dataset

from mltk import AbstractImageDataset, MapDataset


class _TupleSource(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float32), idx


class _SingleSource(Dataset):
    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float32)


def test_map_dataset_unpacks_tuple_items():
    src = _TupleSource()
    md = MapDataset(src, fn=lambda x, y: (x * 2, y + 100))
    out_x, out_y = md[1]
    assert torch.equal(out_x, torch.tensor([2.0]))
    assert out_y == 101


def test_map_dataset_passes_single_item_whole():
    src = _SingleSource()
    md = MapDataset(src, fn=lambda x: x * -1)
    out = md[2]
    assert torch.equal(out, torch.tensor([-2.0]))


def test_map_dataset_explicit_unpack_false_passes_tuple_whole():
    src = _TupleSource()
    md = MapDataset(src, fn=lambda item: (item[0] + 10, item[1]), unpack=False)
    a, b = md[0]
    assert torch.equal(a, torch.tensor([10.0]))
    assert b == 0


def test_map_dataset_train_propagates_to_source_when_supported():
    """If the inner dataset is an AbstractDataset, toggling train() on the
    MapDataset must propagate to the source so the augment flag flips."""
    from PIL import Image

    from mltk import STANDARD_AUGMENT

    class _Inner(AbstractImageDataset):
        def __init__(self):
            super().__init__(size=(8, 8), train=True, augment=STANDARD_AUGMENT)
        def __len__(self):
            return 2
        def load_raw(self, idx):
            return Image.new("RGB", (8, 8)), idx

    inner = _Inner()
    md = MapDataset(inner, fn=lambda x, y: (x, y))
    assert inner.is_training and md.is_training
    md.eval()
    assert not inner.is_training and not md.is_training
    md.train()
    assert inner.is_training and md.is_training


def test_map_dataset_rejects_non_callable_fn():
    with pytest.raises(TypeError):
        MapDataset(_SingleSource(), fn=42)  # type: ignore[arg-type]


def test_map_dataset_len_passes_through():
    md = MapDataset(_SingleSource(), fn=lambda x: x)
    assert len(md) == 3


def test_map_dataset_no_caching_reapplies_fn():
    """Same idx -> fn called fresh every time."""
    calls = {"n": 0}
    def fn(x):
        calls["n"] += 1
        return x
    md = MapDataset(_SingleSource(), fn=fn)
    _ = md[0]
    _ = md[0]
    _ = md[0]
    assert calls["n"] == 3
