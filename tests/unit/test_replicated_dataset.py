"""ReplicatedDataset: length math, modular access, train/eval propagation,
and the augmented-cache composition pattern."""
import pytest
import torch
from PIL import Image
from torch.utils.data import Dataset

from mltk import (
    AbstractImageDataset,
    DiskCachedDataset,
    MapDataset,
    ReplicatedDataset,
    STANDARD_AUGMENT,
)


class _Source(Dataset):
    def __len__(self):
        return 4

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float32), torch.tensor(idx)


def test_length_is_n_times_source():
    rep = ReplicatedDataset(_Source(), n=3)
    assert len(rep) == 12


def test_modular_access_maps_into_source():
    rep = ReplicatedDataset(_Source(), n=3)
    for i in range(12):
        x, lbl = rep[i]
        assert int(lbl) == i % 4
        assert torch.equal(x, torch.tensor([float(i % 4)]))


def test_n_must_be_positive():
    with pytest.raises(ValueError):
        ReplicatedDataset(_Source(), n=0)


def test_oob_raises():
    rep = ReplicatedDataset(_Source(), n=2)
    with pytest.raises(IndexError):
        _ = rep[8]


def test_train_propagates_to_source():
    class _Aug(AbstractImageDataset):
        def __init__(self):
            super().__init__(size=(8, 8), train=True, augment=STANDARD_AUGMENT)
        def __len__(self):
            return 2
        def load_raw(self, idx):
            return Image.new("RGB", (8, 8)), idx

    inner = _Aug()
    rep = ReplicatedDataset(inner, n=4)
    assert inner.is_training and rep.is_training
    rep.eval()
    assert not inner.is_training and not rep.is_training
    rep.train()
    assert inner.is_training and rep.is_training


def test_replicated_with_disk_cache_freezes_first_visit_per_index(tmp_path):
    """The exact pattern users will run: replicate × source augment, cache
    once. Each replica idx gets its own augmented snapshot; subsequent
    reads return the same snapshot."""
    class _NoisyAug(AbstractImageDataset):
        def __init__(self):
            super().__init__(size=(8, 8), train=True, augment=None)
            # Inject deterministic noise via a Lambda so we can detect it.
            import torchvision.transforms as T
            self._augment_with_noise = T.Compose([
                T.Lambda(lambda t: t + torch.randn_like(t) * 0.5),
            ])

        def __len__(self):
            return 2

        def load_raw(self, idx):
            return Image.new("RGB", (8, 8), (100, 100, 100)), torch.tensor(idx)

        def transform(self, image):
            t = super().transform(image)
            if self.is_training:
                t = self._augment_with_noise(t)
            return t

    src = _NoisyAug()
    rep = ReplicatedDataset(src, n=4)               # 4 replicas of 2 items
    cached = DiskCachedDataset(rep, cache_dir=tmp_path, chunk_size=4)

    # Pass 1: caches the first random draw per replica idx.
    pass1 = [cached[i][0] for i in range(8)]

    # Pass 2: must reproduce pass 1 exactly (cached) — no fresh randomness.
    pass2 = [cached[i][0] for i in range(8)]
    for a, b in zip(pass1, pass2):
        torch.testing.assert_close(a, b)

    # Different replica indices for the SAME source item should produce
    # different cached values — that's the whole point of replicating.
    # Replica 0 of source 0 = cached[0]; replica 1 of source 0 = cached[2].
    assert not torch.equal(cached[0][0], cached[2][0])


def test_compose_with_map_dataset_round_trip(tmp_path):
    """ReplicatedDataset → MapDataset → DiskCachedDataset, the production
    composition. Cache must serve every replicated index post-fill."""
    src = _Source()
    rep = ReplicatedDataset(src, n=3)
    fn_calls = {"n": 0}

    def fn(x, lbl):
        fn_calls["n"] += 1
        return x * 100, lbl

    mapped = MapDataset(rep, fn=fn)
    cached = DiskCachedDataset(mapped, cache_dir=tmp_path, chunk_size=4)
    cached.precompute()
    assert fn_calls["n"] == 12  # 3 × 4

    # Subsequent passes do not invoke fn.
    [cached[i] for i in range(12)]
    assert fn_calls["n"] == 12
