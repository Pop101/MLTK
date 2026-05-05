"""Data primitives: dataset bases that own augmentation, cache wrappers.

Design contract
---------------
The MLTK transform paradigm splits responsibility cleanly:

- **Datasets own randomization** (flip, rotate, affine, color jitter, erasing,
  etc.). It belongs next to the data because it's part of how an item is
  drawn, and because train vs eval datasets need different randomness.
- **Models own normalization** (per-backbone mean/std). Different backbones
  expect different normalization — DINOv2 wants ImageNet stats, SigLIP
  wants 0.5/0.5/0.5, custom backbones want per-dataset stats — so the
  knowledge lives on the model that needs it. ``AbstractModel``
  auto-applies ``self.normalize`` inside ``compute_loss`` / ``predict``.

This means a dataset's ``__getitem__`` returns a *non-normalized* tensor in
[0, 1]-ish range. The caller hands that to the model; the model normalizes
right before forward.
"""
from mltk.data.abstract_dataset import (
    AbstractDataset,
    AbstractImageDataset,
    LIGHT_AUGMENT,
    STANDARD_AUGMENT,
    STRONG_AUGMENT,
    pad_to_square,
)
from mltk.data.cache import DeviceLRUCache
from mltk.data.disk_cached_dataset import DiskCachedDataset
from mltk.data.map_dataset import MapDataset
from mltk.data.replicated_dataset import ReplicatedDataset

__all__ = [
    "AbstractDataset",
    "AbstractImageDataset",
    "DeviceLRUCache",
    "DiskCachedDataset",
    "LIGHT_AUGMENT",
    "MapDataset",
    "ReplicatedDataset",
    "STANDARD_AUGMENT",
    "STRONG_AUGMENT",
    "pad_to_square",
]
