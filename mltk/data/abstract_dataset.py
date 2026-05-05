"""Abstract dataset bases that codify the MLTK transform paradigm.

Why this exists
---------------
The split is intentional and load-bearing:

- ``AbstractDataset`` carries a train/eval mode flag. When ``True``, subclasses
  apply randomization; when ``False``, they don't. Toggle via
  ``dataset.train()`` / ``dataset.eval()``, mirroring ``nn.Module``.
- ``AbstractImageDataset`` provides the standard image pipeline (PIL ->
  tensor -> pad-to-square -> resize -> augment). Subclasses only implement
  ``load_raw(idx) -> (PIL.Image | Tensor, label)``.
- Normalization is **deliberately not** in this layer. Models own it (see
  ``AbstractModel.normalize``) so per-backbone mean/std stays attached
  to the model that needs it.

Augmentation presets are module-level ``transforms.Compose`` constants
(``LIGHT_AUGMENT``, ``STANDARD_AUGMENT``, ``STRONG_AUGMENT``); pass any
``transforms.Compose``-style callable to override.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def pad_to_square(image: torch.Tensor) -> torch.Tensor:
    """Pad a CHW tensor with zeros so H == W. Centers the original image."""
    if image.dim() != 3:
        raise ValueError(
            f"pad_to_square expects a 3D CHW tensor, got shape {tuple(image.shape)}"
        )
    _, h, w = image.shape
    side = max(h, w)
    pad_h, rem_h = divmod(side - h, 2)
    pad_w, rem_w = divmod(side - w, 2)
    return torch.nn.functional.pad(
        image,
        (pad_w, pad_w + rem_w, pad_h, pad_h + rem_h),
        mode='constant',
        value=0,
    )


# ---------------------------------------------------------------------------
# Augmentation presets — float-tensor friendly. Applied AFTER pad+resize and
# BEFORE normalization. Each preset is a stand-alone Compose so callers can
# inspect or extend it.
# ---------------------------------------------------------------------------
LIGHT_AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
])

STANDARD_AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
])

# SOTA-ish: stronger jitter + perspective + erasing. Erasing is last because
# it stamps zeros onto the tensor and we want every geometric op to act on
# the un-erased image.
STRONG_AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])


# ---------------------------------------------------------------------------
# Abstract bases
# ---------------------------------------------------------------------------
class AbstractDataset(Dataset, ABC):
    """Base for MLTK datasets that distinguish train vs eval mode for the
    purpose of *augmentation* (randomization).

    Subclasses implement ``__len__`` and ``__getitem__``, and consult
    ``self.is_training`` to decide whether to draw with randomness. Toggle
    via ``dataset.train()`` / ``dataset.eval()`` (mirrors ``nn.Module``).
    """

    def __init__(self, train: bool = True):
        self._train_mode = bool(train)

    def train(self, mode: bool = True) -> "AbstractDataset":
        self._train_mode = bool(mode)
        return self

    def eval(self) -> "AbstractDataset":
        return self.train(False)

    @property
    def is_training(self) -> bool:
        return self._train_mode

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int): ...


class AbstractImageDataset(AbstractDataset):
    """Image-dataset base implementing the canonical pipeline:

        load_raw(idx) -> PIL/tensor + label
            -> ToTensor (if PIL)
            -> pad_to_square
            -> Resize(size)
            -> augment (only when training)
            -> return (image_tensor, label)

    Notably absent: normalization. Models own that — see
    ``AbstractModel.normalize``.

    Args:
        size: target spatial size (H, W) after pad+resize.
        train: initial mode. Toggle later with ``train()`` / ``eval()``.
        augment: a ``Compose``-style callable (or ``None``) applied only
                 when ``self.is_training`` is True. Use one of
                 ``LIGHT_AUGMENT`` / ``STANDARD_AUGMENT`` / ``STRONG_AUGMENT``
                 for the predefined options, or pass any callable.
    """

    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        train: bool = True,
        augment: Optional[Callable] = STANDARD_AUGMENT,
    ):
        super().__init__(train=train)
        if not (
            isinstance(size, (tuple, list)) and len(size) == 2
            and all(isinstance(s, int) and s > 0 for s in size)
        ):
            raise ValueError(
                f"size must be a (H, W) tuple of positive ints, got {size!r}"
            )
        self.size: Tuple[int, int] = (int(size[0]), int(size[1]))

        if augment is not None and not callable(augment):
            raise TypeError(
                "augment must be a callable (e.g. transforms.Compose) or None; "
                f"got {type(augment).__name__}"
            )
        self._augment: Optional[Callable] = augment

        self._to_input_tensor = transforms.ToTensor()
        self._resize = transforms.Resize(self.size)

    @abstractmethod
    def load_raw(self, idx: int) -> Tuple[Union[Image.Image, torch.Tensor], Any]:
        """Return (image, label). ``image`` may be a PIL Image or a CHW
        float tensor. ``label`` is forwarded as-is."""

    @abstractmethod
    def __len__(self) -> int: ...

    def _to_tensor(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, Image.Image):
            return self._to_input_tensor(image)
        if torch.is_tensor(image):
            if image.dim() != 3:
                raise ValueError(
                    f"image tensor must be CHW, got shape {tuple(image.shape)}"
                )
            return image
        raise TypeError(
            "load_raw must return a PIL.Image.Image or a CHW torch.Tensor; "
            f"got {type(image).__name__}"
        )

    def transform(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        """Apply the full pipeline to a single image. Honors ``is_training``
        for the augment step. Useful for ad-hoc evaluation paths."""
        tensor = self._to_tensor(image)
        tensor = pad_to_square(tensor)
        tensor = self._resize(tensor)
        if self.is_training and self._augment is not None:
            tensor = self._augment(tensor)
        return tensor

    def __getitem__(self, idx: int):
        image, label = self.load_raw(idx)
        return self.transform(image), label

    @property
    def augment(self) -> Optional[Callable]:
        """The augmentation callable (or None). Read-only."""
        return self._augment
