"""AbstractDataset / AbstractImageDataset / augmentation presets.

Verifies the MLTK transform paradigm contract: dataset toggles augment via
train()/eval(), image base runs the pad+resize pipeline deterministically,
and presets are stable strings. Normalization is *not* tested here — that
lives on the classifier."""
import pytest
import torch
from PIL import Image

from mltk import (
    AbstractDataset,
    AbstractImageDataset,
    LIGHT_AUGMENT,
    STANDARD_AUGMENT,
    STRONG_AUGMENT,
    pad_to_square,
)


# --------------------------------------------------------------------------
# pad_to_square
# --------------------------------------------------------------------------
def test_pad_to_square_makes_square():
    img = torch.rand(3, 10, 30)
    out = pad_to_square(img)
    assert out.shape == (3, 30, 30)


def test_pad_to_square_centers_content():
    img = torch.ones(3, 10, 30)
    out = pad_to_square(img)
    # Top and bottom rows should be padded zeros; middle rows are the original.
    assert out[:, 0].sum() == 0 and out[:, -1].sum() == 0
    assert out[:, 10:20].sum() > 0


def test_pad_to_square_rejects_non_chw():
    with pytest.raises(ValueError):
        pad_to_square(torch.rand(10, 30))  # HW only


# --------------------------------------------------------------------------
# Augmentation presets
# --------------------------------------------------------------------------
def test_augmentation_presets_apply_to_tensor():
    for preset in (LIGHT_AUGMENT, STANDARD_AUGMENT, STRONG_AUGMENT):
        out = preset(torch.rand(3, 64, 64))
        assert out.shape == (3, 64, 64)


# --------------------------------------------------------------------------
# AbstractDataset — train/eval mode
# --------------------------------------------------------------------------
class _ToyDataset(AbstractDataset):
    def __init__(self, n=4, **kw):
        super().__init__(**kw)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.tensor([idx], dtype=torch.float32), idx


def test_abstract_dataset_default_is_training():
    assert _ToyDataset().is_training is True


def test_abstract_dataset_eval_toggle_chains():
    d = _ToyDataset()
    assert d.eval() is d  # chainable
    assert d.is_training is False
    d.train(True)
    assert d.is_training is True


def test_abstract_dataset_cannot_instantiate_directly():
    """ABC — abstract methods must be implemented."""
    with pytest.raises(TypeError):
        AbstractDataset()  # type: ignore[abstract]


# --------------------------------------------------------------------------
# AbstractImageDataset — pipeline + augment toggle
# --------------------------------------------------------------------------
class _PILImageDataset(AbstractImageDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        # Two arbitrary-sized PIL images; pad+resize must normalize them.
        self._items = [
            (Image.new("RGB", (40, 80), (200, 100, 50)), 0.5),
            (Image.new("RGB", (200, 100), (10, 10, 10)), 1.5),
        ]

    def __len__(self):
        return len(self._items)

    def load_raw(self, idx):
        return self._items[idx]


def test_image_dataset_pipeline_yields_target_size():
    ds = _PILImageDataset(size=(32, 32), train=False)
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(label, float) and label == 0.5


def test_image_dataset_eval_mode_is_deterministic():
    """No augment in eval mode -> same idx returns the exact same tensor."""
    ds = _PILImageDataset(size=(16, 16), train=False)
    a, _ = ds[0]
    b, _ = ds[0]
    torch.testing.assert_close(a, b)


def test_image_dataset_train_mode_with_strong_augment_changes_output():
    ds = _PILImageDataset(size=(32, 32), train=True, augment=STRONG_AUGMENT)
    samples = [ds[0][0] for _ in range(4)]
    differs = any(not torch.equal(samples[0], s) for s in samples[1:])
    assert differs, "augmentation appears not to be applied in train mode"


def test_image_dataset_train_off_disables_augment_even_with_preset():
    ds = _PILImageDataset(size=(32, 32), train=False, augment=STRONG_AUGMENT)
    a, _ = ds[0]
    b, _ = ds[0]
    torch.testing.assert_close(a, b)


def test_image_dataset_accepts_tensor_from_load_raw():
    class _TensorDS(AbstractImageDataset):
        def __init__(self):
            super().__init__(size=(8, 8), train=False)
        def __len__(self):
            return 1
        def load_raw(self, idx):
            return torch.rand(3, 20, 30), idx

    img, _ = _TensorDS()[0]
    assert img.shape == (3, 8, 8)


def test_image_dataset_rejects_bad_size():
    with pytest.raises(ValueError):
        _PILImageDataset(size=(0, 32))
    with pytest.raises(ValueError):
        _PILImageDataset(size=(32,))  # type: ignore[arg-type]


def test_image_dataset_rejects_bad_augment_type():
    with pytest.raises(TypeError):
        _PILImageDataset(augment=42)  # type: ignore[arg-type]


def test_image_dataset_accepts_callable_augment():
    sentinel = {"called": 0}
    def aug(t):
        sentinel["called"] += 1
        return t
    ds = _PILImageDataset(size=(16, 16), train=True, augment=aug)
    _ = ds[0]
    assert sentinel["called"] == 1
