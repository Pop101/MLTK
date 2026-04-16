"""Save / load / resume lifecycle on a minimal AbstractClassifier subclass."""
import os
import tempfile

import torch
import torch.nn as nn

from mltk import AbstractClassifier


class TinyClassifier(AbstractClassifier):
    """Minimal concrete subclass that exercises save/load + optimizer plumbing."""

    def __init__(self, in_dim: int = 4, num_classes: int = 3, lr: float = 1e-2,
                 device=None, dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.head = nn.Linear(in_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.init_params = {"in_dim": in_dim, "num_classes": num_classes, "lr": lr}

    def train_batch(self, batch, transforms=None):
        x, y = batch
        logits = self.head(x)
        loss = self.criterion(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_batches_trained += 1
        return float(loss.item())

    def evaluate(self, data_loader, transforms=None):
        correct = total = 0
        with torch.no_grad():
            for x, y in data_loader:
                correct += (self.head(x).argmax(-1) == y).sum().item()
                total += y.numel()
        return 0.0, correct / max(1, total)

    def predict(self, image):
        return self.head(image)

    def _get_additional_save_state(self):
        return {"head_state_dict": self.head.state_dict()}

    def _load_additional_state(self, checkpoint):
        self.head.load_state_dict(checkpoint["head_state_dict"])


def test_save_load_roundtrip_preserves_weights():
    m = TinyClassifier()
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    for _ in range(5):
        m.train_batch((x, y))

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.pth")
        m.save(path)
        # Atomic save means no .tmp should remain.
        assert not os.path.exists(path + ".tmp")
        loaded = TinyClassifier.load(path)

    for pa, pb in zip(m.head.parameters(), loaded.head.parameters()):
        torch.testing.assert_close(pa, pb)
    assert loaded.total_batches_trained == 5


def test_save_is_atomic_on_interrupt_simulation(monkeypatch):
    """If torch.save raises, we must not leave a half-written real file."""
    m = TinyClassifier()
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.pth")

        real_save = torch.save

        def kaboom(state, target):
            real_save(state, target)  # write the .tmp fully
            raise RuntimeError("simulated crash before rename")

        monkeypatch.setattr("mltk.base_classifier.torch.save", kaboom)
        try:
            m.save(path)
        except RuntimeError:
            pass

        # Real file must not exist; .tmp may linger.
        assert not os.path.exists(path)
