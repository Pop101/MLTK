"""Lightning lifecycle on a minimal AbstractModel subclass."""
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

from mltk import AbstractModel


class TinyClassifier(AbstractModel):
    """Minimal subclass that exercises Lightning checkpoint + optimizer plumbing."""

    def __init__(self, in_dim: int = 4, num_classes: int = 3, lr: float = 1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Linear(in_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)

    def forward(self, inputs):
        return self.head(inputs)


def _quiet_trainer(**kwargs) -> L.Trainer:
    defaults = dict(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    defaults.update(kwargs)
    return L.Trainer(**defaults)


def test_trains_and_checkpoint_roundtrips():
    m = TinyClassifier()
    x = torch.randn(40, 4)
    y = torch.randint(0, 3, (40,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=8)
    val_loader = DataLoader(TensorDataset(x, y), batch_size=8)
    trainer = _quiet_trainer()
    trainer.fit(m, train_loader, val_loader)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "ckpt.ckpt")
        trainer.save_checkpoint(path)
        loaded = TinyClassifier.load_from_checkpoint(path, map_location="cpu")

    for pa, pb in zip(m.head.parameters(), loaded.head.parameters()):
        torch.testing.assert_close(pa, pb)


def test_forward_is_abstract():
    class _NoForward(AbstractModel):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2, 1)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.head.parameters(), lr=1e-2)

    with pytest.raises(TypeError):
        _NoForward()  # type: ignore[abstract]


def test_predict_runs_in_eval_mode_no_grad_with_normalize():
    """``predict`` should set eval, no_grad, normalize, then forward."""
    class _Reg(AbstractModel):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2, 1)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.head.parameters(), lr=1e-2)
            self.normalize = lambda x: x - 1.0  # subtract 1 from every input

        def forward(self, x):
            return self.head(x)

    m = _Reg()
    x = torch.tensor([[1.0, 1.0]])
    out = m.predict(x)
    assert not m.head.training
    expected = m.head.bias.detach()
    torch.testing.assert_close(out, expected.unsqueeze(0))


def test_compute_loss_normalizes_before_forward():
    """``compute_loss`` must normalize before calling forward."""
    seen = {"normalize_called": 0}

    class _SpyNormalize:
        def __call__(self, x):
            seen["normalize_called"] += 1
            return x

    class _Reg(AbstractModel):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2, 1)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.head.parameters(), lr=1e-2)
            self.normalize = _SpyNormalize()

        def forward(self, x):
            return self.head(x)

    m = _Reg()
    batch = (torch.randn(4, 2), torch.randn(4))
    m.compute_loss(batch)
    assert seen["normalize_called"] == 1


def test_configure_optimizers_returns_scheduler_config_when_set():
    m = TinyClassifier()
    cfg = m.configure_optimizers()
    assert isinstance(cfg, dict)
    assert cfg["optimizer"] is m.optimizer
    assert cfg["lr_scheduler"]["scheduler"] is m.scheduler


def test_configure_optimizers_returns_bare_optimizer_when_no_scheduler():
    class _Plain(AbstractModel):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2, 1)
            self.criterion = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.head.parameters(), lr=1e-2)

        def forward(self, x):
            return self.head(x)

    m = _Plain()
    assert m.configure_optimizers() is m.optimizer


def test_missing_optimizer_raises():
    class _NoOpt(AbstractModel):
        def __init__(self):
            super().__init__()
            self.criterion = nn.MSELoss()

        def forward(self, x):
            return x

    m = _NoOpt()
    with pytest.raises(RuntimeError, match="optimizer"):
        m.configure_optimizers()


def test_missing_criterion_raises_on_compute_loss():
    class _NoCrit(AbstractModel):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2, 1)
            self.optimizer = torch.optim.SGD(self.head.parameters(), lr=1e-2)

        def forward(self, x):
            return self.head(x)

    m = _NoCrit()
    with pytest.raises(RuntimeError, match="criterion"):
        m.compute_loss((torch.randn(2, 2), torch.randn(2, 1)))
