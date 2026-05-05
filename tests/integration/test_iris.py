"""Iris integration test: a ``SkipAttentionMLP`` + ``AbstractModel`` subclass
must clear **97%** test accuracy using ``SmoothReduceLROnPlateau``.

Small enough to run on CPU in seconds — purpose is to catch regressions in
the model lifecycle, the block composition, AND the scheduler. If we
can't hit 97% on Iris with this stack, something in the library is broken.
"""
import pytest
import torch
import torch.nn as nn
import lightning as L
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from mltk import AbstractModel, SkipAttentionMLP, SmoothReduceLROnPlateau

pytestmark = pytest.mark.integration


class IrisClassifier(AbstractModel):
    def __init__(self, in_dim: int = 4, num_classes: int = 3, lr: float = 1e-2):
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Sequential(
            SkipAttentionMLP(in_features=in_dim, out_features=64, depth=3),
            nn.Linear(64, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-3)
        self.scheduler = SmoothReduceLROnPlateau(
            self.optimizer,
            smoothing_window=5,
            historical_window=20,
            reduction_threshold=0.98,
            cooldown=10,
            factor=0.5,
            min_lr=1e-5,
        )

    def forward(self, inputs):
        return self.head(inputs)


def _accuracy(model: IrisClassifier, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(-1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def _iris_loaders(batch_size: int = 16):
    iris = load_iris()
    x = torch.tensor(iris.data, dtype=torch.float32)
    y = torch.tensor(iris.target, dtype=torch.long)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std
    return (
        DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size),
    )


def _quiet_trainer(max_epochs: int) -> L.Trainer:
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )


def test_iris_trains_to_high_accuracy():
    train_loader, test_loader = _iris_loaders()
    model = IrisClassifier()
    trainer = _quiet_trainer(max_epochs=200)
    trainer.fit(model, train_loader, test_loader)
    acc = _accuracy(model, test_loader)
    assert acc >= 0.97, (
        f"Iris test accuracy only {acc:.4f}. "
        "SVC hits 100% on this split — something in the stack is broken."
    )


def test_iris_classifier_checkpoint_roundtrip(tmp_path):
    train_loader, test_loader = _iris_loaders()
    model = IrisClassifier()
    trainer = _quiet_trainer(max_epochs=5)
    trainer.fit(model, train_loader, test_loader)
    acc_before = _accuracy(model, test_loader)

    ckpt = tmp_path / "iris.ckpt"
    trainer.save_checkpoint(str(ckpt))
    loaded = IrisClassifier.load_from_checkpoint(str(ckpt), map_location="cpu")
    acc_loaded = _accuracy(loaded, test_loader)
    assert abs(acc_loaded - acc_before) < 1e-6, "reloaded model must produce identical eval acc"
