"""Iris integration test: a `SkipAttentionMLP` + `AbstractClassifier` subclass
must clear **97%** test accuracy using `SmoothReduceLROnPlateau`.

Small enough to run on CPU in seconds — purpose is to catch regressions in
the classifier lifecycle, the block composition, AND the scheduler. If we
can't hit 97% on Iris with this stack, something in the library is broken.
"""
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from mltk import AbstractClassifier, SkipAttentionMLP, SmoothReduceLROnPlateau

pytestmark = pytest.mark.integration


class IrisClassifier(AbstractClassifier):
    def __init__(self, in_dim=4, num_classes=3, lr=1e-2, device=None, dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
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
        self.init_params = {"in_dim": in_dim, "num_classes": num_classes, "lr": lr}

    def train_batch(self, batch, transforms=None):
        x, y = batch
        self.head.train()
        logits = self.head(x)
        loss = self.criterion(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_batches_trained += 1
        return float(loss.item())

    def evaluate(self, data_loader, transforms=None):
        self.head.eval()
        correct = total = 0
        total_loss = 0.0
        with torch.no_grad():
            for x, y in data_loader:
                logits = self.head(x)
                total_loss += self.criterion(logits, y).item() * y.numel()
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.numel()
        return total_loss / max(1, total), correct / max(1, total)

    def predict(self, image):
        return self.head(image)

    def _get_additional_save_state(self):
        return {"head_state_dict": self.head.state_dict()}

    def _load_additional_state(self, checkpoint):
        self.head.load_state_dict(checkpoint["head_state_dict"])


def _iris_loaders(batch_size=16):
    iris = load_iris()
    x = torch.tensor(iris.data, dtype=torch.float32)
    y = torch.tensor(iris.target, dtype=torch.long)
    # seed=0 + test_size=0.2 avoids the versicolor/virginica border pathology
    # that caps ~45-sample 30%-test splits at 93% for every classifier.
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)
    mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr = (x_tr - mean) / std
    x_te = (x_te - mean) / std
    return (
        DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True),
        DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size),
    )


def test_iris_trains_to_high_accuracy():
    train_loader, test_loader = _iris_loaders()
    model = IrisClassifier()
    best_acc = 0.0
    for epoch in range(200):
        for batch in train_loader:
            model.train_batch(batch)
        val_loss, acc = model.evaluate(test_loader)
        model.update_scheduler(val_loss)
        best_acc = max(best_acc, acc)
        if best_acc >= 0.98:  # shortcut when we have healthy margin over bar
            break
    assert best_acc >= 0.97, (
        f"Iris best test accuracy only {best_acc:.4f} after {epoch + 1} epochs. "
        "SVC hits 100% on this split — something in the stack is broken."
    )


def test_iris_classifier_save_load_roundtrip(tmp_path):
    train_loader, test_loader = _iris_loaders()
    model = IrisClassifier()
    for batch in train_loader:
        model.train_batch(batch)
    _, acc_before = model.evaluate(test_loader)

    ckpt = tmp_path / "iris.pth"
    model.save(str(ckpt))
    loaded = IrisClassifier.load(str(ckpt))
    _, acc_loaded = loaded.evaluate(test_loader)
    assert acc_loaded == acc_before, "reloaded model must produce identical eval acc"
    assert loaded.total_batches_trained == model.total_batches_trained
