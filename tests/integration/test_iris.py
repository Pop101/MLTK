"""Iris integration test: a SkipAttentionMLP trained by a minimal
AbstractClassifier subclass must clear 90% test accuracy.

Small enough to run on CPU in seconds — purpose is to catch regressions in
block composition + the classifier lifecycle, not to benchmark Iris.
"""
import pytest
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from mltk import AbstractClassifier, FeaturePerspective, SkipAttentionMLP


pytestmark = pytest.mark.integration


class IrisClassifier(AbstractClassifier):
    def __init__(self, in_dim=4, num_classes=3, lr=5e-3, device=None, dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.head = nn.Sequential(
            SkipAttentionMLP(in_features=in_dim, out_features=32, depth=2),
            nn.Linear(32, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr, weight_decay=1e-4)
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


def _iris_loaders(batch_size=16):
    iris = load_iris()
    x = torch.tensor(iris.data, dtype=torch.float32)
    y = torch.tensor(iris.target, dtype=torch.long)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=17, stratify=y)
    # Standardize on train stats.
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
    # Train until we beat 90% or we hit a budget.
    best_acc = 0.0
    for epoch in range(120):
        for batch in train_loader:
            model.train_batch(batch)
        if epoch >= 20 and epoch % 5 == 0:
            _, acc = model.evaluate(test_loader)
            best_acc = max(best_acc, acc)
            if best_acc >= 0.9:
                break
    _, acc = model.evaluate(test_loader)
    best_acc = max(best_acc, acc)
    assert best_acc >= 0.9, f"Iris best test accuracy only {best_acc:.3f} after {epoch + 1} epochs"


def test_iris_classifier_save_load_roundtrip(tmp_path):
    train_loader, test_loader = _iris_loaders()
    model = IrisClassifier()
    for batch in train_loader:
        model.train_batch(batch)
    _, acc_before = model.evaluate(test_loader)

    ckpt = tmp_path / "iris.pth"
    model.save(str(ckpt))
    loaded = IrisClassifier.load(str(ckpt))
    _, acc_after = model.evaluate(test_loader)
    _, acc_loaded = loaded.evaluate(test_loader)
    assert acc_loaded == acc_after, "reloaded model must produce identical eval acc"
    assert loaded.total_batches_trained == model.total_batches_trained
