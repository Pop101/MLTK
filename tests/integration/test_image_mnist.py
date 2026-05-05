"""MNIST integration: a ``SuperModel`` with a 2-level parity/digit hierarchy
must clear **95%** full-digit accuracy using ``SmoothReduceLROnPlateau``.

The hierarchy:
    root  -> { even, odd }                       (out=2)
    even  -> { 0, 2, 4, 6, 8 }                   (out=5)
    odd   -> { 1, 3, 5, 7, 9 }                   (out=5)
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader

from mltk import SuperModel, SkipAttentionMLP, SmoothReduceLROnPlateau

pytestmark = pytest.mark.integration


PARITY_OF = {d: d % 2 for d in range(10)}
EVEN_DIGITS = [0, 2, 4, 6, 8]
ODD_DIGITS = [1, 3, 5, 7, 9]


class MnistHierModel(SuperModel):
    HEAD_OUT = {"root": 2, "even": 5, "odd": 5}

    def __init__(self, input_dim: int = 784, hidden: int = 256, lr: float = 3e-3):
        trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            SkipAttentionMLP(in_features=hidden, out_features=hidden, depth=2),
        )
        super().__init__(input_dim=input_dim, trunk=trunk)
        self.save_hyperparameters({"input_dim": input_dim, "hidden": hidden, "lr": lr})
        self.hidden = hidden
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        # Pre-create heads so they're in the optimizer from the start.
        self.precreate_heads(["root", "even", "odd"])
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = SmoothReduceLROnPlateau(
            self.optimizer, smoothing_window=3, historical_window=10,
            reduction_threshold=0.98, cooldown=2, factor=0.5, min_lr=1e-5,
        )

    def _build_head(self, key) -> nn.Module:
        return nn.Linear(self.hidden, self.HEAD_OUT[key])

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.extract_features(image.view(image.size(0), -1))
        root_probs = F.softmax(self.get_head("root")(feats), dim=-1)
        even_probs = F.softmax(self.get_head("even")(feats), dim=-1)
        odd_probs = F.softmax(self.get_head("odd")(feats), dim=-1)
        out = torch.zeros(image.size(0), 10, device=feats.device, dtype=feats.dtype)
        for i, d in enumerate(EVEN_DIGITS):
            out[:, d] = root_probs[:, 0] * even_probs[:, i]
        for i, d in enumerate(ODD_DIGITS):
            out[:, d] = root_probs[:, 1] * odd_probs[:, i]
        return out

    def compute_loss(self, batch) -> torch.Tensor:
        x, digit = batch
        x = x.view(x.size(0), -1).to(self.device, self.dtype)
        digit = digit.to(self.device)
        feats = self.extract_features(x)

        parity = torch.tensor(
            [PARITY_OF[int(d)] for d in digit.tolist()],
            dtype=torch.long, device=x.device,
        )
        loss = self.criterion(self.get_head("root")(feats), parity)

        for p_name, digits in [("even", EVEN_DIGITS), ("odd", ODD_DIGITS)]:
            p_val = 0 if p_name == "even" else 1
            mask = (parity == p_val)
            if not mask.any():
                continue
            idx_map = {d: i for i, d in enumerate(digits)}
            y_group = torch.tensor(
                [idx_map[int(d)] for d in digit[mask].tolist()],
                dtype=torch.long, device=x.device,
            )
            logits = self.get_head(p_name)(feats[mask])
            loss = loss + self.criterion(logits, y_group)
        return loss


def _accuracy(model: MnistHierModel, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            preds = model(x).argmax(-1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def _mnist_loaders(data_root, batch_size: int = 128, train_n: int = 20000, test_n: int = 2000):
    from torchvision import datasets, transforms as T
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    train = datasets.MNIST(str(data_root), train=True, download=True, transform=tfm)
    test = datasets.MNIST(str(data_root), train=False, download=True, transform=tfm)
    train = torch.utils.data.Subset(train, list(range(min(train_n, len(train)))))
    test = torch.utils.data.Subset(test, list(range(min(test_n, len(test)))))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=batch_size),
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


def test_mnist_hierarchy_trains_above_threshold(tmp_path_factory):
    pytest.importorskip("torchvision")
    data_root = tmp_path_factory.mktemp("mnist_data")
    train_loader, test_loader = _mnist_loaders(data_root)
    model = MnistHierModel()
    trainer = _quiet_trainer(max_epochs=20)
    trainer.fit(model, train_loader, test_loader)
    acc = _accuracy(model, test_loader)
    assert acc >= 0.95, (
        f"MNIST hierarchical accuracy only {acc:.4f}. "
        "A plain MLP hits 97% on this subset."
    )


def test_mnist_supermodel_checkpoint_roundtrip(tmp_path_factory, tmp_path):
    pytest.importorskip("torchvision")
    data_root = tmp_path_factory.mktemp("mnist_data")
    train_loader, test_loader = _mnist_loaders(data_root, train_n=1000, test_n=500)
    model = MnistHierModel()
    trainer = _quiet_trainer(max_epochs=1)
    trainer.fit(model, train_loader, test_loader)
    acc_before = _accuracy(model, test_loader)

    ckpt = tmp_path / "mnist_super.ckpt"
    trainer.save_checkpoint(str(ckpt))
    loaded = MnistHierModel.load_from_checkpoint(str(ckpt), map_location="cpu")
    acc_loaded = _accuracy(loaded, test_loader)
    assert abs(acc_loaded - acc_before) < 1e-6
