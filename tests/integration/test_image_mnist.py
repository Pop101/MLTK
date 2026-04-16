"""MNIST integration: a `SuperModel` with a 2-level parity/digit hierarchy
must clear **95%** full-digit accuracy using `SmoothReduceLROnPlateau`.

The hierarchy:
    root  -> { even, odd }                       (out=2)
    even  -> { 0, 2, 4, 6, 8 }                   (out=5)
    odd   -> { 1, 3, 5, 7, 9 }                   (out=5)

Purpose: exercise `SuperModel`'s lazy-head dispatch, the scheduler, the
save/load path, and the block composition on real image tensors. Not a
SOTA MNIST run — we use a subset and a modest trunk, so the 95% bar is
the floor a working library has to clear, not its ceiling.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mltk import SuperModel, SkipAttentionMLP, SmoothReduceLROnPlateau

pytestmark = pytest.mark.integration


PARITY_OF = {d: d % 2 for d in range(10)}
EVEN_DIGITS = [0, 2, 4, 6, 8]
ODD_DIGITS = [1, 3, 5, 7, 9]


class MnistHierModel(SuperModel):
    """SuperModel with:
        key='root'   -> head over {even, odd}            (out=2)
        key='even'   -> head over EVEN_DIGITS            (out=5)
        key='odd'    -> head over ODD_DIGITS             (out=5)
    """

    HEAD_OUT = {"root": 2, "even": 5, "odd": 5}

    def __init__(self, input_dim=784, hidden=256, lr=3e-3, device=None, dtype=torch.float32):
        # Attention-augmented MLP trunk — a tiny-but-real model so the
        # 95% bar says something about library quality, not test patience.
        trunk = nn.Sequential(
            nn.Linear(input_dim, hidden),
            SkipAttentionMLP(in_features=hidden, out_features=hidden, depth=2),
        )
        super().__init__(input_dim=input_dim, trunk=trunk, device=device, dtype=dtype)
        self.hidden = hidden
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.trunk.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = SmoothReduceLROnPlateau(
            self.optimizer,
            smoothing_window=3,
            historical_window=10,
            reduction_threshold=0.98,
            cooldown=2,
            factor=0.5,
            min_lr=1e-5,
        )
        self.lr = lr
        self.init_params = {"input_dim": input_dim, "hidden": hidden, "lr": lr}

    def _build_head(self, key):
        return nn.Linear(self.hidden, self.HEAD_OUT[key])

    def _create_head(self, key):
        head = super()._create_head(key)
        self.optimizer.add_param_group({"params": list(head.parameters()), "lr": self.lr})
        return head

    def train_batch(self, batch, transforms=None):
        x, digit = batch
        x = x.view(x.size(0), -1)
        self.trunk.train()
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

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_batches_trained += 1
        return float(loss.item())

    @torch.no_grad()
    def predict(self, image):
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

    def evaluate(self, data_loader, transforms=None):
        self.trunk.eval()
        for h in self._heads.values():
            h.eval()
        correct = total = 0
        total_loss = 0.0
        with torch.no_grad():
            for x, y in data_loader:
                preds = self.predict(x)
                # Cross-entropy on the combined 10-class distribution.
                total_loss += F.nll_loss(preds.clamp_min(1e-9).log(), y, reduction="sum").item()
                correct += (preds.argmax(-1) == y).sum().item()
                total += y.numel()
        return total_loss / max(1, total), correct / max(1, total)


def _mnist_loaders(data_root, batch_size=128, train_n=20000, test_n=2000):
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


def test_mnist_hierarchy_trains_above_threshold(tmp_path_factory):
    pytest.importorskip("torchvision")
    data_root = tmp_path_factory.mktemp("mnist_data")
    train_loader, test_loader = _mnist_loaders(data_root)
    model = MnistHierModel()

    best_acc = 0.0
    for epoch in range(20):
        for batch in train_loader:
            model.train_batch(batch)
        val_loss, acc = model.evaluate(test_loader)
        model.update_scheduler(val_loss)
        best_acc = max(best_acc, acc)
        if best_acc >= 0.97:  # shortcut when we have healthy margin over bar
            break
    assert best_acc >= 0.95, (
        f"MNIST hierarchical accuracy only {best_acc:.4f} after {epoch + 1} epochs. "
        "A plain MLP hits 97% on this subset — something in SuperModel or "
        "the scheduler is underperforming."
    )


def test_mnist_supermodel_save_load(tmp_path_factory, tmp_path):
    pytest.importorskip("torchvision")
    data_root = tmp_path_factory.mktemp("mnist_data")
    train_loader, test_loader = _mnist_loaders(data_root, train_n=1000, test_n=500)
    model = MnistHierModel()
    for batch in train_loader:
        model.train_batch(batch)
    _, acc_before = model.evaluate(test_loader)

    ckpt = tmp_path / "mnist_super.pth"
    model.save(str(ckpt))
    loaded = MnistHierModel.load(str(ckpt))
    _, acc_loaded = loaded.evaluate(test_loader)
    assert abs(acc_loaded - acc_before) < 1e-6, "save/load must preserve eval accuracy exactly"
    assert loaded.total_batches_trained == model.total_batches_trained
