"""SuperModel: lazy heads via nn.ModuleDict, Lightning-driven lifecycle."""
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from mltk import ModelFactory, SuperModel


class _LinearFactory(ModelFactory):
    """Picklable factory that builds an ``nn.Linear`` per key."""

    def __init__(self, in_dim: int = 3, out_dim: int = 2):
        self._in_dim = in_dim
        self._out_dim = out_dim

    @property
    def input_dim(self) -> int:
        return self._in_dim

    def __call__(self, key) -> nn.Module:
        del key
        return nn.Linear(self._in_dim, self._out_dim)


class _MultiHeadModel(SuperModel):
    def __init__(self):
        super().__init__(
            input_dim=3,
            trunk=nn.Linear(3, 3),
            head_factory=_LinearFactory(),
        )
        self.criterion = nn.MSELoss()
        self.created: list = []
        # We must build the head before constructing the optimizer so its
        # params are registered.
        self.get_head("main")
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def _on_head_created(self, key, head):
        self.created.append(key)

    def forward(self, inputs):
        return self.get_head("main")(self.extract_features(inputs))


class _HeadOnlyModel(SuperModel):
    def __init__(self):
        super().__init__(input_dim=3, head_factory=_LinearFactory())
        self.get_head("main")
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01)

    def forward(self, inputs):
        return self.get_head("main")(self.extract_features(inputs))


def test_get_head_caches_and_creates_once():
    m = _MultiHeadModel()
    head_a = m.get_head("main")
    head_b = m.get_head("main")
    assert head_a is head_b
    assert m.created == ["main"]


def test_parameters_includes_trunk_and_cached_heads():
    m = _MultiHeadModel()
    head = m.get_head("main")
    params = list(m.parameters())
    trunk_params = list(m.trunk.parameters())
    head_params = list(head.parameters())
    assert len(params) == len(trunk_params) + len(head_params)


def test_train_eval_propagates_to_heads():
    m = _MultiHeadModel()
    head = m.get_head("main")
    m.eval()
    assert not m.trunk.training and not head.training
    m.train()
    assert m.trunk.training and head.training


def test_supermodel_trains_with_lightning():
    m = _MultiHeadModel()
    x = torch.randn(8, 3)
    y = torch.randn(8, 2)
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    trainer.fit(m, loader, loader)


def test_head_only_supermodel_runs_forward():
    m = _HeadOnlyModel()
    x = torch.randn(4, 3)
    y = torch.randn(4, 2)
    assert m.input_dim == 3
    torch.testing.assert_close(m.extract_features(x), x)
    loss = m.compute_loss((x, y))
    assert float(loss.item()) >= 0


def test_head_keys_round_trip_through_checkpoint(tmp_path):
    m = _MultiHeadModel()
    m.precreate_heads(["a", "b"])
    assert set(m.head_keys) == {"main", "a", "b"}

    trainer = L.Trainer(
        max_epochs=0,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    # Run a no-op fit to give the trainer something to checkpoint against.
    x = torch.randn(2, 3)
    y = torch.randn(2, 2)
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    trainer.fit(m, loader, loader)

    ckpt = tmp_path / "ckpt.ckpt"
    trainer.save_checkpoint(str(ckpt))
    raw = torch.load(str(ckpt), map_location="cpu")
    assert set(raw["mltk_head_keys"]) == {"main", "a", "b"}
