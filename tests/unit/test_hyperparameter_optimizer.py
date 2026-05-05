"""HyperparameterOptimizer over Lightning Trainer."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mltk import (
    AbstractModel,
    ChoiceParam,
    FloatRange,
    HyperparameterOptimizer,
    OptimizationMode,
    SearchStrategy,
)


# Top-level so the model_factory is importable in spawned subprocesses.
class TinyRegressor(AbstractModel):
    def __init__(self, lr: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Linear(1, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.head.parameters(), lr=lr)

    def forward(self, x):
        return self.head(x)


def make_tiny_regressor(params):
    return TinyRegressor(**params)


def make_train_loader():
    x = torch.linspace(-1, 1, 24).unsqueeze(1)
    y = 2.0 * x + 0.5
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def make_val_loader():
    x = torch.linspace(-1, 1, 24).unsqueeze(1)
    y = 2.0 * x + 0.5
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def _quiet_trainer_kwargs():
    return {
        "accelerator": "cpu",
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
    }


def test_random_search_runs_in_process():
    opt = HyperparameterOptimizer(
        model_factory=make_tiny_regressor,
        search_space={"lr": ChoiceParam([0.01, 0.05])},
        train_dataloader_factory=make_train_loader,
        val_dataloader_factory=make_val_loader,
        epochs=1,
        strategy=SearchStrategy.RANDOM,
        max_trials=2,
        max_concurrent_trials=1,
        trainer_kwargs=_quiet_trainer_kwargs(),
        seed=0,
    )
    result = opt.optimize()
    assert len(result.trials) == 2
    assert all(t.status == "completed" for t in result.trials)
    assert result.best_trial.params["lr"] in (0.01, 0.05)
    assert result.best_trial.score < float("inf")


def test_grid_search_enumerates_full_grid():
    opt = HyperparameterOptimizer(
        model_factory=make_tiny_regressor,
        search_space={"lr": ChoiceParam([0.01, 0.05, 0.1])},
        train_dataloader_factory=make_train_loader,
        val_dataloader_factory=make_val_loader,
        epochs=1,
        strategy=SearchStrategy.GRID,
        max_concurrent_trials=1,
        trainer_kwargs=_quiet_trainer_kwargs(),
        seed=0,
    )
    result = opt.optimize()
    assert len(result.trials) == 3


def test_successive_halving_prunes_intermediate_trials():
    opt = HyperparameterOptimizer(
        model_factory=make_tiny_regressor,
        search_space={"lr": FloatRange(1e-3, 1e-1, log=True)},
        train_dataloader_factory=make_train_loader,
        val_dataloader_factory=make_val_loader,
        epochs=4,
        min_epochs=1,
        reduction_factor=2,
        strategy=SearchStrategy.SUCCESSIVE_HALVING,
        max_trials=4,
        max_concurrent_trials=1,
        mode=OptimizationMode.MINIMIZE,
        trainer_kwargs=_quiet_trainer_kwargs(),
        seed=0,
    )
    result = opt.optimize()
    statuses = [t.status for t in result.trials]
    # Some trials should be pruned, at least one completed.
    assert "completed" in statuses
    assert "pruned" in statuses
    assert result.best_trial.status == "completed"
