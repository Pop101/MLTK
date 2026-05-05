"""Hyperparameter search over Lightning ``Trainer.fit``.

The optimizer takes a model factory ``params -> LightningModule`` plus a
search space, and trains many models — optionally many at the same time —
using ``lightning.Trainer``. Successive halving is the headline strategy:
short the bad trials early, give the survivors more epochs.

Concurrency
-----------
``max_concurrent_trials`` controls how many ``Trainer.fit`` calls run at
the same time. Each runs in its own subprocess (``multiprocessing`` with
the ``spawn`` start method), so:

- they don't fight over the GIL,
- each gets its own CUDA / DirectML / MPS context (no fragmentation),
- after a trial finishes, its process exits and the OS reclaims the RAM,
- workers share OS-page-cached files (e.g. ``DiskCachedDataset``) for free.

Per-trial state is persisted as Lightning checkpoints so successive-halving
rungs resume from the previous rung's weights instead of restarting.

Picklability
------------
Anything passed to a worker process must be picklable. In practice this
means:

- ``model_factory`` is a top-level function (or a picklable callable),
- ``train_dataloader_factory`` / ``val_dataloader_factory`` are top-level
  callables that build a fresh ``DataLoader`` inside the worker,
- search-space ``Hyperparameter`` instances and ``params`` dicts are
  picklable (the dataclasses here are).

For ``max_concurrent_trials=1`` the work runs in-process, no pickling.
"""

from __future__ import annotations

import itertools
import math
import multiprocessing as mp
import os
import random
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import lightning as L
import torch
from torch.utils.data import DataLoader


MetricValue = Mapping[str, float]
ObjectiveFn = Callable[[MetricValue], float]
ModelFactoryFn = Callable[[Mapping[str, Any]], L.LightningModule]
DataLoaderFactoryFn = Callable[[], DataLoader]


class SearchStrategy(Enum):
    """Hyperparameter candidate generation strategy."""
    RANDOM = "random"
    GRID = "grid"
    SUCCESSIVE_HALVING = "successive_halving"


class OptimizationMode(Enum):
    """Whether lower or higher objective scores are better."""
    MINIMIZE = "min"
    MAXIMIZE = "max"


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
class Hyperparameter:
    def sample(self, rng: random.Random) -> Any:
        raise NotImplementedError

    def grid_values(self) -> Sequence[Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class ChoiceParam(Hyperparameter):
    """Categorical hyperparameter."""
    values: Sequence[Any]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("ChoiceParam requires at least one value")

    def sample(self, rng: random.Random) -> Any:
        return rng.choice(list(self.values))

    def grid_values(self) -> Sequence[Any]:
        return list(self.values)


@dataclass(frozen=True)
class FloatRange(Hyperparameter):
    """Continuous float hyperparameter; ``log=True`` for log-uniform."""
    low: float
    high: float
    log: bool = False
    grid_size: int = 5

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("FloatRange low must be <= high")
        if self.log and self.low <= 0:
            raise ValueError("log FloatRange requires low > 0")
        if self.grid_size < 1:
            raise ValueError("FloatRange grid_size must be >= 1")

    def sample(self, rng: random.Random) -> float:
        if self.low == self.high:
            return float(self.low)
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(rng.uniform(lo, hi)))
        return float(rng.uniform(self.low, self.high))

    def grid_values(self) -> Sequence[float]:
        if self.grid_size == 1 or self.low == self.high:
            return [float(self.low)]
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return [
                float(math.exp(lo + (hi - lo) * i / (self.grid_size - 1)))
                for i in range(self.grid_size)
            ]
        return [
            float(self.low + (self.high - self.low) * i / (self.grid_size - 1))
            for i in range(self.grid_size)
        ]


@dataclass(frozen=True)
class IntRange(Hyperparameter):
    """Integer hyperparameter with inclusive bounds."""
    low: int
    high: int
    step: int = 1

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError("IntRange low must be <= high")
        if self.step < 1:
            raise ValueError("IntRange step must be >= 1")

    def sample(self, rng: random.Random) -> int:
        return int(rng.choice(list(self.grid_values())))

    def grid_values(self) -> Sequence[int]:
        return list(range(self.low, self.high + 1, self.step))


# --------------------------------------------------------------------------- #
# Trial state
# --------------------------------------------------------------------------- #
@dataclass
class TrialResult:
    """Outcome of one hyperparameter trial."""
    trial_id: int
    params: dict[str, Any]
    metrics: Optional[MetricValue]
    score: float
    epochs_completed: int
    score_history: list[float]
    status: str
    elapsed_seconds: float
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None


@dataclass
class _TrialState:
    trial_id: int
    params: dict[str, Any]
    epochs_completed: int = 0
    last_metrics: Optional[MetricValue] = None
    score_history: list[float] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    started: float = field(default_factory=time.perf_counter)


@dataclass
class OptimizationResult:
    """Complete hyperparameter search result."""
    best_trial: TrialResult
    trials: list[TrialResult]

    @property
    def best_params(self) -> dict[str, Any]:
        return dict(self.best_trial.params)

    @property
    def best_score(self) -> float:
        return self.best_trial.score


# --------------------------------------------------------------------------- #
# Worker entry point — runs in a subprocess
# --------------------------------------------------------------------------- #
def _run_trial(
    model_factory: ModelFactoryFn,
    params: Mapping[str, Any],
    train_dataloader_factory: DataLoaderFactoryFn,
    val_dataloader_factory: Optional[DataLoaderFactoryFn],
    target_epochs: int,
    resume_from: Optional[str],
    checkpoint_dir: str,
    trial_id: int,
    trainer_kwargs: Mapping[str, Any],
    seed: Optional[int],
) -> dict[str, Any]:
    """Train ``model_factory(params)`` to ``target_epochs`` and save a checkpoint.

    If ``resume_from`` is given, training resumes from that Lightning checkpoint
    so the trainer's ``current_epoch`` continues where the previous rung
    left off.
    """
    if seed is not None:
        L.seed_everything(seed, workers=True)

    model = model_factory(dict(params))
    if not isinstance(model, L.LightningModule):
        raise TypeError(
            "model_factory must return a LightningModule, "
            f"got {type(model).__name__}"
        )

    ckpt_path = os.path.join(checkpoint_dir, f"trial_{trial_id:04d}.ckpt")
    kwargs: dict[str, Any] = {
        "max_epochs": target_epochs,
        "default_root_dir": checkpoint_dir,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
    }
    kwargs.update(dict(trainer_kwargs))
    trainer = L.Trainer(**kwargs)

    trainer.fit(
        model,
        train_dataloaders=train_dataloader_factory(),
        val_dataloaders=val_dataloader_factory() if val_dataloader_factory else None,
        ckpt_path=resume_from,
    )

    trainer.save_checkpoint(ckpt_path)
    metrics = {
        k: float(v.detach().cpu().item() if torch.is_tensor(v) else v)
        for k, v in trainer.callback_metrics.items()
    }

    return {
        "trial_id": trial_id,
        "metrics": metrics,
        "epochs_completed": int(trainer.current_epoch),
        "checkpoint_path": ckpt_path,
    }


# --------------------------------------------------------------------------- #
# HyperparameterOptimizer
# --------------------------------------------------------------------------- #
@dataclass
class HyperparameterOptimizer:
    """Train and evaluate many ``LightningModule`` instances.

    Parameters
    ----------
    model_factory:
        Top-level callable that receives the sampled parameter dict and
        returns a fresh ``LightningModule``. Must be picklable when
        ``max_concurrent_trials > 1``.
    search_space:
        Mapping from parameter name to ``ChoiceParam`` / ``FloatRange`` /
        ``IntRange``.
    train_dataloader_factory / val_dataloader_factory:
        Top-level callables that build a fresh ``DataLoader``. A factory is
        required (rather than a constructed loader) so each subprocess can
        construct its own loader and so ``DiskCachedDataset`` and friends
        are mmap-shared across workers via the OS page cache.
    objective:
        Callable that maps ``trainer.callback_metrics`` (a dict of
        ``metric_name -> float``) to the scalar to optimize. Defaults to
        ``metrics["val_loss"]`` and falls back to the first numeric value.
    mode:
        ``MINIMIZE`` for losses, ``MAXIMIZE`` for accuracy/F1/etc.
    epochs:
        Maximum training epochs per trial.
    strategy:
        ``RANDOM``, ``GRID``, or ``SUCCESSIVE_HALVING``.
    max_trials:
        How many trials to draw (random search). Required for ``RANDOM`` and
        ``SUCCESSIVE_HALVING``; optional for ``GRID``.
    max_concurrent_trials:
        How many trials run in parallel. Each runs in its own subprocess.
        Set to 1 to keep everything in-process (no pickling required).
    work_dir:
        Where Lightning checkpoints land. Default: a fresh tempdir.
    trainer_kwargs:
        Extra keyword arguments forwarded to ``lightning.Trainer``. Useful
        for ``accelerator="gpu"``, ``devices=1``, ``precision="16-mixed"``,
        etc. ``max_epochs`` is set per-rung and overrides any value passed.
    seed:
        Optional seed for hyperparameter sampling and per-trial determinism.
    min_epochs / reduction_factor:
        Successive-halving knobs. Rungs are
        ``min_epochs, min_epochs * r, ..., epochs``; at each rung the top
        ``ceil(active / r)`` trials advance.
    raise_on_failed_trial:
        Re-raise the worker exception instead of marking the trial failed.
    """

    model_factory: ModelFactoryFn
    search_space: Mapping[str, Hyperparameter]
    train_dataloader_factory: DataLoaderFactoryFn
    val_dataloader_factory: Optional[DataLoaderFactoryFn] = None
    objective: Optional[ObjectiveFn] = None
    mode: OptimizationMode = OptimizationMode.MINIMIZE
    epochs: int = 1
    strategy: SearchStrategy = SearchStrategy.RANDOM
    max_trials: Optional[int] = None
    max_concurrent_trials: int = 1
    work_dir: Optional[Path] = None
    trainer_kwargs: Mapping[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    min_epochs: int = 1
    reduction_factor: int = 3
    raise_on_failed_trial: bool = False

    _owned_tempdir: Optional[tempfile.TemporaryDirectory] = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.mode, OptimizationMode):
            raise TypeError("mode must be an OptimizationMode")
        if not isinstance(self.strategy, SearchStrategy):
            raise TypeError("strategy must be a SearchStrategy")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.min_epochs < 1:
            raise ValueError("min_epochs must be >= 1")
        if self.min_epochs > self.epochs:
            raise ValueError("min_epochs must be <= epochs")
        if self.reduction_factor < 2:
            raise ValueError("reduction_factor must be >= 2")
        if self.max_trials is not None and self.max_trials < 1:
            raise ValueError("max_trials must be >= 1 when provided")
        if self.max_concurrent_trials < 1:
            raise ValueError("max_concurrent_trials must be >= 1")
        if not self.search_space:
            raise ValueError("search_space must not be empty")
        for name, param in self.search_space.items():
            if not isinstance(name, str) or not name:
                raise ValueError("search_space keys must be non-empty strings")
            if not isinstance(param, Hyperparameter):
                raise TypeError(
                    f"search_space[{name!r}] must be a Hyperparameter, "
                    f"got {type(param).__name__}"
                )
        if self.work_dir is None:
            self._owned_tempdir = tempfile.TemporaryDirectory(prefix="mltk_hpo_")
            self.work_dir = Path(self._owned_tempdir.name)
        else:
            self.work_dir = Path(self.work_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)

    # ---- public API ----

    def optimize(self) -> OptimizationResult:
        rng = random.Random(self.seed)
        states: list[_TrialState] = [
            _TrialState(trial_id=i, params=dict(p))
            for i, p in enumerate(self._iter_param_sets(rng))
        ]
        if not states:
            raise RuntimeError("search produced zero parameter sets")

        if self.strategy is SearchStrategy.SUCCESSIVE_HALVING:
            results = self._optimize_halving(states)
        else:
            results = self._optimize_flat(states)

        completed = [r for r in results if r.status == "completed"]
        if not completed:
            raise RuntimeError("all hyperparameter trials failed or were pruned")
        best = completed[0]
        for r in completed[1:]:
            if self._is_better(r, best):
                best = r
        return OptimizationResult(
            best_trial=best,
            trials=sorted(results, key=lambda r: r.trial_id),
        )

    # ---- strategies ----

    def _optimize_flat(self, states: list[_TrialState]) -> list[TrialResult]:
        """RANDOM/GRID: train every trial to ``self.epochs``."""
        return self._run_rung(states, target_epoch=self.epochs)

    def _optimize_halving(self, states: list[_TrialState]) -> list[TrialResult]:
        """Synchronous successive halving."""
        milestones = self._rung_milestones()
        results: list[TrialResult] = []
        active = states
        for i, milestone in enumerate(milestones):
            terminal = (i == len(milestones) - 1)
            rung = self._run_rung(active, target_epoch=milestone)
            active_by_id = {s.trial_id: s for s in active}
            ranked = sorted(
                (r for r in rung if r.status == "completed"),
                key=lambda r: r.score,
                reverse=(self.mode is OptimizationMode.MAXIMIZE),
            )
            keep_count = max(1, math.ceil(len(active) / self.reduction_factor))
            survivors = ranked if terminal else ranked[:keep_count]
            survivor_ids = {r.trial_id for r in survivors}

            for r in rung:
                if r.status != "completed":
                    results.append(r)
                elif r.trial_id not in survivor_ids:
                    results.append(_with_status(r, "pruned"))

            if terminal:
                results.extend(survivors)
                break

            next_active: list[_TrialState] = []
            for r in survivors:
                state = active_by_id[r.trial_id]
                state.epochs_completed = r.epochs_completed
                state.last_metrics = r.metrics
                state.score_history = list(r.score_history)
                state.checkpoint_path = r.checkpoint_path
                next_active.append(state)
            if not next_active:
                break
            active = next_active

        return results

    # ---- rung executor (handles concurrency) ----

    def _run_rung(
        self, states: list[_TrialState], *, target_epoch: int,
    ) -> list[TrialResult]:
        """Train each ``state`` to ``target_epoch`` and return TrialResults.

        Runs ``self.max_concurrent_trials`` trials at once. With 1, the work
        happens in-process (no pickling); with >1, spawns subprocesses.
        """
        if self.max_concurrent_trials == 1:
            return [self._run_trial_inproc(s, target_epoch) for s in states]
        return self._run_trials_parallel(states, target_epoch)

    def _run_trial_inproc(self, state: _TrialState, target_epoch: int) -> TrialResult:
        try:
            out = _run_trial(
                model_factory=self.model_factory,
                params=state.params,
                train_dataloader_factory=self.train_dataloader_factory,
                val_dataloader_factory=self.val_dataloader_factory,
                target_epochs=target_epoch,
                resume_from=state.checkpoint_path,
                checkpoint_dir=str(self.work_dir),
                trial_id=state.trial_id,
                trainer_kwargs=dict(self.trainer_kwargs),
                seed=(self.seed + state.trial_id) if self.seed is not None else None,
            )
        except Exception as exc:
            if self.raise_on_failed_trial:
                raise
            return self._failed_result(state, exc)
        return self._completed_result(state, out)

    def _run_trials_parallel(
        self, states: list[_TrialState], target_epoch: int,
    ) -> list[TrialResult]:
        ctx = mp.get_context("spawn")
        results: list[TrialResult] = []
        with ProcessPoolExecutor(
            max_workers=self.max_concurrent_trials, mp_context=ctx
        ) as ex:
            futures = {
                ex.submit(
                    _run_trial,
                    self.model_factory,
                    state.params,
                    self.train_dataloader_factory,
                    self.val_dataloader_factory,
                    target_epoch,
                    state.checkpoint_path,
                    str(self.work_dir),
                    state.trial_id,
                    dict(self.trainer_kwargs),
                    (self.seed + state.trial_id) if self.seed is not None else None,
                ): state
                for state in states
            }
            for fut in as_completed(futures):
                state = futures[fut]
                try:
                    out = fut.result()
                except Exception as exc:
                    if self.raise_on_failed_trial:
                        raise
                    results.append(self._failed_result(state, exc))
                    continue
                results.append(self._completed_result(state, out))
        return results

    # ---- result helpers ----

    def _completed_result(self, state: _TrialState, out: Mapping[str, Any]) -> TrialResult:
        metrics = out["metrics"]
        score = self._score(metrics)
        history = list(state.score_history) + [score]
        return TrialResult(
            trial_id=state.trial_id,
            params=dict(state.params),
            metrics=dict(metrics),
            score=score,
            epochs_completed=int(out["epochs_completed"]),
            score_history=history,
            status="completed",
            elapsed_seconds=time.perf_counter() - state.started,
            checkpoint_path=out.get("checkpoint_path"),
        )

    def _failed_result(self, state: _TrialState, error: BaseException) -> TrialResult:
        return TrialResult(
            trial_id=state.trial_id,
            params=dict(state.params),
            metrics=state.last_metrics,
            score=self._fallback_score(),
            epochs_completed=state.epochs_completed,
            score_history=list(state.score_history),
            status="failed",
            elapsed_seconds=time.perf_counter() - state.started,
            checkpoint_path=state.checkpoint_path,
            error=f"{type(error).__name__}: {error}",
        )

    def _score(self, metrics: Mapping[str, float]) -> float:
        if self.objective is not None:
            return float(self.objective(metrics))
        for key in ("val_loss", "val_loss_epoch", "train_loss", "train_loss_epoch"):
            if key in metrics:
                return float(metrics[key])
        for v in metrics.values():
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        raise RuntimeError(
            "could not infer a numeric objective from trainer.callback_metrics; "
            "pass objective=... explicitly"
        )

    def _is_better(self, candidate: TrialResult, incumbent: TrialResult) -> bool:
        if self.mode is OptimizationMode.MINIMIZE:
            return candidate.score < incumbent.score
        return candidate.score > incumbent.score

    def _fallback_score(self) -> float:
        return math.inf if self.mode is OptimizationMode.MINIMIZE else -math.inf

    # ---- search-space iteration ----

    def _iter_param_sets(self, rng: random.Random) -> Iterable[dict[str, Any]]:
        if self.strategy is SearchStrategy.GRID:
            yield from self._iter_grid_param_sets(self.max_trials)
            return
        if self.max_trials is None:
            raise ValueError(f"{self.strategy.value} search requires max_trials")
        yield from self._iter_random_param_sets(rng, self.max_trials)

    def _iter_grid_param_sets(self, limit: Optional[int]) -> Iterable[dict[str, Any]]:
        names = list(self.search_space.keys())
        value_sets = [self.search_space[n].grid_values() for n in names]
        combos = (dict(zip(names, values)) for values in itertools.product(*value_sets))
        if limit is None:
            yield from combos
        else:
            yield from itertools.islice(combos, limit)

    def _iter_random_param_sets(
        self, rng: random.Random, max_trials: int,
    ) -> Iterable[dict[str, Any]]:
        for _ in range(max_trials):
            yield {n: p.sample(rng) for n, p in self.search_space.items()}

    def _rung_milestones(self) -> list[int]:
        milestones: list[int] = []
        budget = self.min_epochs
        while budget < self.epochs:
            milestones.append(budget)
            budget *= self.reduction_factor
        if not milestones or milestones[-1] != self.epochs:
            milestones.append(self.epochs)
        return sorted(set(milestones))


def _with_status(r: TrialResult, status: str) -> TrialResult:
    return TrialResult(
        trial_id=r.trial_id,
        params=dict(r.params),
        metrics=r.metrics,
        score=r.score,
        epochs_completed=r.epochs_completed,
        score_history=list(r.score_history),
        status=status,
        elapsed_seconds=r.elapsed_seconds,
        checkpoint_path=r.checkpoint_path,
        error=r.error,
    )


__all__ = [
    "ChoiceParam",
    "FloatRange",
    "HyperparameterOptimizer",
    "IntRange",
    "OptimizationMode",
    "OptimizationResult",
    "SearchStrategy",
    "TrialResult",
]
