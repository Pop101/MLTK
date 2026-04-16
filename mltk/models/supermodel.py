"""Shared trunk + lazy-cached heads ("supermodel") pattern.

Generic, task-agnostic abstraction for "one shared feature producer + many
classifier heads keyed by arbitrary hashables". The hierarchical geo
classifier is one specialization (keys = LevelPath tuples); a multi-task
classifier with one head per task id would be another.

Design:
    - Subclasses implement `_create_head(key) -> nn.Module`.
    - `get_head(key)` caches the result. Heads are created on first use.
    - `send_to_device`, `compile`, and save/load propagate to the trunk +
      every cached head.

TODO(ml-library audit): heads are created on GPU and never evicted. For
bounded key spaces this is fine; for unbounded ones a future version
should expose an LRU/CPU-offload policy.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Hashable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from mltk.learning.base_classifier import AbstractClassifier


class SuperModel(AbstractClassifier):
    """Shared trunk + lazy per-key heads. Subclass and implement `_create_head`."""

    def __init__(
        self,
        *,
        input_dim: int,
        trunk: Optional[nn.Module] = None,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__(device=device, dtype=dtype)
        self._input_dim = int(input_dim)
        self.trunk: nn.Module = trunk if trunk is not None else nn.Identity()
        self._heads: Dict[Hashable, nn.Module] = {}
        self._compile_args: Optional[dict] = None
        self.init_params: Dict[str, Any] = {"input_dim": self._input_dim}

        if device is not None:
            self.trunk = self.trunk.to(device=device, dtype=dtype)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run the trunk. Override seam for test-time transforms."""
        return self.trunk(x)

    # ---- head management ----

    def _canonical_key(self, key: Hashable) -> Hashable:
        """Override to canonicalize keys (e.g. tuple coercion)."""
        return key

    @abstractmethod
    def _build_head(self, key: Hashable) -> nn.Module:
        """Build a fresh, unwrapped head for `key`. Subclass implementation."""
        ...

    def _create_head(self, key: Hashable) -> nn.Module:
        """Build a head, move to device/dtype, apply any pending `torch.compile`.

        Subclasses override `_build_head` for the actual construction; this
        method wraps the result with device + compile plumbing that all
        subclasses share.
        """
        head = self._build_head(key)
        if self.device is not None:
            head = head.to(device=self.device, dtype=self.dtype)
        if self._compile_args is not None:
            head = torch.compile(head, **self._compile_args)
        return head

    def get_head(self, key: Hashable) -> nn.Module:
        """Return the cached head for `key`, creating it lazily on miss."""
        canonical = self._canonical_key(key)
        cached = self._heads.get(canonical)
        if cached is not None:
            return cached
        head = self._create_head(canonical)
        self._heads[canonical] = head
        return head

    def precreate_heads(self, keys: Iterable[Hashable]) -> None:
        """Eagerly instantiate heads for every key in `keys`. Useful for
        deterministic warmup or ensuring a checkpoint covers every key."""
        for key in keys:
            self.get_head(key)

    # ---- device / compile plumbing ----

    @staticmethod
    def _unwrap(module: nn.Module) -> nn.Module:
        return getattr(module, "_orig_mod", module)

    def send_to_device(self, device, dtype=None) -> None:
        super().send_to_device(device, dtype)
        self.trunk = self.trunk.to(device=device, dtype=self.dtype)
        for key, head in list(self._heads.items()):
            self._heads[key] = head.to(device=device, dtype=self.dtype)

    def compile(self, **kwargs) -> None:
        self._compile_args = dict(kwargs)
        if not isinstance(self.trunk, nn.Identity):
            self.trunk = torch.compile(self.trunk, **kwargs)
        for key, head in list(self._heads.items()):
            self._heads[key] = torch.compile(head, **kwargs)

    # ---- save / load ----

    def _get_additional_save_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "head_state_dicts": {k: self._unwrap(h).state_dict() for k, h in self._heads.items()}
        }
        if not isinstance(self.trunk, nn.Identity):
            state["trunk_state_dict"] = self._unwrap(self.trunk).state_dict()
        return state

    def _load_additional_state(self, checkpoint: Dict[str, Any]) -> None:
        # Re-create heads in insertion order so optimizer param groups line up.
        for key, head_state in (checkpoint.get("head_state_dicts") or {}).items():
            self._unwrap(self.get_head(key)).load_state_dict(head_state)
        trunk_state = checkpoint.get("trunk_state_dict")
        if trunk_state is not None and not isinstance(self.trunk, nn.Identity):
            self._unwrap(self.trunk).load_state_dict(trunk_state)

    def get_model_size(self) -> int:
        total = 0
        for module in (self.trunk, *self._heads.values()):
            unwrapped = self._unwrap(module)
            for p in unwrapped.parameters():
                total += p.numel() * p.element_size()
            for b in unwrapped.buffers():
                total += b.numel() * b.element_size()
        return total
