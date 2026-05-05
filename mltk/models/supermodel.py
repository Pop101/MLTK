"""Shared trunk + lazy-cached heads ("supermodel") pattern.

Generic, task-agnostic abstraction for "one shared feature producer + many
heads keyed by arbitrary hashables". The hierarchical geo classifier is
one specialization (keys = LevelPath tuples); a multi-task classifier with
one head per task id would be another.

No trunk is required: omit ``trunk`` and inputs are passed directly to heads.

Heads live in an ``nn.ModuleDict`` keyed by a stringified canonical key.
Lightning/PyTorch see them as children, so ``parameters()``, ``.to(...)``,
``.train()``/``.eval()``, and ``state_dict()`` work for free.

Subclass ``_build_head(key)`` to construct heads on demand. ``_canonical_key``
is the override seam for normalising arbitrary key types (e.g. tuple
coercion). The ``str`` form is what ``ModuleDict`` actually stores; the
canonical (untyped) key is preserved for hierarchy queries via
``self.head_keys``.
"""
from __future__ import annotations

from typing import Callable, Hashable, Iterable, Optional

import torch
import torch.nn as nn

from mltk.learning.abstract_model import AbstractModel


def _key_to_str(key: Hashable) -> str:
    """Stringify a canonical key for ``nn.ModuleDict`` storage.

    ``ModuleDict`` requires string keys. ``repr`` is deterministic for tuples,
    ints, strings, frozensets — i.e. the things people use as routing keys.
    """
    return repr(key)


class SuperModel(AbstractModel):
    """Shared trunk + lazy per-key heads.

    Use this when one trunk feeds many heads selected by an arbitrary key
    (task id, hierarchy node, language, domain, etc.). The class is still
    abstract through ``AbstractModel.forward``; subclasses define
    task-specific routing.
    """

    def __init__(
        self,
        *,
        input_dim: Optional[int] = None,
        trunk: Optional[nn.Module] = None,
        head_factory: Optional[Callable[[Hashable], nn.Module]] = None,
    ):
        super().__init__()
        if input_dim is None and trunk is None:
            raise ValueError(
                "SuperModel needs input_dim (when trunk is None or Identity) "
                "so callers can introspect the head input width."
            )
        self._input_dim = int(input_dim) if input_dim is not None else None
        self.trunk: nn.Module = trunk if trunk is not None else nn.Identity()
        self._head_factory = head_factory
        self.heads = nn.ModuleDict()
        # canonical (untyped) keys, in insertion order, parallel to self.heads
        self._head_keys: list[Hashable] = []

    @property
    def input_dim(self) -> Optional[int]:
        return self._input_dim

    @property
    def head_keys(self) -> tuple[Hashable, ...]:
        """Canonical keys of every cached head, in insertion order."""
        return tuple(self._head_keys)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)

    # ---- head management ----

    def _canonical_key(self, key: Hashable) -> Hashable:
        """Override to canonicalize keys (e.g. tuple coercion)."""
        return key

    def _build_head(self, key: Hashable) -> nn.Module:
        """Build a fresh head for ``key``. Default: call ``head_factory(key)``.

        Override for key-derived dimensions (e.g. hierarchy children counts).
        """
        if self._head_factory is None:
            raise NotImplementedError(
                f"{type(self).__name__} must pass head_factory or override _build_head"
            )
        return self._head_factory(key)

    def _on_head_created(self, key: Hashable, head: nn.Module) -> None:
        """Hook for attaching lazy heads to optimizers, schedulers, or logs.

        Default: no-op. Override when you need to register newly-created
        heads with ``self.optimizer`` or wire them up to per-head metrics.
        """
        del key, head

    def get_head(self, key: Hashable) -> nn.Module:
        """Return the cached head for ``key``, creating it lazily on miss."""
        canonical = self._canonical_key(key)
        sk = _key_to_str(canonical)
        if sk in self.heads:
            return self.heads[sk]
        head = self._build_head(canonical)
        head = head.to(device=self.device, dtype=self.dtype)
        self.heads[sk] = head
        self._head_keys.append(canonical)
        self._on_head_created(canonical, head)
        return head

    def precreate_heads(self, keys: Iterable[Hashable]) -> None:
        """Eagerly instantiate heads for every key in ``keys``."""
        for key in keys:
            self.get_head(key)

    # ---- checkpoint hooks: persist canonical keys so ``head_keys`` round-trips ----

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["mltk_head_keys"] = list(self._head_keys)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        # ``state_dict`` already restored the head modules themselves; we just
        # need to recover the canonical-key list parallel to self.heads.
        keys = checkpoint.get("mltk_head_keys")
        if keys is not None:
            self._head_keys = list(keys)
