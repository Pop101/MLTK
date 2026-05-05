"""Lightning-backed lifecycle base for trainable MLTK models.

MLTK is building blocks on top of Lightning. ``AbstractModel`` is a
``LightningModule`` that adds a small set of conveniences:

- ``self.normalize`` — model-owned normalization callable (typically a
  ``transforms.Normalize`` with backbone-specific mean/std). Datasets feed
  un-normalized tensors; the model normalizes inside ``compute_loss`` and
  ``predict`` right before ``forward``.
- ``compute_loss(batch)`` — default supervised ``(inputs, targets)`` loss
  that runs ``self.normalize`` then ``self.forward`` then ``self.criterion``.
  Override for non-standard loss shapes; Lightning's ``training_step`` calls
  it directly.
- ``predict(inputs)`` — eval + ``no_grad`` + ``self.normalize`` + ``forward``.
  This is what plain ``model(inputs)`` should be at inference time.

Everything else — train loops, device transfer, checkpointing, optimizer
configuration — comes from Lightning. Train with ``lightning.Trainer``;
restore with ``cls.load_from_checkpoint``.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import lightning as L


class AbstractModel(L.LightningModule, ABC):
    """Lightning module + ``self.normalize`` + ``predict`` + default ``compute_loss``.

    Subclasses set ``self.criterion`` (an ``nn.Module``) and implement
    ``forward``. ``configure_optimizers`` is required by Lightning;
    subclasses can override it directly or set ``self.optimizer`` /
    ``self.scheduler`` and use the default implementation here.
    """

    def __init__(self):
        super().__init__()
        self.criterion: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.normalize: Optional[Callable] = None
        self.grad_clip: Optional[float] = 1.0

    # ------------------------------------------------------------------ #
    # Lightning hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the model on a (batched) input. Inputs are assumed normalized
        — ``compute_loss`` and ``predict`` normalize before calling here."""

    def training_step(self, batch, batch_idx):
        del batch_idx
        loss = self.compute_loss(batch)
        self.log("train_loss", loss.detach(), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        del batch_idx
        loss = self.compute_loss(batch)
        self.log("val_loss", loss.detach(), prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """Default: return ``self.optimizer`` (and ``self.scheduler`` if set).

        Override for multi-optimizer or fully-custom schedules.
        """
        if self.optimizer is None:
            raise RuntimeError(
                f"{type(self).__name__} must set self.optimizer or override "
                "configure_optimizers()."
            )
        if self.scheduler is None:
            return self.optimizer
        return {"optimizer": self.optimizer, "lr_scheduler": self._scheduler_config()}

    def _scheduler_config(self) -> dict:
        """Build a Lightning scheduler config. Subclasses can override."""
        config: dict[str, Any] = {"scheduler": self.scheduler}
        interval = getattr(self.scheduler, "interval", None)
        if interval is not None:
            config["interval"] = interval
        monitor = getattr(self.scheduler, "monitor", None)
        if monitor is not None:
            config["monitor"] = monitor
            config["reduce_on_plateau"] = True
        return config

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        clip = gradient_clip_val if gradient_clip_val is not None else self.grad_clip
        if clip is None:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=clip,
            gradient_clip_algorithm=gradient_clip_algorithm or "norm",
        )

    # ------------------------------------------------------------------ #
    # Default supervised loss path. Override for non-standard batch shapes.
    # ------------------------------------------------------------------ #
    def compute_loss(self, batch) -> torch.Tensor:
        """Compute loss for ``(inputs, targets)`` supervised batch.

        Pipeline: device move → ``self.normalize`` → ``self.forward`` →
        ``self.criterion``. Output is reshaped to match ``targets`` if it
        only differs by trailing singleton dims.
        """
        if self.criterion is None:
            raise RuntimeError(
                f"{type(self).__name__} must set self.criterion or override "
                "compute_loss()."
            )
        inputs, targets = batch
        inputs = inputs.to(device=self.device, dtype=self.dtype)
        targets = targets.to(device=self.device)
        if self.normalize is not None:
            inputs = self.normalize(inputs)
        outputs = self.forward(inputs)
        if outputs.shape != targets.shape and outputs.numel() == targets.numel():
            outputs = outputs.view(targets.shape)
        return self.criterion(outputs, targets)

    # ------------------------------------------------------------------ #
    # Inference shorthand: eval + no_grad + normalize + forward.
    # Leaves the model in eval mode — call ``self.train()`` explicitly to
    # resume training afterwards.
    # ------------------------------------------------------------------ #
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            inputs = inputs.to(device=self.device, dtype=self.dtype)
            if self.normalize is not None:
                inputs = self.normalize(inputs)
            return self.forward(inputs)
