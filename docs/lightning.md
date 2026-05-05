# Lightning Integration

MLTK is a thin set of building blocks on top of PyTorch Lightning.
`AbstractModel` is a `LightningModule`, so it composes with
`lightning.Trainer`, callbacks, loggers, accelerators, profilers, and the
standard Lightning checkpoint mechanism.

## Model Style

Subclass `AbstractModel`, set `self.criterion` and `self.optimizer`,
implement `forward`. Lightning's `training_step` is wired to MLTK's
`compute_loss` by default — supervised `(inputs, targets)` batches just
work.

```python
import torch
import torch.nn as nn
from mltk import AbstractModel

class Regressor(AbstractModel):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Linear(4, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        return self.head(x)
```

`save_hyperparameters()` (Lightning) records constructor arguments so
`Regressor.load_from_checkpoint(path)` reconstructs the model with the
right shape.

## Training

```python
import lightning as L

trainer = L.Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model, train_loader, val_loader)
```

For non-standard batch shapes (multi-output, contrastive, custom
unpacking) override `compute_loss`; reach for `training_step` only when
the whole Lightning step needs custom logic.

## Device And Dtype

Use Lightning/PyTorch semantics — `model.to(device=...)` or let the
trainer place the model via `accelerator="auto"` /
`accelerator="gpu"`. Pick a device with `auto_device()` from
`mltk.devices` if you want to detect CUDA / ROCm / DirectML / MPS / CPU
in one call.

## Schedulers

MLTK schedulers expose `interval` (and `monitor` where relevant) so the
default `configure_optimizers` can build a Lightning scheduler config
without per-class branches. `SmoothReduceLROnPlateau` uses
`interval="epoch"` + `monitor="val_loss"`; `CosineWarmupScheduler` uses
`interval="step"`. Set both fields on any custom scheduler before assigning
it to `self.scheduler`.

## Checkpointing

Lightning's `Trainer.save_checkpoint(path)` and
`Cls.load_from_checkpoint(path)` are the canonical save/load. Models built
on `SuperModel` automatically round-trip the canonical-key list of cached
heads via `on_save_checkpoint` / `on_load_checkpoint`.

## Inference

Plain `model(x)` calls `forward` directly. Use `model.predict(x)` for
MLTK's eval + `no_grad` + `self.normalize` + `forward` shorthand. After
`predict()` the model is left in eval mode — call `model.train()` to
resume training.

## Recipe: frozen-backbone, augmented features, disk cache

The canonical pattern when training a small head over a large frozen
backbone:

1. **Augment in the dataset.** Train mode applies the augment; eval mode
   skips it. Use `STANDARD_AUGMENT` (or `LIGHT_/STRONG_AUGMENT`, or pass
   any callable).
2. **Replicate** the source `n` times so each replica index draws an
   independent random augment. With `n=4`, each source image produces 4
   distinct augmented views.
3. **Map** every replicated item through the frozen backbone to extract
   features.
4. **DiskCache** the result. After the first epoch, training reads
   features from mmap-backed disk; the backbone never runs again.
5. **Train** a small head Lightning module on the cached features.

```python
import lightning as L
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mltk import (
    AbstractModel,
    AbstractImageDataset,
    DiskCachedDataset,
    MapDataset,
    ReplicatedDataset,
    STANDARD_AUGMENT,
    auto_device,
)

device = auto_device()

# 1) image dataset with augmentation
class MyImages(AbstractImageDataset):
    def __init__(self, paths, labels):
        super().__init__(size=(224, 224), train=True, augment=STANDARD_AUGMENT)
        self.paths, self.labels = paths, labels

    def __len__(self): return len(self.paths)

    def load_raw(self, idx):
        from PIL import Image
        return Image.open(self.paths[idx]).convert("RGB"), self.labels[idx]


images = MyImages(train_paths, train_labels)

# 2) repeat 4× — each replica draws its own augment
replicated = ReplicatedDataset(images, n=4)

# 3) feature extraction through the frozen backbone
backbone = load_my_backbone(device)  # returns a frozen module + normalize fn
def to_features(img, lbl):
    img = backbone.normalize(img.to(device))
    feats = backbone.forward(img.unsqueeze(0))   # [1, feat_dim]
    return feats.squeeze(0).detach().cpu(), lbl

mapped = MapDataset(replicated, fn=to_features)

# 4) on-disk mmap cache (chunked) — fills lazily on first read, or eagerly:
cached = DiskCachedDataset(mapped, cache_dir="cache/my_features", chunk_size=512)
cached.precompute(verbose=True)   # optional eager fill of the whole cache

# 5) head-only Lightning module trained on the cached features
class HeadModel(AbstractModel):
    def __init__(self, feat_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU(), nn.Linear(256, 1))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(self, features):
        return self.head(features).squeeze(-1)


loader = DataLoader(cached, batch_size=64, shuffle=True)
trainer = L.Trainer(max_epochs=20, accelerator="auto")
trainer.fit(HeadModel(feat_dim=backbone.feat_dim), loader, val_loader)
```

Key properties of this pipeline:

* The backbone runs **once per replica**, ever — first-epoch fill writes
  to the cache, subsequent epochs are pure mmap reads.
* `n` augmented views are *frozen* into the cache the first time each
  replica index is touched, so subsequent epochs see the same `n` views
  per source image (deterministic but augmented).
* The OS page cache transparently shares cache files across worker
  processes, so the same on-disk cache is fast for parallel
  `HyperparameterOptimizer` trials too.
* Inference still goes through the full image → backbone → head model,
  loading the head weights from the head-only checkpoint with
  `strict=False` so the backbone stays freshly fetched.
