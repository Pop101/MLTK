# MLTK

Small PyTorch toolkit of building blocks on top of [PyTorch
Lightning](https://lightning.ai/). `AbstractModel` is a `LightningModule`;
the rest of the toolkit (datasets, schedulers, `SuperModel`,
hyperparameter search) sits on top.

Train with `lightning.Trainer`. Restore with `cls.load_from_checkpoint`.

## Layout

```
mltk/
├── learning/          # Lightning lifecycle base, losses, schedulers,
│                      #   samplers, hyperparameter search
├── building_blocks/   # reusable nn.Module primitives: MLP blocks,
│                      #   attention-augmented MLP, classifier heads,
│                      #   gradient-checkpointed Sequential
├── models/            # composed architectures: SuperModel (shared trunk
│                      #   + lazy heads), ModelFactory protocol
├── data/              # AbstractDataset / AbstractImageDataset, lazy
│                      #   MapDataset, ReplicatedDataset, mmap
│                      #   DiskCachedDataset, DeviceLRUCache
├── hierarchical/      # tree-structured dispatch: HierarchyInformation,
│                      #   HierarchicDataset, beam-search inference
└── devices.py         # cross-platform accelerator selection
                       #   (CUDA / ROCm / DirectML / MPS / CPU)
```

Every important symbol is also re-exported at the top level (`from mltk
import …`).

| Where | Symbol | Notes |
|---|---|---|
| `learning` | `AbstractModel` | Lightning module + `self.normalize` + default `compute_loss` + `predict` |
| `learning` | `HyperparameterOptimizer`, `ChoiceParam`, `FloatRange`, `IntRange`, `SearchStrategy`, `OptimizationMode` | Trainer-driven HPO with concurrent trials and successive halving |
| `learning` | `SmoothReduceLROnPlateau`, `CosineWarmupScheduler`, `EMAWeightTracker` | Schedulers + EMA |
| `learning` | `KLDivLossWithSoftmax` | KL divergence over soft targets |
| `learning` | `AdaptiveFrequencySampler`, `LossAwareSampler`, `make_weighted_sampler`, `LINEAR_TARGET`/`SQRT_TARGET`/`LOG_TARGET`/`UNIFORM_TARGET` | Class-rebalancing samplers (static + feedback-driven) |
| `building_blocks` | `SkipAttentionMLP`, `FeaturePerspective` | Attention-augmented blocks |
| `building_blocks` | `ModernMLPBlock`, `SwiGLU`, `DropPath`, `LayerScale` | Modern MLP primitives |
| `building_blocks` | `CosineClassifier`, `ArcFaceHead`, `SubCenterArcFaceHead` | Cosine + margin-based heads |
| `building_blocks` | `CheckpointedSequential` | `nn.Sequential` with per-layer gradient checkpointing |
| `models` | `SuperModel` | Shared trunk + lazy per-key heads in an `nn.ModuleDict` |
| `models` | `ModelFactory` | Optional typed-protocol for head factories |
| `data` | `AbstractDataset`, `AbstractImageDataset` | Train/eval mode toggle; image base bundles pad+resize+augment |
| `data` | `LIGHT_AUGMENT`, `STANDARD_AUGMENT`, `STRONG_AUGMENT` | First-class `transforms.Compose` augmentation presets |
| `data` | `MapDataset`, `ReplicatedDataset`, `DiskCachedDataset`, `DeviceLRUCache` | Lazy fn map, n-replication, on-disk mmap+chunk cache, device-aware LRU |
| `hierarchical` | `HierarchyInformation`, `HierarchicDataset`, `PerLevelSampler` | Tree metadata + per-level sampler |
| `hierarchical` | `HierarchicInference`, `BeamCandidate` | Teacher-forced descent at train, beam-search at inference |
| `devices` | `auto_device`, `device_label`, `is_rocm`, `synchronize`, `empty_cache` | Cross-backend accelerator helpers |

## Install

```bash
pip install -e .              # core
pip install -e '.[test]'      # + sklearn + torchvision for tests
pip install -e '.[docs]'      # + sphinx + furo for the docs
```

## Docs

```bash
pip install -e '.[docs]'
sphinx-build -b html docs docs/_build/html
```

The HTML lands in `docs/_build/html/index.html`. The build pulls
docstrings via `sphinx.ext.autodoc` + `napoleon`; module-level and class
docstrings are the source of truth.

## Quick start

### 1. A trainable model

```python
import torch
import torch.nn as nn
import lightning as L
from mltk import AbstractModel

class Regressor(AbstractModel):
    def __init__(self, in_dim, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.head = nn.Sequential(nn.Linear(in_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def forward(self, x):
        return self.head(x).squeeze(-1)

model = Regressor(in_dim=768)
trainer = L.Trainer(max_epochs=10, accelerator="auto")
trainer.fit(model, train_loader, val_loader)
```

`save_hyperparameters()` records constructor args so
`Regressor.load_from_checkpoint(path)` restores the model with the right
shape. `model.predict(x)` is MLTK's eval + `no_grad` + `self.normalize` +
`forward` shorthand.

### 2. Frozen-backbone, augmented features, disk cache

The canonical pipeline when training a small head over a large frozen
backbone (see `docs/lightning.md` for the full version):

```python
from mltk import (
    AbstractImageDataset, DiskCachedDataset, MapDataset,
    ReplicatedDataset, STANDARD_AUGMENT,
)

class MyImages(AbstractImageDataset):
    def __init__(self, paths, labels):
        super().__init__(size=(224, 224), train=True, augment=STANDARD_AUGMENT)
        self.paths, self.labels = paths, labels
    def __len__(self): return len(self.paths)
    def load_raw(self, idx):
        from PIL import Image
        return Image.open(self.paths[idx]).convert("RGB"), self.labels[idx]

images     = MyImages(train_paths, train_labels)
replicated = ReplicatedDataset(images, n=4)        # 4 augmented views per item
mapped     = MapDataset(replicated, fn=to_features) # frozen backbone
cached     = DiskCachedDataset(mapped, cache_dir="cache/feats")
cached.precompute(verbose=True)                     # eager fill
```

The first epoch fills the on-disk mmap cache; every subsequent epoch
reads features straight from disk.

### 3. Hyperparameter search (parallel + halving)

```python
from mltk import (
    ChoiceParam, FloatRange, HyperparameterOptimizer,
    OptimizationMode, SearchStrategy,
)

opt = HyperparameterOptimizer(
    model_factory=lambda p: Regressor(in_dim=768, **p),
    search_space={"lr": FloatRange(1e-5, 1e-2, log=True)},
    train_dataloader_factory=lambda: build_train_loader(),
    val_dataloader_factory=lambda: build_val_loader(),
    epochs=20,
    min_epochs=2,
    reduction_factor=3,
    strategy=SearchStrategy.SUCCESSIVE_HALVING,
    max_trials=24,
    max_concurrent_trials=4,        # 4 trials × Trainer.fit at once
    mode=OptimizationMode.MINIMIZE,
)
result = opt.optimize()
print(result.best_params, result.best_score)
```

`max_concurrent_trials > 1` runs each `Trainer.fit` in its own
subprocess (spawn context — safe with CUDA). Survivors of each rung
resume from their previous Lightning checkpoint instead of restarting.

### 4. Hierarchical classifier over a tree

See `HierarchicInference` + `HierarchyInformation`. One head per internal
node; output dim derived from the hierarchy (number of children, or leaf
size at a leaf). Descent works teacher-forced at training and
beam-search at inference.

## Tests

```bash
pytest                         # unit + integration (CPU)
pytest -m "not integration"    # unit only (~10s)
pytest -m integration          # iris + mnist
```

The integration suite trains two real models so regressions surface
immediately:

- **Iris** — `SkipAttentionMLP` + `SmoothReduceLROnPlateau` through an
  `AbstractModel` subclass. **Must clear 97%** on a separable split.
- **MNIST** — `SuperModel` parity/digit hierarchy + `SkipAttentionMLP`
  trunk + `SmoothReduceLROnPlateau`. **Must clear 95%** on a 20K/2K
  subset.
