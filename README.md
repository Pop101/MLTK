# MLTK — Machine Learning Toolkit

Small PyTorch toolkit extracted from the DoxAI geolocation project. The parts
that aren't specific to any one domain live here so they can be reused without
pulling in 5 GB of image data and a hierarchical Haversine loss.

## Layout

```
mltk/
├── learning/          # training lifecycle: classifier base, losses,
│                      #   schedulers, samplers
├── building_blocks/   # reusable nn.Module primitives: MLP blocks,
│                      #   attention-augmented MLP, classifier heads,
│                      #   gradient-checkpointed Sequential
├── models/            # composed architectures: SuperModel, model
│                      #   factories, frozen vision backbone loader
└── hierarchical/      # tree-structured dispatch: HierarchyInformation,
                       #   HierarchicDataset, beam-search inference
```

Every important symbol is also re-exported at the top level (`from mltk import …`).

| Where | Symbol | Notes |
|---|---|---|
| `learning` | `AbstractClassifier` | Save/load, optimizer + scheduler plumbing, atomic save |
| `learning` | `SmoothReduceLROnPlateau` | Plateau-aware LR schedule that smooths the loss history before deciding to drop |
| `learning` | `KLDivLossWithSoftmax` | KL divergence over soft targets |
| `learning` | `create_sqrt_sampler` | sqrt-weight sampler for imbalanced label distributions |
| `building_blocks` | `SkipAttentionMLP` | Attention-augmented MLP with skip connections |
| `building_blocks` | `FeaturePerspective` | Multi-activation feature transform |
| `building_blocks` | `ModernMLPBlock`, `SwiGLU`, `DropPath`, `LayerScale` | Modern MLP primitives |
| `building_blocks` | `CosineClassifier` | Cosine-similarity classifier head |
| `building_blocks` | `CheckpointedSequential` | `nn.Sequential` with per-layer gradient checkpointing |
| `models` | `SuperModel` | Shared trunk + lazy per-key heads. Heads are created on first `get_head(key)` |
| `models` | `ModelFactory`, `SkipAttentionMLPFactory` | Factories for per-level head construction |
| `models` | `BackboneSpec`, `load_backbone` | SigLIP / SigLIP2 (+ NaFlex) / CLIP / DINOv2 / DINOv3 / generic — one call returns forward fn + mean/std + feat dim |
| `hierarchical` | `HierarchyInformation`, `HierarchicDataset`, `PerLevelSampler` | Tree metadata, per-level sampler that descends the tree |
| `hierarchical` | `HierarchicInference`, `BeamCandidate` | Teacher-forced descent at train time, beam-search descent at eval |

## Install

```bash
pip install -e .              # core
pip install -e '.[hf]'        # + huggingface backbones
pip install -e '.[test]'      # + test dependencies (sklearn, torchvision)
```

## Quick start

### 1. Frozen-feature classifier

```python
import torch, torch.nn as nn
from mltk import AbstractClassifier

class SimpleClassifier(AbstractClassifier):
    def __init__(self, in_dim: int, num_classes: int, lr: float = 1e-3, device=None, dtype=torch.float32):
        super().__init__(device=device, dtype=dtype)
        self.head = nn.Linear(in_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.head.parameters(), lr=lr)
        self.init_params = {"in_dim": in_dim, "num_classes": num_classes, "lr": lr}

    def train_batch(self, batch, transforms=None):
        x, y = batch
        logits = self.head(x)
        loss = self.criterion(logits, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.total_batches_trained += 1
        return float(loss.item())

    def evaluate(self, data_loader, transforms=None):
        correct = total = 0
        self.head.eval()
        with torch.no_grad():
            for x, y in data_loader:
                correct += (self.head(x).argmax(-1) == y).sum().item()
                total += y.numel()
        return 0.0, correct / max(1, total)

    def predict(self, image):
        return self.head(image)
```

### 2. Hierarchical classifier over a tree

See `HierarchicInference` + `HierarchyInformation`. Head per internal node; output
dim at each node is `len(children)` (or `leaf_size` at the leaf). Descent works
teacher-forced during training and beam-search at inference time.

### 3. Load a frozen vision backbone

```python
from mltk.backbones import load_backbone
import torch

spec = load_backbone("google/siglip2-so400m-patch16-naflex",
                     device=torch.device("cuda"),
                     dtype=torch.float16,
                     max_num_patches=1024)

# spec.forward takes a dict for NaFlex models, a tensor for fixed-res ones.
# spec.feat_dim is the real output dim, probed empirically.
```

## Tests

```bash
pytest                         # unit + integration (CPU only)
pytest -m "not integration"    # unit only
pytest -m integration          # slow tests (iris + mnist)
```

The integration suite trains two real models on common benchmarks so regressions
in the core building blocks surface immediately:

- **Iris** — `SkipAttentionMLP` + `SmoothReduceLROnPlateau` through an `AbstractClassifier` subclass. **Must clear 97%** on a separable split (seed=0 / 20% test) where SVC ceilings at 100%. If it misses, the stack is broken.
- **MNIST** — a `SuperModel` 2-level hierarchy (root → parity → digit) with a `SkipAttentionMLP` trunk and `SmoothReduceLROnPlateau`. **Must clear 95%** on a 20K/2K subset. A plain MLP hits ~97% here; anything below 95% means the scheduler, the head dispatch, or block composition is underperforming.

Both tests also roundtrip the checkpoint through `save` / `load` and assert bit-exact evaluation after reload.

## License

MIT.
