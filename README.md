# MLTK — Machine Learning Toolkit

Small PyTorch toolkit extracted from the DoxAI geolocation project. The parts
that aren't specific to any one domain live here so they can be reused without
pulling in 5 GB of image data and a hierarchical Haversine loss.

## What's inside

| Concept | Module | Notes |
|---|---|---|
| Base classifier lifecycle | `mltk.base_classifier.AbstractClassifier` | Save / load, optimizer + scheduler plumbing, atomic save |
| Super model (dispatch over many heads) | `mltk.supermodel.SuperModel` | Heads created lazily by key, shared trunk optional |
| Hierarchical dispatch + beam search inference | `mltk.hierarchic_inference.HierarchicInference` | Teacher-forced descent at train time, beam-search descent at eval |
| Hierarchy metadata + per-level sampler | `mltk.hierarchic_dataset` | `HierarchyInformation`, `HierarchicDataset`, `PerLevelSampler` |
| Blocks | `mltk.skipattnmlp.SkipAttentionMLP`, `mltk.feature_perspective.FeaturePerspective`, `mltk.mlp_blocks.*` | Attention-augmented MLP, SwiGLU MLP block, DropPath, LayerScale |
| Heads | `mltk.classifier_heads.CosineClassifier` | Cosine-similarity classifier with learnable temperature |
| Losses | `mltk.kldivlosssoftmax.KLDivLossWithSoftmax` | KL divergence over soft targets |
| Schedulers | `mltk.schedulers.SmoothReduceLROnPlateau` | Smoothed plateau-LR |
| Samplers | `mltk.samplers.create_sqrt_sampler`, `mltk.smart_samplers` | sqrt-weight sampler for imbalanced label distributions |
| Frozen vision backbones | `mltk.backbones.load_backbone` | SigLIP / SigLIP2 (+ NaFlex) / CLIP / DINOv2 / DINOv3 / generic — one call returns a `BackboneSpec` with a forward fn, preprocessing mean/std, and feat dim |

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
in the core building blocks surface fast:

- **Iris** — a `SkipAttentionMLP` trained by a minimal `AbstractClassifier` subclass
  to >90 % test accuracy in ~5 seconds on CPU.
- **MNIST** — a `SuperModel` hierarchical classifier (parity / digit two-level
  tree) trained on a small MNIST slice.

## License

MIT.
