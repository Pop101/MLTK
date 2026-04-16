"""Shape + autograd sanity for the reusable blocks."""
import torch

from mltk import SkipAttentionMLP, FeaturePerspective, CosineClassifier, CheckpointedSequential


def test_skipattnmlp_forward_shape():
    mlp = SkipAttentionMLP(in_features=16, out_features=32, depth=2)
    x = torch.randn(4, 16)
    y = mlp(x)
    assert y.shape == (4, 32)


def test_skipattnmlp_backward_is_trainable():
    mlp = SkipAttentionMLP(in_features=8, out_features=8, depth=1)
    x = torch.randn(2, 8, requires_grad=False)
    y = mlp(x)
    y.sum().backward()
    grads = [p.grad for p in mlp.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), "no gradient flow"


def test_feature_perspective_shape():
    fp = FeaturePerspective(input_dim=32, output_dim=32, num_heads=4)
    x = torch.randn(3, 32)
    y = fp(x)
    assert y.shape == (3, 32)


def test_cosine_classifier_bounded():
    head = CosineClassifier(feat_dim=16, num_classes=5, scale=10.0)
    x = torch.randn(2, 16)
    logits = head(x)
    assert logits.shape == (2, 5)
    # scale=10 bounds raw cosines in [-10, 10].
    assert logits.abs().max().item() <= 10.0 + 1e-4


def test_checkpointed_sequential_matches_plain():
    import torch.nn as nn

    seq = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
    ckpt = CheckpointedSequential(*seq)
    x = torch.randn(2, 8)
    # Matching param init: reuse seq's weights.
    ckpt.load_state_dict(seq.state_dict())
    torch.testing.assert_close(seq(x), ckpt(x))
