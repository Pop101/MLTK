"""KLDivLossWithSoftmax + SmoothReduceLROnPlateau behave sanely."""
import torch

from mltk import KLDivLossWithSoftmax, SmoothReduceLROnPlateau
from mltk import CosineWarmupScheduler, EMAWeightTracker


def test_kldiv_reaches_zero_when_distributions_match():
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    target = torch.softmax(logits, dim=-1)
    loss_fn = KLDivLossWithSoftmax()
    loss = loss_fn(logits, target)
    assert float(loss.item()) < 1e-5


def test_kldiv_positive_when_distributions_differ():
    logits = torch.tensor([[2.0, 0.0, -1.0]])
    target = torch.tensor([[0.0, 0.5, 0.5]])
    loss_fn = KLDivLossWithSoftmax()
    loss = loss_fn(logits, target)
    assert float(loss.item()) > 0.0


def test_smooth_plateau_reduces_lr_on_flat_losses():
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = SmoothReduceLROnPlateau(
        opt,
        smoothing_window=3,
        historical_window=6,
        reduction_threshold=0.99,
        cooldown=0,
        factor=0.5,
        min_lr=1e-6,
    )
    # Feed a long flat sequence — loss stops improving, lr must drop.
    for _ in range(30):
        sched.step(1.0)
    final_lr = max(g["lr"] for g in opt.param_groups)
    assert final_lr < 1e-2, f"LR should have dropped; still {final_lr}"


def test_custom_schedulers_follow_torch_scheduler_protocol():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sched = CosineWarmupScheduler(opt, warmup_steps=1, total_steps=4)

    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler)
    assert sched.interval == "step"
    sched.step()
    assert max(group["lr"] for group in opt.param_groups) <= 0.1


def test_ema_weight_tracker_roundtrips_state():
    """state_dict carries dynamic state only; config (decay/warmup_steps) is
    owned by the constructor."""
    model = torch.nn.Linear(2, 1)
    ema = EMAWeightTracker(model.parameters(), decay=0.9, warmup_steps=0)
    ema.update(model.parameters())
    saved_updates = ema._num_updates

    restored = EMAWeightTracker(model.parameters(), decay=0.9, warmup_steps=0)
    restored.load_state_dict(ema.state_dict())

    assert restored._num_updates == saved_updates
    for live, src in zip(restored._shadow, ema._shadow):
        torch.testing.assert_close(live, src)
