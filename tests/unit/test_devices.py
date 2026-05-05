"""Cross-platform device selection. Asserts shape of the API, not which
backend the test host happens to expose."""
import torch

from mltk import auto_device, device_label, empty_cache, is_rocm, synchronize


def test_auto_device_returns_torch_device():
    d = auto_device()
    assert isinstance(d, torch.device)


def test_auto_device_picks_known_backend():
    d = auto_device()
    assert d.type in {"cuda", "privateuseone", "mps", "cpu"}


def test_auto_device_is_pure():
    """No caching, no global state — repeated calls return equivalent devices."""
    a, b = auto_device(), auto_device()
    assert a.type == b.type


def test_device_label_for_every_supported_type():
    """Pass in synthetic devices to lock the label table without depending
    on what the test host happens to have."""
    assert device_label(torch.device("cpu")) == "CPU"
    assert device_label(torch.device("cuda")) in {"CUDA", "ROCm"}
    # synthetic types we might see; label() must not crash
    for fake in ("mps", "privateuseone"):
        try:
            label = device_label(torch.device(fake))
        except RuntimeError:
            # torch may refuse to construct devices for unavailable backends;
            # that's acceptable, the label fn itself isn't on the hot path then.
            continue
        assert label in {"DirectML", "MPS"}


def test_is_rocm_consistent_with_torch_version():
    d = auto_device()
    assert is_rocm(d) == (d.type == "cuda" and bool(getattr(torch.version, "hip", None)))


def test_synchronize_and_empty_cache_no_throw_on_cpu():
    """CPU path must be a clean no-op so callers don't need a guard."""
    cpu = torch.device("cpu")
    synchronize(cpu)
    empty_cache(cpu)


def test_synchronize_default_arg_uses_auto_device():
    # Should not raise regardless of host configuration.
    synchronize()
    empty_cache()
