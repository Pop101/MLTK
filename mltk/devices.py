"""Cross-platform accelerator selection.

A small, side-effect-free module that picks the best available device for
PyTorch on the current host. Supports CUDA (covers AMD ROCm transparently —
ROCm builds expose the same `torch.cuda` API and set `torch.version.hip`),
DirectML (AMD/Intel on Windows, via the optional `torch_directml`), Apple
MPS, and CPU as the final fallback.

Typical use:

    from mltk import auto_device, device_label

    device = auto_device()
    print(f"Running on {device_label(device)}")

`synchronize()` and `empty_cache()` dispatch to the right backend so callers
don't litter their training loop with `if torch.cuda.is_available()` guards.
"""
from __future__ import annotations

from typing import Optional

import torch


def auto_device() -> torch.device:
    """Return the best available accelerator.

    Preference order: CUDA / ROCm → DirectML → MPS → CPU. Pure function:
    no warnings, no caching, no global state. Call once and assign to a
    module-level variable in the caller if you want to reuse it.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    try:
        import torch_directml  # type: ignore
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def is_rocm(device: Optional[torch.device] = None) -> bool:
    """True iff `device` (default: `auto_device()`) is an AMD ROCm GPU.

    PyTorch ROCm builds use the same `torch.cuda` namespace as NVIDIA, so a
    bare `device.type == "cuda"` check can't distinguish them. This looks at
    `torch.version.hip` to disambiguate.
    """
    d = device if device is not None else auto_device()
    return d.type == "cuda" and bool(getattr(torch.version, "hip", None))


def device_label(device: Optional[torch.device] = None) -> str:
    """Human-readable name for the backend behind `device`."""
    d = device if device is not None else auto_device()
    if d.type == "cuda":
        return "ROCm" if is_rocm(d) else "CUDA"
    if d.type == "privateuseone":
        return "DirectML"
    if d.type == "mps":
        return "MPS"
    if d.type == "cpu":
        return "CPU"
    return d.type.upper()


def synchronize(device: Optional[torch.device] = None) -> None:
    """Cross-backend `torch.*.synchronize()`. CPU and DirectML are no-ops."""
    d = device if device is not None else auto_device()
    if d.type == "cuda":
        torch.cuda.synchronize()
    elif d.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: Optional[torch.device] = None) -> None:
    """Cross-backend cache flush. CPU is a no-op; DirectML doesn't expose
    a flush API."""
    d = device if device is not None else auto_device()
    if d.type == "cuda":
        torch.cuda.empty_cache()
    elif d.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


__all__ = [
    "auto_device",
    "is_rocm",
    "device_label",
    "synchronize",
    "empty_cache",
]
