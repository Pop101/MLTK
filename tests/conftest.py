"""Shared test fixtures / utilities."""
import random
import numpy as np
import torch


def pytest_configure(config):
    # Deterministic-ish runs so integration tests don't flake on accuracy thresholds.
    seed = 17
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
