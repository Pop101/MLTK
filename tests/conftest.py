"""Shared test fixtures / utilities."""
import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_everything():
    """Reset every RNG before each test so ordering doesn't flake the
    accuracy-threshold integration tests."""
    seed = 17
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield
