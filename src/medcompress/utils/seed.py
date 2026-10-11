"""Seeding for reproducible runs (3 seeds per training arm)."""

from __future__ import annotations

import os
import random


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and (if installed) PyTorch.

    torch is imported lazily so this works with light dependencies only.

    Args:
        seed: Seed value (training arms use 0, 1, 2).
        deterministic: Also request deterministic cuDNN / torch algorithms.
            Slower; use for final runs if the kernels support it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is in base requirements
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
