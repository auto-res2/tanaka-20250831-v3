"""src/utils.py
Utility functions shared across the project.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch

__all__ = [
    "set_seed",
    "measure_peak_mem_mb",
    "reset_peak_mem",
    "ensure_dir",
]


# -----------------------------------------------------------------------------
# Reproducibility helpers
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:  # noqa: D401 – simple function
    """Set *all* relevant random seeds to ensure full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe on CPU as well


# -----------------------------------------------------------------------------
# Memory-tracking helpers
# -----------------------------------------------------------------------------

def measure_peak_mem_mb(device: str | torch.device | None = None) -> float:
    """Return the *peak* GPU memory in **MB** that was allocated so far.

    If CUDA is not available we fall back to ``0.0`` so that CPU users can
    still run the script without modifications.
    """
    if torch.cuda.is_available():
        dev = torch.device(device) if device is not None else torch.device("cuda")
        return float(torch.cuda.max_memory_allocated(dev) / 1e6)
    return 0.0


def reset_peak_mem(device: str | torch.device | None = None) -> None:
    """Reset CUDA peak-memory statistics (no-op on CPU)."""
    if torch.cuda.is_available():
        dev = torch.device(device) if device is not None else torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(dev)


# -----------------------------------------------------------------------------
# File-system helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create *path* (a directory) if it does not yet exist and return it."""
    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p
