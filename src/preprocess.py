"""src/preprocess.py
Data & utility helpers used by train / evaluate.
"""
from __future__ import annotations

import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----------------------------------------------------------------------------
# reproducibility ------------------------------------------------------------
# ----------------------------------------------------------------------------

def set_seed(seed: int = 2024):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# memory helpers -------------------------------------------------------------
# ----------------------------------------------------------------------------

def gpu_mem_mb() -> float:
    """Returns max *allocated* memory since program start (in MB)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


# ----------------------------------------------------------------------------
# dataloaders ----------------------------------------------------------------
# ----------------------------------------------------------------------------

def build_train_loader(batch_size: int = 2) -> DataLoader:
    """Very small FakeData loader so experiment becomes self-contained."""
    transform = transforms.Compose(
        [
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = datasets.FakeData(size=4_000, image_size=(3, 64, 64), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return loader
