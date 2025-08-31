"""src/preprocess.py
----------------------------------
Dataset helpers.  If the user supplies real image folders in the config we
load them with `torchvision.datasets.ImageFolder`; otherwise we fall back
to a synthetic Gaussian blobs dataset so that the whole pipeline remains
fully runnable even without external data.
"""
from __future__ import annotations

import pathlib
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

# -----------------------------------------------------------
# Synthetic fall-back dataset
# -----------------------------------------------------------


class RandomImageSet(Dataset):
    def __init__(self, length: int, image_size: int = 64):
        self.length = length
        self.image_size = image_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):  # pylint: disable=unused-argument
        img = torch.randn(3, self.image_size, self.image_size)
        return {
            "pixel_values": img,
        }


# -----------------------------------------------------------
# Public API
# -----------------------------------------------------------

def get_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns (train_loader, val_loader, test_loader)."""

    from torchvision import datasets, transforms as T

    batch_size = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 2)
    img_size = cfg.get("image_size", 64)

    tf = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
    ])

    data_root = pathlib.Path(cfg.get("data_root", ""))
    if data_root.exists() and any(data_root.iterdir()):
        print(f"[preprocess]  Loading real images from {data_root}")
        full_ds = datasets.ImageFolder(data_root, transform=tf)
    else:
        print("[preprocess]  Falling back to synthetic random images ✨")
        full_ds = RandomImageSet(length=5000, image_size=img_size)

    n = len(full_ds)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(full_ds, [int(0.8 * n), int(0.1 * n), n - int(0.9 * n)])

    def _loader(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return _loader(train_ds), _loader(val_ds), _loader(test_ds)
