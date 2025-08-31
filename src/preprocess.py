"""
preprocess.py – very small wrappers around torchvision datasets so that the
main training / evaluation code remains clean. If ImageNet or LAION are not
available locally we fall back to a dummy CIFAR-10 subset; this allows the
code to run *out-of-the-box* on the execution platform.
All images are re-scaled to (256, 256) and mapped to (-1, 1).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Tuple

import torch
import torchvision
import torchvision.transforms as T

__all__ = [
    "get_dataloaders",
]

def _cifar10_loader(batch: int, workers: int = 4):
    tf = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(256),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    train = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    dl_train = torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True, num_workers=workers, pin_memory=True)
    dl_test = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return dl_train, dl_test


def get_dataloaders(dataset: Literal["cifar10", "imagenet"], batch: int = 8):
    """Factory that returns (train_loader, val_loader). For brevity only two datasets
    are wired-in, but others can be added trivially.
    """
    if dataset == "cifar10":
        return _cifar10_loader(batch)
    elif dataset == "imagenet":
        if not (Path("data/imagenet/train").exists() and Path("data/imagenet/val").exists()):
            print("[Preprocess] ImageNet directory not found – falling back to CIFAR-10.")
            return _cifar10_loader(batch)
        tf = T.Compose(
            [
                T.Resize(286),
                T.RandomCrop(256),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )
        train_ds = torchvision.datasets.ImageFolder("data/imagenet/train", transform=tf)
        val_ds = torchvision.datasets.ImageFolder("data/imagenet/val", transform=tf)
        dl_train = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=8, pin_memory=True)
        dl_val = torch.utils.data.DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
        return dl_train, dl_val
    else:
        raise ValueError(dataset)
