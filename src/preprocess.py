"""
preprocess.py – downloads CIFAR-10 (if not yet) and prepares training
DataLoader for downstream tasks.  As this dataset is tiny, no further
processing is needed; still we keep the module for completeness.
"""
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_dataloaders(batch_size: int = 128):
    tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_ds = datasets.CIFAR10(root="data", train=True, download=True, transform=tf)
    val_ds   = datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader
