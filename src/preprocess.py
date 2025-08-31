"""src/preprocess.py
Data-loading utilities.  For the purpose of the ReST memory experiment we use
synthetic latent tensors instead of an expensive VAE + real images.  This
keeps the runtime small while still exercising the full UNet-2D network.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


class DummyLatentDataset(Dataset):
    """Returns random latent tensors of shape (4, H, W) plus a dummy text       
    embedding (token ids) so that the Stable-Diffusion UNet's conditioning
    branch is exercised.
    """

    def __init__(self, length: int, latent_shape: Tuple[int, int, int]):
        self.length = length
        self.latent_shape = latent_shape

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        latent = torch.randn(self.latent_shape)
        cond = torch.randint(0, 49408, (77,))  # hard-coded SD-v1 tokenizer length
        return latent, cond


def get_dataloaders(batch_size: int = 2, resolution: int = 512) -> Tuple[DataLoader, DataLoader]:
    latent_shape = (4, resolution // 8, resolution // 8)
    train_set = DummyLatentDataset(length=512, latent_shape=latent_shape)
    val_set = DummyLatentDataset(length=128, latent_shape=latent_shape)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader
