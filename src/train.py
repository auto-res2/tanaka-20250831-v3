"""src/train.py
Training utilities – contains a very small toy denoising network plus the
run_training() helper that is used by src.main.  The implementation is *not*
a full Stable-Diffusion UNet; it is a light-weight convolutional network that
is sufficient for demonstrating the training / evaluation / plotting pipeline
inside the limited CI environment.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from diffusers import DDPMScheduler

# We intentionally *duplicate* the simple helper functions that were already
# present in the previous revision so that existing imports from other modules
# (e.g. ``src.utils``) continue to work after a new, dedicated ``utils.py``
# file has been introduced.
__all__ = [
    "set_seed",
    "measure_peak_mem_mb",
    "reset_peak_mem",
    "ensure_dir",
    "run_training",
]

# -----------------------------------------------------------------------------
# Reproducibility helpers (kept for backward-compatibility)
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:  # noqa: D401
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Memory helpers (kept for backward-compatibility)
# -----------------------------------------------------------------------------


def measure_peak_mem_mb(device: str | torch.device | None = None) -> float:  # noqa: D401
    if torch.cuda.is_available():
        dev = torch.device(device) if device is not None else torch.device("cuda")
        return float(torch.cuda.max_memory_allocated(dev) / 1e6)
    return 0.0


def reset_peak_mem(device: str | torch.device | None = None) -> None:  # noqa: D401
    if torch.cuda.is_available():
        dev = torch.device(device) if device is not None else torch.device("cuda")
        torch.cuda.reset_peak_memory_stats(dev)


# -----------------------------------------------------------------------------
# File-system helpers (kept for backward-compatibility)
# -----------------------------------------------------------------------------


def ensure_dir(path):  # noqa: D401 – inline minimal helper
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------------
# Tiny convolutional model
# -----------------------------------------------------------------------------


class _SimpleDenoiser(nn.Module):
    """A *tiny* UNet-like model that predicts the added noise.

    The network purposefully keeps the parameter count small so that unit tests
    can finish quickly on the CI GPU.  It completely ignores the textual
    conditioning and the diffusion time-step – both are only accepted so that
    the call signature matches that of *diffusers.UNet2DConditionModel*.
    """

    def __init__(self, in_channels: int = 4, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, in_channels, 3, padding=1),
        )

    def forward(self, x, timesteps=None, encoder_hidden_states=None):  # noqa: D401
        return _ModelOutput(sample=self.net(x))


@dataclass
class _ModelOutput:
    sample: torch.Tensor


# -----------------------------------------------------------------------------
# Training entry-point
# -----------------------------------------------------------------------------


def run_training(cfg: Dict, train_loader, val_loader):
    """Runs a very small training loop and returns the model plus statistics."""

    device = cfg.get("device", "cpu")
    set_seed(cfg.get("seed", 42))

    model = _SimpleDenoiser().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-4))
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    scaler = GradScaler(enabled=torch.cuda.is_available())
    loss_curve: List[float] = []

    # ------------------------------------------------------------------
    # Iterate over the *loader* indefinitely until *train_steps* is met.
    # ------------------------------------------------------------------
    train_iter = itertools.cycle(train_loader)
    pbar = tqdm(range(cfg["train_steps"]), desc="train", ncols=80)

    model.train()
    for _ in pbar:
        latents, cond = next(train_iter)
        latents = latents.to(device)
        # The dummy conditioning is ignored by the tiny model, but we still
        # send it to the correct device to avoid potential device mismatch.
        cond = cond.to(device)

        noise = torch.randn_like(latents)
        tsteps = torch.randint(0, 1000, (latents.size(0),), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, tsteps)

        optim.zero_grad(set_to_none=True)
        with autocast(dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(noisy_latents, tsteps, encoder_hidden_states=cond)
            loss = nn.functional.mse_loss(out.sample, noise)

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        loss_curve.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    stats = {
        "final_loss": float(loss_curve[-1]),
        "loss_curve": loss_curve,
    }
    return model, stats
