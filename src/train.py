"""
train.py – model construction, training loop, memory-profiling & visualisation
All heavy-lifting lives here so that src.main can orchestrate the whole
pipeline with only a few lines of code.
"""
from __future__ import annotations
import os, time, math, random, pathlib, json
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# Use the new (recommended) AMP API
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------
#  Reversible blocks – very light generic implementation
#  (additive coupling – RevNet style)
# -------------------------------------------------------

class _RevFn(torch.autograd.Function):
    """Low-level autograd function that rebuilds activations in backward."""
    @staticmethod
    def forward(ctx, x, f, g):
        x1, x2 = torch.chunk(x, 2, dim=1)
        with torch.enable_grad():
            x2_detached = x2.detach().requires_grad_()
            y1 = x1 + f(x2_detached)
            y2 = x2 + g(y1)
        ctx.save_for_backward(y1.detach(), y2.detach())
        ctx.f, ctx.g = f, g
        return torch.cat([y1, y2], dim=1)

    @staticmethod
    def backward(ctx, dy):
        y1, y2 = ctx.saved_tensors
        f, g = ctx.f, ctx.g
        dy1, dy2 = torch.chunk(dy, 2, dim=1)
        with torch.enable_grad():
            # Re-enable gradient tracking on the saved tensor
            y1 = y1.detach().requires_grad_()
            gy = g(y1)
            torch.autograd.backward(gy, dy2)
            dx2 = y1.grad.clone()
            y1.grad.zero_()
            torch.autograd.backward(y1, dy1 + dx2)
            dx1 = y1.grad.clone()
        return torch.cat([dx1, dx2], dim=1), None, None

class RevBlock(nn.Module):
    """Wrapper that turns (F,G) into a reversible block."""
    def __init__(self, in_channels: int):
        super().__init__()
        mid = in_channels // 2
        self.f = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1))
        self.g = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.GELU(),
            nn.Conv2d(mid, mid, 3, padding=1))

    def forward(self, x):
        return _RevFn.apply(x, self.f, self.g)

# -------------------------------------------------------
#  Two tiny CNNs: baseline vs reversible (same depth)
# -------------------------------------------------------

class TinyCNN(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 3, 3, padding=1))

    def forward(self, x):
        return self.net(x)

class TinyRevCNN(nn.Module):
    def __init__(self, channels: int = 64):
        super().__init__()
        assert channels % 2 == 0, "channels must be divisible by 2 for RevNet"
        blocks = []
        blocks.append(nn.Conv2d(3, channels, 3, padding=1))
        for _ in range(4):
            blocks.append(RevBlock(channels))
        blocks.append(nn.Conv2d(channels, 3, 3, padding=1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------------
#  Helper to build model from cfg
# -------------------------------------------------------


def build_model(cfg: Dict) -> nn.Module:
    if cfg["model"] == "baseline":
        return TinyCNN(cfg["channels"]).to("cuda")
    if cfg["model"] == "reversible":
        return TinyRevCNN(cfg["channels"]).to("cuda")
    raise ValueError(f"unknown model {cfg['model']}")

# -------------------------------------------------------
#  Training loop – returns statistics
# -------------------------------------------------------


def train(model: nn.Module, loader: DataLoader, cfg: Dict) -> Dict[str, List[float]]:
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    # GradScaler does not accept `device_type`; it infers the device automatically.
    scaler = GradScaler()

    mem, losses = [], []
    for epoch in range(cfg["epochs"]):
        for batch, (x, _) in enumerate(loader):
            x = x.to("cuda", dtype=torch.float16)
            torch.cuda.reset_peak_memory_stats()
            with autocast(device_type="cuda", dtype=torch.float16):
                out = model(x)
                loss = F.mse_loss(out, x)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
            losses.append(loss.item())
            mem.append(torch.cuda.max_memory_allocated() / 1e6)  # MB
    return {"loss": losses, "mem": mem}

# -------------------------------------------------------
#  Plot helpers – saved under .research/iteration15/images
# -------------------------------------------------------


def _make_img_dir():
    img_dir = pathlib.Path(".research/iteration15/images")
    img_dir.mkdir(parents=True, exist_ok=True)
    return img_dir


def save_plots(stats: Dict[str, List[float]], tag: str):
    img_dir = _make_img_dir()
    sns.set_style("whitegrid")
    # 1) Loss curve
    plt.figure(figsize=(6,4))
    plt.title(f"Training loss – {tag}")
    plt.plot(stats["loss"]) ; plt.xlabel("step") ; plt.ylabel("MSE loss")
    plt.savefig(img_dir / f"loss_{tag}.pdf", bbox_inches="tight")
    # 2) Memory curve
    plt.figure(figsize=(6,4))
    plt.title(f"Peak memory – {tag}")
    plt.plot(stats["mem"]) ; plt.xlabel("step") ; plt.ylabel("MB")
    plt.savefig(img_dir / f"memory_{tag}.pdf", bbox_inches="tight")
    plt.close("all")
