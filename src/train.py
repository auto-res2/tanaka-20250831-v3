"""src/train.py
----------------------------------
Training utilities for the ReChuNet study.
The code is deliberately light-weight so that it can run on a single
Tesla-T4 (16 GB).  Instead of the full Stable-Diffusion UNet we use a very
small toy-UNet that keeps exactly the same tensor interface
(pixel_values, timesteps, encoder_hidden_states) so that functions can be
swapped later with the real backbone without touching training logic.

If the optional dependency `rechuwrapper` is installed, a reversible /
chunked variant of the toy-UNet is built by calling
```
rechuwrapper.make_rechunked(model, rev=True, chunk_hw=2, chunk_tb=2)
```
otherwise a warning is emitted and the baseline model is trained.
"""
from __future__ import annotations

import math
import pathlib
import time
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import dataset helpers (works whether the module is executed as a package
# or as a plain script)
# ---------------------------------------------------------------------------
try:
    from .preprocess import get_dataloaders  # type: ignore
except ImportError:  # fallback when executed as a plain script
    from preprocess import get_dataloaders  # type: ignore

# -----------------------------------------------------------
# Small, GPU-friendly UNet we can really train in a few minutes
# -----------------------------------------------------------

def double_conv(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.GroupNorm(4, out_c),
        nn.SiLU(),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.GroupNorm(4, out_c),
        nn.SiLU(),
    )


class TinyUNet(nn.Module):
    """Tiny 4-layer UNet that mimics the signature of Diffusers UNet."""

    def __init__(self, base_channels: int = 32):
        super().__init__()
        self.down1 = double_conv(3, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = double_conv(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        # bottleneck
        self.mid = double_conv(base_channels * 2, base_channels * 4)

        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.conv_up1 = double_conv(base_channels * 4, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.conv_up2 = double_conv(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, 3, 1)

    # the signature expected by diffusion training loops
    def forward(self, x, timesteps=None, encoder_hidden_states=None):  # pylint: disable=unused-argument
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        mid = self.mid(self.pool2(d2))
        u1 = self.conv_up1(torch.cat([self.up1(mid), d2], dim=1))
        u2 = self.conv_up2(torch.cat([self.up2(u1), d1], dim=1))
        return torch.tanh(self.out_conv(u2))


# -----------------------------------------------------------
# Helper
# -----------------------------------------------------------

def _apply_rechu_if_available(model: nn.Module, cfg: Dict):
    if cfg.get("use_rechunet", False):
        try:
            from rechuwrapper import make_rechunked  # type: ignore

            model = make_rechunked(
                model,
                rev=True,
                chunk_hw=cfg.get("chunk_hw", 2),
                chunk_tb=cfg.get("chunk_tb", 2),
                flash_attn=cfg.get("flash", False),
                gated=cfg.get("gated", False),
            )
            print("[train]  ReChuNet wrapper successfully applied ✔")
        except ImportError:
            print("[train]  rechuwrapper not installed – falling back to baseline model")
    return model


# -----------------------------------------------------------
# Public training entry-point
# -----------------------------------------------------------

def train(cfg: Dict):
    """Main training routine; returns path to the saved model."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train]  Using device: {device}")

    # ---------------------------------------------------------------------
    # 1. Data
    # ---------------------------------------------------------------------
    train_loader: DataLoader
    _, train_loader, _ = get_dataloaders(cfg)

    # ---------------------------------------------------------------------
    # 2. Model + optimiser
    # ---------------------------------------------------------------------
    model = TinyUNet(base_channels=cfg.get("base_channels", 32))
    model = _apply_rechu_if_available(model, cfg).to(device)

    if device.type == "cuda":
        model = model.half()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3))

    # ---------------------------------------------------------------------
    # 3. Training loop
    # ---------------------------------------------------------------------
    losses: List[float] = []
    mem_peak = 0
    num_steps = cfg.get("num_steps", 500)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    model.train()
    pbar = tqdm(train_loader, total=num_steps, desc="training", unit="step")
    step = 0
    while step < num_steps:
        for batch in pbar:
            step += 1
            if step > num_steps:
                break
            imgs = batch["pixel_values"].to(device)
            if device.type == "cuda":
                imgs = imgs.half()
            noise = torch.randn_like(imgs)
            noisy_imgs = imgs + 0.1 * noise  # fake diffusion noise

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                preds = model(noisy_imgs)
                loss = F.mse_loss(preds.float(), imgs.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            losses.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if device.type == "cuda":
                torch.cuda.synchronize()
                mem_peak = max(mem_peak, torch.cuda.max_memory_allocated() // (1024 ** 2))

    if device.type == "cuda":
        print(f"[train]  Peak GPU memory during training: {mem_peak} MB")

    # ---------------------------------------------------------------------
    # 4. Save artefacts
    # ---------------------------------------------------------------------
    models_dir = pathlib.Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "toy_unet.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[train]  Model saved → {model_path.relative_to(pathlib.Path.cwd())}")

    # plot loss curve for paper-ready pdf
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    # ------------------------------------------------------------------
    # NOTE: all experiment images are now saved under iteration13
    # ------------------------------------------------------------------
    img_dir = Path(".research/iteration13/images")
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("iteration")
    plt.ylabel("MSE loss")
    plt.title("Training loss curve")
    plt.tight_layout()
    fig_path = img_dir / "training_loss_curve.pdf"
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"[train]  Loss curve saved → {fig_path.relative_to(Path.cwd())}")

    return str(model_path)
