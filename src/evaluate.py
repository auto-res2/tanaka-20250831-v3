"""
evaluate.py – lightweight evaluation helpers.
For demonstration we compute a *proxy* metric (MSE on a held-out noise-prediction
 task) to avoid the heavy FID/CLIP computation that would exceed the 16-GB / 15-min
 execution budget on this platform. Hooks are provided so that a proper FID
 implementation can be plugged-in later without changing main.py.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from tqdm import tqdm

from .utils import get_peak_memory_gb, save_bar_plot

__all__ = ["evaluate"]


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    max_batches: int = 50,
    out_dir: str | Path = "outputs",
):
    """Runs a fast evaluation loop that measures:
    • proxy-MSE loss (noise-prediction)
    • peak GPU memory during eval
    """
    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1_000)
    losses = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=max_batches)):
            if i >= max_batches:
                break
            img = batch["image"].to(accelerator.device)
            labels = batch.get("label")
            if labels is not None:
                labels = labels.to(accelerator.device)
            noise = torch.randn_like(img)
            ts = torch.randint(0, 1_000, (img.size(0),), device=img.device)
            noisy = scheduler.add_noise(img, noise, ts)
            with torch.autocast("cuda"):
                pred = model(noisy, ts, class_labels=labels).sample
                loss = F.mse_loss(pred, noise, reduction="none").mean([1, 2, 3]).cpu()
            losses.extend(loss.tolist())

    proxy_mse = float(np.mean(losses))
    peak_mem = get_peak_memory_gb()

    if accelerator.is_local_main_process:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        res_path = Path(out_dir) / "eval_results.json"
        with res_path.open("w") as f:
            json.dump({"proxy_mse": proxy_mse, "peak_mem_gb": peak_mem}, f, indent=2)
        print(f"[Eval] proxy-MSE: {proxy_mse:.4f} | peak-mem {peak_mem:.2f} GB (results → {res_path})")

    return proxy_mse, peak_mem
