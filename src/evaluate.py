"""src/evaluate.py
Very small evaluation helper – computes a toy FID proxy and generates a figure.
This is **not** a rigorous evaluation; it is only included so that the pipeline
produces some quantitative output even without heavy datasets.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from diffusers import UNet2DConditionModel, DDPMScheduler

from .preprocess import gpu_mem_mb


@torch.no_grad()
def _simple_quality_score(unet: torch.nn.Module, device: torch.device) -> float:
    """A ridiculous *proxy* for quality: negative MSE on 16 generated images."""
    unet.eval()
    scheduler = DDPMScheduler(num_train_timesteps=50)
    noise = torch.randn(16, 3, 64, 64, device=device)
    for t in scheduler.timesteps:
        with torch.cuda.amp.autocast():
            noise_pred = unet(noise, t).sample
        noise = scheduler.step(noise_pred, t, noise).prev_sample
    score = -noise.float().pow(2).mean().item()
    return score


def evaluate_model(args, model_ckpt: Path) -> Tuple[Path, dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    ).to(device, dtype=torch.float16)
    unet.load_state_dict(torch.load(model_ckpt, map_location="cpu"))

    score = _simple_quality_score(unet, device)
    mem = gpu_mem_mb()

    # save to CSV -------------------------------------------------------
    out_dir = Path("logs"); out_dir.mkdir(exist_ok=True, parents=True)
    eval_path = out_dir / f"eval_metrics_{args.model}.csv"
    pd.DataFrame([{"quality": score, "peak_mem": mem}]).to_csv(eval_path, index=False)

    # bar figure --------------------------------------------------------
    fig_dir = Path(".research/iteration7/images")
    fig_path = fig_dir / f"eval_{args.model}.pdf"
    plt.figure(figsize=(2.5,3))
    sns.barplot(x=[""], y=[score], palette=["#4C72B0"])
    plt.ylabel("proxy quality ↑")
    plt.title("Eval score")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f"[eval]  quality={score:.3f}  peak_mem={mem:.0f}MB")
    return fig_path, {"quality": score, "peak_mem": mem}
