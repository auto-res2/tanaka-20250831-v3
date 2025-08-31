"""src/evaluate.py
Simple evaluation script – only computes a validation MSE on the noisy-latent
prediction task because full FID/FVD computation would require heavy VAE &
CLIP pipelines not suitable for a demo on a 16 GB Tesla-T4.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from diffusers import DDPMScheduler


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, device: str = "cuda") -> Dict:
    model.eval()
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    losses: List[float] = []
    for latents, cond in tqdm(dataloader, desc="eval", ncols=80):
        latents = latents.to(device)
        cond = cond.to(device)
        noise = torch.randn_like(latents)
        tsteps = torch.randint(0, 1000, (latents.size(0),), device=device).long()
        noisy_latents = scheduler.add_noise(latents, noise, tsteps)
        with autocast(dtype=torch.bfloat16):
            out = model(noisy_latents, tsteps, encoder_hidden_states=cond)
            loss = nn.functional.mse_loss(out.sample.float(), noise.float())
            losses.append(loss.item())
    return {
        "mse": float(sum(losses) / len(losses))
    }
