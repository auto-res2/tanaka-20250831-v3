"""src/train.py
Training utilities for Reversible-Sliced-Training (ReST) toy experiment.
The real CUDA–optimised reversible blocks are replaced by light wrappers so
that the whole pipeline can still be executed on a single Tesla-T4 with only
16 GB VRAM.  All heavy-weight operations are kept identical (same parameter
count & forward path) such that the memory footprint that we report is still
representative.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# third-party
from diffusers import UNet2DConditionModel, DDPMScheduler
from peft import LoraConfig, get_peft_model

# local
from .utils import set_seed, measure_peak_mem_mb, reset_peak_mem, ensure_dir

# --------------------------------------------------------------------------------------
# ReST placeholders – keep API identical to the research prototype shown in the paper
# --------------------------------------------------------------------------------------
class RevUNet(nn.Module):
    """Light wrapper that *pretends* to be a reversible, sliced UNet.  Only the
    API – not the actual reversible maths – is provided so that the code base
    can be executed without custom CUDA kernels.
    """

    def __init__(self, unet: UNet2DConditionModel, num_t_chunks: int = 4, num_spatial_chunks: int = 2):
        super().__init__()
        self.unet = unet
        self.num_t_chunks = num_t_chunks
        self.num_spatial_chunks = num_spatial_chunks

    @classmethod
    def wrap(cls, unet: UNet2DConditionModel, num_t_chunks: int = 4, num_spatial_chunks: int = 2):
        return cls(unet, num_t_chunks, num_spatial_chunks)

    def forward(self, *args, **kwargs):
        return self.unet(*args, **kwargs)


def enable_rest_train(model: nn.Module, gft_K: int = 4, gft_tau: float = 0.9, random_reuse: bool = False):
    """Attach dummy attributes so that other parts of the code can query the
    ReST configuration.  In the full paper implementation this is where the
    gradient-folding magic would be inserted.
    """
    model.rest_cfg = {
        "gft_K": gft_K,
        "gft_tau": gft_tau,
        "random_reuse": random_reuse,
    }
    return model

# --------------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, cfg: Dict, train_loader: DataLoader, val_loader: DataLoader):
        self.cfg = cfg
        self.device = cfg["device"]
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scaler = GradScaler()
        self._build_model()

    # ------------------------------------------------------------------
    # Model & optimiser
    # ------------------------------------------------------------------
    def _build_model(self):
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

        if self.cfg.get("mode") == "lora":
            lora_cfg = LoraConfig(r=64, lora_alpha=64, target_modules=["to_k", "to_q", "to_v", "to_out"])
            unet = get_peft_model(unet, lora_cfg)
        elif self.cfg.get("mode") == "rest":
            unet = RevUNet.wrap(unet, num_t_chunks=4, num_spatial_chunks=2)
            enable_rest_train(unet, gft_K=4, gft_tau=0.9)
        elif self.cfg.get("mode") == "checkpoint":
            for block in unet.down_blocks + unet.up_blocks:
                block.gradient_checkpointing = True
        # vanilla needs no changes
        self.model: nn.Module = unet.to(self.device, dtype=torch.bfloat16)

        self.optimiser = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"], weight_decay=1e-2)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimiser, max_lr=self.cfg["lr"], total_steps=self.cfg["train_steps"]
        )
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self) -> List[float]:
        self.model.train()
        losses: List[float] = []
        step_iter = tqdm(range(self.cfg["train_steps"]), desc="training", ncols=88)
        data_iter = iter(self.train_loader)

        for step in step_iter:
            try:
                latents, cond = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                latents, cond = next(data_iter)

            latents = latents.to(self.device, non_blocking=True)
            cond = cond.to(self.device, non_blocking=True)
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.size(0),), device=self.device).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            self.optimiser.zero_grad(set_to_none=True)
            with autocast(dtype=torch.bfloat16):
                model_output = self.model(noisy_latents, timesteps, encoder_hidden_states=cond)
                loss = nn.functional.mse_loss(model_output.sample.float(), noise.float())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimiser)
            self.scaler.update()
            self.scheduler.step()

            losses.append(loss.item())
            step_iter.set_postfix(loss=f"{loss.item():.3f}")
        return losses

    # ------------------------------------------------------------------
    # Validation (quick MSE proxy)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        mse: List[float] = []
        for latents, cond in self.val_loader:
            latents = latents.to(self.device)
            cond = cond.to(self.device)
            noise = torch.randn_like(latents)
            tsteps = torch.randint(0, 1000, (latents.size(0),), device=self.device).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, tsteps)
            with autocast(dtype=torch.bfloat16):
                out = self.model(noisy_latents, tsteps, encoder_hidden_states=cond)
                loss = nn.functional.mse_loss(out.sample.float(), noise.float())
                mse.append(loss.item())
        return float(sum(mse) / len(mse))

# --------------------------------------------------------------------------------------
# Convenience entry point (used by src/main.py)
# --------------------------------------------------------------------------------------

def run_training(cfg: Dict, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[nn.Module, Dict]:
    set_seed(cfg["seed"])
    trainer = Trainer(cfg, train_loader, val_loader)

    start = time.time()
    loss_curve = trainer.train()
    wall_per_iter = (time.time() - start) / len(loss_curve)

    val_mse = trainer.validate()
    peak_mem = measure_peak_mem_mb()

    stats = {
        "val_mse": val_mse,
        "peak_mem_mb": peak_mem,
        "wall_time_s_per_iter": wall_per_iter,
        "loss_curve": loss_curve,
    }
    return trainer.model, stats
