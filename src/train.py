"""
train.py – implements the training loop that is used by src.main.
All heavy-lifting (mixed precision, distributed, gradient accumulation …)
 is delegated to 🤗 `accelerate` so that the same code runs on 1 GPU or many.
The trainer is intentionally general-purpose: it receives
  • a UNet (from diffusers or a lightweight custom one);
  • a Scheduler (DDPM, DDIM …);
  • a PyTorch DataLoader that yields dicts with an `image` tensor and, for
    class-conditional models, an optional `label` tensor.
The trainer logs
  – running loss,
  – peak GPU memory (GB),
  – iterations / second
and stores them as a CSV so that downstream visualisation is trivial.
Figures are saved as PDF (Vector) under .research/iteration5/images.
"""
from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import DDPMScheduler
from tqdm import tqdm

# Relative import to get helper utilities / datasets
from .utils import (
    get_peak_memory_gb,
    save_line_plot,
    setup_seed,
)

__all__ = ["Trainer"]


class Trainer:
    """A very small but flexible trainer that supports
    • fp16/bf16 via accelerate
    • gradient accumulation
    • arbitrary UNet-like models from diffusers
    """

    def __init__(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        lr: float = 1e-4,
        gradient_accumulation_steps: int = 1,
        num_train_steps: int = 1_000,
        output_dir: str | Path = "outputs",
        seed: int = 0,
        log_every: int = 50,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.dataloader = dataloader
        self.lr = lr
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_steps = num_train_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        setup_seed(seed)

        # ======== optimiser & scheduler ==========
        self.scheduler = DDPMScheduler(num_train_timesteps=1_000)
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=1e-2,
        )

        # prepare everything with accelerate (handles DDP, AMP …)
        (
            self.model,
            self.optim,
            self.dataloader,
        ) = accelerator.prepare(self.model, self.optim, self.dataloader)

        # tracking containers – will be serialised at the end
        self.metrics: Dict[str, list] = defaultdict(list)

    # ------------------------------------------------------------------
    #                           TRAINING LOOP
    # ------------------------------------------------------------------
    def _step(self, batch) -> torch.Tensor:
        images = batch["image"].to(self.accelerator.device)
        # optional labels for class-conditional models
        labels = batch.get("label")
        if labels is not None:
            labels = labels.to(self.accelerator.device)

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, 1_000, (images.size(0),), device=images.device)
        noisy_images = self.scheduler.add_noise(images, noise, timesteps)

        with torch.autocast("cuda"):
            preds = self.model(noisy_images, timesteps, class_labels=labels).sample
            loss = F.mse_loss(preds, noise)
        return loss

    def train(self):
        self.model.train()
        total_time = 0.0
        data_iter = iter(self.dataloader)
        pbar = tqdm(range(self.num_train_steps), disable=not self.accelerator.is_local_main_process)

        for step in pbar:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            tic = time.time()
            with self.accelerator.accumulate(self.model):
                loss = self._step(batch)
                self.accelerator.backward(loss)
                self.optim.step()
                self.optim.zero_grad()
            toc = time.time()
            step_time = toc - tic
            total_time += step_time

            # --------------- logging (only main process) ----------------
            if self.accelerator.is_local_main_process and step % self.log_every == 0:
                peak = get_peak_memory_gb()
                self.metrics["step"].append(step)
                self.metrics["loss"].append(loss.item())
                self.metrics["peak_mem"].append(peak)
                self.metrics["sec_per_step"].append(step_time)
                torch.cuda.reset_peak_memory_stats()
                pbar.set_description(f"step {step} | loss {loss.item():.4f} | mem {peak:.2f} GB")

        # ======================= end ‑- save metrics ======================
        if self.accelerator.is_local_main_process:
            csv_path = self.output_dir / "training_metrics.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.metrics.keys())
                for row in zip(*self.metrics.values()):
                    writer.writerow(row)
            print(f"[Trainer] metrics saved → {csv_path}")

            # plot loss curve & memory curve -----------------------------
            img_dir = Path(".research/iteration5/images")
            img_dir.mkdir(parents=True, exist_ok=True)
            save_line_plot(
                self.metrics["step"],
                [self.metrics["loss"],],
                ["train loss"],
                xlabel="step",
                ylabel="loss",
                title="Training loss curve",
                filename=img_dir / "loss_curve.pdf",
            )
            save_line_plot(
                self.metrics["step"],
                [self.metrics["peak_mem"],],
                ["peak memory"],
                xlabel="step",
                ylabel="GB",
                title="Peak GPU memory",
                filename=img_dir / "memory_curve.pdf",
            )

        mean_t = total_time / self.num_train_steps
        if self.accelerator.is_local_main_process:
            print(f"Mean seconds / step: {mean_t:.3f}")
