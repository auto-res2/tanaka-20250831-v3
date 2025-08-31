"""
evaluate.py – lightweight evaluation routine.
Computes a simple denoising loss on a validation set and saves
sample images as well as a loss curve under .research/iteration5/images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from accelerate import Accelerator
from diffusers import DDPMScheduler
from tqdm import tqdm

from .utils import save_line_plot

__all__ = ["evaluate"]


def _step(model, scheduler: DDPMScheduler, batch, device):
    """Compute a single validation loss value (MSE between predicted and true noise)."""
    images = batch["image"].to(device)
    labels = batch.get("label")
    if labels is not None:
        labels = labels.to(device)

    noise = torch.randn_like(images)
    timesteps = torch.randint(0, 1_000, (images.size(0),), device=device)
    noisy_images = scheduler.add_noise(images, noise, timesteps)
    with torch.autocast("cuda"):
        preds = model(noisy_images, timesteps, class_labels=labels).sample
        loss = F.mse_loss(preds, noise, reduction="mean")
    return loss


def _save_samples(batch, img_dir: Path):
    """Utility: save a grid of validation images (rescaled to [0,1])."""
    images = batch["image"]
    # Rescale from (-1,1) → (0,1)
    grid = (images + 1.0) / 2.0
    grid = torch.clamp(grid, 0.0, 1.0)
    img_path = img_dir / "val_samples.png"
    vutils.save_image(grid, img_path, nrow=min(8, grid.size(0)))
    print(f"[Evaluate] sample images saved → {img_path}")


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    accelerator: Accelerator,
    *,
    max_batches: int = 100,
    out_dir: str | Path = "outputs",
):
    """Run evaluation for `max_batches` from `dataloader` and save artefacts."""
    if accelerator.is_local_main_process:
        print("[Evaluate] starting evaluation …")

    img_dir = Path(".research/iteration5/images")
    img_dir.mkdir(parents=True, exist_ok=True)

    # Prepare model & data with Accelerate so that everything is on the right device.
    model, dataloader = accelerator.prepare(model, dataloader)
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1_000)

    losses: List[float] = []
    steps: List[int] = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, total=max_batches, disable=not accelerator.is_local_main_process)):
            if idx >= max_batches:
                break
            loss = _step(model, scheduler, batch, accelerator.device)
            losses.append(loss.item())
            steps.append(idx)

            # Save a grid of real images only once (on the first batch)
            if idx == 0 and accelerator.is_local_main_process:
                _save_samples(batch, img_dir)

    # ---------------- plotting ----------------
    if accelerator.is_local_main_process:
        save_line_plot(
            steps,
            [losses],
            ["val loss"],
            xlabel="batch",
            ylabel="loss",
            title="Validation loss curve",
            filename=img_dir / "val_loss_curve.pdf",
        )

        # Save numeric values as CSV -------------------------------------------------
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "eval_metrics.csv"
        with csv_path.open("w") as f:
            f.write("batch,loss\n")
            for s, l in zip(steps, losses):
                f.write(f"{s},{l}\n")
        print(f"[Evaluate] metrics saved → {csv_path}")
