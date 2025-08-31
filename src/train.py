"""src/train.py
Train script implementing memory-efficient diffusion UNet fine-tuning.
Run only through ``python -m src.main`` – do **NOT** execute this file directly.
All heavy-lifting (dataloaders, evaluation, fig generation) is kept extremely
light-weight so everything fits into a 16-GB T4 whilst still showcasing the
Reversible-Chunked-UNet (ReChuNet) idea described in the paper draft.

Because the public implementation of ReChuNet is assumed to live in the helper
package ``rechuwrapper`` we gracefully fall back to the vanilla UNet when the
wrapper is not found so that the code remains runnable even without the
research prototype installed.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from diffusers import DDPMScheduler, UNet2DConditionModel

from .preprocess import build_train_loader, gpu_mem_mb

# ----------------------------------------------------------------------------
# internal helpers
# ----------------------------------------------------------------------------

def _get_unet(model_name: str = "rechu") -> torch.nn.Module:
    """Returns a UNet wrapped in ReChuNet if available, else baseline UNet."""
    base_unet = UNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
    )
    if model_name == "rechu":
        try:
            from rechuwrapper import apply_rechu  # type: ignore

            print("[train]  Applying ReChuWrapper …")
            unet = apply_rechu(base_unet, chunk_size=2, gate_lambda=1e-3, nf4_optim=True)
        except (ImportError, ModuleNotFoundError):
            print(
                "[train][warning] rechuwrapper not found – falling back to vanilla UNet."
            )
            unet = base_unet
    else:
        unet = base_unet

    # checkpointing for baseline to reduce memory so everything still fits T4
    if model_name != "rechu":
        unet.enable_gradient_checkpointing()

    return unet


def _null_encoder(batch: int, device: torch.device, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Creates an all-zero encoder_hidden_states tensor expected by the SD UNet."""
    return torch.zeros(batch, 77, 768, device=device, dtype=dtype)


def _ensure_four_channels(x: torch.Tensor) -> torch.Tensor:
    """Pads input tensor to 4 channels as required by the SD UNet."""
    if x.shape[1] == 4:
        return x
    if x.shape[1] == 3:
        pad = torch.zeros_like(x[:, :1])
        return torch.cat([x, pad], dim=1)
    raise ValueError("Input to UNet must have 3 or 4 channels.")


def _diffusion_loss(
    unet: torch.nn.Module,
    scheduler: DDPMScheduler,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Standard MSE diffusion objective (simplified)."""
    images, _ = batch  # FakeData returns (img, label)
    images = images.to(device)
    images = _ensure_four_channels(images)
    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (images.size(0),), device=device
    )
    noise = torch.randn_like(images)
    noisy = scheduler.add_noise(images, noise, timesteps)

    encoder_hidden_states = _null_encoder(images.size(0), device, noisy.dtype)

    with autocast(device_type="cuda"):
        noise_pred = unet(noisy, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())
    return loss


# ----------------------------------------------------------------------------
# public entry
# ----------------------------------------------------------------------------

def train_model(args) -> Tuple[Path, Path]:
    """Full training routine – returns (csv_path, pdf_path)."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # model + optimiser -------------------------------------------------
    # ------------------------------------------------------------------
    unet = _get_unet(args.model).to(device, dtype=torch.float16)

    if args.model == "rechu":
        try:
            import bitsandbytes as bnb  # type: ignore

            optim_cls = bnb.optim.AdamW8bit
        except ImportError:
            optim_cls = torch.optim.AdamW
    else:
        optim_cls = torch.optim.AdamW

    optimizer = optim_cls(unet.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000, beta_schedule="linear", beta_start=1e-4, beta_end=0.02
    )
    scaler = GradScaler()

    # ------------------------------------------------------------------
    # data --------------------------------------------------------------
    # ------------------------------------------------------------------
    loader = build_train_loader(batch_size=args.batch_size)

    # ------------------------------------------------------------------
    # training loop -----------------------------------------------------
    # ------------------------------------------------------------------
    unet.train()
    log_buffer = []
    tic = time.time()
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, 1):
        loss = _diffusion_loss(unet, scheduler, batch, device)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            torch.cuda.synchronize()
            mem = gpu_mem_mb()
            it_s = step / (time.time() - tic)
            print(f"step={step:04d} | loss={loss.item():.4f} | it/s={it_s:.2f} | mem={mem:.0f}MB")
            log_buffer.append({"step": step, "loss": loss.item(), "it_s": it_s, "mem": mem})

        if step >= args.max_steps:
            break

    # ------------------------------------------------------------------
    # save artefacts ----------------------------------------------------
    # ------------------------------------------------------------------
    models_dir = Path("models"); models_dir.mkdir(exist_ok=True, parents=True)
    model_path = models_dir / f"unet_{args.model}.pt"
    torch.save(unet.state_dict(), model_path)

    # logs → CSV --------------------------------------------------------
    logs_dir = Path("logs"); logs_dir.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(log_buffer)
    csv_path = logs_dir / f"train_log_{args.model}.csv"
    df.to_csv(csv_path, index=False)

    # plot loss curve ---------------------------------------------------
    fig_dir = Path(".research/iteration8/images"); fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / f"loss_curve_{args.model}.pdf"
    plt.figure(figsize=(6,4))
    sns.lineplot(data=df, x="step", y="loss")
    plt.title(f"Training loss – {args.model}")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print("[train]  finished – artefacts saved\n")
    return csv_path, fig_path
