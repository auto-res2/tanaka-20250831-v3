"""
main.py – entry-point (`python -m src.main`).
Orchestrates preprocessing ➜ model building ➜ training ➜ evaluation.
Keeping things minimal yet extensible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from accelerate import Accelerator
from diffusers import UNet2DConditionModel  # kept for potential future use

from .evaluate import evaluate
from .preprocess import get_dataloaders
from .train import Trainer
from .utils import build_unet, print_experiment_header

# --------------------------------------------------------------------------------------
#                                   CLI
# --------------------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("ReChuNet research prototype")
    p.add_argument("--dataset", choices=["cifar10", "imagenet"], default="cifar10")
    p.add_argument("--variant", choices=["baseline", "rechunet_rev", "rechunet_full"], default="baseline")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grad_accum", type=int, default=1)
    return p.parse_args()


# --------------------------------------------------------------------------------------
#                                   MAIN
# --------------------------------------------------------------------------------------

def main():
    args = parse_args()

    print_experiment_header(
        "ReChuNet – memory-efficient diffusion U-Net", f"variant = {args.variant}")

    # ---------- data ----------
    train_loader, val_loader = get_dataloaders(args.dataset, batch=args.batch)

    # ---------- model ----------
    model = build_unet(args.variant, chunk_size=16)

    # ---------- train ----------
    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum, mixed_precision="fp16")
    trainer = Trainer(
        accelerator=accelerator,
        model=model,
        dataloader=train_loader,
        lr=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        num_train_steps=args.steps,
        output_dir="outputs",
        seed=args.seed,
        log_every=50,
    )
    trainer.train()

    # ---------- evaluate ----------
    evaluate(model, val_loader, accelerator, max_batches=50, out_dir="outputs")


if __name__ == "__main__":
    main()
