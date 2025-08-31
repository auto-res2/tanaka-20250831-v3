"""src/main.py
Entry point for  *python -m src.main*  as required by the instructions.  The
script orchestrates one end-to-end run: synthetic-data creation ➜ training ➜
evaluation ➜ PDF plot written to .research/iteration8/images/.

For quick validation on a Tesla-T4 the default configuration trains only 300
iterations.  Use the *--fast False* flag if you want the full 2-epoch demo.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

# local imports (must be relative)
from .preprocess import get_dataloaders
from .train import run_training
from .evaluate import evaluate
from .utils import ensure_dir


def plot_loss(loss_curve, out_path: Path):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_curve, label="train-loss", color="tab:blue")
    plt.title("Training loss curve")
    plt.xlabel("Iteration")
    plt.ylabel("MSE loss")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "checkpoint", "lora", "rest"], default="rest",
                        help="Training mode – baseline or ReST")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    cfg: Dict = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "resolution": args.resolution,
        "device": device,
        "lr": 1e-4,
        "train_steps": args.train_steps,
        "seed": args.seed,
    }

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size, resolution=args.resolution)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    model, stats = run_training(cfg, train_loader, val_loader)

    # ------------------------------------------------------------------
    # Evaluation (extra sanity-check)
    # ------------------------------------------------------------------
    eval_stats = evaluate(model, val_loader, device=device)
    stats.update(eval_stats)

    # ------------------------------------------------------------------
    # Output handling
    # ------------------------------------------------------------------
    img_dir = Path(".research/iteration8/images")
    ensure_dir(img_dir)
    loss_fig = img_dir / "training_loss_curve.pdf"
    plot_loss(stats["loss_curve"], loss_fig)

    print("\n================ Experiment Summary ================")
    print(json.dumps({k: v for k, v in stats.items() if k != "loss_curve"}, indent=2))
    print(f"\nTraining loss curve saved to: {loss_fig.relative_to(Path('.'))}")


if __name__ == "__main__":
    main()
