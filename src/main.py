"""src/main.py
Main entry point.  Execute experiments via
    python -m src.main --model rechu   # or baseline
All paths/figures are stored inside the repository structure demanded by the
assignment.  The script is intentionally *very* light so that CI pipelines or
course VMs with a single Tesla T4 can finish quickly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .preprocess import set_seed
from .train import train_model
from .evaluate import evaluate_model


# ----------------------------------------------------------------------------
# CLI ------------------------------------------------------------------------
# ----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ReChuNet – tiny experimental runner")
    p.add_argument("--model", default="rechu", choices=["rechu", "baseline"], help="which UNet variant to use")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=200, help="number of update steps (very small by default)")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=2024)
    return p


# ----------------------------------------------------------------------------
# main -----------------------------------------------------------------------
# ----------------------------------------------------------------------------

def main():
    args = build_parser().parse_args()

    set_seed(args.seed)

    print("==========  TRAIN  ==========")
    csv_path, loss_fig = train_model(args)

    print("==========  EVAL   ==========")
    eval_fig, metrics = evaluate_model(args, model_ckpt=Path("models") / f"unet_{args.model}.pt")

    # final stdout -----------------------------------------------------------
    print("\n==========  SUMMARY  ==========")
    print(f"train log : {csv_path}")
    print(f"loss plot : {loss_fig}")
    print(f"eval plot : {eval_fig}")
    print(f"metrics   : {metrics}")


if __name__ == "__main__":
    main()
