"""src/main.py
----------------------------------
Entry-point that is executed via `python -m src.main` *or* `python src/main.py`.

Usage examples
--------------
Train a baseline model
    python src/main.py                # default is training mode

Train a memory-efficient ReChuNet model and then evaluate it
    python src/main.py --use_rechunet 1 --num_steps 1000
    python src/main.py --mode eval
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict

# ---------------------------------------------------------------------------
# Imports that work both in package and script mode
# ---------------------------------------------------------------------------
try:
    from .train import train as train_fn  # type: ignore
    from .evaluate import evaluate as eval_fn  # type: ignore
except ImportError:  # script mode fallback
    from train import train as train_fn  # type: ignore
    from evaluate import evaluate as eval_fn  # type: ignore

from rich import print  # pretty std-out


# ---------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------

def _parse_cfg() -> Dict:
    p = argparse.ArgumentParser(description="ReChuNet study runner")
    p.add_argument("--mode", choices=["train", "eval"], default="train", help="Execution mode (default: train)")

    # generic hyper-params
    p.add_argument("--data_root", type=str, default="")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--num_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)

    # ReChuNet flags
    p.add_argument("--use_rechunet", type=int, default=0, help="set to 1 to wrap the model with ReChuNet")
    p.add_argument("--chunk_hw", type=int, default=2)
    p.add_argument("--chunk_tb", type=int, default=2)

    args = p.parse_args()
    cfg = vars(args)
    return cfg


# ---------------------------------------------------------------------
# main dispatch
# ---------------------------------------------------------------------

def main():
    cfg: Dict = _parse_cfg()
    mode = cfg.pop("mode")

    print("[main]  Configuration")
    print(json.dumps(cfg, indent=2))

    if mode == "train":
        model_path = train_fn(cfg)
        print(f"[main]  Training finished; model saved at {model_path}")
    elif mode == "eval":
        mse = eval_fn(cfg)
        print(f"[main]  Evaluation finished; MSE={mse:.6f}")
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
