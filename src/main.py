"""
main.py – experiment entry-point (python -m src.main).
It wires together preprocessing, model creation, training,
visualisation and (optional) evaluation.
"""
from __future__ import annotations
import argparse, json, pathlib, torch, random, os

from .preprocess import get_dataloaders
from .train import build_model, train, save_plots
from .evaluate import evaluate as eval_fn


def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run(cfg):
    # Data
    train_loader, _ = get_dataloaders(batch_size=cfg["batch"])
    # Model
    model = build_model(cfg)
    # Training
    stats = train(model, train_loader, cfg)
    # Save artefacts
    models_dir = pathlib.Path("models"); models_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_dir / f"{cfg['run_name']}.pt")
    save_plots(stats, cfg['run_name'])
    # Persist config for evaluation script
    pathlib.Path("config").mkdir(exist_ok=True)
    with open(f"config/{cfg['run_name']}.json", "w") as f:
        json.dump(cfg, f, indent=2)


def make_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "reversible"], default="baseline")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = make_argparser().parse_args()
    set_seed(args.seed)
    cfg = vars(args)
    cfg["run_name"] = f"{args.model}_c{args.channels}_s{args.seed}"
    run(cfg)

if __name__ == "__main__":
    main()
