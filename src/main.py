"""
main.py – central CLI entry-point that wires together preprocessing, training
and evaluation for the three experiments described in the research plan.
Run from the project root via
    $ python -m src.main --exp 1
"""
from __future__ import annotations

import argparse, json, os
import pandas as pd

from .train import run_image_finetune, run_video_finetune
from .evaluate import save_lineplot, save_barplot

# -----------------------------------------------------------------------------
# configuration helpers – tiny, yaml could be added later via the config/ dir
# -----------------------------------------------------------------------------
_DEF_IMG_CFG = {
    "coco_root": "data/coco2017",   # change to your path
    "img_res": 768,
    "batch_size": 4,
    "epochs": 3,
    "seed": 0,
    "lr": 1e-4,
    "use_rest": True,   # toggle ReST
}

_DEF_VID_CFG = {
    "ucf_root": "data/ucf101/videos",
    "train_split": "data/ucf101/trainlist01.txt",
    "img_res": 256,
}

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _json(msg, obj):
    print(msg)
    print(json.dumps(obj, indent=2))


def _exp1():
    cfg = _DEF_IMG_CFG
    _json("[Experiment-1] configuration", cfg)
    metrics = run_image_finetune(cfg)

    # -------- plots --------
    xs = list(range(1, len(metrics["fid"]) + 1))
    save_lineplot(xs, {"ReST": metrics["fid"]}, "FID over epochs", "Epoch", "FID", "fid_curve.pdf")

    df = pd.DataFrame(metrics)
    df.to_csv("exp1_metrics.csv", index=False)
    print("Figures saved: fid_curve.pdf  |  CSV: exp1_metrics.csv")


def _exp2():
    cfg = _DEF_VID_CFG
    _json("[Experiment-2] configuration", cfg)
    m = run_video_finetune(cfg)
    save_barplot(["sec/iter", "peak-mem (GB)"], [sum(m["sec_iter"]) / len(m["sec_iter"]), max(m["mem"]) / 1024 ** 3], "Speed & Memory", "value", "video_speed_mem.pdf")
    print("Figure saved: video_speed_mem.pdf")


def _exp3():
    # Very small ablation – simply compares memory consumption with/without ReST
    cfg_base = _DEF_IMG_CFG.copy()
    cfg_base.update(img_res=512, epochs=1)
    results = {}
    for name, rest_flag in [("plain", False), ("rest", True)]:
        cfg = cfg_base.copy(); cfg["use_rest"] = rest_flag
        r = run_image_finetune(cfg)
        results[name] = max(r["mem"]) / 1024 ** 3
    save_barplot(list(results.keys()), list(results.values()), "Peak Memory @512²", "GB", "ablation_memory.pdf")
    print("Figure saved: ablation_memory.pdf")


def main():
    parser = argparse.ArgumentParser(description="Run ReST experiments")
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3], help="Which experiment to run")
    args = parser.parse_args()

    if args.exp == 1:
        _exp1()
    elif args.exp == 2:
        _exp2()
    else:
        _exp3()


if __name__ == "__main__":
    main()
