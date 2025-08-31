"""src/evaluate.py
----------------------------------
Simple evaluation script that loads the model produced by `train.py` and
computes a reconstruction MSE on a held-out validation set.
Figures are saved as PDF in `.research/iteration14/images`.
"""
from __future__ import annotations

import pathlib
from typing import Dict

import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import helpers (compatible with both package and script execution)
# ---------------------------------------------------------------------------
try:
    from .preprocess import get_dataloaders  # type: ignore
    from .train import TinyUNet, _apply_rechu_if_available  # type: ignore
except ImportError:  # executed as a script
    from preprocess import get_dataloaders  # type: ignore
    from train import TinyUNet, _apply_rechu_if_available  # type: ignore


def evaluate(cfg: Dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval]  Using device: {device}")

    # 1. data
    val_loader = get_dataloaders(cfg)[2]

    # 2. model
    model = TinyUNet(base_channels=cfg.get("base_channels", 32))
    model_path = pathlib.Path(cfg.get("model_path", "models/toy_unet.pt"))
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model = _apply_rechu_if_available(model, cfg).to(device)
    if device.type == "cuda":
        model = model.half()
    model.eval()

    # 3. loop
    mse_total, n_pixels = 0.0, 0
    imgs, preds = None, None  # for visualisation later
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="evaluating"):
            imgs = batch["pixel_values"].to(device)
            if device.type == "cuda":
                imgs = imgs.half()
            preds = model(imgs)
            mse_total += F.mse_loss(preds.float(), imgs.float(), reduction="sum").item()
            n_pixels += imgs.numel()

    mse = mse_total / n_pixels
    print(f"[eval]  MSE reconstruction error: {mse:.6f}")

    # 4. save a qualitative visualisation (first 4 images)
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    imgs_vis = imgs[:4].cpu() * 0.5 + 0.5  # type: ignore[arg-type]
    preds_vis = preds[:4].cpu() * 0.5 + 0.5  # type: ignore[arg-type]

    fig, axes = plt.subplots(4, 2, figsize=(4, 8))
    for i in range(4):
        axes[i, 0].imshow(imgs_vis[i].permute(1, 2, 0))
        axes[i, 0].axis("off")
        axes[i, 1].imshow(preds_vis[i].permute(1, 2, 0))
        axes[i, 1].axis("off")
    fig.suptitle("Ground-truth (left) vs. reconstruction (right)")
    img_dir = pathlib.Path(".research/iteration14/images")
    img_dir.mkdir(parents=True, exist_ok=True)
    fig_path = img_dir / "qualitative_eval.pdf"
    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"[eval]  Qualitative figure saved → {fig_path.relative_to(pathlib.Path.cwd())}")

    return mse
