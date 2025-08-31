"""
evaluate.py – small helper module with quantitative metrics and plotting
"""
from __future__ import annotations

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import seaborn as sns, matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path

__all__ = [
    "FIDEvaluator",
    "save_lineplot",
    "save_barplot",
    "human_readable_size",
]

# -----------------------------------------------------------------------------
# All figures must be stored in this directory (created on the fly)
# -----------------------------------------------------------------------------
_IM_DIR = Path(".research/iteration2/images")
_IM_DIR.mkdir(parents=True, exist_ok=True)


class FIDEvaluator:
    """Thin wrapper around torchmetrics' FrechetInceptionDistance so that we can
    accumulate results across the whole training run.
    """

    def __init__(self, device: str | torch.device = "cuda"):
        self._impl = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    @torch.no_grad()
    def update(self, imgs: torch.Tensor, recons: torch.Tensor):
        self._impl.update(imgs, real=True)
        self._impl.update(recons, real=False)

    def compute(self) -> float:
        return float(self._impl.compute())


# ----------------------------------------------------------------------------
# Helper plotting utilities (PDF, high-quality)
# ----------------------------------------------------------------------------

def _resolve_path(filename: str | Path) -> Path:
    """Return absolute path inside the mandated images directory."""
    return _IM_DIR / Path(filename).name


def save_lineplot(
    x: List[int],
    ys: Dict[str, List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
):
    plt.figure(figsize=(6, 4))
    for label, y in ys.items():
        sns.lineplot(x=x, y=y, label=label)
        plt.text(x[-1], y[-1], f"{y[-1]:.2f}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(_resolve_path(filename), bbox_inches="tight", dpi=300)
    plt.close()


def save_barplot(
    categories: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    filename: str,
):
    plt.figure(figsize=(6, 4))
    sns.barplot(x=categories, y=values, palette="viridis")
    for idx, val in enumerate(values):
        plt.text(idx, val, f"{val:.2f}", ha="center", va="bottom")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(_resolve_path(filename), bbox_inches="tight", dpi=300)
    plt.close()


def human_readable_size(num_bytes: int) -> str:
    return f"{num_bytes / 1024 ** 3:.2f} GB"
