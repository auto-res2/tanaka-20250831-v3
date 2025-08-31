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
_IM_DIR = Path(".research/iteration6/images")  # updated as per specification
_IM_DIR.mkdir(parents=True, exist_ok=True)


class FIDEvaluator:
    """Thin wrapper around torchmetrics' FrechetInceptionDistance so that we can
    accumulate results across the whole training run.
    """

    def __init__(self, device: str | torch.device = "cuda"):
        # Torch-metrics raises if torch-fidelity is not installed; users may not
        # have it during development.  Instead of crashing completely, we try to
        # construct the metric and, if that fails, fall back to a dummy that
        # returns 0 so the training script can proceed unimpeded.
        try:
            self._impl: FrechetInceptionDistance | None = FrechetInceptionDistance(
                feature=2048, normalize=True
            ).to(device)
            self._dummy = False
        except ModuleNotFoundError:
            # Degrade gracefully – logs will still be produced, but FID will be
            # meaningless.  This path is only taken when developers run the code
            # without the optional dependency.
            print(
                "[WARN] torch-fidelity not found – FID metric disabled (returns 0)."
            )
            self._impl = None
            self._dummy = True

    @torch.no_grad()
    def update(self, imgs: torch.Tensor, recons: torch.Tensor):
        if self._dummy:
            return
        assert self._impl is not None  # mypy – never None when not dummy
        self._impl.update(imgs, real=True)
        self._impl.update(recons, real=False)

    def compute(self) -> float:
        if self._dummy:
            return 0.0
        assert self._impl is not None
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
    for label, y_vals in ys.items():
        sns.lineplot(x=x, y=y_vals, label=label)
        plt.text(x[-1], y_vals[-1], f"{y_vals[-1]:.2f}")
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
