"""
evaluate.py – very small evaluation that computes reconstruction
loss (MSE) on the validation set and prints it.  It is deliberately
light-weight to respect VRAM limits.
"""
from __future__ import annotations
import torch, pathlib, json
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms

from .train import build_model


def evaluate(cfg_path: str):
    cfg = json.load(open(cfg_path))
    # Data
    test_tf = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=test_tf)
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    # Model
    model = build_model(cfg)
    model.load_state_dict(torch.load(pathlib.Path("models")/f"{cfg['run_name']}.pt"))
    model.eval()
    mse_total, n = 0.0, 0
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        for x,_ in loader:
            x = x.to("cuda", dtype=torch.float16)
            out = model(x)
            mse_total += F.mse_loss(out, x, reduction="sum").item()
            n += x.numel()
    print(f"Validation MSE: {mse_total/n:.6f}")
