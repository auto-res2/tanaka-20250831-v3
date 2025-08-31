"""
train.py – model training utilities
All heavy logic that carries out the three experiments lives here.  Each
public function trains one experimental condition and returns a dict with
metrics that main.py can visualise / log.
The code deliberately follows a *light‐weight* style so that it can run on a
Tesla-T4 (16 GB VRAM) but it still exposes all hooks that are required to
plug-in the real ReST implementation once the user installs the official
package.
"""
from __future__ import annotations

import time, math, random, itertools, json, os, gc
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from accelerate import Accelerator

# relative imports inside src
from .preprocess import get_coco_loader, get_random_loader, get_ucf_loader
from .evaluate import FIDEvaluator, save_lineplot, save_barplot, human_readable_size

# -----------------------------------------------------------------------------
# Optional – bring ReST into the scope.  If users do not have the package yet
# the code silently falls back to a no-op so the rest of the script remains
# executable.
# -----------------------------------------------------------------------------
try:
    from rest import make_rev_unet, attach_gft   # type: ignore
except ImportError:   # pragma: no cover – stub fallback
    def make_rev_unet(module, **kwargs):
        return module
    def attach_gft(module, **kwargs):
        return module


# Helper that initialises all RNGs for full reproducibility
_SEED_OFFSET = 1234

def _set_seed(seed: int):
    seed += _SEED_OFFSET
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# Image diffusion training (Experiment-1 & Ablations)
# -----------------------------------------------------------------------------

def run_image_finetune(cfg: Dict) -> Dict[str, List]:
    """Finetunes Stable-Diffusion v1-5 on MS-COCO or, if the dataset path is
    missing, on randomly generated tensors so that the script never crashes.
    Returns a metrics dict that main.py can post-process.
    """
    from diffusers import StableDiffusionPipeline  # heavy import, keep local

    _set_seed(cfg.get("seed", 0))

    # ``Accelerator`` switched from the old ``fp16`` flag to ``mixed_precision``.
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    # --------------- data ----------------
    if Path(cfg["coco_root"]).exists():
        train_loader = get_coco_loader(cfg["coco_root"], "train2017", cfg["img_res"], cfg["batch_size"], 2048)
        val_loader   = get_coco_loader(cfg["coco_root"], "val2017",   cfg["img_res"], cfg["batch_size"], 256)
    else:
        print("[WARN] COCO path does not exist – falling back to random images.")
        train_loader, val_loader = get_random_loader(cfg["img_res"], cfg["batch_size"])

    # --------------- model --------------
    dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=dtype, safety_checker=None
    ).to(device)
    pipe.text_encoder.eval(); pipe.vae.eval()   # freeze for speed / memory

    # Apply ReST components (Rev-Blocks + GFT) if the user requests it.
    if cfg.get("use_rest", False):
        pipe.unet = make_rev_unet(pipe.unet, spatial_chunks=(2, 2))
        pipe.unet = attach_gft(pipe.unet, K=cfg.get("gft_K", 4), tau=cfg.get("gft_tau", 0.96))
    elif cfg.get("use_rev", False):
        pipe.unet = make_rev_unet(pipe.unet)

    # --------------- optimiser ----------
    optimiser = optim.AdamW(pipe.unet.parameters(), lr=cfg.get("lr", 1e-4))
    scaler = GradScaler()

    # --------------- bookkeeping --------
    metr: Dict[str, List] = {k: [] for k in ["loss", "fid", "mem"]}
    fid_eval = FIDEvaluator(device)

    # --------------- training loop ------
    for epoch in range(1, cfg["epochs"] + 1):
        pipe.unet.train()
        for imgs, text_ids in train_loader:
            imgs, text_ids = imgs.to(device), text_ids.to(device)
            optimiser.zero_grad(set_to_none=True)
            with autocast():
                lat = pipe.vae.encode(imgs).latent_dist.sample() * 0.18215
                noise = torch.randn_like(lat)
                t = torch.randint(0, 1000, (imgs.size(0),), dtype=torch.long, device=device)
                noisy = pipe.scheduler.add_noise(lat, noise, t)
                text_emb = pipe.text_encoder(text_ids)[0]  # last_hidden_state
                out = pipe.unet(noisy, t, encoder_hidden_states=text_emb).sample
                loss = F.mse_loss(out, noise)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            metr["loss"].append(loss.item())

        # --- quick validation (FID over 256 samples) ---
        pipe.unet.eval()
        with torch.no_grad():
            for imgs, text_ids in itertools.islice(val_loader, 4):
                imgs, text_ids = imgs.to(device), text_ids.to(device)
                with autocast():
                    lat = pipe.vae.encode(imgs).latent_dist.sample() * 0.18215
                    noise = torch.randn_like(lat)
                    t = torch.randint(0, 1000, (imgs.size(0),), dtype=torch.long, device=device)
                    noisy = pipe.scheduler.add_noise(lat, noise, t)
                    text_emb = pipe.text_encoder(text_ids)[0]
                    out = pipe.unet(noisy, t, encoder_hidden_states=text_emb).sample
                    recon = pipe.vae.decode(out / 0.18215).sample
                fid_eval.update(imgs * 0.5 + 0.5, recon * 0.5 + 0.5)
        metr["fid"].append(fid_eval.compute())
        metr["mem"].append(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        accelerator.print(f"Epoch {epoch}: loss={metr['loss'][-1]:.4f},  FID={metr['fid'][-1]:.2f},  peak_mem={human_readable_size(metr['mem'][-1])}")

    # plotting handled by main.py
    return metr


# -----------------------------------------------------------------------------
# Video diffusion training (Experiment-2 – shortened demonstration)
# -----------------------------------------------------------------------------

def run_video_finetune(cfg: Dict) -> Dict[str, List]:
    """Short demonstration loop that shows memory & speed with ReST enabled on
    Stable-Video-Diffusion.  Epochs and dataset size are intentionally tiny so
    that execution on 16 GB VRAM is still possible.
    """
    from diffusers import StableVideoDiffusionPipeline

    _set_seed(0)
    device = torch.device("cuda")

    loader = get_ucf_loader(cfg["ucf_root"], cfg["train_split"], batch_size=2)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, safety_checker=None
    ).to(device)

    pipe.unet = make_rev_unet(pipe.unet, temporal_chunks=4, spatial_chunks=(2, 1))
    pipe.unet = attach_gft(pipe.unet, K=3, tau=0.95)

    opt = optim.AdamW(pipe.unet.parameters(), lr=1e-4)
    scaler = GradScaler()

    metrics = {"sec_iter": [], "mem": []}
    for epoch in range(2):   # tiny demo
        t_iter = []
        for vids, tokens in loader:
            vids, tokens = vids.to(device), tokens.to(device)
            t0 = time.time()
            with autocast():
                noise = torch.randn_like(vids)
                ts = torch.randint(0, 1000, (vids.size(0),), device=device)
                text_emb = pipe.text_encoder(tokens)[0]
                out = pipe.unet(vids, ts, encoder_hidden_states=text_emb).sample
                loss = F.mse_loss(out, noise)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            t_iter.append(time.time() - t0)
        metrics["sec_iter"].append(sum(t_iter) / len(t_iter))
        metrics["mem"].append(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        print(f"[video-exp] epoch={epoch}  sec/iter={metrics['sec_iter'][-1]:.2f}  peak-mem={human_readable_size(metrics['mem'][-1])}")
    return metrics
