"""
preprocess.py – data-loading helpers
Only *very* light preprocessing is required because we delegate all heavy
transformations to torchvision transforms and decord.
"""
from __future__ import annotations

import random, os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image

from transformers import CLIPTokenizer
import decord

def _basic_img_transform(res: int):
    return T.Compose([
        T.Resize(res, max_size=res),
        T.CenterCrop(res),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])

# -----------------------------------------------------------------------------
# MS-COCO (captions only) – used for experiment-1
# -----------------------------------------------------------------------------

class _CocoDataset(Dataset):
    def __init__(self, root: str, split: str, img_res: int, max_samples: int):
        ann_file = Path(root) / f"annotations/captions_{split}.json"
        if not ann_file.exists():
            raise FileNotFoundError("COCO annotation file not found. Please download the dataset first.")
        from torchvision.datasets import CocoCaptions
        self.ds = CocoCaptions(root=str(Path(root) / split), annFile=str(ann_file))
        idx = list(range(len(self.ds)))
        random.shuffle(idx)
        self.idxs = idx[:max_samples]
        self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.proc = _basic_img_transform(img_res)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        img, caps = self.ds[self.idxs[i]]
        img = self.proc(img.convert("RGB"))
        sent = caps[0].lower().strip()
        tokens = self.tok(sent, padding="max_length", max_length=77, return_tensors="pt", truncation=True)
        return img, tokens["input_ids"].squeeze(0)


def get_coco_loader(root: str, split: str, img_res: int, batch_size: int, max_samples: int):
    ds = _CocoDataset(root, split, img_res, max_samples)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True), None

# -----------------------------------------------------------------------------
# Fallback random-tensor loader – keeps the script runnable anywhere
# -----------------------------------------------------------------------------

class _RandomDS(Dataset):
    def __init__(self, img_res: int):
        self.img_res = img_res
        self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    def __len__(self):
        return 1024

    def __getitem__(self, _):
        img = torch.randn(3, self.img_res, self.img_res)
        txt = "a photo of nothing"
        tok = self.tok(txt, padding="max_length", max_length=77, return_tensors="pt")
        return img, tok["input_ids"].squeeze(0)


def get_random_loader(img_res: int, batch_size: int):
    ds = _RandomDS(img_res)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    return loader, loader

# -----------------------------------------------------------------------------
# UCF-101 clips – used for experiment-2 (tiny demo)
# -----------------------------------------------------------------------------

class _UCFClips(Dataset):
    def __init__(self, root: str, split_file: str, frames: int = 16, res: Tuple[int, int] = (256, 448)):
        self.frames = frames; self.res = res
        with open(split_file) as fp:
            self.paths = [Path(root) / line.strip() for line in fp.readlines()]
        self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        vr = decord.VideoReader(str(path), ctx=decord.cpu())
        tot = len(vr)
        start = random.randint(0, max(0, tot - self.frames * 3))
        inds = [start + i * 3 for i in range(self.frames)]
        frames = vr.get_batch(inds).permute(0, 3, 1, 2).float() / 255.0
        frames = torch.nn.functional.interpolate(frames, size=self.res, mode="bilinear", align_corners=False)
        frames = frames * 2 - 1
        txt = path.parent.name.replace("_", " ")
        tok = self.tok(txt, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        return frames, tok["input_ids"].squeeze(0)


def get_ucf_loader(root: str, split_file: str, batch_size: int):
    ds = _UCFClips(root, split_file)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
