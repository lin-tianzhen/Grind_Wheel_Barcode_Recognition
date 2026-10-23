"""Torch dataset for the synthetic one-shot barcode scenes."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import load_json


class BarcodeSceneDataset(Dataset):
    def __init__(self, root: Path, split: str = "train", augment: bool = False,
                 size: int | None = None):
        self.root = Path(root)
        self.split = split
        self.augment = augment
        self.size = size
        manifest = load_json(self.root / f"{split}_manifest.json")
        self.items = [m for m in manifest if m.get("has_barcode", True)]
        negatives = [m for m in manifest if not m.get("has_barcode", True)]
        # Keep a few pure-negative scenes: they teach the network not to fire on
        # the abrasive texture.  They are oversampled here as hard negatives.
        if negatives:
            self.items += negatives * max(1, len(self.items) // max(1, len(negatives)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        name = item["name"]
        img = cv2.imread(str(self.root / self.split / "images" / f"{name}.jpg"),
                         cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.root / self.split / "masks" / f"{name}.png"),
                          cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(name)
        if self.size is not None and img.shape[0] != self.size:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            # Light on-the-fly photometric augmentation (geometry is already
            # randomised by the scene generator).
            if np.random.rand() < 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-15, 15)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
            if np.random.rand() < 0.25:
                img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(0.2, 0.8))
            if np.random.rand() < 0.20:
                img = np.clip(img.astype(np.float32) +
                              np.random.normal(0, 4, img.shape), 0, 255).astype(np.uint8)

        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        x = (x - 0.5) / 0.5
        return x, y
