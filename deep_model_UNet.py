#!/usr/bin/env python3
"""深度学习砂轮条形码检测识别 — 训练脚本。

技术路线:
  1. U-Net 语义分割网络定位条码区域 (输出像素级 mask)
  2. 损失函数: BCEWithLogitsLoss + DiceLoss
  3. 优化器: AdamW + CosineAnnealingLR
  4. 推理: 滑窗推理 → 连通域 → 最小外接矩形 → 透视矫正 → EAN-13 解码

用法:
  python deep_model_UNet.py                          # 训练 (默认30 epochs)
  python deep_model_UNet.py --epochs 50 --size 224   # 自定义参数
  python deep_model_UNet.py --eval-only              # 仅评估已训练模型
  python deep_model_UNet.py --infer data/raw/RegulaImg.bmp  # 单张推理
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "deep"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GT_TEXT = "4014855012527"

# ============================================================================
# U-Net 模型定义
# ============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """轻量 U-Net: 4 层编码器-解码器, base_ch=16, ~1.94M 参数."""

    def __init__(self, in_ch: int = 1, base_ch: int = 16, dropout: float = 0.1):
        super().__init__()
        c = base_ch
        d = dropout
        self.enc1 = ConvBlock(in_ch, c, dropout=d * 0.5)
        self.enc2 = ConvBlock(c, c * 2, dropout=d * 0.5)
        self.enc3 = ConvBlock(c * 2, c * 4, dropout=d)
        self.enc4 = ConvBlock(c * 4, c * 8, dropout=d)
        self.bottleneck = ConvBlock(c * 8, c * 16, dropout=d * 1.5)
        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = ConvBlock(c * 16, c * 8, dropout=d)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = ConvBlock(c * 8, c * 4, dropout=d)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = ConvBlock(c * 4, c * 2, dropout=d * 0.5)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = ConvBlock(c * 2, c, dropout=d * 0.5)
        self.head = nn.Conv2d(c, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)

# ============================================================================
# 损失函数
# ============================================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (2, 3)
        inter = (probs * target).sum(dims)
        card = probs.sum(dims) + target.sum(dims)
        dice = (2 * inter + self.smooth) / (card + self.smooth)
        return (1 - dice).mean()

class CombinedLoss(nn.Module):
    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight_val = pos_weight
        self._pos_weight = torch.tensor([pos_weight])
        self._bce = None
        self._bce_device = None
        self.dice = DiceLoss()

    def _get_bce(self, device: torch.device) -> nn.BCEWithLogitsLoss:
        if self._bce is None or self._bce_device != device:
            self._bce = nn.BCEWithLogitsLoss(
                pos_weight=self._pos_weight.to(device))
            self._bce_device = device
        return self._bce

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self._get_bce(logits.device)
        return bce(logits, target) + self.dice(logits, target)

# ============================================================================
# 数据集
# ============================================================================

class BarcodeDataset(Dataset):
    def __init__(self, root: Path, split: str = "train",
                 size: int = 224, augment: bool = False):
        self.root = Path(root)
        self.split = split
        self.size = size
        self.augment = augment
        manifest_path = self.root / f"{split}_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        self.items = [m for m in manifest if m.get("has_barcode", True)]
        negatives = [m for m in manifest if not m.get("has_barcode", True)]
        if negatives:
            ratio = max(1, len(self.items) // max(1, len(negatives)))
            self.items += negatives * min(ratio, 5)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        name = item["name"]
        img_path = self.root / self.split / "images" / f"{name}.jpg"
        mask_path = self.root / self.split / "masks" / f"{name}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        if mask is None:
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-30, 30)
                h_, w_ = img.shape[:2]
                M = cv2.getRotationMatrix2D((w_ / 2, h_ / 2), angle, 1.0)
                cos_a = abs(M[0, 0])
                sin_a = abs(M[0, 1])
                new_w = int(h_ * sin_a + w_ * cos_a)
                new_h = int(h_ * cos_a + w_ * sin_a)
                M[0, 2] += (new_w - w_) / 2
                M[1, 2] += (new_h - h_) / 2
                img = cv2.warpAffine(img, M, (new_w, new_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.warpAffine(mask, M, (new_w, new_h),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_REFLECT_101)
                img = cv2.resize(img, (self.size, self.size),
                                 interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask, (self.size, self.size),
                                  interpolation=cv2.INTER_NEAREST)
            if np.random.rand() < 0.5:
                img = img[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[::-1, :].copy()
                mask = mask[::-1, :].copy()
            if np.random.rand() < 0.3:
                sx = np.random.uniform(0.85, 1.15)
                sy = np.random.uniform(0.85, 1.15)
                h_, w_ = img.shape[:2]
                M = np.float32([[sx, 0, (1 - sx) * w_ / 2],
                                [0, sy, (1 - sy) * h_ / 2]])
                img = cv2.warpAffine(img, M, (w_, h_),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT_101)
                mask = cv2.warpAffine(mask, M, (w_, h_),
                                      flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_REFLECT_101)
            if np.random.rand() < 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-15, 15)
                img = np.clip(img.astype(np.float32) * alpha + beta,
                              0, 255).astype(np.uint8)
            if np.random.rand() < 0.25:
                img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(0.3, 0.8))
            if np.random.rand() < 0.20:
                noise = np.random.normal(0, 4, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        x = (x - 0.5) / 0.5
        y = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return x, y

# ============================================================================
# 训练逻辑
# ============================================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / max(1, n)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    iou_sum = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        inter = (preds * y).sum(dim=(1, 2, 3))
        union = ((preds + y) > 0).float().sum(dim=(1, 2, 3))
        iou_sum += (inter / (union + 1e-6)).sum().item()
        n += x.size(0)
    return total_loss / max(1, n), iou_sum / max(1, n)

def train(args):
    device = get_device()
    print(f"设备: {device}")

    train_ds = BarcodeDataset(SYNTH_DIR, "train", size=args.size, augment=True)
    val_ds = BarcodeDataset(SYNTH_DIR, "val", size=args.size, augment=False)
    print(f"训练集: {len(train_ds)} 张, 验证集: {len(val_ds)} 张")

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=0, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=0, pin_memory=pin_mem)

    model = UNet(in_ch=1, base_ch=args.base_ch, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net 参数量: {n_params / 1e6:.2f}M")

    criterion = CombinedLoss(pos_weight=args.pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_iou = 0.0
    patience_counter = 0
    patience = args.patience
    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    model_path = MODEL_DIR / "unet_barcode.pt"

    print(f"\n开始训练: {args.epochs} epochs, batch={args.batch}, "
          f"lr={args.lr}, size={args.size}, dropout={args.dropout}")
    print(f"早停: patience={patience}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        marker = ""
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_iou": val_iou,
                "args": vars(args),
                "train_size": args.size,
            }, str(model_path))
            marker = " *"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_iou={val_iou:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
              f"{elapsed:.1f}s{marker}")

        if patience_counter >= patience:
            print(f"\n早停触发: val_iou 连续 {patience} 轮未提升")
            break

    hist_path = MODEL_DIR / "train_history.json"
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("-" * 60)
    print(f"训练完成! 最佳 val IoU = {best_iou:.4f}")
    print(f"模型已保存: {model_path}")
    print(f"训练历史: {hist_path}")

# ============================================================================
# EAN-13 解码器 
# ============================================================================

L_PATTERNS = [
    "0001101", "0011001", "0010011", "0111101", "0100011",
    "0110001", "0101111", "0111011", "0110111", "0001011",
]
G_PATTERNS = [
    "0100111", "0110011", "0011011", "0100001", "0011101",
    "0111001", "0000101", "0010001", "0001001", "0010111",
]
R_PATTERNS = [
    "1110010", "1100110", "1101100", "1000010", "1011100",
    "1001110", "1010000", "1000100", "1001000", "1110100",
]
PARITY_PATTERNS = [
    "LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
    "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL",
]

def ean13_checksum(text: str) -> int:
    digits = [int(c) for c in text[:12]]
    s = sum(digits[::2]) + 3 * sum(digits[1::2])
    return (10 - s % 10) % 10

def valid_ean13(text: str) -> bool:
    return (len(text) == 13 and text.isdigit()
            and ean13_checksum(text) == int(text[-1]))

def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))

def _decode_95_bits(bits: str) -> Optional[tuple]:
    if len(bits) != 95:
        return None
    best = None
    for inv in (False, True):
        b = "".join("1" if c == "0" else "0" for c in bits) if inv else bits
        if b[:3] != "101" or b[45:50] != "01010" or b[-3:] != "101":
            continue
        cost = 0
        left_digits, parity = [], ""
        for i in range(6):
            seg = b[3 + 7 * i: 10 + 7 * i]
            cl, dl = min((_hamming(seg, L_PATTERNS[d]), d) for d in range(10))
            cg, dg = min((_hamming(seg, G_PATTERNS[d]), d) for d in range(10))
            if cl <= cg:
                cost += cl
                left_digits.append(dl)
                parity += "L"
            else:
                cost += cg
                left_digits.append(dg)
                parity += "G"
        if parity not in PARITY_PATTERNS:
            continue
        first = PARITY_PATTERNS.index(parity)
        right_digits = []
        for i in range(6):
            seg = b[50 + 7 * i: 57 + 7 * i]
            cr, dr = min((_hamming(seg, R_PATTERNS[d]), d) for d in range(10))
            cost += cr
            right_digits.append(dr)
        text = "".join(str(d) for d in [first] + left_digits + right_digits)
        if valid_ean13(text) and (best is None or cost < best[1]):
            best = (text, cost)
    return best

def _binarize_line(line: np.ndarray) -> np.ndarray:
    line = np.asarray(line).ravel()
    blur = cv2.GaussianBlur(line.reshape(1, -1), (3, 3), 0).ravel()
    th, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th < 30:
        return np.zeros_like(bw)
    if np.mean(bw > 0) > 0.50:
        bw = 255 - bw
    return bw

def _decode_scanline(line: np.ndarray) -> Optional[tuple]:
    bw = _binarize_line(line)
    idx = np.flatnonzero(bw > 0)
    if len(idx) < 50:
        return None
    gaps = np.flatnonzero(np.diff(idx) > max(12, int(0.05 * len(line))))
    groups = np.split(idx, gaps + 1)
    groups = [g for g in groups if len(g) > 40 and g[-1] - g[0] > 80]
    if not groups:
        return None
    group = max(groups, key=lambda g: len(g))
    x0, x1 = int(group[0]), int(group[-1])
    if not (180 <= x1 - x0 <= 0.95 * len(line)):
        return None
    a_values = np.arange(x0 - 6, x0 + 3, 1.0)
    z_values = np.arange(x1 - 3, x1 + 6, 1.0)
    best = None
    for a in a_values:
        for z in z_values:
            width = (z - a + 1) / 95.0
            if not (2.0 <= width <= 16.0):
                continue
            pos = a + (np.arange(95) + 0.5) * width
            pos = np.clip(np.rint(pos).astype(np.int32), 0, len(line) - 1)
            bits = "".join("1" if bw[p] > 0 else "0" for p in pos)
            res = _decode_95_bits(bits)
            if res is None:
                continue
            text, cost = res
            score = cost + 0.15 * abs(width - 4.6)
            if best is None or score < best[0]:
                best = (score, text, float(a), float(z))
    if best is None:
        return None
    score, text, a, z = best
    conf = max(0.0, min(1.0, 1.0 - score / 95.0))
    return text, conf, a, z

def decode_ean13(gray: np.ndarray) -> Optional[str]:
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    if h < 12 or w < 60:
        return None
    up = max(1.0, 520.0 / max(1, w))
    if up > 1:
        g = cv2.resize(g, None, fx=min(up, 3.0), fy=min(up, 3.0),
                       interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    g = clahe.apply(g)
    blur = cv2.GaussianBlur(g, (0, 0), 2)
    g = cv2.addWeighted(g, 1.8, blur, -0.8, 0)
    h, w = g.shape[:2]
    votes: dict[str, list[float]] = {}
    for frac in np.linspace(0.10, 0.90, 15):
        y = int(round((h - 1) * frac))
        res = _decode_scanline(g[y])
        if res is None:
            continue
        text, conf, a, z = res
        votes.setdefault(text, []).append(conf)
    if not votes:
        return None
    text, confs = max(votes.items(), key=lambda kv: (len(kv[1]), max(kv[1])))
    if valid_ean13(text):
        return text
    return None

def decode_barcode_crop(gray: np.ndarray) -> Optional[str]:
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    text = decode_ean13(g)
    if text:
        return text
    text = decode_ean13(cv2.bitwise_not(g))
    if text:
        return text
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    enh = clahe.apply(g)
    blur = cv2.GaussianBlur(enh, (0, 0), 2)
    enh = cv2.addWeighted(enh, 1.8, blur, -0.8, 0)
    text = decode_ean13(enh)
    if text:
        return text
    try:
        import zxingcpp
        for v in (g, cv2.bitwise_not(g)):
            for scale in (1.0, 2.0):
                img = cv2.resize(v, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC) if scale > 1 else v
                try:
                    results = zxingcpp.read_barcodes(
                        img, try_harder=True, try_rotate=True,
                        formats=zxingcpp.BarcodeFormat.EAN13)
                except TypeError:
                    try:
                        results = zxingcpp.read_barcodes(img)
                    except Exception:
                        results = []
                except Exception:
                    results = []
                for r in results:
                    t = str(r.text).strip()
                    if valid_ean13(t):
                        return t
    except ImportError:
        pass
    return None

# ============================================================================
# 推理逻辑
# ============================================================================

@torch.no_grad()
def sliding_window_infer(model, image: np.ndarray,
                         window: int = 512, stride: int = 384,
                         train_size: int = 224,
                         device: torch.device = None) -> np.ndarray:
    """滑窗推理: 避免整图缩放导致条码过小."""
    if device is None:
        device = get_device()
    model.eval()
    h, w = image.shape[:2]
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    positions = []
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = min(y + window, h)
            x1 = min(x + window, w)
            y0 = max(0, y1 - window)
            x0 = max(0, x1 - window)
            positions.append((y0, x0, y1, x1))

    for y0, x0, y1, x1 in positions:
        crop = image[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        if ch < 32 or cw < 32:
            continue
        input_tensor = torch.from_numpy(
            crop.astype(np.float32) / 255.0
        ).unsqueeze(0).unsqueeze(0)
        input_tensor = (input_tensor - 0.5) / 0.5
        input_tensor = F.interpolate(
            input_tensor.to(device), size=(train_size, train_size),
            mode="bilinear", align_corners=False)
        logits = model(input_tensor)
        prob = torch.sigmoid(logits)
        prob = F.interpolate(
            prob, size=(ch, cw), mode="bilinear", align_corners=False)
        prob_np = prob.squeeze().cpu().numpy()
        prob_map[y0:y1, x0:x1] += prob_np
        count_map[y0:y1, x0:x1] += 1.0

    count_map = np.maximum(count_map, 1.0)
    prob_map = prob_map / count_map
    return prob_map

def extract_barcode_from_mask(image: np.ndarray, prob_map: np.ndarray,
                              threshold: float = 0.45) -> Optional[tuple]:
    """从概率图中提取最大条码区域的旋转矩形."""
    mask = (prob_map > threshold).astype(np.uint8) * 255
    if mask.sum() == 0:
        return None
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return None
    candidates = sorted(range(1, num), key=lambda i: stats[i, cv2.CC_STAT_AREA],
                        reverse=True)
    candidates = candidates[:3]
    results = []
    for lbl in candidates:
        comp = (labels == lbl).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        w_r, h_r = rect[1]
        if w_r < 20 or h_r < 5:
            continue
        aspect = max(w_r, h_r) / max(min(w_r, h_r), 1)
        results.append((box, rect, stats[lbl, cv2.CC_STAT_AREA], aspect))
    if not results:
        return None
    results.sort(key=lambda x: x[2], reverse=True)
    box, rect = results[0][0], results[0][1]
    return box, rect

def extract_barcode_multi_threshold(image: np.ndarray,
                                     prob_map: np.ndarray) -> Optional[tuple]:
    """多阈值尝试提取条码区域."""
    for threshold in [0.45, 0.35, 0.55, 0.30, 0.60, 0.25]:
        result = extract_barcode_from_mask(image, prob_map, threshold)
        if result is not None:
            return result
    return None

def perspective_correct(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    """透视矫正: 将旋转矩形矫正为水平条码."""
    box = np.asarray(box, dtype=np.float32)
    center = box.mean(axis=0)
    rel = box - center
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    order = np.argsort(angles)
    box = box[order]
    best_box = None
    best_score = -1
    for shift in range(4):
        trial = np.roll(box, -shift, axis=0)
        w_top = np.linalg.norm(trial[0] - trial[1])
        w_bot = np.linalg.norm(trial[3] - trial[2])
        h_left = np.linalg.norm(trial[0] - trial[3])
        h_right = np.linalg.norm(trial[1] - trial[2])
        avg_w = (w_top + w_bot) / 2
        avg_h = (h_left + h_right) / 2
        if avg_w >= avg_h:
            score = avg_w / max(avg_h, 1)
        else:
            trial = trial[[3, 0, 1, 2]]
            w_top = np.linalg.norm(trial[0] - trial[1])
            w_bot = np.linalg.norm(trial[3] - trial[2])
            h_left = np.linalg.norm(trial[0] - trial[3])
            h_right = np.linalg.norm(trial[1] - trial[2])
            avg_w = (w_top + w_bot) / 2
            avg_h = (h_left + h_right) / 2
            score = avg_w / max(avg_h, 1)
        if score > best_score:
            best_score = score
            best_box = trial.copy()
            best_max_w = max(int(w_top), int(w_bot))
            best_max_h = max(int(h_left), int(h_right))
    if best_box is None:
        best_box = box
        best_max_w = max(int(np.linalg.norm(box[0] - box[1])),
                         int(np.linalg.norm(box[3] - box[2])))
        best_max_h = max(int(np.linalg.norm(box[0] - box[3])),
                         int(np.linalg.norm(box[1] - box[2])))
    best_max_w = max(best_max_w, 1)
    best_max_h = max(best_max_h, 1)
    dst = np.array([[0, 0], [best_max_w - 1, 0],
                    [best_max_w - 1, best_max_h - 1], [0, best_max_h - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(best_box, dst)
    warped = cv2.warpPerspective(image, M, (best_max_w, best_max_h),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped

def infer_single(model, image_path: Path, device: torch.device = None,
                 window: int = 512, stride: int = 384,
                 train_size: int = 224) -> dict:
    """对单张图像进行端到端推理: 定位 → 矫正 → 解码."""
    if device is None:
        device = get_device()
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"ok": False, "text": "", "error": "无法读取图像"}
    h, w = img.shape[:2]
    t0 = time.perf_counter()

    if max(h, w) > 2048:
        scale = 2048 / max(h, w)
        img_infer = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
    else:
        img_infer = img
        scale = 1.0

    prob_map = sliding_window_infer(model, img_infer, window, stride,
                                    train_size, device)
    result = extract_barcode_multi_threshold(img_infer, prob_map)

    text = None
    bbox = None
    if result is not None:
        box, rect = result
        bbox = (box / scale).tolist()
        warped = perspective_correct(img_infer, box)
        text = decode_barcode_crop(warped)
        if text is None:
            for angle in [90, 180, 270]:
                M_rot = cv2.getRotationMatrix2D(
                    (warped.shape[1] / 2, warped.shape[0] / 2), angle, 1.0)
                rotated = cv2.warpAffine(warped, M_rot,
                                         (warped.shape[1], warped.shape[0]),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REPLICATE)
                text = decode_barcode_crop(rotated)
                if text:
                    break
        if text is None:
            for pad in [10, 20, 30, 50]:
                x0 = int(box[:, 0].min()) - pad
                y0 = int(box[:, 1].min()) - pad
                x1 = int(box[:, 0].max()) + pad
                y1 = int(box[:, 1].max()) + pad
                x0, y0 = max(0, x0), max(0, y0)
                x1 = min(img_infer.shape[1], x1)
                y1 = min(img_infer.shape[0], y1)
                crop = img_infer[y0:y1, x0:x1]
                text = decode_barcode_crop(crop)
                if text:
                    break

    if text is None:
        text = decode_barcode_crop(img)

    elapsed = (time.perf_counter() - t0) * 1000
    ok = text is not None and text == GT_TEXT
    return {
        "ok": ok,
        "text": text or "",
        "bbox": bbox,
        "elapsed_ms": elapsed,
        "checksum_ok": valid_ean13(text) if text else False,
    }

def evaluate(model, device: torch.device = None, train_size: int = 224):
    """在验证集上评估模型."""
    if device is None:
        device = get_device()
    val_ds = BarcodeDataset(SYNTH_DIR, "val", size=train_size, augment=False)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    criterion = CombinedLoss()
    val_loss, val_iou = validate(model, val_loader, criterion, device)
    print(f"\n验证集评估: val_loss={val_loss:.4f}, val_iou={val_iou:.4f}")

    raw_images = sorted(RAW_DIR.glob("*.bmp"))
    if not raw_images:
        return
    print(f"\n在 {len(raw_images)} 张原始图像上测试:")
    print("-" * 60)
    success = 0
    for img_path in raw_images:
        res = infer_single(model, img_path, device, train_size=train_size)
        status = "✓" if res["ok"] else "✗"
        text_disp = res["text"] if res["text"] else "—"
        match = " (匹配)" if res["ok"] else ""
        print(f"  {status} {img_path.name}: {text_disp}{match} "
              f"({res['elapsed_ms']:.0f}ms)")
        if res["ok"]:
            success += 1
    print(f"\n原始图像识别率: {success}/{len(raw_images)}")

# ============================================================================
# 可视化
# ============================================================================

def visualize_prediction(image: np.ndarray, prob_map: np.ndarray,
                         result_dict: dict, save_path: Path):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    prob_color = cv2.applyColorMap(
        (np.clip(prob_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(vis, 0.6, prob_color, 0.4, 0)
    if result_dict.get("bbox") is not None:
        box = np.array(result_dict["bbox"], dtype=np.int32)
        cv2.drawContours(overlay, [box], 0, (0, 255, 0), 2)
        text = result_dict.get("text", "")
        if text:
            x = int(box[:, 0].min())
            y = int(box[:, 1].min()) - 10
            cv2.putText(overlay, text, (x, max(y, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    h, w = overlay.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        overlay = cv2.resize(overlay, None, fx=s, fy=s,
                             interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(save_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])

# ============================================================================
# 主程序
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="深度学习砂轮条形码检测识别")
    parser.add_argument("--epochs", type=int, default=80,
                        help="训练轮数 (默认80)")
    parser.add_argument("--batch", type=int, default=8,
                        help="批大小 (默认8)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率 (默认1e-3)")
    parser.add_argument("--wd", type=float, default=1e-4,
                        help="权重衰减 (默认1e-4)")
    parser.add_argument("--size", type=int, default=224,
                        help="训练图像尺寸 (默认224)")
    parser.add_argument("--base-ch", type=int, default=16,
                        help="U-Net 基础通道数 (默认16)")
    parser.add_argument("--pos-weight", type=float, default=2.0,
                        help="正样本权重 (默认2.0)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 率 (默认0.1)")
    parser.add_argument("--patience", type=int, default=15,
                        help="早停耐心轮数 (默认15)")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估已训练模型")
    parser.add_argument("--infer", type=str, default=None,
                        help="对单张图像推理")
    args = parser.parse_args()

    print("=" * 60)
    print("  深度学习砂轮条形码检测识别")
    print("  U-Net 分割 + 滑窗推理 + EAN-13 解码")
    print("=" * 60)

    device = get_device()
    print(f"设备: {device}\n")

    if args.infer:
        model = UNet(in_ch=1, base_ch=args.base_ch, dropout=0.0).to(device)
        model_path = MODEL_DIR / "unet_barcode.pt"
        if not model_path.exists():
            print(f"[错误] 模型文件不存在: {model_path}")
            print("请先运行训练: python train_deep.py")
            return
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        train_size = ckpt.get("train_size", ckpt.get("args", {}).get("size", 224))
        print(f"已加载模型: {model_path} (epoch {ckpt['epoch']}, "
              f"val_iou={ckpt['val_iou']:.4f}, train_size={train_size})")
        img_path = Path(args.infer)
        res = infer_single(model, img_path, device, train_size=train_size)
        status = "✓ 成功" if res["ok"] else "✗ 失败"
        print(f"\n结果: {status}")
        print(f"条码值: {res['text'] or '—'}")
        print(f"校验位: {'通过' if res['checksum_ok'] else '未通过'}")
        print(f"耗时: {res['elapsed_ms']:.0f} ms")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            if max(img.shape) > 2048:
                s = 2048 / max(img.shape)
                img_r = cv2.resize(img, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_AREA)
            else:
                img_r = img
            prob_map = sliding_window_infer(model, img_r, train_size=train_size,
                                            device=device)
            vis_path = OUTPUT_DIR / f"{img_path.stem}_deep_result.jpg"
            visualize_prediction(img_r, prob_map, res, vis_path)
            print(f"可视化已保存: {vis_path}")
        return

    if args.eval_only:
        model = UNet(in_ch=1, base_ch=args.base_ch, dropout=0.0).to(device)
        model_path = MODEL_DIR / "unet_barcode.pt"
        if not model_path.exists():
            print(f"[错误] 模型文件不存在: {model_path}")
            return
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        train_size = ckpt.get("train_size", ckpt.get("args", {}).get("size", 224))
        print(f"已加载模型: {model_path}, train_size={train_size}")
        evaluate(model, device, train_size=train_size)
        return

    train(args)

    print("\n训练完成，正在评估...")
    model = UNet(in_ch=1, base_ch=args.base_ch, dropout=0.0).to(device)
    ckpt = torch.load(str(MODEL_DIR / "unet_barcode.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    train_size = ckpt.get("train_size", ckpt.get("args", {}).get("size", 224))
    evaluate(model, device, train_size=train_size)

if __name__ == "__main__":
    main()