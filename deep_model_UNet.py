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
import math
import re
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
REAL_DIR = PROJECT_ROOT / "data" / "real_synth"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "deep"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GT_TEXT = "4014855012527"


def imread_unicode(path, flags: int = cv2.IMREAD_GRAYSCALE):
    """读取图像, 兼容含中文等非 ASCII 字符的 Windows 路径.

    cv2.imread 在 Windows 下无法打开非 ASCII 路径, 改为按字节流读取后解码.
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


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
    """U-Net: 4 层编码器-解码器, base_ch=24, ~4.4M 参数."""

    def __init__(self, in_ch: int = 1, base_ch: int = 24, dropout: float = 0.1):
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
    def __init__(self, pos_weight: float = 4.0, dice_weight: float = 1.0):
        super().__init__()
        self.pos_weight_val = pos_weight
        self.dice_weight = dice_weight
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
        return bce(logits, target) + self.dice_weight * self.dice(logits, target)

# ============================================================================
# 数据集
# ============================================================================

class BarcodeDataset(Dataset):
    """条码定位数据集. preload=True 时一次性解码并缓存, 避免每轮重复解码大图."""

    def __init__(self, root: Path, split: str = "train",
                 size: int = 384, augment: bool = False,
                 preload: bool = True,balance: bool = True):
        self.root = Path(root)
        self.split = split
        self.size = size
        self.augment = augment
        self.balance = balance
        manifest_path = self.root / f"{split}_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        pos = [m for m in manifest if m.get("has_barcode", True)]
        neg = [m for m in manifest if not m.get("has_barcode", True)]
        if balance and pos and neg:
            ratio = max(1, len(pos) // len(neg))
            self.items = pos + neg * min(ratio, 4)
        else:
            self.items = list(manifest)
        self._cache = None
        if preload:
            self._cache = [self._load(i) for i in range(len(self.items))]

    def __len__(self):
        return len(self.items)

    def _load(self, idx: int):
        item = self.items[idx]
        name = item["name"]
        img = imread_unicode(self.root / self.split / "images" / f"{name}.jpg",
                             cv2.IMREAD_GRAYSCALE)
        mask = imread_unicode(self.root / self.split / "masks" / f"{name}.png",
                              cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {name}")
        if mask is None:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (self.size, self.size),
                         interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size),
                          interpolation=cv2.INTER_NEAREST)
        return img, mask

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        s = self.size
        if np.random.rand() < 0.5:
            k = np.random.uniform(0.55, 1.0)
            side = max(16, int(round(s * k)))
            full_area = int((mask > 127).sum())
            best = None
            for _ in range(6):
                y0 = np.random.randint(0, s - side + 1)
                x0 = np.random.randint(0, s - side + 1)
                sub = mask[y0:y0 + side, x0:x0 + side]
                keep = 1.0 if full_area == 0 else \
                    int((sub > 127).sum()) / float(full_area)
                if best is None or keep > best[0]:
                    best = (keep, y0, x0)
                if keep >= 0.65:
                    break
            keep, y0, x0 = best
            if keep >= 0.5:
                img = cv2.resize(img[y0:y0 + side, x0:x0 + side], (s, s),
                                 interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask[y0:y0 + side, x0:x0 + side], (s, s),
                                  interpolation=cv2.INTER_NEAREST)
        ang = np.random.uniform(-20, 20)
        zoom = np.random.uniform(0.80, 1.15)
        M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, zoom)
        jit = np.random.uniform(-0.05, 0.05, (4, 2)) * s
        src = np.array([[0, 0], [s, 0], [s, s], [0, s]], dtype=np.float32)
        Hp = cv2.getPerspectiveTransform(src, (src + jit).astype(np.float32))
        M3 = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64) @ \
            Hp.astype(np.float64)
        img = cv2.warpPerspective(img, M3, (s, s), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpPerspective(mask, M3, (s, s), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_REPLICATE)
        if np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if np.random.rand() < 0.2:
            img = img[::-1, :].copy()
            mask = mask[::-1, :].copy()
        if np.random.rand() < 0.7:
            gy, gx = np.mgrid[0:s, 0:s].astype(np.float32)
            ax, ay = np.random.uniform(-0.45, 0.45, 2)
            phase = np.random.uniform(0, 2 * np.pi)
            shade = 1.0 + ax * (gx / s - 0.5) * 2 + ay * (gy / s - 0.5) * 2
            shade += 0.15 * np.sin(
                2 * np.pi * (gx / s) * np.random.uniform(0.5, 2.5) + phase)
            img = np.clip(img.astype(np.float32) * shade, 0, 255).astype(np.uint8)
        alpha = np.random.uniform(0.75, 1.25)
        beta = np.random.uniform(-20, 20)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255)
        if np.random.rand() < 0.30:
            img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(0.3, 0.8))
        if np.random.rand() < 0.25:
            img = np.clip(img + np.random.normal(0, 4, img.shape), 0, 255)
        return img.astype(np.uint8), mask

    def __getitem__(self, idx):
        if self._cache is not None:
            img = self._cache[idx][0].copy()
            mask = self._cache[idx][1].copy()
        else:
            img, mask = self._load(idx)
        if self.augment:
            img, mask = self._augment(img, mask)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        x = (x - 0.5) / 0.5
        y = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return x, y

# ============================================================================
# 真实域数据集生成
# ============================================================================

def _crop_replicate(img: np.ndarray, cx: float, cy: float, side: int):
    h, w = img.shape[:2]
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - w), max(0, y1 - h)
    crop = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if pl or pt or pr or pb:
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return crop, x0, y0

def _build_sample(img: np.ndarray, box, size: int, rng, positive: bool):
    """从真实图中裁出场景块并做几何/光度增强, 返回 (图像, mask, 四角点).

    mask 由真值框经同一变换矩阵映射得到, 因此是像素级精确的紧贴 mask.
    """
    h, w = img.shape[:2]
    base = max(box[2] - box[0], box[3] - box[1])
    if positive:
        side = base * rng.uniform(1.7, 3.0)
        cx = (box[0] + box[2]) / 2 + rng.uniform(-0.28, 0.28) * side
        cy = (box[1] + box[3]) / 2 + rng.uniform(-0.28, 0.28) * side
    else:
        side = base * rng.uniform(1.4, 3.0)
        cx = cy = None
        for _ in range(8):
            tx, ty = rng.uniform(0, w), rng.uniform(0, h)
            if max(abs(tx - (box[0] + box[2]) / 2),
                   abs(ty - (box[1] + box[3]) / 2)) > 0.9 * side:
                cx, cy = tx, ty
                break
        if cx is None:
            cx, cy = w / 2, h / 2
    side = int(max(96, min(side, max(h, w))))
    crop, ox, oy = _crop_replicate(img, cx, cy, side)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    k = size / float(side)
    quad = np.array([[box[0] - ox, box[1] - oy], [box[2] - ox, box[1] - oy],
                     [box[2] - ox, box[3] - oy], [box[0] - ox, box[3] - oy]],
                    dtype=np.float32) * k

    src = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    ang = rng.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((size / 2, size / 2), ang, rng.uniform(0.85, 1.10))
    jit = rng.uniform(-0.06, 0.06, (4, 2)) * size
    Hp = cv2.getPerspectiveTransform(src, (src + jit).astype(np.float32))
    M3 = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float64) @ Hp.astype(np.float64)
    crop = cv2.warpPerspective(crop, M3, (size, size), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    quad = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), M3).reshape(-1, 2)
    mask = np.zeros((size, size), dtype=np.uint8)
    if positive:
        cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 255)

    if rng.random() < 0.75:
        gy, gx = np.mgrid[0:size, 0:size].astype(np.float32)
        ax, ay = rng.uniform(-0.5, 0.5, 2)
        shade = 1.0 + ax * (gx / size - 0.5) * 2 + ay * (gy / size - 0.5) * 2
        shade += 0.18 * np.sin(
            2 * np.pi * (gx / size) * rng.uniform(0.5, 3.0) + rng.uniform(0, 6.3))
        crop = np.clip(crop.astype(np.float32) * shade, 0, 255).astype(np.uint8)
    crop = np.clip(crop.astype(np.float32) * rng.uniform(0.75, 1.25)
                   + rng.uniform(-20, 20), 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        crop = cv2.GaussianBlur(crop, (0, 0), rng.uniform(0.4, 1.3))
    if rng.random() < 0.30:
        crop = np.clip(crop.astype(np.float32) + rng.normal(0, 7, crop.shape),
                       0, 255).astype(np.uint8)
    return crop, mask, quad

def build_real_dataset(size: int = 384, n_pos: int = 40, n_neg: int = 12,
                       seed: int = 0, val_index: int = 1):
    """用真实图像 + 精确真值框构建定位数据集 (紧贴 mask, 贴合真实分布)."""
    import barcode_recognize as br

    print("提取真实图像条码真值框 (结构张量定位 + 解码验证)...")
    sources = []
    for p in sorted(RAW_DIR.glob("*.bmp")):
        g = imread_unicode(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"  [跳过] {p.name}: 读取失败")
            continue
        res = br.recognize_barcode(g)
        if not res.ok or getattr(res, "bbox", None) is None:
            print(f"  [跳过] {p.name}: 未能定位条码")
            continue
        bbox = [float(v) for v in res.bbox]
        print(f"  {p.name}: box=({bbox[0]:.0f},{bbox[1]:.0f})-"
              f"({bbox[2]:.0f},{bbox[3]:.0f}) text={res.text}")
        sources.append((p.stem, g, bbox))
    if not sources:
        print("[错误] 没有可用的真值框, 请检查 data/raw")
        return

    val_name = sources[min(val_index, len(sources) - 1)][0]
    print(f"验证集来源图像(留出): {val_name}")
    for split in ("train", "val"):
        (REAL_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (REAL_DIR / split / "masks").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    manifest = {"train": [], "val": []}
    for name, g, bbox in sources:
        split = "val" if name == val_name else "train"
        cnt_p = n_pos if split == "train" else max(16, n_pos // 3)
        cnt_n = n_neg if split == "train" else max(6, n_neg // 3)
        for i in range(cnt_p + cnt_n):
            positive = i < cnt_p
            crop, mask, quad = _build_sample(g, bbox, size, rng, positive)
            tag = re.sub(r"[^0-9A-Za-z_.-]", "_",
                         f"{name}_{'pos' if positive else 'neg'}{i:03d}")
            cv2.imwrite(str(REAL_DIR / split / "images" / f"{tag}.jpg"),
                        crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(REAL_DIR / split / "masks" / f"{tag}.png"), mask)
            manifest[split].append({
                "name": tag,
                "source": name,
                "mode": "real_" + ("pos" if positive else "neg"),
                "quad": [[round(float(x), 1), round(float(y), 1)]
                         for x, y in quad],
                "has_barcode": bool(positive),
            })

    for split in ("train", "val"):
        with (REAL_DIR / f"{split}_manifest.json").open("w",
                                                        encoding="utf-8") as f:
            json.dump(manifest[split], f, indent=2, ensure_ascii=False)
        n = len(manifest[split])
        npos = sum(1 for m in manifest[split] if m["has_barcode"])
        print(f"{split}: {n} 张 (正样本 {npos}, 负样本 {n - npos})")
    print(f"数据集已生成: {REAL_DIR}")
    print(f"提示: 训练请运行 python deep_model_UNet.py --size {size}")

# ============================================================================
# 训练逻辑
# ============================================================================

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss = 0.0
    n = 0
    amp = scaler is not None and device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if amp:
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    n_pos = 0
    fp = 0
    n_neg = 0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) > 0.5).float()
        inter = (preds * y).sum(dim=(1, 2, 3))
        union = ((preds + y) > 0).float().sum(dim=(1, 2, 3))
        has = y.sum(dim=(1, 2, 3)) > 0
        iou = inter / union.clamp(min=1)
        if has.any():
            iou_sum += iou[has].sum().item()
            n_pos += int(has.sum().item())
        if (~has).any():
            neg_pred = preds[~has].sum(dim=(1, 2, 3)) > 0
            fp += int(neg_pred.sum().item())
            n_neg += int((~has).sum().item())
        n += x.size(0)
    mean_iou = iou_sum / max(1, n_pos)
    fp_rate = (fp / n_neg) if n_neg else 0.0
    return total_loss / max(1, n), mean_iou, fp_rate

def train(args):
    device = get_device()
    print(f"设备: {device}")

    if (REAL_DIR / "train_manifest.json").exists():
        data_root = REAL_DIR
    else:
        data_root = SYNTH_DIR
    print(f"数据集目录: {data_root}")

    train_ds = BarcodeDataset(data_root, "train", size=args.size, augment=True)
    val_ds = BarcodeDataset(data_root, "val", size=args.size, augment=False,balance=False)
    print(f"训练集: {len(train_ds)} 张, 验证集: {len(val_ds)} 张")

    cover = [float((m > 127).mean()) for _, m in train_ds._cache] if train_ds._cache else []
    if cover:
        print(f"训练 mask 覆盖率: 中位数={np.median(cover):.3f} "
              f"范围=[{min(cover):.3f}, {max(cover):.3f}]")

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
    warmup = max(8, args.epochs // 15)

    def lr_lambda(ep: int) -> float:
        if ep < warmup:
            return float(ep + 1) / float(warmup)
        p = (ep - warmup) / max(1, args.epochs - warmup)
        return 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_iou = 0.0
    patience_counter = 0
    patience = args.patience
    history = {"train_loss": [], "val_loss": [], "val_iou": []}
    model_path = MODEL_DIR / "unet_barcode.pt"

    print(f"\n开始训练: {args.epochs} epochs, batch={args.batch}, "
          f"lr={args.lr}, size={args.size}, dropout={args.dropout}")
    print(f"学习率: 线性 warmup {warmup} 轮 + 余弦退火; 早停 patience={patience}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou, fp_rate = validate(model, val_loader, criterion, device)
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
                "val_loss": val_loss,
                "args": vars(args),
                "train_size": args.size,
            }, str(model_path))
            marker = " *"
        else:
            patience_counter += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_iou={val_iou:.4f} fp_rate={fp_rate:.4f} | lr={scheduler.get_last_lr()[0]:.2e} | "
              f"best_iou={best_iou:.4f} | "
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
    for frac in np.linspace(0.20, 0.80, 9):
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

def _zxing_decode(gray: np.ndarray) -> Optional[str]:
    """zxingcpp 兜底解码 """
    try:
        import zxingcpp
    except ImportError:
        return None
    h, w = gray.shape[:2]
    if max(h, w) > 1600:
        k = 1600 / float(max(h, w))
        gray = cv2.resize(gray, None, fx=k, fy=k,
                          interpolation=cv2.INTER_AREA)
        w = gray.shape[1]
    scales = (1.0,) if w >= 300 else (1.0, 2.0)
    for scale in scales:
        img = (cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
               if scale > 1 else gray)
        try:
            results = zxingcpp.read_barcodes(img, try_rotate=True,
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
    return None

def decode_barcode_crop(gray: np.ndarray,use_zxing: bool = True) -> Optional[str]:
    """条码小图解码: 自研扫描线解码 → 反色 → zxing 兜底."""
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    if g.shape[0] < 12 or g.shape[1] < 60:
        return None
    for v in (g, cv2.bitwise_not(g)):
        text = decode_ean13(v)
        if text:
            return text
    return _zxing_decode(g) if use_zxing else None

# ============================================================================
# 推理逻辑
# ============================================================================

def _pad_to_grid(image: np.ndarray, window: int, stride: int):
    """右下侧复制填充, 使所有滑窗裁剪尺寸一致, 便于整批前向."""
    h, w = image.shape[:2]

    def need(n: int) -> int:
        if n <= window:
            return window
        k = int(np.ceil((n - window) / stride))
        return window + k * stride

    ph, pw = need(h), need(w)
    if ph == h and pw == w:
        return image, h, w
    padded = cv2.copyMakeBorder(image, 0, ph - h, 0, pw - w,
                                cv2.BORDER_REPLICATE)
    return padded, h, w

@torch.no_grad()
def sliding_window_infer(model, image: np.ndarray,
                         window: int = 384, stride: int = 256,
                         train_size: int = 384,
                         device: torch.device = None,
                         batch_size: int = 16) -> np.ndarray:
    """滑窗批量推理: 填充对齐后窗口同尺寸, 整批一次前向."""
    if device is None:
        device = get_device()
    model.eval()
    padded, h0, w0 = _pad_to_grid(image, window, stride)
    ph, pw = padded.shape[:2]
    ys = list(range(0, ph - window + 1, stride))
    xs = list(range(0, pw - window + 1, stride))
    if not ys or ys[-1] != ph - window:
        ys.append(max(0, ph - window))
    if not xs or xs[-1] != pw - window:
        xs.append(max(0, pw - window))
    positions = [(y, x) for y in ys for x in xs]

    prob_map = np.zeros((ph, pw), dtype=np.float32)
    count_map = np.zeros((ph, pw), dtype=np.float32)
    for i in range(0, len(positions), batch_size):
        chunk = positions[i:i + batch_size]
        batch = np.stack([padded[y:y + window, x:x + window] for y, x in chunk])
        t = torch.from_numpy(batch.astype(np.float32) / 255.0).unsqueeze(1)
        t = ((t - 0.5) / 0.5).to(device)
        if window != train_size:
            t = F.interpolate(t, size=(train_size, train_size),
                              mode="bilinear", align_corners=False)
        prob = torch.sigmoid(model(t))
        if window != train_size:
            prob = F.interpolate(prob, size=(window, window),
                                 mode="bilinear", align_corners=False)
        prob = prob.squeeze(1).cpu().numpy()
        for (y, x), p in zip(chunk, prob):
            prob_map[y:y + window, x:x + window] += p
            count_map[y:y + window, x:x + window] += 1.0
    prob_map = prob_map / np.maximum(count_map, 1.0)
    return prob_map[:h0, :w0]

def _vertical_edge_ratio(crop: np.ndarray) -> float:
    """矫正后的条码应以水平方向梯度为主."""
    g = cv2.GaussianBlur(crop, (3, 3), 0)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    ex, ey = float(gx.mean()), float(gy.mean())
    return ex / (ex + ey + 1e-6)

def extract_candidates(prob_map: np.ndarray, threshold: float = 0.40,
                       max_keep: int = 3) -> list:
    """从概率图提取候选旋转矩形, 剔除面积/长宽比不合理的区域."""
    H, W = prob_map.shape
    mask = (prob_map > threshold).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        ratio = area / float(H * W)
        if ratio < 0.0004 or ratio > 0.60:
            continue
        ar = max(bw, bh) / max(1, min(bw, bh))
        if not (1.1 <= ar <= 14.0):
            continue
        comp = (labels[y:y + bh, x:x + bw] == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea) + np.array([[x, y]])
        rect = cv2.minAreaRect(cnt)
        if min(rect[1]) < 8:
            continue
        box = cv2.boxPoints(rect)
        mean_p = float(prob_map[y:y + bh, x:x + bw][comp > 0].mean())
        out.append({"box": box, "score": area * mean_p, "prob": mean_p,
                    "aspect": ar, "area": int(area)})
    out.sort(key=lambda d: d["score"], reverse=True)
    return out[:max_keep]

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
                 window: int = 384, stride: int = 256,
                 train_size: int = 384, long_side: int = 1280,
                 use_fallback: bool = True) -> dict:
    """端到端推理: 缩放图粗定位 → 候选框 → 全分辨率透视矫正 → 解码.

    关键点: 定位在缩放图上做 (快), 解码在全分辨率原图上做 (清晰),
    避免把小条码缩到无法解码的分辨率.
    """
    t0 = time.perf_counter()
    result = {"ok": False, "text": "", "bbox": None, "elapsed_ms": 0.0,
              "checksum_ok": False, "method": ""}
    if device is None:
        device = get_device()
    img = imread_unicode(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        result["error"] = f"无法读取图像: {image_path}"
        return result

    h, w = img.shape[:2]
    scale = min(1.0, long_side / float(max(h, w)))
    small = (cv2.resize(img, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_AREA)
             if scale < 1.0 else img)

    prob_map = sliding_window_infer(model, small, window, stride,
                                    train_size, device)
    result["prob_map"] = prob_map
    result["scale"] = scale

    text = None
    bbox = None
    for cand_i,cand in enumerate(extract_candidates(prob_map)):
        box_full = np.asarray(cand["box"], dtype=np.float32) / scale
        warped = perspective_correct(img, box_full)
        wh, ww = warped.shape[:2]
        if ww < 150 or wh < 40:
            continue
        if ww / max(1, wh) < 1.3:
            continue
        if max(warped.shape) > 1400:
            k = 1400 / float(max(warped.shape))
            warped = cv2.resize(warped, None, fx=k, fy=k,
                                interpolation=cv2.INTER_AREA)
        text = decode_barcode_crop(warped, use_zxing=(cand_i == 0))
        if text is None:
            for k_rot in (1, 2, 3):
                text = decode_barcode_crop(np.rot90(warped, k_rot).copy(), use_zxing=False)
                if text:
                    break
        if text:
            bbox = box_full.tolist()
            result["method"] = "unet"
            break

    if text is None and use_fallback:
        try:
            import barcode_recognize as br
            res = br.recognize_barcode(img)
            if res.ok:
                text = res.text
                b = getattr(res, "bbox", None)
                bbox = [float(v) for v in b] if b is not None else None
                result["method"] = "classical"
        except Exception:
            pass

    result["text"] = text or ""
    result["ok"] = bool(text) and text == GT_TEXT
    result["checksum_ok"] = valid_ean13(text) if text else False
    result["bbox"] = bbox
    result["elapsed_ms"] = (time.perf_counter() - t0) * 1000
    return result

def evaluate(model, device: torch.device = None, train_size: int = 384,
             long_side: int = 1280, use_fallback: bool = True):
    """在验证集和原始图像上评估模型."""
    if device is None:
        device = get_device()
    data_root = REAL_DIR if (REAL_DIR / "val_manifest.json").exists() else SYNTH_DIR
    val_ds = BarcodeDataset(data_root, "val", size=train_size, augment=False,balance=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    criterion = CombinedLoss()
    val_loss, val_iou, fp_rate = validate(model, val_loader, criterion, device)
    print(f"\n验证集评估: val_loss={val_loss:.4f}, "
          f"val_iou(正样本)={val_iou:.4f}, 负样本误报率={fp_rate:.2%}")
    raw_images = sorted(RAW_DIR.glob("*.bmp"))
    if not raw_images:
        return
    print(f"\n在 {len(raw_images)} 张原始图像上测试:")
    print("-" * 60)
    success = 0
    total_ms = 0.0
    for img_path in raw_images:
        res = infer_single(model, img_path, device, train_size=train_size,
                           long_side=long_side, use_fallback=use_fallback)
        status = "✓" if res["ok"] else "✗"
        text_disp = res["text"] if res["text"] else "—"
        match = " (匹配)" if res["ok"] else ""
        elapsed_ms = res.get("elapsed_ms", 0.0)
        total_ms += elapsed_ms
        method = res.get("method") or "-"
        err = res.get("error")
        err_disp = f" [{err}]" if err else ""
        print(f"  {status} {img_path.name}: {text_disp}{match} "
              f"({elapsed_ms:.0f}ms, {method}){err_disp}")
        if res["ok"]:
            success += 1
    print(f"\n原始图像识别率: {success}/{len(raw_images)}")
    print(f"平均耗时: {total_ms / len(raw_images):.0f} ms/张")

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
    parser.add_argument("--build-data", action="store_true",
                        help="从 data/raw 构建真实域定位数据集后退出")
    parser.add_argument("--data-pos", type=int, default=60,
                        help="每张源图生成的正样本数 (默认60)")
    parser.add_argument("--data-neg", type=int, default=18,
                        help="每张源图生成的负样本数 (默认18)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="训练轮数 (默认200)")
    parser.add_argument("--batch", type=int, default=16,
                        help="批大小 (默认16)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率 (默认2e-4)")
    parser.add_argument("--wd", type=float, default=1e-5,
                        help="权重衰减 (默认1e-5)")
    parser.add_argument("--size", type=int, default=384,
                        help="训练图像尺寸 (默认384)")
    parser.add_argument("--base-ch", type=int, default=24,
                        help="U-Net 基础通道数 (默认24)")
    parser.add_argument("--pos-weight", type=float, default=4.0,
                        help="正样本权重 (默认4.0)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 率 (默认0.1)")
    parser.add_argument("--patience", type=int, default=40,
                        help="早停耐心轮数 (默认40)")
    parser.add_argument("--long-side", type=int, default=1280,
                        help="推理定位时的图像长边 (默认1280)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="关闭传统方法兜底 (仅评估纯 U-Net 效果)")
    parser.add_argument("--eval-only", action="store_true",
                        help="仅评估已训练模型")
    parser.add_argument("--infer", type=str, default=None,
                        help="对单张图像推理")
    args = parser.parse_args()

    print("=" * 60)
    print("  深度学习砂轮条形码检测识别")
    print("  U-Net 分割 + 批量滑窗推理 + EAN-13 解码")
    print("=" * 60)

    device = get_device()
    print(f"设备: {device}\n")

    if args.build_data:
        build_real_dataset(size=args.size, n_pos=args.data_pos,
                           n_neg=args.data_neg)
        return

    use_fallback = not args.no_fallback

    if args.infer:
        model = UNet(in_ch=1, base_ch=args.base_ch, dropout=0.0).to(device)
        model_path = MODEL_DIR / "unet_barcode.pt"
        if not model_path.exists():
            print(f"[错误] 模型文件不存在: {model_path}")
            print("请先运行训练: python deep_model_UNet.py")
            return
        ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        train_size = ckpt.get("train_size", ckpt.get("args", {}).get("size", 384))
        print(f"已加载模型: {model_path} (epoch {ckpt['epoch']}, "
              f"val_iou={ckpt['val_iou']:.4f}, train_size={train_size})")
        img_path = Path(args.infer)
        res = infer_single(model, img_path, device, train_size=train_size,
                           long_side=args.long_side, use_fallback=use_fallback)
        status = "✓ 成功" if res["ok"] else "✗ 失败"
        print(f"\n结果: {status}")
        print(f"条码值: {res['text'] or '—'}")
        print(f"校验位: {'通过' if res['checksum_ok'] else '未通过'}")
        print(f"定位方式: {res.get('method') or '—'}")
        print(f"耗时: {res['elapsed_ms']:.0f} ms")
        img = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
        prob_map = res.get("prob_map")
        if img is not None and prob_map is not None:
            scale = res.get("scale", 1.0)
            img_r = (cv2.resize(img, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
                     if scale < 1.0 else img)
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
        train_size = ckpt.get("train_size", ckpt.get("args", {}).get("size", 384))
        print(f"已加载模型: {model_path}, train_size={train_size}")
        evaluate(model, device, train_size=train_size,
                 long_side=args.long_side, use_fallback=use_fallback)
        return

    train(args)

    print("\n训练完成，正在评估...")
    model = UNet(in_ch=1, base_ch=args.base_ch, dropout=0.0).to(device)
    ckpt = torch.load(str(MODEL_DIR / "unet_barcode.pt"),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    train_size = ckpt.get("train_size", ckpt.get("args", {}).get("size", 384))
    evaluate(model, device, train_size=train_size,
             long_side=args.long_side, use_fallback=use_fallback)

if __name__ == "__main__":
    main()