"""条码定位数据集与真实域数据集生成."""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import RAW_DIR, REAL_DIR, imread_unicode


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