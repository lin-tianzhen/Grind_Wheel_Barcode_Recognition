#!/usr/bin/env python3
"""基于原始砂轮条形码图片的数据增广程序。

对 data/raw/ 下的所有原始图片施加几何、光度、环境等增广变换，
生成大量训练/验证样本，保存到 data/synthetic/ 目录。

增广策略:
  1. 几何变换: 旋转、透视畸变、仿射变换、弹性形变、缩放
  2. 光度变换: 亮度、对比度、伽马校正、高斯噪声、椒盐噪声、
               高斯模糊、运动模糊、JPEG 压缩伪影
  3. 环境模拟: 局部遮挡、阴影、反光、不均匀光照
  4. 组合变换: 随机组合上述变换以模拟真实场景

输出目录结构:
  data/synthetic/
    train/
      images/   -- 增广后的场景图
      masks/    -- 对应的条形码区域掩码
    val/
      images/
      masks/
    train_manifest.json
    val_manifest.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SYNTH_DIR = PROJECT_ROOT / "data" / "synthetic"
SEED = 202610

def imread_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 几何增广
# ---------------------------------------------------------------------------
def rotate_image(img: np.ndarray, angle: float,
                 border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=border)

def random_perspective(img: np.ndarray, rng: np.random.Generator,
                       strength: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    d = int(0.04 * min(h, w) * strength)
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst = src.copy()
    for i in range(4):
        dst[i] += rng.integers(-d, d + 1, size=2).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

def random_affine(img: np.ndarray, rng: np.random.Generator,
                  strength: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    shear_x = rng.uniform(-0.12, 0.12) * strength
    shear_y = rng.uniform(-0.08, 0.08) * strength
    scale_x = rng.uniform(0.88, 1.12) ** strength
    scale_y = rng.uniform(0.88, 1.12) ** strength
    M = np.float32([
        [scale_x, shear_x, 0],
        [shear_y, scale_y, 0]
    ])
    M[0, 2] = (1 - M[0, 0]) * w / 2 - M[0, 1] * h / 2
    M[1, 2] = -M[1, 0] * w / 2 + (1 - M[1, 1]) * h / 2
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)

def elastic_deform(img: np.ndarray, rng: np.random.Generator,
                   alpha: float = 30.0, sigma: float = 6.0,
                   grid_size: int = 4) -> np.ndarray:
    h, w = img.shape[:2]
    gh, gw = h // grid_size, w // grid_size
    dx = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
    dy = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC) * alpha
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC) * alpha
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = np.clip(xx + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(yy + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)

def random_scale(img: np.ndarray, rng: np.random.Generator,
                 range_: tuple = (0.6, 1.5)) -> np.ndarray:
    h, w = img.shape[:2]
    s = rng.uniform(*range_)
    new_w, new_h = max(16, int(w * s)), max(16, int(h * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

# ---------------------------------------------------------------------------
# 光度增广
# ---------------------------------------------------------------------------
def adjust_brightness_contrast(img: np.ndarray, rng: np.random.Generator,
                               strength: float = 1.0) -> np.ndarray:
    alpha = rng.uniform(0.65, 1.45) ** strength
    beta = rng.uniform(-35, 35) * strength
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

def adjust_gamma(img: np.ndarray, rng: np.random.Generator,
                 strength: float = 1.0) -> np.ndarray:
    gamma = rng.uniform(0.55, 1.8) ** strength
    lut = np.array([255.0 * (i / 255.0) ** (1.0 / gamma)
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)

def add_gaussian_noise(img: np.ndarray, rng: np.random.Generator,
                       strength: float = 1.0) -> np.ndarray:
    sigma = rng.uniform(2.0, 18.0) * strength
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_salt_pepper(img: np.ndarray, rng: np.random.Generator,
                    prob: float = 0.008) -> np.ndarray:
    out = img.copy()
    h, w = out.shape
    n_salt = int(prob * h * w * 0.5)
    n_pepper = int(prob * h * w * 0.5)
    ys, xs = rng.integers(0, h, n_salt), rng.integers(0, w, n_salt)
    out[ys, xs] = 255
    ys, xs = rng.integers(0, h, n_pepper), rng.integers(0, w, n_pepper)
    out[ys, xs] = 0
    return out

def add_gaussian_blur(img: np.ndarray, rng: np.random.Generator,
                      strength: float = 1.0) -> np.ndarray:
    sigma = rng.uniform(0.3, 2.5) * strength
    return cv2.GaussianBlur(img, (0, 0), sigma)

def add_motion_blur(img: np.ndarray, rng: np.random.Generator,
                    strength: float = 1.0) -> np.ndarray:
    kernel_size = int(rng.integers(5, 21))
    if kernel_size % 2 == 0:
        kernel_size += 1
    angle = rng.uniform(0, 180)
    kernel = np.zeros((kernel_size, kernel_size), np.float32)
    mid = kernel_size // 2
    cos_a, sin_a = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    for t in range(-mid, mid + 1):
        x = int(round(mid + t * cos_a))
        y = int(round(mid + t * sin_a))
        if 0 <= x < kernel_size and 0 <= y < kernel_size:
            kernel[y, x] = 1.0
    kernel /= kernel.sum() + 1e-9
    return cv2.filter2D(img, -1, kernel)

def jpeg_compression(img: np.ndarray, rng: np.random.Generator,
                     quality_range: tuple = (25, 75)) -> np.ndarray:
    quality = int(rng.integers(*quality_range))
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)

def clahe_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    clip = rng.uniform(1.5, 4.0)
    grid = int(rng.integers(4, 10))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(img)

# ---------------------------------------------------------------------------
# 环境模拟增广
# ---------------------------------------------------------------------------
def add_shadow(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = out.shape
    x0 = int(rng.integers(0, w))
    y0 = int(rng.integers(0, h))
    shadow_w = int(rng.integers(w // 6, w // 2))
    shadow_h = int(rng.integers(h // 6, h // 2))
    intensity = rng.uniform(0.3, 0.7)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist_x = np.clip(1.0 - np.abs(xx - x0) / max(1, shadow_w), 0, 1)
    dist_y = np.clip(1.0 - np.abs(yy - y0) / max(1, shadow_h), 0, 1)
    shadow_mask = (dist_x * dist_y * intensity)
    out *= (1.0 - shadow_mask)
    return np.clip(out, 0, 255).astype(np.uint8)

def add_glare(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = out.shape
    cx = int(rng.integers(w // 4, 3 * w // 4))
    cy = int(rng.integers(h // 4, 3 * h // 4))
    radius = int(rng.integers(min(h, w) // 8, min(h, w) // 3))
    intensity = rng.uniform(40, 120)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    glare = np.clip(1.0 - dist / max(1, radius), 0, 1) * intensity
    out += glare
    return np.clip(out, 0, 255).astype(np.uint8)

def add_illumination_gradient(img: np.ndarray, rng: np.random.Generator,
                              strength: float = 1.0) -> np.ndarray:
    out = img.astype(np.float32)
    h, w = out.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx = rng.uniform(-0.25, 0.25) * strength
    gy = rng.uniform(-0.25, 0.25) * strength
    ramp = gx * (xx / max(1, w) - 0.5) + gy * (yy / max(1, h) - 0.5)
    out *= (1.0 + ramp)
    return np.clip(out, 0, 255).astype(np.uint8)

def add_occlusion(img: np.ndarray, rng: np.random.Generator,
                  max_n: int = 5) -> np.ndarray:
    out = img.copy()
    h, w = out.shape
    n = int(rng.integers(1, max_n + 1))
    for _ in range(n):
        kind = rng.choice(["rect", "ellipse", "scratch"])
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        color = int(rng.integers(15, 100))
        if kind == "rect":
            bw = int(rng.integers(5, max(6, w // 8)))
            bh = int(rng.integers(3, max(4, h // 12)))
            cv2.rectangle(out, (x0, y0), (min(w - 1, x0 + bw), min(h - 1, y0 + bh)),
                          color, -1)
        elif kind == "ellipse":
            rx = int(rng.integers(5, max(6, w // 10)))
            ry = int(rng.integers(3, max(4, h // 10)))
            cv2.ellipse(out, (x0, y0), (rx, ry),
                        rng.uniform(0, 180), 0, 360, color, -1)
        else:
            x1 = min(w - 1, x0 + int(rng.integers(10, max(11, w // 4))))
            y1 = min(h - 1, y0 + int(rng.integers(1, max(2, h // 20))))
            cv2.line(out, (x0, y0), (x1, y1), color,
                     thickness=int(rng.integers(1, 4)))
    return cv2.GaussianBlur(out, (3, 3), 0)

# ---------------------------------------------------------------------------
# 组合增广管线
# ---------------------------------------------------------------------------
def augment_pipeline(img: np.ndarray, rng: np.random.Generator,
                     strength: float = 1.0) -> np.ndarray:
    out = img.copy()

    if rng.random() < 0.85:
        angle = rng.uniform(-35, 35) * strength
        out = rotate_image(out, angle)

    if rng.random() < 0.6 * strength:
        out = random_perspective(out, rng, strength)

    if rng.random() < 0.5 * strength:
        out = random_affine(out, rng, strength)

    if rng.random() < 0.25 * strength:
        alpha = rng.uniform(15, 50) * strength
        sigma = rng.uniform(4, 10)
        out = elastic_deform(out, rng, alpha, sigma)

    if rng.random() < 0.3:
        out = random_scale(out, rng, (0.5, 1.6))
        max_side = 1024
        h, w = out.shape
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    if rng.random() < 0.9:
        out = adjust_brightness_contrast(out, rng, strength)

    if rng.random() < 0.7 * strength:
        out = adjust_gamma(out, rng, strength)

    if rng.random() < 0.6 * strength:
        out = add_illumination_gradient(out, rng, strength)

    if rng.random() < 0.55 * strength:
        out = add_gaussian_noise(out, rng, strength)

    if rng.random() < 0.2 * strength:
        out = add_salt_pepper(out, rng)

    if rng.random() < 0.5 * strength:
        out = add_gaussian_blur(out, rng, strength)

    if rng.random() < 0.3 * strength:
        out = add_motion_blur(out, rng, strength)

    if rng.random() < 0.25 * strength:
        out = jpeg_compression(out, rng, (20, 70))

    if rng.random() < 0.35 * strength:
        out = add_shadow(out, rng)

    if rng.random() < 0.2 * strength:
        out = add_glare(out, rng)

    if rng.random() < 0.3 * strength:
        out = add_occlusion(out, rng)

    if rng.random() < 0.3 * strength:
        out = clahe_augment(out, rng)

    return out

# ---------------------------------------------------------------------------
# 条形码区域掩码生成（基于阈值 + 形态学）
# ---------------------------------------------------------------------------
def generate_barcode_mask(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    opened_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_v)
    opened_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_h)
    combined = cv2.bitwise_or(opened_v, opened_h)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel_dilate, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_c = max(contours, key=cv2.contourArea)
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [max_c], -1, 255, -1)
        return clean_mask
    return mask

# ---------------------------------------------------------------------------
# 场景合成：将条形码贴到不同背景上
# ---------------------------------------------------------------------------
def paste_on_background(fg: np.ndarray, mask: np.ndarray,
                        bg: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bh, bw = bg.shape[:2]
    fh, fw = fg.shape[:2]
    if fh > bh or fw > bw:
        scale = min(bh / fh, bw / fw) * 0.85
        new_w, new_h = max(16, int(fw * scale)), max(16, int(fh * scale))
        fg = cv2.resize(fg, (new_w, new_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        fh, fw = new_h, new_w
    px = int(rng.integers(0, max(1, bw - fw // 2)))
    py = int(rng.integers(0, max(1, bh - fh // 2)))
    x0, y0 = max(0, px), max(0, py)
    x1, y1 = min(bw, px + fw), min(bh, py + fh)
    sx0, sy0 = x0 - px, y0 - py
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    result = bg.copy()
    roi = result[y0:y1, x0:x1]
    fg_region = fg[sy0:sy1, sx0:sx1]
    mask_region = mask[sy0:sy1, sx0:sx1]
    valid = mask_region > 127
    roi[valid] = fg_region[valid]
    result[y0:y1, x0:x1] = roi
    return result

# ---------------------------------------------------------------------------
# 主增广流程
# ---------------------------------------------------------------------------
def load_raw_images(raw_dir: Path) -> dict[str, np.ndarray]:
    images = {}
    for p in sorted(raw_dir.glob("*.bmp")):
        img = imread_gray(p)
        images[p.stem] = img
        print(f"  loaded: {p.name}  shape={img.shape}")
    if not images:
        raise FileNotFoundError(f"No .bmp images found in {raw_dir}")
    return images

def build_synthetic_dataset(raw_dir: Path, synth_dir: Path,
                            n_train: int = 500, n_val: int = 100,
                            seed: int = SEED) -> None:
    print("=" * 60)
    print("砂轮条形码数据增广")
    print("=" * 60)

    print(f"\n[1/4] 加载原始图片 from {raw_dir}")
    raw_images = load_raw_images(raw_dir)
    names = list(raw_images.keys())
    print(f"  共 {len(names)} 张原始图片\n")

    rng = np.random.default_rng(seed)

    for split, n_samples in [("train", n_train), ("val", n_val)]:
        print(f"[2/4] 生成 {split} 集 ({n_samples} 张)")
        img_dir = synth_dir / split / "images"
        mask_dir = synth_dir / split / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        manifest = []

        for i in range(n_samples):
            src_name = names[int(rng.integers(len(names)))]
            src_img = raw_images[src_name]

            mode = rng.choice([
                "direct_augment",
                "direct_augment",
                "direct_augment",
                "compose_scene",
            ])

            if mode == "direct_augment":
                aug_img = augment_pipeline(src_img, rng, strength=1.0)
                aug_mask = generate_barcode_mask(aug_img)
            else:
                fg_name = names[int(rng.integers(len(names)))]
                fg = raw_images[fg_name]
                fg_mask = generate_barcode_mask(fg)
                bg_name = names[int(rng.integers(len(names)))]
                bg = raw_images[bg_name]
                composed = paste_on_background(fg, fg_mask, bg, rng)
                aug_img = augment_pipeline(composed, rng, strength=0.7)
                aug_mask = generate_barcode_mask(aug_img)

            name = f"{split}_{i:05d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), aug_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(mask_dir / f"{name}.png"), aug_mask)

            ys, xs = np.nonzero(aug_mask > 127)
            has_barcode = len(xs) > 50
            if has_barcode:
                rect = cv2.minAreaRect(
                    np.stack([xs, ys], axis=1).astype(np.float32))
                quad = cv2.boxPoints(rect).tolist()
            else:
                quad = []

            manifest.append({
                "name": name,
                "source": src_name,
                "mode": mode,
                "quad": quad,
                "has_barcode": has_barcode,
            })

            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{n_samples} done")

        save_json(synth_dir / f"{split}_manifest.json", manifest)
        n_pos = sum(1 for m in manifest if m["has_barcode"])
        print(f"  {split}: {n_samples} images, {n_pos} with barcode detected\n")

    print(f"[3/4] 生成增广统计报告")
    for split in ["train", "val"]:
        manifest_path = synth_dir / f"{split}_manifest.json"
        with manifest_path.open("r") as f:
            manifest = json.load(f)
        n_total = len(manifest)
        n_pos = sum(1 for m in manifest if m["has_barcode"])
        modes = {}
        for m in manifest:
            modes[m["mode"]] = modes.get(m["mode"], 0) + 1
        print(f"  {split}: {n_total} images, {n_pos} positive, modes={modes}")

    print(f"\n[4/4] 完成！增广数据已保存到 {synth_dir}")
    print("=" * 60)

def main():
    ap = argparse.ArgumentParser(
        description="砂轮条形码数据增广程序 — 基于原始图片生成训练/验证集")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR,
                    help="原始图片目录")
    ap.add_argument("--synth-dir", type=Path, default=SYNTH_DIR,
                    help="增广输出目录")
    ap.add_argument("--n-train", type=int, default=500,
                    help="训练集样本数")
    ap.add_argument("--n-val", type=int, default=100,
                    help="验证集样本数")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="随机种子")
    args = ap.parse_args()

    build_synthetic_dataset(args.raw_dir, args.synth_dir,
                            args.n_train, args.n_val, args.seed)

if __name__ == "__main__":
    main()