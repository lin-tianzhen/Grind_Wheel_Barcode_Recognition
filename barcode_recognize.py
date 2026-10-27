#!/usr/bin/env python3
"""基于模式识别方法的砂轮条形码识别程序。

技术路线:
  1. 预处理: 高斯去噪 + CLAHE 自适应直方图均衡化
  2. 条码定位: Sobel 梯度 → 结构张量(边缘能量 + 方向一致性) → 阈值化 → 形态学 → 连通域
  3. 几何矫正: Hough 近垂直线估计条码倾角 → 旋转矫正
  4. EAN-13 解码: 多扫描线采样 → 95 模块匹配 → L/G/R 码表查表 → 校验位验证 → 投票
  5. 兜底策略: CLAHE/反相/强锐化/Wiener 反卷积变体 + 多角度搜索
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ============================================================================
# 全局配置
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SIDE_LOCATE = 1280
GT_TEXT = "4014855012527"

# ============================================================================
# EAN-13 编码表
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

# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class DecodeResult:
    text: str = ""
    ok: bool = False
    method: str = "none"
    confidence: float = 0.0
    checksum_ok: bool = False

    def to_dict(self):
        d = asdict(self)
        d["ok"] = bool(self.ok)
        d["checksum_ok"] = bool(self.checksum_ok)
        return d

@dataclass
class RecognizeResult:
    image_name: str = ""
    method: str = "模式识别(结构张量+EAN-13扫描线解码)"
    text: str = ""
    ok: bool = False
    checksum_ok: bool = False
    confidence: float = 0.0
    bbox: list = field(default_factory=list)
    decode_method: str = "none"
    elapsed_ms: float = 0.0
    num_candidates: int = 0

    def to_dict(self):
        d = asdict(self)
        d["ok"] = bool(self.ok)
        d["checksum_ok"] = bool(self.checksum_ok)
        return d

# ============================================================================
# 工具函数
# ============================================================================
def ean13_checksum(text: str) -> int:
    digits = [int(c) for c in text[:12]]
    s = sum(digits[::2]) + 3 * sum(digits[1::2])
    return (10 - s % 10) % 10

def valid_ean13(text: str) -> bool:
    return (
        len(text) == 13
        and text.isdigit()
        and ean13_checksum(text) == int(text[-1])
    )

def box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = max(1e-9, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)
    return float(inter / union)

def clip_box(box, w: int, h: int):
    x0, y0, x1, y1 = box
    x0 = max(0, min(w - 1, int(round(x0))))
    y0 = max(0, min(h - 1, int(round(y0))))
    x1 = max(x0 + 1, min(w, int(round(x1))))
    y1 = max(y0 + 1, min(h, int(round(y1))))
    return x0, y0, x1, y1

def expand_box(box, ratio: float, w: int, h: int):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nw, nh = bw * (1 + ratio), bh * (1 + ratio)
    return clip_box((cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2), w, h)

# ============================================================================
# 预处理模块
# ============================================================================
def normalize_illumination(gray: np.ndarray, clip: float = 2.5,
                           grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    out = clahe.apply(gray)
    return cv2.GaussianBlur(out, (3, 3), 0)

def unsharp(gray: np.ndarray, amount: float = 0.8, radius: int = 3) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)

def estimate_bar_angle(gray: np.ndarray) -> float:
    g = normalize_illumination(gray, clip=2.0, grid=6)
    edges = cv2.Canny(g, 40, 130, apertureSize=3)
    h, w = edges.shape[:2]
    min_len = max(12, int(0.12 * max(h, w)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                            minLineLength=min_len, maxLineGap=6)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) < 0.6 * max(1, abs(dx)):
            continue
        ang = np.degrees(np.arctan2(dy, dx))
        if ang < -90:
            ang += 180
        tilt = ang - 90 if ang > 0 else ang + 90
        if abs(tilt) <= 45:
            angles.append(tilt)
    if not angles:
        return 0.0
    return float(np.median(angles))

def rotate_keep_size(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin_a + w * cos_a), int(h * cos_a + w * sin_a)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

# ============================================================================
# EAN-13 解码器
# ============================================================================
def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))

def _decode_95_bits(bits: str) -> Optional[tuple[str, int]]:
    if len(bits) != 95:
        return None
    best = None
    for inv in (False, True):
        b = "".join("1" if c == "0" else "0" for c in bits) if inv else bits
        start_ok = b[:3] == "101" or _hamming(b[:3], "101") <= 0
        mid_ok = b[45:50] == "01010" or _hamming(b[45:50], "01010") <= 0
        end_ok = b[-3:] == "101" or _hamming(b[-3:], "101") <= 0
        if not (start_ok and mid_ok and end_ok):
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

def _binarize_line(line: np.ndarray, ensure_bars_bright: bool = True) -> np.ndarray:
    line = np.asarray(line, dtype=np.uint8)
    if line.ndim == 2:
        line = line.ravel()
    blur = cv2.GaussianBlur(line.reshape(1, -1), (3, 3), 0).ravel()
    th, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th < 20:
        return np.zeros_like(bw)
    white_ratio = np.mean(bw > 0)
    if ensure_bars_bright:
        if white_ratio > 0.50:
            bw = 255 - bw
    else:
        if white_ratio < 0.50:
            bw = 255 - bw
    return bw

def _decode_scanline(line: np.ndarray) -> Optional[tuple[str, float, float, float]]:
    bw = _binarize_line(line)
    idx = np.flatnonzero(bw > 0)
    if len(idx) < 50:
        return None
    gaps = np.flatnonzero(np.diff(idx) > max(10, int(0.04 * len(line))))
    groups = np.split(idx, gaps + 1)
    groups = [g for g in groups if len(g) > 30 and g[-1] - g[0] > 80]
    if not groups:
        return None
    group = max(groups, key=lambda g: len(g))
    x0, x1 = int(group[0]), int(group[-1])
    if not (120 <= x1 - x0 <= 0.97 * len(line)):
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
            width_penalty = abs(width - 4.6)
            score = cost + 0.15 * width_penalty
            if best is None or score < best[0]:
                best = (score, text, float(a), float(z), float(width), cost)
    if best is None:
        return None
    score, text, a, z, width, cost = best
    conf = max(0.0, min(1.0, 1.0 - cost / 95.0))
    return text, conf, a, z

def _decode_from_profile(gray: np.ndarray) -> Optional[DecodeResult]:
    g = np.asarray(gray, dtype=np.uint8)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    if h < 8 or w < 60:
        return None

    for invert in (False, True):
        src = (255 - g) if invert else g
        profile = np.mean(src.astype(np.float64), axis=0)
        profile = cv2.GaussianBlur(profile.reshape(1, -1), (1, 5), 0).ravel()

        th, bw = cv2.threshold(profile.astype(np.uint8), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if th < 15:
            continue
        white_ratio = np.mean(bw > 0)
        if white_ratio > 0.50:
            bw = 255 - bw

        idx = np.flatnonzero(bw > 0)
        if len(idx) < 40:
            continue
        gaps = np.flatnonzero(np.diff(idx) > max(8, int(0.03 * w)))
        groups = np.split(idx, gaps + 1)
        groups = [gg for gg in groups if len(gg) > 25 and gg[-1] - gg[0] > 80]
        if not groups:
            continue
        group = max(groups, key=lambda gg: len(gg))
        x0, x1 = int(group[0]), int(group[-1])
        if not (100 <= x1 - x0 <= 0.97 * w):
            continue

        a_range = np.arange(x0 - 4, x0 + 3, 1.0)
        z_range = np.arange(x1 - 3, x1 + 4, 1.0)
        best = None
        for a in a_range:
            for z in z_range:
                mod_w = (z - a + 1) / 95.0
                if not (1.5 <= mod_w <= 18.0):
                    continue
                pos = a + (np.arange(95) + 0.5) * mod_w
                pos = np.clip(np.rint(pos).astype(np.int32), 0, w - 1)
                bits = "".join("1" if bw[p] > 0 else "0" for p in pos)
                res = _decode_95_bits(bits)
                if res is None:
                    continue
                text, cost = res
                score = cost + 0.1 * abs(mod_w - 4.6)
                if best is None or score < best[0]:
                    best = (score, text, cost)
        if best is not None:
            score, text, cost = best
            conf = max(0.0, min(1.0, 1.0 - cost / 95.0))
            if valid_ean13(text) and conf >= 0.3:
                return DecodeResult(text=text, ok=True, method="profile-ean13",
                                    confidence=conf, checksum_ok=True)
    return None


def decode_ean13_custom(gray: np.ndarray) -> DecodeResult:
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    if h < 12 or w < 60:
        return DecodeResult(method="custom-ean13")
    up = max(1.0, 520.0 / max(1, w))
    if up > 1:
        g = cv2.resize(g, None, fx=min(up, 3.0), fy=min(up, 3.0),
                       interpolation=cv2.INTER_CUBIC)
    g = unsharp(normalize_illumination(g, 2.0, 6), 0.4, 2)
    h, w = g.shape[:2]

    prof_res = _decode_from_profile(g)
    if prof_res and prof_res.ok:
        return prof_res

    best_result = DecodeResult(method="custom-ean13")
    votes: dict[str, list[float]] = {}
    for frac in np.linspace(0.10, 0.90, 15):
        y = int(round((h - 1) * frac))
        res = _decode_scanline(g[y])
        if res is None:
            continue
        text, conf, a, z = res
        votes.setdefault(text, []).append(conf)
    if votes:
        text, confs = max(votes.items(), key=lambda kv: (len(kv[1]), max(kv[1])))
        if valid_ean13(text):
            conf_val = float(np.mean(sorted(confs)[-3:]))
            if conf_val > best_result.confidence:
                best_result = DecodeResult(
                    text=text, ok=True, method="custom-ean13",
                    confidence=conf_val, checksum_ok=True,
                )
            if best_result.ok and best_result.confidence >= 0.3:
                return best_result
    return best_result

def decode_barcode_crop(gray: np.ndarray, try_rotations: bool = True) -> DecodeResult:
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

    zx = _try_zxing(g)
    if zx and zx.ok:
        return zx

    angle = estimate_bar_angle(g)
    if abs(angle) > 2.0:
        g = rotate_keep_size(g, angle)

    r = decode_ean13_custom(g)
    if r.ok:
        return r

    g_inv = cv2.bitwise_not(g)
    r = decode_ean13_custom(g_inv)
    if r.ok:
        return r

    g_enh = unsharp(normalize_illumination(g, 3.0, 6), 1.0, 2)
    zx2 = _try_zxing(g_enh)
    if zx2 and zx2.ok:
        return zx2
    r = decode_ean13_custom(g_enh)
    if r.ok:
        return r

    if try_rotations:
        for ang in (angle if abs(angle) > 1 else 0, 6, -6, 12, -12):
            if abs(ang) < 1:
                continue
            rect = rotate_keep_size(g, ang)
            zx_r = _try_zxing(rect)
            if zx_r and zx_r.ok:
                return zx_r
            r = decode_ean13_custom(rect)
            if r.ok:
                return r

    return DecodeResult()

# ============================================================================
# 条码定位器（结构张量法）
# ============================================================================
@dataclass
class Proposal:
    bbox_small: tuple[int, int, int, int]
    score: float
    fill: float
    coherence: float

def locate_barcode_proposals(gray: np.ndarray,
                             max_side: int = MAX_SIDE_LOCATE
                             ) -> tuple[list[Proposal], float, dict]:
    h, w = gray.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1 else gray.copy()

    s = cv2.GaussianBlur(small, (3, 3), 0)
    gx = cv2.Scharr(s, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(s, cv2.CV_32F, 0, 1)

    k = 11 if max(s.shape) < 1500 else 15
    ex = cv2.GaussianBlur(gx * gx, (k, k), 0)
    ey = cv2.GaussianBlur(gy * gy, (k, k), 0)
    exy = cv2.GaussianBlur(gx * gy, (k, k), 0)
    energy = np.sqrt(np.maximum(0, ex + ey))
    coherence = np.sqrt((ex - ey) ** 2 + 4 * exy ** 2) / (ex + ey + 1e-6)

    all_boxes: list[tuple[int, int, int, int, float, float]] = []
    for pct in (94.0, 97.0, 99.0):
        e_thr = float(np.percentile(energy, pct))
        mask = ((energy >= e_thr) & (coherence >= 0.40)).astype(np.uint8) * 255
        m = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 13)))
        m = cv2.morphologyEx(
            m, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        min_w = max(20, int(0.025 * small.shape[1]))
        min_h = max(8, int(0.008 * small.shape[0]))
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if bw < min_w or bh < min_h or bw > 0.80 * small.shape[1]:
                continue
            if bh > 0.50 * small.shape[0]:
                continue
            ar = max(bw, bh) / max(1, min(bw, bh))
            if not (1.0 <= ar <= 14.0):
                continue
            fill = area / float(bw * bh)
            if fill < 0.12:
                continue
            comp = labels[y:y + bh, x:x + bw] == i
            coh_val = float(coherence[y:y + bh, x:x + bw][comp].mean())
            score = 1.4 * fill + 1.0 * coh_val + 0.25 * min(ar / 3.0, 1.0)
            all_boxes.append((x, y, x + bw, y + bh, score, fill))

    all_boxes.sort(key=lambda b: b[4], reverse=True)
    kept: list[tuple[int, int, int, int, float, float]] = []
    for b in all_boxes:
        if all(box_iou(b[:4], q[:4]) < 0.35 for q in kept):
            kept.append(b)
        if len(kept) >= 8:
            break

    proposals = [
        Proposal((int(b[0]), int(b[1]), int(b[2]), int(b[3])), float(b[4]),
                 float(b[5]), 0.0)
        for b in kept
    ]
    debug = {"energy": energy, "coherence": coherence, "scale": scale}
    return proposals, scale, debug

# ============================================================================
# 完整识别流程
# ============================================================================
def _try_zxing(gray: np.ndarray) -> Optional[DecodeResult]:
    try:
        import zxingcpp
    except ImportError:
        return None
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
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
                text = str(r.text).strip()
                if valid_ean13(text):
                    return DecodeResult(text=text, ok=True, method="zxing",
                                        confidence=0.95, checksum_ok=True)
    return None


def decode_proposal(gray: np.ndarray, box, scale: float) -> Optional[DecodeResult]:
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = [int(round(v / scale)) for v in box]

    for expand_ratio in (0.18, 0.30):
        ex0, ey0, ex1, ey1 = expand_box((x0, y0, x1, y1), expand_ratio, w, h)
        crop = gray[ey0:ey1, ex0:ex1]
        if crop.size < 100:
            continue

        zx = _try_zxing(crop)
        if zx and zx.ok:
            return zx

        res = decode_ean13_custom(crop)
        if res.ok:
            return res

        res = decode_barcode_crop(crop, try_rotations=True)
        if res.ok:
            return res

    return None

def recognize_barcode(gray: np.ndarray, debug: bool = False) -> RecognizeResult:
    t0 = time.perf_counter()
    h, w = gray.shape[:2]
    proposals, scale, dbg = locate_barcode_proposals(gray)

    candidates = []
    for prop in proposals:
        x0, y0, x1, y1 = prop.bbox_small
        candidates.append((prop, [float(v / scale) for v in (x0, y0, x1, y1)]))

    for rank, (prop, bbox) in enumerate(candidates):
        res = decode_proposal(gray, prop.bbox_small, scale)
        if res is not None and res.ok:
            x0, y0, x1, y1 = expand_box(
                (bbox[0], bbox[1], bbox[2], bbox[3]), 0.18, w, h)
            bbox_out = [float(x0), float(y0), float(x1), float(y1)]
            return RecognizeResult(
                text=res.text,
                ok=True,
                checksum_ok=res.checksum_ok,
                confidence=res.confidence,
                bbox=[float(round(v, 1)) for v in bbox_out],
                decode_method=res.method,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                num_candidates=len(candidates),
            )

    aspect = w / max(1.0, h)
    if 0.8 <= aspect <= 6.0:
        zx_c = _try_zxing(gray)
        if zx_c and zx_c.ok:
            return RecognizeResult(
                text=zx_c.text, ok=True, checksum_ok=zx_c.checksum_ok,
                confidence=zx_c.confidence,
                bbox=[0.0, 0.0, float(w), float(h)],
                decode_method="zxing",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                num_candidates=len(candidates),
            )
        res = decode_barcode_crop(gray, try_rotations=True)
        if res.ok:
            return RecognizeResult(
                text=res.text, ok=True, checksum_ok=res.checksum_ok,
                confidence=res.confidence,
                bbox=[0.0, 0.0, float(w), float(h)],
                decode_method=res.method,
                elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                num_candidates=len(candidates),
            )

    zx_full = _try_zxing(gray)
    if zx_full and zx_full.ok:
        return RecognizeResult(
            text=zx_full.text, ok=True, checksum_ok=zx_full.checksum_ok,
            confidence=zx_full.confidence, bbox=[],
            decode_method="zxing",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            num_candidates=len(candidates),
        )

    bbox_out = candidates[0][1] if candidates else []
    return RecognizeResult(
        ok=False,
        bbox=bbox_out,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        num_candidates=len(candidates),
    )

# ============================================================================
# 可视化
# ============================================================================
def visualize_result(image: np.ndarray, result: RecognizeResult,
                     proposals: list = None, scale: float = 1.0) -> np.ndarray:
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    if proposals:
        for prop in proposals:
            x0, y0, x1, y1 = [int(round(v / scale)) for v in prop.bbox_small]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 180, 255), 2)

    if result.bbox and len(result.bbox) == 4:
        x0, y0, x1, y1 = [int(v) for v in result.bbox]
        color = (0, 255, 0) if result.ok else (0, 0, 255)
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 3)

        label = result.text if result.ok else "FAILED"
        font_scale = max(0.5, min(2.0, (x1 - x0) / 200.0))
        thickness = max(1, int(font_scale * 1.5))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, thickness)
        ty = max(th + 5, y0 - 8)
        cv2.rectangle(vis, (x0, ty - th - 5), (x0 + tw + 6, ty + 5), color, -1)
        cv2.putText(vis, label, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    h, w = vis.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        vis = cv2.resize(vis, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return vis

def visualize_debug(gray: np.ndarray, proposals, scale, debug_info) -> np.ndarray:
    h, w = gray.shape[:2]
    energy = debug_info.get("energy")
    coherence = debug_info.get("coherence")
    if energy is None or coherence is None:
        return None

    e_norm = cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    c_norm = cv2.normalize(coherence, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    e_color = cv2.applyColorMap(e_norm, cv2.COLORMAP_JET)
    c_color = cv2.applyColorMap(c_norm, cv2.COLORMAP_VIRIDIS)

    for prop in proposals:
        x0, y0, x1, y1 = prop.bbox_small
        cv2.rectangle(e_color, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.rectangle(c_color, (x0, y0), (x1, y1), (0, 255, 0), 2)

    target_h = min(400, h)
    s = target_h / max(h, w)
    e_color = cv2.resize(e_color, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    c_color = cv2.resize(c_color, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    combined = np.hstack([e_color, c_color])
    return combined

# ============================================================================
# 主程序
# ============================================================================
def process_single_image(image_path: Path, debug: bool = False) -> RecognizeResult:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"  [错误] 无法读取图像: {image_path}")
        return RecognizeResult(image_name=image_path.name, ok=False)

    print(f"  图像尺寸: {img.shape[1]}×{img.shape[0]}")

    result = recognize_barcode(img, debug=debug)
    result.image_name = image_path.name

    proposals, scale, dbg = locate_barcode_proposals(img)
    vis = visualize_result(img, result, proposals, scale)
    vis_path = OUTPUT_DIR / f"{image_path.stem}_result.jpg"
    cv2.imwrite(str(vis_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 92])

    if debug:
        debug_vis = visualize_debug(img, proposals, scale, dbg)
        if debug_vis is not None:
            debug_path = OUTPUT_DIR / f"{image_path.stem}_debug.jpg"
            cv2.imwrite(str(debug_path), debug_vis, [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"  调试图已保存: {debug_path}")

    return result

def main():
    parser = argparse.ArgumentParser(
        description="基于模式识别方法的砂轮条形码识别程序")
    parser.add_argument("--image", type=str, default=None,
                        help="指定单张图像路径（不指定则处理 data/raw/ 下所有图像）")
    parser.add_argument("--debug", action="store_true",
                        help="输出调试可视化（梯度能量图 + 方向一致性图）")
    args = parser.parse_args()

    print("=" * 70)
    print("  砂轮条形码识别 — 基于模式识别方法")
    print("  技术路线: 结构张量定位 + Hough倾角矫正 + EAN-13扫描线解码")
    print("=" * 70)
    print()

    if args.image:
        image_paths = [Path(args.image)]
    else:
        image_paths = sorted(RAW_DIR.glob("*.bmp")) + sorted(RAW_DIR.glob("*.jpg"))
        image_paths = sorted(set(image_paths))

    if not image_paths:
        print(f"[错误] 在 {RAW_DIR} 下未找到任何图像文件")
        return

    print(f"共发现 {len(image_paths)} 张图像待处理\n")

    all_results = []
    success_count = 0

    for i, path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] 处理: {path.name}")
        result = process_single_image(path, debug=args.debug)
        all_results.append(result)

        status = "✓ 成功" if result.ok else "✗ 失败"
        text_display = result.text if result.ok else "—"
        match = " (匹配真值)" if result.ok and result.text == GT_TEXT else ""
        if result.ok and result.text == GT_TEXT:
            success_count += 1

        print(f"  结果: {status}")
        print(f"  条码值: {text_display}{match}")
        print(f"  置信度: {result.confidence:.3f}")
        print(f"  校验位: {'通过' if result.checksum_ok else '未通过'}")
        print(f"  候选区域: {result.num_candidates} 个")
        print(f"  耗时: {result.elapsed_ms:.0f} ms")
        if result.bbox:
            print(f"  定位框: [{result.bbox[0]:.0f}, {result.bbox[1]:.0f}] "
                  f"→ [{result.bbox[2]:.0f}, {result.bbox[3]:.0f}]")
        print()

    results_path = OUTPUT_DIR / "recognize_results.json"
    results_data = [r.to_dict() for r in all_results]
    results_path.write_text(
        json.dumps(results_data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 70)
    print(f"  识别完成!")
    print(f"  成功: {success_count}/{len(image_paths)}")
    print(f"  真值: {GT_TEXT} (EAN-13)")
    print(f"  结果已保存至: {OUTPUT_DIR}")
    print(f"  详细结果: {results_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()