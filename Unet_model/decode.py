"""EAN-13 解码器."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

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