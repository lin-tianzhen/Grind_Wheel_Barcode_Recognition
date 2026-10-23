"""Small image / geometry helpers shared by both pipelines."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def imread_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise OSError(f"Cannot write image: {path}")


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def order_quad(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Return 4 points in TL, TR, BR, BL order."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def quad_to_bbox(quad: np.ndarray) -> tuple[int, int, int, int]:
    q = np.asarray(quad)
    return (
        int(np.floor(q[:, 0].min())),
        int(np.floor(q[:, 1].min())),
        int(np.ceil(q[:, 0].max())),
        int(np.ceil(q[:, 1].max())),
    )


def clip_box(box: tuple[float, float, float, float], w: int, h: int):
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


def box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = max(1e-9, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)
    return float(inter / union)


def rotate_points(points: np.ndarray, angle: float, center: tuple[float, float]):
    c, s = math.cos(angle), math.sin(angle)
    pts = np.asarray(points, dtype=np.float32).copy()
    pts -= np.asarray(center, dtype=np.float32)
    x = pts[:, 0] * c - pts[:, 1] * s + center[0]
    y = pts[:, 0] * s + pts[:, 1] * c + center[1]
    return np.stack([x, y], axis=1)


def warp_perspective_image(img: np.ndarray, H: np.ndarray, dsize: tuple[int, int],
                           border=cv2.BORDER_REFLECT_101) -> np.ndarray:
    return cv2.warpPerspective(img, H, dsize, flags=cv2.INTER_LINEAR,
                               borderMode=border)


def homography_from_quad(src_quad: np.ndarray, dst_quad: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(
        np.asarray(src_quad, dtype=np.float32), np.asarray(dst_quad, dtype=np.float32)
    )


def rectify_quad(img: np.ndarray, quad: np.ndarray, out_w: int | None = None,
                 out_h: int | None = None, pad: int = 0) -> np.ndarray:
    """Perspective-rectify a rotated barcode quadrilateral to an upright image."""
    q = order_quad(quad)
    if out_w is None:
        out_w = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    if out_h is None:
        out_h = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    out_w, out_h = max(16, out_w), max(16, out_h)
    dst = np.array(
        [[pad, pad], [out_w - 1 + pad, pad], [out_w - 1 + pad, out_h - 1 + pad],
         [pad, out_h - 1 + pad]], dtype=np.float32)
    H = homography_from_quad(q, dst)
    return cv2.warpPerspective(
        img, H, (out_w + 2 * pad, out_h + 2 * pad),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Barcode text helpers
# ---------------------------------------------------------------------------
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


def char_error_rate(pred: str, gt: str) -> float:
    """Levenshtein distance / len(gt)."""
    if not gt:
        return 0.0 if not pred else 1.0
    prev = list(range(len(pred) + 1))
    for i, gc in enumerate(gt, 1):
        cur = [i]
        for j, pc in enumerate(pred, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (gc != pc)))
        prev = cur
    return prev[-1] / len(gt)


def timer() -> float:
    import time
    return time.perf_counter()
