"""推理逻辑: 滑窗推理、候选框提取、透视矫正与端到端推理."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .config import GT_TEXT, imread_unicode
from .decode import decode_barcode_crop, valid_ean13
from .train import get_device


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