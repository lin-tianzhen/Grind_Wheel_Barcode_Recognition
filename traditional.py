"""Method 1: classical computer-vision barcode localisation + EAN-13 decode.

The detector does not use any learned parameters.  It exploits a property that
is very stable for this industrial part: barcodes are the only image region
with a *dense, strongly oriented* set of edges.  A structure tensor gives the
local edge energy and orientation coherence; thresholding + morphology turns
that into a handful of rectangular proposals.  Each proposal is deskewed and
passed to the transparent EAN-13 scan-line decoder.

This is deliberately different from the deep-learning method, which learns the
appearance of the barcode rather than relying on this hand-designed cue.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import cv2
import numpy as np

from .config import MAX_SIDE_LOCATE
from .decode import DecodeResult, decode_barcode_crop, decode_ean13_custom
from .preprocess import (estimate_bar_angle, normalize_illumination,
                         rotate_keep_size, unsharp, wiener_deconvolve)
from .utils import box_iou, clip_box, expand_box


@dataclass
class MethodResult:
    method: str
    text: str = ""
    ok: bool = False
    checksum_ok: bool = False
    confidence: float = 0.0
    bbox: list[float] = field(default_factory=list)   # x0,y0,x1,y1 in input image
    decode_method: str = "none"
    elapsed_ms: float = 0.0
    num_candidates: int = 0

    def to_dict(self):
        d = asdict(self)
        d["ok"] = bool(self.ok)
        d["checksum_ok"] = bool(self.checksum_ok)
        return d


@dataclass
class Proposal:
    bbox_small: tuple[int, int, int, int]
    score: float
    fill: float
    coherence: float


class TraditionalBarcodeRecognizer:
    name = "传统 CV（梯度能量+结构张量+EAN 扫描线解码）"

    def __init__(self, max_side: int = MAX_SIDE_LOCATE, debug: bool = False):
        self.max_side = max_side
        self.debug = debug
        self.last_debug: dict = {}

    # ------------------------------------------------------------------
    # Proposal generation
    # ------------------------------------------------------------------
    def _proposals(self, gray: np.ndarray) -> tuple[list[Proposal], float, dict]:
        h, w = gray.shape[:2]
        scale = min(1.0, self.max_side / max(h, w))
        small = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA) if scale < 1 else gray.copy()

        # Light denoising keeps the fine bars but suppresses abrasive grain.
        s = cv2.GaussianBlur(small, (3, 3), 0)
        gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3)

        k = 11 if max(s.shape) < 1500 else 15
        ex = cv2.GaussianBlur(gx * gx, (k, k), 0)
        ey = cv2.GaussianBlur(gy * gy, (k, k), 0)
        exy = cv2.GaussianBlur(gx * gy, (k, k), 0)
        energy = np.sqrt(np.maximum(0, ex + ey))
        coherence = np.sqrt((ex - ey) ** 2 + 4 * exy ** 2) / (ex + ey + 1e-6)

        all_boxes: list[tuple[int, int, int, int, float, float]] = []
        for pct in (94.5, 95.5, 96.5, 97.5, 98.5, 99.2):
            e_thr = float(np.percentile(energy, pct))
            mask = ((energy >= e_thr) & (coherence >= 0.45)).astype(np.uint8) * 255
            # Barcode bars are close together: close along the bar direction.
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (25, 13)))
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

            min_w = max(24, int(0.035 * small.shape[1]))
            min_h = max(12, int(0.012 * small.shape[0]))
            for i in range(1, num):
                x, y, bw, bh, area = stats[i]
                if bw < min_w or bh < min_h or bw > 0.55 * small.shape[1]:
                    continue
                if bh > 0.4 * small.shape[0]:
                    continue
                ar = max(bw, bh) / max(1, min(bw, bh))
                if not (1.25 <= ar <= 8.0):
                    continue
                fill = area / float(bw * bh)
                if fill < 0.22:
                    continue
                comp = labels[y:y + bh, x:x + bw] == i
                coh = float(coherence[y:y + bh, x:x + bw][comp].mean())
                # Prefer rectangular, dense and highly coherent regions.
                score = 1.4 * fill + 1.0 * coh + 0.25 * min(ar / 3.0, 1.0)
                all_boxes.append((x, y, x + bw, y + bh, score, fill))

        # Merge duplicates from different thresholds.
        all_boxes.sort(key=lambda b: b[4], reverse=True)
        kept: list[tuple[int, int, int, int, float, float]] = []
        for b in all_boxes:
            if all(box_iou(b[:4], q[:4]) < 0.35 for q in kept):
                kept.append(b)
            if len(kept) >= 14:
                break

        proposals = [
            Proposal((int(b[0]), int(b[1]), int(b[2]), int(b[3])), float(b[4]),
                     float(b[5]), 0.0)
            for b in kept
        ]
        debug = {
            "energy": energy,
            "coherence": coherence,
            "scale": scale,
        }
        return proposals, scale, debug

    # ------------------------------------------------------------------
    # Decoding one proposal
    # ------------------------------------------------------------------
    def _decode_proposal(self, gray: np.ndarray, box, scale: float) -> Optional[DecodeResult]:
        h, w = gray.shape[:2]
        x0, y0, x1, y1 = [int(round(v / scale)) for v in box]
        x0, y0, x1, y1 = expand_box((x0, y0, x1, y1), 0.18, w, h)
        crop = gray[y0:y1, x0:x1]
        if crop.size < 100:
            return None

        # Try the unrotated crop first: for mild skew a horizontal scan line
        # still crosses all bars, and this avoids trusting a bad Hough angle.
        res = decode_ean13_custom(crop)
        if res.ok:
            return res

        angle = estimate_bar_angle(crop)
        angles = [0.0, angle, -angle, angle + 6, angle - 6,
                  angle + 12, angle - 12, -angle + 6, -angle - 6]
        for ang in angles:
            if abs(ang) < 0.8:
                continue
            rectified = rotate_keep_size(crop, ang)
            res = decode_ean13_custom(rectified)
            if res.ok:
                return res

        # Strong sharpening / deconvolution helps defocused labels.
        sharpened = unsharp(normalize_illumination(crop, 4.0, 6), 2.2, 2)
        for ang in (0.0, angle, -angle):
            rectified = rotate_keep_size(sharpened, ang) if abs(ang) > 0.8 else sharpened
            res = decode_ean13_custom(rectified)
            if res.ok:
                return res

        # Defocused labels: Wiener deconvolution + a small angle search.
        try:
            deconv = unsharp(normalize_illumination(
                wiener_deconvolve(crop, 2.8, 0.005), 3.0, 6), 1.5, 2)
            for ang in range(-10, 11, 2):
                rectified = (rotate_keep_size(deconv, ang) if ang
                             else deconv)
                res = decode_ean13_custom(rectified)
                if res.ok:
                    return res
        except Exception:
            pass

        # Last resort: the shared crop decoder (may try small extra rotations).
        return decode_barcode_crop(crop, try_rotations=True)

    # ------------------------------------------------------------------
    def predict(self, image: np.ndarray) -> MethodResult:
        t0 = time.perf_counter()
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        proposals, scale, debug = self._proposals(gray)
        self.last_debug = debug

        candidates = []
        for prop in proposals:
            x0, y0, x1, y1 = prop.bbox_small
            candidates.append((prop, [float(v / scale) for v in (x0, y0, x1, y1)]))

        best: Optional[MethodResult] = None
        for rank, (prop, bbox) in enumerate(candidates):
            res = self._decode_proposal(gray, prop.bbox_small, scale)
            if res is not None and res.ok:
                # The high-energy blob often covers only the central bars; the
                # decoder was run on an 18%-expanded crop.  Report that crop as
                # the detected barcode region.
                x0, y0, x1, y1 = expand_box(
                    (bbox[0], bbox[1], bbox[2], bbox[3]), 0.18, w, h)
                bbox = [float(x0), float(y0), float(x1), float(y1)]
                out = MethodResult(
                    method=self.name,
                    text=res.text,
                    ok=True,
                    checksum_ok=res.checksum_ok,
                    confidence=res.confidence,
                    bbox=[float(round(v, 1)) for v in bbox],
                    decode_method=res.method,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    num_candidates=len(candidates),
                )
                self.last_debug["proposals"] = [p.bbox_small for p in proposals]
                self.last_debug["chosen"] = prop.bbox_small
                return out

        # Close-up fallback: when the barcode occupies most of the frame there
        # is no meaningful "search" problem left.  Decode the whole image (and
        # a slightly cropped centre) directly.  This is what makes the method
        # work on BarAreaRoted.bmp as well as on the full wheel photo.
        aspect = w / max(1.0, h)
        if 1.2 <= aspect <= 4.5:
            for ratio in (0.0, 0.08, 0.15):
                x0, y0, x1, y1 = expand_box((0, 0, w, h), -ratio, w, h)
                sub = gray[y0:y1, x0:x1]
                res = decode_barcode_crop(sub, try_rotations=True)
                if res.ok:
                    return MethodResult(
                        method=self.name,
                        text=res.text,
                        ok=True,
                        checksum_ok=res.checksum_ok,
                        confidence=res.confidence,
                        bbox=[float(x0), float(y0), float(x1), float(y1)],
                        decode_method=res.method,
                        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                        num_candidates=len(candidates),
                    )

        # Nothing decoded: report the visually most plausible proposal so that
        # detection IoU can still be measured separately from decoding.
        if candidates:
            bbox = candidates[0][1]
        else:
            bbox = []
        return MethodResult(
            method=self.name,
            ok=False,
            bbox=bbox,
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            num_candidates=len(candidates),
        )

    # ------------------------------------------------------------------
    def debug_visualisation(self, image: np.ndarray) -> np.ndarray:
        """Draw proposals + chosen box on a BGR copy for the report."""
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
        proposals = self.last_debug.get("proposals", [])
        scale = self.last_debug.get("scale", 1.0)
        for b in proposals:
            x0, y0, x1, y1 = [int(round(v / scale)) for v in b]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 180, 255), 2)
        chosen = self.last_debug.get("chosen")
        if chosen:
            x0, y0, x1, y1 = [int(round(v / scale)) for v in chosen]
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 4)
        return vis
