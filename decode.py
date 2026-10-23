"""Barcode decoding back-end.

Two independent decoders are provided:

* ``decode_ean13_custom`` - a small, transparent EAN-13 scan-line decoder
  written with NumPy/OpenCV.  It is used by the classical pipeline and makes
  the method independent of black-box barcode SDKs.
* ``decode_with_zxing`` - ZXing-C++ fallback / cross-check.

Both return a :class:`DecodeResult`.  The localization stage is what differs
between the two assignment methods; sharing the decoder makes the comparison
about *detection*, not about third-party decoder quality.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import cv2
import numpy as np

from .preprocess import normalize_illumination, unsharp
from .utils import valid_ean13

try:  # optional dependency, used as a fallback and for verification
    import zxingcpp
except Exception:  # pragma: no cover
    zxingcpp = None


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
PARITY = ["LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
          "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL"]


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


def _hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def _decode_95_bits(bits: str) -> Optional[tuple[str, int]]:
    """Decode an exact 95-module string, trying both polarities."""
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
        if parity not in PARITY:
            continue
        first = PARITY.index(parity)
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
    line = np.asarray(line)
    if line.ndim == 2:
        line = line.ravel()
    blur = cv2.GaussianBlur(line.reshape(1, -1), (3, 3), 0).ravel()
    th, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th < 40:  # a nearly uniform line carries no barcode
        return np.zeros_like(bw)
    # Ensure the bars are the bright foreground.
    if np.mean(bw > 0) > 0.62:
        bw = 255 - bw
    return bw


def _decode_scanline(line: np.ndarray) -> Optional[tuple[str, float, float, float]]:
    bw = _binarize_line(line)
    idx = np.flatnonzero(bw > 0)
    if len(idx) < 80:
        return None

    # Split at long bright gaps; keep the widest connected group.
    gaps = np.flatnonzero(np.diff(idx) > max(12, int(0.05 * len(line))))
    groups = np.split(idx, gaps + 1)
    groups = [g for g in groups if len(g) > 40 and g[-1] - g[0] > 120]
    if not groups:
        return None
    group = max(groups, key=lambda g: len(g))
    x0, x1 = int(group[0]), int(group[-1])
    if not (180 <= x1 - x0 <= 0.95 * len(line)):
        return None

    # The first/last bar can be partially clipped; search a small neighbourhood.
    a_values = np.arange(x0 - 6, x0 + 2.5, 0.5)
    z_values = np.arange(x1 - 2.5, x1 + 6, 0.5)
    best = None
    for a in a_values:
        for z in z_values:
            width = (z - a + 1) / 95.0
            if not (3.0 <= width <= 12.0):
                continue
            pos = a + (np.arange(95) + 0.5) * width
            pos = np.clip(np.rint(pos).astype(np.int32), 0, len(line) - 1)
            bits = "".join("1" if bw[p] > 0 else "0" for p in pos)
            res = _decode_95_bits(bits)
            if res is None:
                continue
            text, cost = res
            # Prefer a clean legal code and a plausible module width.
            width_penalty = abs(width - 4.6)
            score = cost + 0.15 * width_penalty
            if best is None or score < best[0]:
                best = (score, text, float(a), float(z), float(width), cost)
    if best is None:
        return None
    score, text, a, z, width, cost = best
    conf = max(0.0, min(1.0, 1.0 - cost / 95.0))
    return text, conf, a, z


def decode_ean13_custom(gray: np.ndarray) -> DecodeResult:
    """Decode EAN-13 by scanning many horizontal lines of an upright crop.

    The crop should contain bars + a small quiet zone; the human-readable
    digits may be present.  Rotation must already be corrected to roughly
    +/-3 degrees.
    """
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    h, w = g.shape[:2]
    if h < 12 or w < 60:
        return DecodeResult(method="custom-ean13")

    # Normalise and enlarge without changing the aspect ratio.
    up = max(1.0, 520.0 / max(1, w))
    if up > 1:
        g = cv2.resize(g, None, fx=min(up, 3.0), fy=min(up, 3.0),
                       interpolation=cv2.INTER_CUBIC)
    g = unsharp(normalize_illumination(g, 2.0, 6), 0.4, 2)
    h, w = g.shape[:2]

    votes: dict[str, list[float]] = {}
    for frac in np.linspace(0.12, 0.88, 19):
        y = int(round((h - 1) * frac))
        res = _decode_scanline(g[y])
        if res is None:
            continue
        text, conf, a, z = res
        votes.setdefault(text, []).append(conf)

    if not votes:
        return DecodeResult(method="custom-ean13")

    text, confs = max(votes.items(), key=lambda kv: (len(kv[1]), max(kv[1])))
    return DecodeResult(
        text=text,
        ok=valid_ean13(text),
        method="custom-ean13",
        confidence=float(np.mean(sorted(confs)[-5:])),
        checksum_ok=valid_ean13(text),
    )


def decode_with_zxing(gray: np.ndarray) -> DecodeResult:
    if zxingcpp is None:
        return DecodeResult(method="zxing")
    gray = np.asarray(gray)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    variants = []
    base = normalize_illumination(gray, 2.0, 6)
    variants.append(base)
    variants.append(cv2.bitwise_not(base))
    variants.append(cv2.threshold(base, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
    variants.append(cv2.threshold(base, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1])

    for v in list(variants):
        h, w = v.shape[:2]
        for scale in (1.0, 1.7, 2.5):
            if scale != 1.0:
                img = cv2.resize(v, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_CUBIC)
            else:
                img = v
            for attempt in range(2):
                try:
                    results = zxingcpp.read_barcodes(
                        img, try_harder=True, try_rotate=True,
                        formats=zxingcpp.BarcodeFormat.EAN13)
                except TypeError:
                    results = zxingcpp.read_barcodes(img)
                except Exception:
                    results = []
                if results:
                    for r in results:
                        text = str(r.text).strip()
                        if valid_ean13(text):
                            return DecodeResult(
                                text=text, ok=True, method="zxing",
                                confidence=0.99, checksum_ok=True)
                if attempt == 0:
                    # Light sharpening, then break out of the attempt loop.
                    img = unsharp(img, 0.8, 3)
                else:
                    break
    return DecodeResult(method="zxing")


def decode_barcode_crop(gray: np.ndarray, try_rotations: bool = True,
                       use_zxing_fallback: bool = False) -> DecodeResult:
    """Best-effort decoding of a barcode crop.

    The image is first deskewed with a Hough-based bar-angle estimate, then the
    transparent EAN-13 scan-line decoder is applied.  ZXing is intentionally
    *optional*: on very low-resolution, inverted crops it can occasionally
    return a checksum-valid but wrong code, so it is only used as a cross-check
    in the demo, not as the default decision maker.
    """
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)

    # Deskew once, using the strong near-vertical bar edges.
    from .preprocess import estimate_bar_angle, rotate_keep_size
    angle = estimate_bar_angle(g)
    if abs(angle) > 2.0:
        g = rotate_keep_size(g, angle)

    candidates = [g]
    if try_rotations:
        for ang in (-4, -2, 2, 4, -7, 7):
            h, w = g.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            candidates.append(cv2.warpAffine(g, M, (w, h),
                                             flags=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_REPLICATE))

    best = DecodeResult()
    for img in candidates:
        r = decode_ean13_custom(img)
        if r.ok and r.confidence >= 0.7:
            return r
        if r.ok and not best.ok:
            best = r

    if use_zxing_fallback and not best.ok:
        z = decode_with_zxing(g)
        if z.ok:
            return z
    return best
