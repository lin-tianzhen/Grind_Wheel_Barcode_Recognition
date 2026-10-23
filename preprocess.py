"""Image pre-processing utilities (classical CV)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_max_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale == 1.0:
        return img.copy(), 1.0
    out = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_AREA)
    return out, scale


def normalize_illumination(gray: np.ndarray, clip: float = 2.5,
                           grid: int = 8) -> np.ndarray:
    """Remove low-frequency lighting changes while keeping bar edges sharp."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    out = clahe.apply(gray)
    # A very light denoise avoids amplifying the abrasive texture.
    return cv2.bilateralFilter(out, d=5, sigmaColor=25, sigmaSpace=3)


def unsharp(gray: np.ndarray, amount: float = 0.8, radius: int = 3) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), radius)
    return cv2.addWeighted(gray, 1 + amount, blur, -amount, 0)


def binarize_otsu(gray: np.ndarray, invert: bool = False) -> np.ndarray:
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)[1]


def local_threshold(gray: np.ndarray, block: int = 51, c: int = 7) -> np.ndarray:
    block = block if block % 2 == 1 else block + 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        max(3, block), c)


def preprocess_for_decode(gray: np.ndarray, upscale: float = 1.5,
                          invert: bool = True) -> np.ndarray:
    """Enhance a small, low-contrast barcode crop before decoding.

    The label used in this assignment has *bright* bars on a *dark* background.
    Most barcode decoders expect the opposite polarity, hence ``invert``.
    """
    g = to_gray(gray)
    g = normalize_illumination(g, clip=2.2, grid=6)
    g = unsharp(g, 0.6, 2)
    if upscale > 1:
        g = cv2.resize(g, None, fx=upscale, fy=upscale,
                       interpolation=cv2.INTER_CUBIC)
    if invert:
        g = cv2.bitwise_not(g)
    return g


def bright_bar_mask(gray: np.ndarray) -> np.ndarray:
    """Mask of the bright bars / white label area (used for angle estimation)."""
    g = normalize_illumination(gray, clip=2.0, grid=6)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5)))
    return m


def estimate_bar_angle(gray: np.ndarray) -> float:
    """Estimate in-plane rotation of near-vertical bars (degrees, OpenCV sense).

    Hough segments are used instead of PCA because the digits below the bars
    would otherwise bias the estimate.
    """
    g = normalize_illumination(gray, clip=2.0, grid=6)
    edges = cv2.Canny(g, 40, 130, apertureSize=3)
    h, w = edges.shape[:2]
    min_len = max(18, int(0.18 * max(h, w)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25,
                            minLineLength=min_len, maxLineGap=4)
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) < 0.8 * max(1, abs(dx)):
            continue  # keep near-vertical lines only
        ang = np.degrees(np.arctan2(dy, dx))  # (-90, 90]
        if ang < -90:
            ang += 180
        # vertical line -> 90 deg.  Convert to a tilt w.r.t. vertical.
        tilt = ang - 90 if ang > 0 else ang + 90
        if abs(tilt) <= 35:
            angles.append(tilt)
    if not angles:
        return 0.0
    return float(np.median(angles))


def rotate_keep_size(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def wiener_deconvolve(gray: np.ndarray, sigma: float = 2.8,
                      nsr: float = 0.005) -> np.ndarray:
    """Frequency-domain Wiener deconvolution for a Gaussian defocus kernel.

    Used only as a last-resort branch for strongly defocused barcodes.
    """
    img = np.asarray(gray, dtype=np.float32)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fft2(img)
    fy = np.fft.fftfreq(img.shape[0]).astype(np.float32)[:, None]
    fx = np.fft.fftfreq(img.shape[1]).astype(np.float32)[None, :]
    H = np.exp(-2 * (np.pi ** 2) * (sigma ** 2) * (fx ** 2 + fy ** 2))
    G = np.conj(H) / (np.abs(H) ** 2 + nsr)
    out = np.real(np.fft.ifft2(f * G))
    return np.clip(out, 0, 255).astype(np.uint8)
