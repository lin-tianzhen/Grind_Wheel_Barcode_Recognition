"""Dataset augmentation / synthetic scene generation.

Only one real barcode example is available.  We therefore use *one-shot*
augmentation:

1. the barcode patch is extracted once;
2. its geometry (rotation, scale, perspective), photometry (brightness,
   contrast, gamma, blur, noise, illumination gradient) and background are
   randomised;
3. a pixel-accurate mask of the bars is produced together with every sample.

The same transforms are used to build the held-out test set with known
ground-truth quadrilaterals.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import CROP_BAR_BOX, CROP_IMAGE, FULL_IMAGE, GT_QUAD_FULL, RAW_DIR
from .utils import order_quad, rotate_points, save_json


# ---------------------------------------------------------------------------
# Basic photometric operations
# ---------------------------------------------------------------------------
def random_photometric(img: np.ndarray, rng: np.random.Generator,
                       strength: float = 1.0) -> np.ndarray:
    out = img.astype(np.float32)
    # brightness / contrast
    alpha = rng.uniform(0.72, 1.32) ** strength
    beta = rng.uniform(-28, 28) * strength
    out = out * alpha + beta
    # gamma
    gamma = rng.uniform(0.72, 1.35) ** strength
    out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, gamma)
    # illumination gradient
    if rng.random() < 0.65 * strength:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        gx = rng.uniform(-0.15, 0.15) * strength
        gy = rng.uniform(-0.15, 0.15) * strength
        ramp = gx * (xx / max(1, w) - 0.5) + gy * (yy / max(1, h) - 0.5)
        out *= (1.0 + ramp)
    out = np.clip(out, 0, 255).astype(np.uint8)
    # blur / sensor noise
    if rng.random() < 0.65 * strength:
        sigma = rng.uniform(0.35, 1.8) * strength
        out = cv2.GaussianBlur(out, (0, 0), sigma)
    if rng.random() < 0.7 * strength:
        noise_sigma = rng.uniform(1.0, 9.0) * strength
        noise = rng.normal(0, noise_sigma, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def random_occlusion(img: np.ndarray, rng: np.random.Generator,
                     max_frac: float = 0.18) -> np.ndarray:
    """Add a few dark scratches / dust specks (does not cover the whole code)."""
    out = img.copy()
    h, w = out.shape[:2]
    for _ in range(rng.integers(0, 4)):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1 = x0 + rng.integers(3, max(4, int(w * max_frac * 0.35)))
        y1 = y0 + rng.integers(1, max(2, int(h * 0.012)))
        color = int(rng.integers(20, 90))
        cv2.rectangle(out, (x0, y0), (x1, y1), color, -1)
        out = cv2.GaussianBlur(out, (3, 3), 0)
    return out


# ---------------------------------------------------------------------------
# Scene composition
# ---------------------------------------------------------------------------
def _extract_barcode_source():
    gray = cv2.imread(str(CROP_IMAGE), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(CROP_IMAGE)
    patch = gray.copy()
    bx0, by0, bx1, by1 = CROP_BAR_BOX
    mask = np.zeros_like(patch)
    mask[by0:by1, bx0:bx1] = 255
    return patch, mask


def _background_pool() -> list[np.ndarray]:
    """Texture backgrounds that do not contain the target barcode."""
    images = []
    for p in [FULL_IMAGE, RAW_DIR / "RegulaImg.bmp"]:
        if not p.exists():
            continue
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        images.append(im)
    if not images:
        raise FileNotFoundError("No background image available")
    return images


class SceneGenerator:
    def __init__(self, seed: int = 2026):
        self.rng = np.random.default_rng(seed)
        self.patch, self.mask = _extract_barcode_source()
        self.backgrounds = _background_pool()
        self.gt_mask_full = np.zeros(self.backgrounds[0].shape[:2], np.uint8)
        cv2.fillPoly(self.gt_mask_full,
                     [np.asarray(GT_QUAD_FULL, dtype=np.int32)], 255)

    def _random_background(self, size: int) -> np.ndarray:
        for _ in range(30):
            bg = self.backgrounds[int(self.rng.integers(len(self.backgrounds)))]
            h, w = bg.shape
            if h < size or w < size:
                bg = cv2.resize(bg, (max(size, w), max(size, h)))
                h, w = bg.shape
            x = int(self.rng.integers(0, w - size + 1))
            y = int(self.rng.integers(0, h - size + 1))
            crop = bg[y:y + size, x:x + size].copy()
            # Exclude the real barcode of RegulaImg2 from backgrounds.
            if bg is self.backgrounds[0] and h == self.gt_mask_full.shape[0]:
                if self.gt_mask_full[y:y + size, x:x + size].mean() > 2:
                    continue
            return crop
        return np.full((size, size), 120, np.uint8)

    def make_scene(self, size: int, strength: float = 1.0) -> tuple[np.ndarray, np.ndarray, list]:
        rng = self.rng
        bg = self._random_background(size)
        bg = random_photometric(bg, rng, 0.7 * strength)
        # Slight background rotation makes the texture less repetitive.
        ang_bg = rng.uniform(-18, 18)
        M = cv2.getRotationMatrix2D((size / 2, size / 2), ang_bg, 1.0)
        bg = cv2.warpAffine(bg, M, (size, size), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)

        # Barcode size: 28%-72% of the scene width.
        target_w = int(rng.uniform(0.28, 0.72) * size)
        src_h, src_w = self.patch.shape
        target_h = max(24, int(round(target_w * src_h / src_w)))
        if target_h > 0.7 * size:
            target_h = int(0.7 * size)
            target_w = int(target_h * src_w / src_h)
        patch = cv2.resize(self.patch, (target_w, target_h),
                           interpolation=cv2.INTER_AREA if target_w < src_w else cv2.INTER_CUBIC)
        mask = cv2.resize(self.mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Rotated placement.
        angle = float(rng.uniform(-32, 32))
        diag = int(math.ceil(math.hypot(target_w, target_h))) + 8
        canvas_p = np.full((diag, diag), 0, np.uint8)
        canvas_m = np.zeros((diag, diag), np.uint8)
        canvas_a = np.zeros((diag, diag), np.uint8)
        ox, oy = (diag - target_w) // 2, (diag - target_h) // 2
        canvas_p[oy:oy + target_h, ox:ox + target_w] = patch
        canvas_m[oy:oy + target_h, ox:ox + target_w] = mask
        canvas_a[oy:oy + target_h, ox:ox + target_w] = 255
        R = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
        rp = cv2.warpAffine(canvas_p, R, (diag, diag), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        rm = cv2.warpAffine(canvas_m, R, (diag, diag), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ra = cv2.warpAffine(canvas_a, R, (diag, diag), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Random placement, keeping the patch inside the scene most of the time.
        max_x = max(0, size - diag)
        max_y = max(0, size - diag)
        px = int(rng.integers(-diag // 8, max_x + 1))
        py = int(rng.integers(-diag // 8, max_y + 1))
        scene = bg.copy()
        out_mask = np.zeros((size, size), np.uint8)
        x0, y0 = max(0, px), max(0, py)
        x1, y1 = min(size, px + diag), min(size, py + diag)
        sx0, sy0 = x0 - px, y0 - py
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
        if x1 > x0 and y1 > y0:
            # Paste the whole label rectangle (validity mask > 0).
            region = scene[y0:y1, x0:x1]
            p = rp[sy0:sy1, sx0:sx1]
            valid = ra[sy0:sy1, sx0:sx1] > 0
            region[valid] = p[valid]
            scene[y0:y1, x0:x1] = region
            out_mask[y0:y1, x0:x1] = rm[sy0:sy1, sx0:sx1]

        # Perspective warp of the whole scene.
        if rng.random() < 0.7 * strength:
            d = int(0.018 * size)
            dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
                           np.float32)
            for i in range(4):
                dst[i] += rng.integers(-d, d + 1, size=2)
            H = cv2.getPerspectiveTransform(
                np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], np.float32),
                dst.astype(np.float32))
            scene = cv2.warpPerspective(scene, H, (size, size),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT_101)
            out_mask = cv2.warpPerspective(out_mask, H, (size, size),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT)

        scene = random_photometric(scene, rng, strength)
        scene = random_occlusion(scene, rng)
        # A mild final blur mimics the shallow depth of field of the wheel photo.
        if rng.random() < 0.55:
            scene = cv2.GaussianBlur(scene, (0, 0), rng.uniform(0.3, 1.2))

        # Hard negative scenes (no barcode) are useful for suppressing texture.
        if rng.random() < 0.12:
            scene = random_photometric(self._random_background(size), rng, 1.0)
            out_mask[:] = 0

        ys, xs = np.nonzero(out_mask > 0)
        if len(xs) == 0:
            quad = []
        else:
            rect = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
            quad = cv2.boxPoints(rect).tolist()
        return scene, out_mask, quad


# ---------------------------------------------------------------------------
# Dataset writers
# ---------------------------------------------------------------------------
def build_synthetic_dataset(out_dir: Path, n: int, size: int, seed: int,
                            strength: float = 1.0, split: str = "train"):
    img_dir = out_dir / split / "images"
    mask_dir = out_dir / split / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    gen = SceneGenerator(seed)
    meta = []
    for i in range(n):
        scene, mask, quad = gen.make_scene(size, strength)
        name = f"{split}_{i:04d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), scene,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(mask_dir / f"{name}.png"), mask)
        meta.append({"name": name, "quad": quad, "has_barcode": len(quad) > 0})
    save_json(out_dir / f"{split}_manifest.json", meta)
    return meta


# ---------------------------------------------------------------------------
# Test-set augmentation with exact ground-truth transformation
# ---------------------------------------------------------------------------
def transform_points_h(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out.astype(np.float32)


def rotate_image_points(img: np.ndarray, points: np.ndarray, angle: float):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    pts = transform_points_h(points, np.vstack([M, [0, 0, 1]]))
    return out, pts


def build_test_set(out_dir: Path, max_side: int = 2000, seed: int = 7):
    """Create real + augmented test cases, each with a GT quadrilateral."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir, gt_dir = out_dir / "images", out_dir / "annotated"
    img_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    full = cv2.imread(str(FULL_IMAGE), cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(str(CROP_IMAGE), cv2.IMREAD_GRAYSCALE)
    if full is None or crop is None:
        raise FileNotFoundError("source images missing")

    scale = max_side / max(full.shape)
    full = cv2.resize(full, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gt_quad = np.asarray(GT_QUAD_FULL, np.float32) * scale
    rng = np.random.default_rng(seed)

    cases = []

    def add(name, img, quad, kind, text, note=""):
        p = img_dir / f"{name}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 94])
        # annotated copy for the report
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if quad is not None and len(quad):
            cv2.polylines(vis, [np.asarray(quad, np.int32).reshape(-1, 1, 2)],
                          True, (0, 0, 255), 3)
        cv2.imwrite(str(gt_dir / f"{name}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cases.append({
            "name": name,
            "image": str(p.relative_to(out_dir.parents[1])),
            "quad": np.asarray(quad).tolist() if quad is not None and len(quad) else [],
            "kind": kind,
            "gt_text": text,
            "note": note,
        })

    add("00_full", full, gt_quad, "real_full", "4014855012527", "原始砂轮图")
    add("01_crop", crop, np.array([[20, 20], [505, 20], [505, 212], [20, 212]]),
        "real_crop", "4014855012527", "给定条码区域")

    # geometric variants
    for i, ang in enumerate([-12, -7, 7, 12], start=1):
        im, q = rotate_image_points(full, gt_quad, ang)
        im = random_photometric(im, rng, 0.55)
        add(f"1{i}_rot{ang:+d}", im, q, "geometry", "4014855012527", f"旋转 {ang}°")

    # perspective
    h, w = full.shape
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst += np.array([[-30, -10], [25, 12], [18, -16], [-22, 9]], np.float32)
    H = cv2.getPerspectiveTransform(
        np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32), dst)
    im = cv2.warpPerspective(full, H, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    add("20_perspective", im, transform_points_h(gt_quad, H), "geometry",
        "4014855012527", "透视畸变")

    # photometric variants
    for name, note, fn in [
        ("30_dark", "低照度", lambda x: np.clip(x.astype(np.float32) * 0.58, 0, 255).astype(np.uint8)),
        ("31_bright", "过曝", lambda x: np.clip(x.astype(np.float32) * 1.35 + 20, 0, 255).astype(np.uint8)),
        ("32_blur", "失焦模糊", lambda x: cv2.GaussianBlur(x, (0, 0), 2.2)),
        ("33_motion", "运动模糊", lambda x: cv2.filter2D(
            x, -1, np.eye(15, dtype=np.float32) / 15.0)),
        ("34_noise", "高斯噪声", lambda x: np.clip(
            x.astype(np.float32) + rng.normal(0, 13, x.shape), 0, 255).astype(np.uint8)),
    ]:
        add(name, fn(full), gt_quad, "photometric", "4014855012527", note)

    # resolution + combined degradation
    small = cv2.resize(full, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)
    add("40_lowres", small, gt_quad * 0.55, "resolution", "4014855012527", "低分辨率 55%")
    im = cv2.GaussianBlur(full, (0, 0), 1.2)
    im = np.clip(im.astype(np.float32) * 0.75 + 8, 0, 255).astype(np.uint8)
    im, q = rotate_image_points(im, gt_quad, -9)
    im = np.clip(im.astype(np.float32) + rng.normal(0, 5, im.shape), 0, 255).astype(np.uint8)
    add("41_combined", im, q, "combined", "4014855012527", "旋转+变暗+模糊")

    # JPEG compression artefact
    ok, enc = cv2.imencode(".jpg", full, [cv2.IMWRITE_JPEG_QUALITY, 35])
    if ok:
        add("42_jpeg35", cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE), gt_quad,
            "compression", "4014855012527", "JPEG 质量 35")

    # partial occlusion
    im = full.copy()
    oc = np.asarray(gt_quad)
    x0, y0 = int(oc[:, 0].min()), int(oc[:, 1].min())
    cv2.rectangle(im, (x0 + 15, y0 + 8), (x0 + 55, y0 + 90), 55, -1)
    add("43_occlusion", im, gt_quad, "occlusion", "4014855012527", "局部遮挡")

    save_json(out_dir / "manifest.json", cases)
    return cases
