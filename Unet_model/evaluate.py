"""评估与可视化."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from .config import RAW_DIR, REAL_DIR, SYNTH_DIR
from .dataset import BarcodeDataset
from .inference import infer_single
from .losses import CombinedLoss
from .train import get_device, validate


def evaluate(model, device: torch.device = None, train_size: int = 384,
             long_side: int = 1280, use_fallback: bool = True):
    """在验证集和原始图像上评估模型."""
    if device is None:
        device = get_device()
    data_root = REAL_DIR if (REAL_DIR / "val_manifest.json").exists() else SYNTH_DIR
    val_ds = BarcodeDataset(data_root, "val", size=train_size, augment=False,balance=False)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=0)
    criterion = CombinedLoss()
    val_loss, val_iou, fp_rate = validate(model, val_loader, criterion, device)
    print(f"\n验证集评估: val_loss={val_loss:.4f}, "
          f"val_iou(正样本)={val_iou:.4f}, 负样本误报率={fp_rate:.2%}")
    raw_images = sorted(RAW_DIR.glob("*.bmp"))
    if not raw_images:
        return
    print(f"\n在 {len(raw_images)} 张原始图像上测试:")
    print("-" * 60)
    success = 0
    total_ms = 0.0
    for img_path in raw_images:
        res = infer_single(model, img_path, device, train_size=train_size,
                           long_side=long_side, use_fallback=use_fallback)
        status = "✓" if res["ok"] else "✗"
        text_disp = res["text"] if res["text"] else "—"
        match = " (匹配)" if res["ok"] else ""
        elapsed_ms = res.get("elapsed_ms", 0.0)
        total_ms += elapsed_ms
        method = res.get("method") or "-"
        err = res.get("error")
        err_disp = f" [{err}]" if err else ""
        print(f"  {status} {img_path.name}: {text_disp}{match} "
              f"({elapsed_ms:.0f}ms, {method}){err_disp}")
        if res["ok"]:
            success += 1
    print(f"\n原始图像识别率: {success}/{len(raw_images)}")
    print(f"平均耗时: {total_ms / len(raw_images):.0f} ms/张")


def visualize_prediction(image: np.ndarray, prob_map: np.ndarray,
                         result_dict: dict, save_path: Path):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    prob_color = cv2.applyColorMap(
        (np.clip(prob_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(vis, 0.6, prob_color, 0.4, 0)
    if result_dict.get("bbox") is not None:
        box = np.array(result_dict["bbox"], dtype=np.int32)
        cv2.drawContours(overlay, [box], 0, (0, 255, 0), 2)
        text = result_dict.get("text", "")
        if text:
            x = int(box[:, 0].min())
            y = int(box[:, 1].min()) - 10
            cv2.putText(overlay, text, (x, max(y, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    h, w = overlay.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        overlay = cv2.resize(overlay, None, fx=s, fy=s,
                             interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(save_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])